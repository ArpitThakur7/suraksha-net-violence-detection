"""Replay files or read up to three live sources, keeping alerts in a local outbox."""
import argparse
import base64
import csv
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

import httpx
import numpy as np
from PIL import Image
import torch
from .video import ROOT, detector, window_tensor, optimized_engine
sys.path.insert(0,str(ROOT/".ml_deps"))
import cv2

def decision(scores,mode,threshold,votes):
    if len(scores)<scores.maxlen:return False
    if mode=="average":return np.mean(scores)>=threshold
    if mode=="consecutive":return all(s>=threshold for s in list(scores)[-votes:])
    return sum(s>=threshold for s in scores)>=votes


def record_event(outbox,camera,location,p,frame):
    identity=uuid4().hex
    ok,jpeg=cv2.imencode(".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    if not ok:raise ValueError("Could not encode snapshot.")
    event=dict(incident_id=identity,camera_id=camera,location=location,timestamp=datetime.now(timezone.utc).isoformat(),
               confidence=p,detection="Violence",snapshot=base64.b64encode(jpeg).decode("ascii"))
    temp=outbox/(identity+".tmp");temp.write_text(json.dumps(event),encoding="utf-8");temp.replace(outbox/(identity+".json"))
    return identity


def deliver(outbox,endpoint):
    headers={"X-API-Key":os.environ["SURAKSHA_API_KEY"]} if os.environ.get("SURAKSHA_API_KEY") else {}
    attempts=[]
    with httpx.Client(timeout=5,trust_env=False,headers=headers) as client:
        for path in sorted(outbox.glob("*.json")):
            try:
                r=client.post(endpoint,json=json.loads(path.read_text(encoding="utf-8")));r.raise_for_status()
                target=outbox/"delivered";target.mkdir(exist_ok=True);path.replace(target/path.name)
                attempts.append((path.stem,"delivered"))
            except (httpx.HTTPError,OSError,ValueError) as exc:
                attempts.append((path.stem,type(exc).__name__))
    return attempts


def capture(spec, args, scorer):
    name,location,source=spec;live=source.isdecimal() or source.lower().startswith(("rtsp://","rtsps://"))
    if source.isdecimal(): cap=cv2.VideoCapture(int(source))
    elif live: cap=cv2.VideoCapture(source,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000])
    else: cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return dict(camera=name,frames=0,windows=0,alerts=0,reason="open failed")
    fps=cap.get(cv2.CAP_PROP_FPS);fps=fps if np.isfinite(fps) and fps>0 else 25.
    buf=deque();scores=deque(maxlen=args.history);last=-float("inf");due=args.window
    start=time.perf_counter();seen=0;count=0;alerts=0;last_scored=-1;reason="end of file"
    logpath=args.output/(name+"-"+uuid4().hex[:8]+".csv")
    cap_time=0.;processing_time=0.;post_time=0.;infer_time=0.
    try:
        with logpath.open("w",newline="",encoding="utf-8") as stream:
            log=csv.writer(stream);log.writerow(["utc","camera","location","end_s","score","fps","forward_ms","smoothed_positive","incident_id"])
            def emit(at):
                nonlocal count,last,alerts,last_scored,post_time,infer_time
                p,elapsed=scorer(buf);infer_time+=elapsed
                t=time.perf_counter();scores.append(p);confirmed=decision(scores,args.smoothing,args.threshold,args.votes)
                identity=""
                if confirmed and at-last>=args.cooldown:
                    identity=record_event(args.output/"outbox",name,location,p,buf[len(buf)//2][1])
                    last=at;alerts+=1
                count+=1;last_scored=seen
                log.writerow([datetime.now(timezone.utc).isoformat(),name,location,round(at,3),p,seen/max(.001,time.perf_counter()-start),
                              elapsed*1000,bool(confirmed),identity]);stream.flush();post_time+=time.perf_counter()-t
            while True:
                t=time.perf_counter();ok,bgr=cap.read();cap_time+=time.perf_counter()-t
                if not ok:
                    if live:reason="stream disconnected or timed out"
                    break
                seen+=1
                at=time.perf_counter()-start if live else (seen-1)/fps
                if (seen-1)%args.skip==0:
                    t=time.perf_counter()
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    buf.append((at,np.asarray(Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR))))
                    while len(buf)>1 and buf[0][0]<at-args.window:buf.popleft()
                    processing_time+=time.perf_counter()-t
                    if at>=due:
                        emit(at);due=at+args.stride
                if args.limit and seen>=args.limit:
                    reason="frame limit";break
            if buf and last_scored!=seen:emit(at)
    finally:cap.release()
    return dict(camera=name,frames=seen,windows=count,alerts=alerts,seconds=time.perf_counter()-start,reason=reason,
                capture_s=cap_time,resize_s=processing_time,model_and_queue_s=infer_time,postprocess_s=post_time,log=str(logpath))


def run(args):
    args.output.mkdir(parents=True,exist_ok=True);outbox=args.output/"outbox";outbox.mkdir(exist_ok=True)
    if not args.camera:
        if args.endpoint:return deliver(outbox,args.endpoint)
        raise ValueError("Supply a camera or an endpoint for outbox retry.")
    if not 1<=len(args.camera)<=3 or len({c[0] for c in args.camera})!=len(args.camera):raise ValueError("Use one to three distinct camera IDs.")
    import re
    if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,60}",c[0]) for c in args.camera):raise ValueError("Invalid camera ID.")
    if not (1<=args.votes<=args.history and args.window>0 and args.stride>0 and args.cooldown>=0 and args.skip>=1 and 0<=args.threshold<=1):raise ValueError("Invalid timing, votes or threshold.")
    model,mean,std=detector();device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA unavailable.")
    import copy
    model=copy.deepcopy(model).to(device).eval() if args.engine is None else None
    optimized=optimized_engine(args.engine) if args.engine else None
    lock=Lock()
    @torch.inference_mode()
    def scorer(frames):
        if optimized is not None:
            t=time.perf_counter()
            with lock:p=optimized(frames)
            return p,time.perf_counter()-t
        x=window_tensor(frames,mean,std)
        t=time.perf_counter()
        with lock:p=float(model(x.to(device)).softmax(1)[0,1])
        return p,time.perf_counter()-t
    from threading import Event,Thread
    stop=Event()
    def send_pending():
        while not stop.is_set():
            deliver(outbox,args.endpoint);stop.wait(3)
    sender=Thread(target=send_pending,daemon=True) if args.endpoint else None
    if sender:sender.start()
    try:
        with ThreadPoolExecutor(len(args.camera)) as pool:
            results=list(pool.map(lambda c:capture(c,args,scorer),args.camera))
    finally:
        stop.set()
        if sender:sender.join()
    if args.endpoint:deliver(outbox,args.endpoint)
    return results

def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--camera",action="append",nargs=3,metavar=("ID","LOCATION","FILE_OR_DEVICE_OR_RTSP"))
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"data"/"cameras")
    p.add_argument("--threshold",type=float,default=.4);p.add_argument("--skip",type=int,default=1)
    p.add_argument("--window",type=float,default=4);p.add_argument("--stride",type=float,default=1)
    p.add_argument("--history",type=int,default=7);p.add_argument("--votes",type=int,default=5)
    p.add_argument("--smoothing",choices=["majority","average","consecutive"],default="majority")
    p.add_argument("--cooldown",type=float,default=30);p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--engine",choices=["torch-cpu","onnx-cpu","cuda-fp16"],help="Override --device with a benchmarked inference engine")
    p.add_argument("--limit",type=int,default=0);p.add_argument("--endpoint",help="Alert intake URL; omit to buffer locally")
    return p

if __name__=="__main__":print(json.dumps(run(parser().parse_args()),indent=2))

