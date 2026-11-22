"""Exercise camera replay, durable delivery, independent splits and the local review API."""
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from core import *
sys.path.insert(0,str(ROOT))
from suraksha import cameras
from fastapi.testclient import TestClient


def main():
    out=ROOT/"optimization/camera_replay";out.mkdir(exist_ok=True,parents=True)
    clips=["datasets/public/avd/violence-detection-dataset/violent/cam1/1.mp4",
           "datasets/public/avd/violence-detection-dataset/non-violent/cam1/1.mp4","datasets/public/sfd/noFight/nofi001.mp4"]
    import psutil
    proc=psutil.Process()
    scale=[];comparisons=[];checks={}
    with tempfile.TemporaryDirectory(dir=out) as temp:
        temp=Path(temp);os.environ["SURAKSHA_STORAGE"]=str(temp/"database")
        from suraksha.app import app
        from suraksha.video import profile
        args=cameras.parser().parse_args([])
        # Short replay clips use a 1-second window and a 2-of-3 rule to exercise alert delivery.
        args.window=1;args.stride=.4;args.votes=2;args.history=3;args.cooldown=30
        args.device="cuda" if torch.cuda.is_available() else "cpu"
        for num,skip in [(1,1),(2,1),(3,1),(3,2),(3,3)]:
            args.skip=skip
            args.camera=[(f"camera-{i+1}",f"Replay {i+1}",str(ROOT/p)) for i,p in enumerate(clips[:num])]
            args.output=temp/f"scale-{num}-{skip}"
            start=time.perf_counter();cpu=proc.cpu_times()
            results=cameras.run(args);wall=time.perf_counter()-start;end=proc.cpu_times()
            assert all(r["frames"] and r["windows"] for r in results)
            scale.append(dict(cameras=num,skip=skip,device=args.device,seconds=wall,aggregate_input_fps=sum(r["frames"] for r in results)/wall,
                              windows=sum(r["windows"] for r in results),alerts=sum(r["alerts"] for r in results),
                              process_cpu_cores=(end.user+end.system-cpu.user-cpu.system)/wall,process_rss_mib=proc.memory_info().rss/2**20))
            if (num,skip)==(3,1):
                for r in results:
                    rows=read_csv(Path(r["log"]));shutil_path=out/(r["camera"]+".csv")
                    import shutil
                    shutil.copyfile(r["log"],shutil_path)
                    for mode in ["majority","average","consecutive"]:
                        history=deque(maxlen=args.history);count=0
                        for row in rows:
                            history.append(float(row["score"]));count+=int(cameras.decision(history,mode,args.threshold,args.votes))
                        comparisons.append(dict(camera=r["camera"],mode=mode,windows=len(rows),confirmed_windows=count))
        expected=sorted((temp/"scale-3-1/outbox").glob("*.json"))
        assert expected,"Reference replay must produce a confirmed incident"
        folder=expected[0].parent
        # Simulate a disconnected upload, then deliver the same IDs to the actual API.
        class Disconnected:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def post(self,*a,**k):raise cameras.httpx.ConnectError("offline")
        with patch.object(cameras.httpx,"Client",return_value=Disconnected()):
            assert all(v=="ConnectError" for _,v in cameras.deliver(folder,"/api/alert"))
        assert all(p.exists() for p in expected)
        checks["offline_buffer"]=len(expected)
        with TestClient(app) as client:
            with patch.object(cameras.httpx,"Client",return_value=NoClose(client)):
                assert all(v=="delivered" for _,v in cameras.deliver(folder,"/api/alert"))
            records=client.get("/api/alerts").json();assert len(records)==len(expected)
            sample=json.loads((folder/"delivered"/expected[0].name).read_text())
            assert client.post("/api/alert",json=sample).status_code==200
            assert client.post("/api/alert",json=sample|{"location":"Other"}).status_code==409
            assert client.post("/api/alert",json=sample|{"snapshot":"bad!"}).status_code==422
            import uuid
            id=str(uuid.UUID(sample["incident_id"]))
            assert client.get(f"/api/alerts/{id}/snapshot").content[:2]==bytes([255,216])
            for state in ["Acknowledged","Resolved","False alarm"]:
                assert client.patch(f"/api/alerts/{id}",json={"status":state}).json()["status"]==state
            with patch.dict(os.environ,SURAKSHA_API_KEY="fixture"):
                assert client.post("/api/alert",json=sample).status_code==401
                assert client.post("/api/alert",json=sample,headers={"X-API-Key":"fixture"}).status_code==200
            checks.update(delivery_and_retry=True,duplicate_and_conflict=True,snapshot_and_validation=True,status_and_api_key=True)
            for release in ["","ml_runs/two_sources/release"]:
                if release:os.environ["SURAKSHA_MODEL_RELEASE"]=str(ROOT/release)
                else:os.environ.pop("SURAKSHA_MODEL_RELEASE",None)
                profile.cache_clear()
                c=client.get("/api/detector").json()
                with (ROOT/"review/vishnu/sample_video.mp4").open("rb") as f:
                    result=client.post("/api/videos",files={"file":("handoff.mp4",f,"video/mp4")})
                assert result.status_code==201,result.text
                body=result.json();assert body["threshold"]==c["threshold"] and body["model"]==c["model"] and body["windows"]
                checks[c["model"]]=dict(threshold=c["threshold"],windows=len(body["windows"]))
            assert client.post("/api/videos",files={"file":("bad.mp4",b"broken","video/mp4")}).status_code==422
    save_csv(out/"scaling.csv",scale);save_csv(out/"smoothing.csv",comparisons);write_json(out/"integration.json",checks)
    print(checks,flush=True)


class NoClose:
    def __init__(self,client):self.client=client
    def __enter__(self):return self.client
    def __exit__(self,*args):pass

if __name__=="__main__":main()

