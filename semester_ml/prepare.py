"""Cache fixed ImageNet features for transfer learning; keep splits in external manifests."""
import argparse
import hashlib
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import torch
from core import *

RWF = Path(os.environ.get("SURAKSHA_RWF_ROOT",r"F:\RWF 2000\suraksha_public_release\clips"))


def inventory_rwf():
    files = sorted(RWF.rglob("*.avi"))
    if not files:
        raise RuntimeError(f"No RWF videos found in {RWF}")
    def inspect(p):
        parts=[s.lower() for s in p.parts]
        upstream=next((s for s in reversed(parts[:-1]) if s in ("train","val","test")),None)
        if upstream is None or p.parent.name.lower() not in ("fight","nonfight"):
            raise ValueError(f"Unrecognized upstream layout: {p}")
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        count=0; issue=""; fps=0.;width=height=0
        try:
            with av.open(str(p)) as c:
                tr=c.streams.video[0]; tr.codec_context.thread_count=1
                fps=float(tr.average_rate or 0);width=tr.width;height=tr.height
                count=sum(1 for _ in c.decode(tr))
        except (av.FFmpegError, IndexError, OSError) as exc:
            issue=type(exc).__name__
        # RWF appends a clip number to the source recording ID (including IDs with underscores).
        group = re.split(r"[_-](?:\d{1,2}[-:]\d{2}[-:]\d{2}|x264)",p.stem,flags=re.I)[0]
        group = re.sub(r"_\d+$","",group)
        return dict(path=str(p),dataset="rwf",label=int(p.parent.name.lower()=="fight"),event_group="rwf:"+group,
                    sha256=sha,frames=count,fps=fps,width=width,height=height,duration_s=count/fps if fps else 0,
                    split="train" if upstream=="train" else "test",
                    exclusion=issue or ("" if count and fps else "empty or missing fps"))
    with ThreadPoolExecutor(4) as pool:
        rows=list(pool.map(inspect,files))
    holdout={r["event_group"] for r in rows if r["split"]=="test"}
    counts=defaultdict(int)
    for r in rows: counts[r["sha256"]]+=1
    existing=source_rows(False)
    prior={r["sha256"] for r in existing}
    sfd_sources={r["event_group"].removeprefix("sfd:") for r in existing if r["dataset"]=="sfd"}
    for r in rows:
        if counts[r["sha256"]]>1 or r["sha256"] in prior or (r["split"]=="train" and r["event_group"] in holdout):
            r["exclusion"]="duplicate or overlaps existing holdout"
        elif r["event_group"].removeprefix("rwf:") in sfd_sources:
            r["exclusion"]="source recording already represented in SFD"
    for label in (0,1):
        groups=sorted({r["event_group"] for r in rows if r["split"]=="train" and r["label"]==label and not r["exclusion"]})
        random.Random(42+label).shuffle(groups)
        chosen=set(groups[:round(len(groups)*.15)])
        for r in rows:
            if r["split"]=="train" and r["event_group"] in chosen: r["split"]="validation"
    save_csv(OUT/"rwf_manifest.csv",rows)


@torch.inference_mode()
def cache(rows):
    torch.set_num_threads(3)
    nets={name:backbone(name)[0].to(DEVICE).eval() for name in ("resnet18","vgg16","efficientnet")}
    (OUT/"features").mkdir(parents=True,exist_ok=True)
    todo=[r for r in rows if not cache_path(r).exists()]
    def decode(r):
        return frames(ROOT/r["path"],int(r["frames"]))
    # Bounded read-ahead keeps decoded video buffers small.
    with ThreadPoolExecutor(3) as pool:
        for offset in range(0,len(todo),12):
            batch=todo[offset:offset+12]
            for r,images in zip(batch,pool.map(decode,batch)):
                x=normalize(images)
                values={}
                for name,net in nets.items():
                    data=x if name=="resnet18" else x[indices(32,16)]
                    values[name]=net(data).float().cpu().numpy()
                    if name=="resnet18" and r["split"]=="train":
                        values["flip"]=net(data.flip(-1)).float().cpu().numpy()
                dest=cache_path(r)
                with dest.with_suffix(".tmp").open("wb") as f: np.savez(f,**values)
                dest.with_suffix(".tmp").replace(dest)
            if offset%120==0 or offset+12>=len(todo):
                print("Cached",min(offset+12,len(todo)),"/",len(todo),flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--rwf",action="store_true")
    opts=arg.parse_args()
    if opts.rwf and not (OUT/"rwf_manifest.csv").exists(): inventory_rwf()
    cache(source_rows(opts.rwf))
