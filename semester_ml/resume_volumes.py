"""Resume Zenodo's range-capable release without discarding interrupted downloads."""
import hashlib
import http.client
import itertools
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fetch_rwf import BASE,ROOT

CHUNK=16*2**20

def verified(path,item):
    if not path.is_file() or path.stat().st_size!=item["size"]:return False
    with path.open("rb") as f:return hashlib.file_digest(f,"md5").hexdigest()==item["checksum"].split(":")[-1]

def request_part(job):
    item,start,end,path=job
    if path.is_file() and path.stat().st_size==end-start+1:return
    tmp=path.with_suffix(".working")
    for attempt in range(8):
        offset=start+(tmp.stat().st_size if tmp.exists() else 0)
        try:
            if offset<=end:
                req=urllib.request.Request(BASE+item["key"],headers={"Range":f"bytes={offset}-{end}"})
                with urllib.request.urlopen(req,timeout=55) as response:
                    want=f"bytes {offset}-{end}/{item['size']}"
                    if response.status!=206 or response.headers.get("Content-Range")!=want:
                        raise IOError("Unexpected range response")
                    with tmp.open("ab") as f:shutil.copyfileobj(response,f,256*1024)
            if tmp.stat().st_size!=end-start+1:raise IOError("Truncated range response")
            tmp.replace(path);return
        except (OSError,EOFError,http.client.HTTPException,urllib.error.URLError) as e:
            if attempt==7:raise
            wait=int(e.headers.get("Retry-After","5")) if isinstance(e,urllib.error.HTTPError) and e.code==429 else min(2**attempt,20)
            time.sleep(wait)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    metadata=json.load(urllib.request.urlopen("https://zenodo.org/api/records/15687512",timeout=30))
    items=sorted(metadata["files"],key=lambda x:x["key"])
    if [i["key"] for i in items]!=[f"RWF-2000.7z.{n:03}" for n in range(1,14)]:raise ValueError("Unexpected release")
    plans=[];chunks=ROOT/"ranges";chunks.mkdir(exist_ok=True)
    for item in items:
        dest=ROOT/item["key"]
        if verified(dest,item):continue
        candidates=[dest.with_suffix(dest.suffix+ext) for ext in (".partial",".prefetch")]
        prefix=max((p for p in candidates if p.exists() and p.stat().st_size<=item["size"]),key=lambda p:p.stat().st_size,default=None)
        offset=prefix.stat().st_size if prefix else 0
        parts=[(item,i,min(item["size"]-1,i+CHUNK-1),chunks/f"{dest.name}.{i}.part") for i in range(offset,item["size"],CHUNK)]
        plans.append((item,prefix,parts))
    jobs=[j for group in itertools.zip_longest(*(p[2] for p in plans)) for j in group if j]
    print("Remaining ranged data (GiB)",round(sum(j[2]-j[1]+1 for j in jobs)/2**30,2),"chunks",len(jobs),flush=True)
    with ThreadPoolExecutor(7) as workers:
        for n,_ in enumerate(workers.map(request_part,jobs),1):
            if n%15==0 or n==len(jobs):print("Verified range lengths",n,"/",len(jobs),flush=True)
    for item,prefix,parts in plans:
        dest=ROOT/item["key"];joined=dest.with_suffix(dest.suffix+".assembled")
        with joined.open("wb") as out:
            for p in ([prefix] if prefix else [])+[j[3] for j in parts]:
                with p.open("rb") as f:shutil.copyfileobj(f,out,1024*1024)
        if not verified(joined,item):raise IOError("Checksum mismatch: "+item["key"])
        joined.replace(dest)
    subprocess.run([r"C:\Program Files\7-Zip\7z.exe","x",str(ROOT/items[0]["key"]),"-o"+str(ROOT/"clips"),"-aos","-r","*.avi","-bso0"],check=True)
    count=sum(1 for p in (ROOT/"clips").rglob("*.avi"))
    if count!=2000:raise ValueError(f"Release contains {count} AVI paths; inspect before training")
    print("Verified all 13 MD5 checksums and extracted",count,"clips.",flush=True)
if __name__=="__main__":main()

