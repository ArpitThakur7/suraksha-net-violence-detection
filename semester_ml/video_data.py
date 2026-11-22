"""Grouped manifest datasets and clip-consistent transforms for PyTorch training."""
import math
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image,ImageEnhance,ImageFilter
from core import ROOT, MEAN, STD, source_rows, av, indices

def sample_positions(count,length,fps,mode,rng,interval=3):
    if count<1 or length<1 or fps<=0 or interval<=0:raise ValueError("Invalid video length, sample length, FPS, or interval")
    if mode=="uniform":return indices(count,length)
    if mode=="consecutive":
        start=(count-length)//2 if count>=length else 0
        return np.minimum(start+np.arange(length),count-1)
    if mode=="random":
        picked=np.sort(rng.choice(count,min(count,length),replace=False))
        return np.pad(picked,(0,length-len(picked)),mode="edge")
    if mode in ("1fps","2fps","interval"):
        step=interval if mode=="interval" else fps/(1 if mode=="1fps" else 2)
        return np.unique(np.minimum(np.round(np.arange(0,count,max(1,step))).astype(int),count-1))
    raise ValueError(mode)


def keyframes(path,length=16,motion_threshold=.07):
    if length<1 or not 0<=motion_threshold<=1:raise ValueError("Invalid keyframe selection")
    motion=[];last=None
    with av.open(str(path)) as c:
        for i,frame in enumerate(c.decode(video=0)):
            gray=np.asarray(frame.to_image().convert("L").resize((64,48)),dtype=np.float32)/255
            score=float(abs(gray-last).mean()) if last is not None else 0.
            motion.append((i,score));last=gray
    if not motion:raise ValueError(f"Empty video: {path}")
    selected=sorted([v for v in motion if v[1]>=motion_threshold],key=lambda v:-v[1])[:length]
    # Include uniform coverage if there are too few motion peaks.
    ids={i for i,s in selected}
    for i in indices(len(motion),length):
        if len(ids)>=length:break
        ids.add(int(i))
    values=sorted(ids)
    return np.pad(values,(0,length-len(values)),mode="edge")


def read_images(path,positions,size):
    wanted=set(positions);images={}
    with av.open(str(path)) as c:
        for i,frame in enumerate(c.decode(video=0)):
            if i in wanted:
                images[i]=frame.to_image().convert("RGB").resize((size,size),Image.Resampling.BILINEAR)
            if len(images)==len(wanted):break
    if set(images)!=wanted:raise ValueError(f"Decoded frames differ from manifest: {path}")
    return [images[i] for i in positions]


def augment_clip(images,rng):
    size=images[0].width
    crop=int(rng.uniform(.84,1)*size)
    x,y=rng.integers(0,size-crop+1,size=2)
    angle=rng.uniform(-7,7);brightness=rng.uniform(.85,1.15);contrast=rng.uniform(.85,1.15)
    flip=rng.random()<.5;blur=rng.uniform(.2,.8) if rng.random()<.2 else 0
    noise=rng.normal(0,2,(size,size,3)) if rng.random()<.2 else None
    result=[]
    for im in images:
        im=im.crop((x,y,x+crop,y+crop)).resize((size,size),Image.Resampling.BILINEAR)
        im=im.rotate(angle,Image.Resampling.BILINEAR)
        if flip:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im=ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(brightness)).enhance(contrast)
        if blur:im=im.filter(ImageFilter.GaussianBlur(blur))
        if noise is not None:im=Image.fromarray(np.clip(np.asarray(im)+noise,0,255).astype(np.uint8))
        result.append(im)
    return result


class ManifestDataset(Dataset):
    def __init__(self,rows,length=16,size=224,sampling="uniform",augmentation=False):
        self.rows=rows;self.length=length;self.size=size;self.sampling=sampling;self.augmentation=augmentation
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];path=ROOT/row["path"];rng=np.random.default_rng(int(torch.randint(2**31,()).item()))
        if self.sampling=="motion":positions=keyframes(path,self.length)
        else:positions=sample_positions(int(row["frames"]),self.length,float(row["fps"]),self.sampling,rng)
        if len(positions)!=self.length:positions=positions[indices(len(positions),self.length)]
        images=read_images(path,positions,self.size)
        if self.augmentation:images=augment_clip(images,rng)
        values=torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0,3,1,2).float()/255
        values=(values-values.new_tensor(MEAN)[None,:,None,None])/values.new_tensor(STD)[None,:,None,None]
        return values,int(row["label"]),row["dataset"],row["path"]


def loaders(include_rwf=False,sources=None,length=16,batch_size=8,training_sampling="uniform",balance="natural",augmentation=False):
    rows=source_rows(include_rwf);result={};rng=torch.Generator().manual_seed(42)
    for split in ("train","validation","test"):
        sub=[r for r in rows if r["split"]==split and (not sources or r["dataset"] in sources)]
        train=split=="train";sampler=None
        if train and balance!="natural":
            if balance not in ("under","over","weighted"):raise ValueError(balance)
            groups=[[i for i,r in enumerate(sub) if int(r["label"])==k] for k in (0,1)]
            if not all(groups):raise ValueError("Both classes are required for balancing.")
            if balance=="under":
                n=min(map(len,groups));selected=[g[i] for g in groups for i in torch.randperm(len(g),generator=rng)[:n].tolist()]
                sub=[sub[i] for i in selected]
            else:
                weights=[1/len(groups[int(r["label"])]) for r in sub]
                sampler=WeightedRandomSampler(weights,2*max(map(len,groups)) if balance=="over" else len(sub),generator=rng)
        ds=ManifestDataset(sub,length=length,sampling=training_sampling if train else "uniform",augmentation=train and augmentation)
        result[split]=DataLoader(ds,batch_size=batch_size,shuffle=train and sampler is None,sampler=sampler,generator=rng)
    return result

