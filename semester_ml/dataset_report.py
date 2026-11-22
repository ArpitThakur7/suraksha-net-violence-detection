"""Summarise actual clip metadata, source grouping and frame-selection methods."""
import argparse
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import *
from video_data import keyframes, read_images, sample_positions, augment_clip

def generate(rwf):
    rows=source_rows(rwf);out=OUT/("three_sources" if rwf else "two_sources")/"dataset";out.mkdir(parents=True,exist_ok=True)
    ds=sorted({r["dataset"] for r in rows})
    counts=Counter((r["dataset"],r["split"],int(r["label"])) for r in rows)
    summary=[dict(dataset=d,split=s,nonviolence=counts[d,s,0],violence=counts[d,s,1],
                  violence_class_weight=(counts[d,s,0]+counts[d,s,1])/(2*counts[d,s,1]),
                  nonviolence_class_weight=(counts[d,s,0]+counts[d,s,1])/(2*counts[d,s,0])) for d in ds for s in ("train","validation","test")]
    save_csv(out/"class_balance.csv",summary)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    normal=[sum(int(r["label"])==0 and r["dataset"]==d for r in rows) for d in ds]
    fight=[sum(int(r["label"])==1 and r["dataset"]==d for r in rows) for d in ds]
    axes[0,0].bar(ds,normal,label="Nonviolence");axes[0,0].bar(ds,fight,bottom=normal,label="Violence")
    axes[0,0].set_ylabel("Clips");axes[0,0].legend()
    for a,feature,label in [(axes[0,1],"fps","FPS"),(axes[1,0],"duration_s","Duration (seconds)"),(axes[1,1],"width","Frame width (pixels)")]:
        vals=[[float(r[feature]) for r in rows if r["dataset"]==d] for d in ds]
        a.hist(vals,bins=14,label=ds);a.set(xlabel=label,ylabel="Clips");a.legend()
    fig.savefig(out/"dataset_statistics.png",dpi=130);plt.close(fig)
    comparisons=[];rng=np.random.default_rng(42)
    fig,axes=plt.subplots(len(ds),3,figsize=(9,3*len(ds)),squeeze=False,layout="constrained")
    for di,d in enumerate(ds):
        train=[r for r in rows if r["split"]=="train" and r["dataset"]==d]
        for j in rng.choice(len(train),min(10,len(train)),replace=False):
            r=train[j];n=int(r["frames"]);path=ROOT/r["path"]
            uniform=indices(n,16);motion=keyframes(path)
            comparisons.append(dict(dataset=d,path=r["path"],uniform_indices=uniform.tolist(),motion_indices=motion.tolist(),
                                    shared_indices=len(set(uniform)&set(motion)),unique_motion_frames=len(set(motion))))
        ri=comparisons[-1];p=ROOT/ri["path"];u=read_images(p,ri["uniform_indices"],224);m=read_images(p,ri["motion_indices"],224)
        for a,img,title in zip(axes[di],[u[8],m[8],augment_clip([u[8]],rng)[0]],["Uniform frame","Motion-selected frame","Training augmentation"]):
            a.imshow(img);a.axis("off");a.set_title(d.upper()+" · "+title)
    save_csv(out/"sampling_comparison.csv",comparisons);fig.savefig(out/"frame_examples.png",dpi=125);plt.close(fig)
    print("Summarised",len(rows),"clips.",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--rwf",action="store_true");generate(p.parse_args().rwf)

