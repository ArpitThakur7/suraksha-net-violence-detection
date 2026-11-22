"""Train the prespecified models, select using validation, then unlock final evaluation."""
import argparse
import copy
import csv
import itertools
import math
import random
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from core import *

DEFAULT=dict(backbone="resnet18",length=16,hidden=256,layers=1,dropout=.3,optimizer="adamw",
             lr=.001,augment=False,weighted=False,epochs=12,seed=42,trainable="head")
GRID=[.30,.40,.50,.60,.70,.80,.90]


def metrics(y, prob, threshold):
    y=np.asarray(y); pred=np.asarray(prob)>=threshold
    tp=int(np.sum(pred & (y==1))); tn=int(np.sum(~pred & (y==0)))
    fp=int(np.sum(pred & (y==0))); fn=int(np.sum(~pred & (y==1)))
    div=lambda a,b: a/b if b else 0.
    return dict(accuracy=div(tp+tn,len(y)),precision=div(tp,tp+fp),recall=div(tp,tp+fn),
                sensitivity=div(tp,tp+fn),specificity=div(tn,tn+fp),f1=div(2*tp,2*tp+fp+fn),
                fpr=div(fp,fp+tn),fnr=div(fn,fn+tp),roc_auc=float(roc_auc_score(y,prob)) if len(set(y))==2 else None,
                tp=tp,tn=tn,fp=fp,fn=fn,n=len(y))


def seed(seed_value):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)


class Features:
    def __init__(self,rows):
        self.rows=rows
        self.y=torch.tensor([int(r["label"]) for r in rows],device=DEVICE)
        self.bank={}
        for r in rows:
            with np.load(cache_path(r)) as f:
                for k in ("resnet18","efficientnet","vgg16","flip"):
                    v=f[k] if k in f else f["resnet18"]
                    self.bank.setdefault(k,[]).append(v)
        self.bank={k:torch.from_numpy(np.stack(v)) for k,v in self.bank.items()}

    def select(self,split,sources):
        return torch.tensor([i for i,r in enumerate(self.rows) if r["split"]==split and r["dataset"] in sources],dtype=torch.long)

    def get(self,idx,c,training=False):
        key=c["backbone"]; x=self.bank[key][idx]
        if training and c["augment"]:
            mask=torch.rand(len(idx))<.5
            if mask.any(): x[mask]=self.bank["flip"][idx[mask]]
        x=x[:,indices(x.shape[1],c["length"])]
        return x.to(DEVICE),self.y[idx.to(DEVICE)]


def predict(head, data, idx, c):
    head.eval()
    with torch.inference_mode():
        return np.concatenate([head(data.get(batch,c)[0]).softmax(1)[:,1].cpu().numpy() for batch in idx.split(64)])


def train(c, data, where):
    where.mkdir(parents=True,exist_ok=True)
    seed(c["seed"])
    tr=data.select("train",c["sources"]); val=data.select("validation",c["sources"])
    head=Head(c).to(DEVICE)
    counts=torch.bincount(data.y[tr.to(DEVICE)],minlength=2)
    loss=nn.CrossEntropyLoss(weight=(len(tr)/(2*counts.float())) if c["weighted"] else None)
    if c["optimizer"]=="sgd":
        opt=torch.optim.SGD(head.parameters(),lr=c["lr"],momentum=.9,weight_decay=.0001)
    else:
        opt=(torch.optim.AdamW if c["optimizer"]=="adamw" else torch.optim.Adam)(head.parameters(),lr=c["lr"],weight_decay=.0001)
    schedule=c.get("scheduler","cosine")
    if schedule=="step": scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=4,gamma=.5)
    elif schedule=="plateau": scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=2)
    else:scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c["epochs"])
    best=(-1.,-1.); history=[]; best_state=None;best_epoch=0
    state_path=where/"resume.pt"
    if state_path.exists():
        state=torch.load(state_path,map_location="cpu",weights_only=True)
        if state["config"]!=c:raise ValueError("Configuration differs from saved training state")
        head.load_state_dict(state["head"]);opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        history=state["history"];best=state["best"];best_state=state["best_head"];best_epoch=state["best_epoch"]
        torch.set_rng_state(state["rng"])
        if DEVICE.type=="cuda" and state["cuda_rng"] is not None:torch.cuda.set_rng_state(state["cuda_rng"])
    for epoch in range(len(history)+1,c["epochs"]+1):
        if history and len(history)-best_epoch>=4:break
        head.train(); total=0.; correct=0
        for batch in tr[torch.randperm(len(tr))].split(32):
            x,y=data.get(batch,c,True)
            opt.zero_grad(); logits=head(x); objective=loss(logits,y); objective.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
            total+=float(objective.detach())*len(batch); correct+=int((logits.argmax(1)==y).sum())
        prob=predict(head,data,val,c); labels=data.y[val.to(DEVICE)].cpu().numpy()
        score=metrics(labels,prob,.5); clipped=np.clip(prob,1.e-6,1.-1.e-6)
        vloss=float(np.mean(-labels*np.log(clipped)-(1-labels)*np.log(1-clipped)))
        history.append(dict(epoch=epoch,train_loss=total/len(tr),train_accuracy=correct/len(tr),
                            validation_loss=vloss,validation_accuracy=score["accuracy"],validation_f1=score["f1"]))
        if (score["f1"],score["roc_auc"])>best:
            best=(score["f1"],score["roc_auc"]); best_epoch=epoch
            best_state={k:v.cpu().clone() for k,v in head.state_dict().items()}
        if schedule=="plateau":scheduler.step(vloss)
        else:scheduler.step()
        tmp=state_path.with_suffix(".tmp")
        torch.save(dict(config=c,head=head.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                        history=history,best=best,best_epoch=best_epoch,best_head=best_state,rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if DEVICE.type=="cuda" else None),tmp)
        tmp.replace(state_path)
        if epoch-best_epoch>=4: break
    head.load_state_dict(best_state)
    prob=predict(head,data,val,c)
    labels=data.y[val.to(DEVICE)].cpu().numpy()
    sweep=[dict(threshold=t,**metrics(labels,prob,t)) for t in GRID]
    picked=max(sweep,key=lambda m:(m["f1"],m["recall"],-abs(m["threshold"]-.5)))
    save_csv(where/"epochs.csv",history);save_csv(where/"validation_thresholds.csv",sweep)
    torch.save(dict(config=c,head=best_state),where/"head.pt")
    result=dict(**c,best_epoch=best_epoch,**picked)
    write_json(where/"validation.json",result)
    return result


def plan(sources):
    runs={}
    def add(name,src,**overrides):
        runs[name]=DEFAULT | {"sources":list(src)} | overrides
    for src in sources: add(src+"_vgg16",[src],backbone="vgg16")
    add("combined_vgg16",sources,backbone="vgg16")
    add("combined_lightweight",sources,backbone="efficientnet")
    add("combined_cnn_lstm",sources)
    for field,values in dict(length=[8,32],hidden=[128,512],layers=[2],dropout=[.5],optimizer=["adam","sgd"],lr=[.0001,.00001]).items():
        for value in values: add(f"lstm_{field}_{value}",sources,**{field:value})
    add("lstm_augmentation",sources,augment=True);add("lstm_aug_weights",sources,augment=True,weighted=True)
    for size in (1,2):
        for src in itertools.combinations(sources,size):
            if len(src)<len(sources): add("_".join(src)+"_lstm_transfer",src)
    return runs


def main(rwf,validation_only=False):
    torch.set_num_threads(3)
    src=["sfd","avd"]+(["rwf"] if rwf else [])
    suite=OUT/("three_sources" if rwf else "two_sources")
    rows=source_rows(rwf)
    groups={k:{r["event_group"] for r in rows if r["split"]==k} for k in ("train","validation","test")}
    for a,b in itertools.combinations(groups,2): assert not groups[a]&groups[b]
    assert len({r["sha256"] for r in rows})==len(rows)
    fingerprint=hashlib.sha256(json.dumps([(r["sha256"],r["split"],r["label"]) for r in rows]).encode()).hexdigest()
    manifest=suite/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["digest"]!=fingerprint:raise RuntimeError("Manifest changed; use a new suite directory.")
    write_json(manifest,dict(digest=fingerprint,counts=dict(Counter(r["split"] for r in rows))))
    runs=plan(src)
    data=Features(rows)
    results={}
    for name,c in runs.items():
        path=suite/name/"validation.json"
        if path.exists():
            item=json.loads(path.read_text())
            assert all(item[k]==v for k,v in c.items())
        else:
            item=train(c,data,suite/name)
        results[name]=item
        print(name,"validation F1",round(item["f1"],3),flush=True)
    eligible=[n for n,c in runs.items() if c["sources"]==src]
    winner=max(eligible,key=lambda n:(results[n]["f1"],results[n]["recall"],results[n]["roc_auc"]))
    write_json(suite/"selection.json",dict(name=winner,threshold=results[winner]["threshold"],criterion="validation F1 then recall then ROC-AUC"))
    save_csv(suite/"validation_comparison.csv",[dict(run=n,**r) for n,r in results.items()])
    if validation_only: return
    scores=[]; transfer=[]
    needed={winner,"combined_vgg16","combined_cnn_lstm","combined_lightweight"}|{s+"_vgg16" for s in src}
    for name,c in runs.items():
        if name not in needed and not name.endswith("transfer"):continue
        saved=torch.load(suite/name/"head.pt",weights_only=True);head=Head(c).to(DEVICE);head.load_state_dict(saved["head"])
        if name in needed:
            base,_=backbone(c["backbone"])
            deployment=dict(config=c,head=saved["head"],backbone=base.state_dict(),threshold=results[name]["threshold"],
                            image_size=SIZE,mean=MEAN,std=STD,input_contract="uniform32_then_uniform_length; RGB bilinear; repeat last on short clips")
            torch.save(deployment,suite/name/"inference.pth")
        targets=[("own",c["sources"])] if name in needed else []
        if name.endswith("transfer"):targets+=[(s,[s]) for s in src if s not in c["sources"]]
        for target,datasets in targets:
            ids=data.select("test",datasets)
            p=predict(head,data,ids,c);y=data.y[ids.to(DEVICE)].cpu().numpy();t=results[name]["threshold"]
            m=dict(run=name,trained_on="+".join(c["sources"]),tested_on="+".join(datasets),threshold=t,**metrics(y,p,t))
            (scores if target=="own" else transfer).append(m)
            csvrows=[dict(path=rows[i]["path"],source=rows[i]["dataset"],label=int(yy),prob=float(pp),prediction=int(pp>=t)) for i,yy,pp in zip(ids.tolist(),y,p)]
            save_csv(suite/name/(target+"_predictions.csv"),csvrows)
    save_csv(suite/"test_results.csv",scores);save_csv(suite/"generalization.csv",transfer)
    print("Selected",winner,flush=True)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--rwf",action="store_true")
    args.add_argument("--validation-only",action="store_true")
    options=args.parse_args()
    main(options.rwf,options.validation_only)
