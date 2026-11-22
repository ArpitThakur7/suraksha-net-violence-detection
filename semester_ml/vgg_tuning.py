"""Controlled SFD VGG16 comparison at three backbone fine-tuning depths."""
import numpy as np
import torch
from torch import nn
from core import *
from train_suite import seed,metrics


def main():
    torch.set_num_threads(3)
    rows=[r for r in source_rows(False) if r["dataset"]=="sfd" and r["split"]!="test"]
    base,_=backbone("vgg16"); features=base[0]
    prefix=features[:17].to(DEVICE).eval(); pooled=[]
    for i,r in enumerate(rows):
        x=normalize(frames(ROOT/r["path"],int(r["frames"]))[indices(32,8)])
        with torch.inference_mode(): pooled.append(prefix(x).half().cpu())
    del prefix
    data=torch.stack(pooled); del pooled
    y=torch.tensor([int(r["label"]) for r in rows],device=DEVICE)
    tr=torch.tensor([i for i,r in enumerate(rows) if r["split"]=="train"])
    val=torch.tensor([i for i,r in enumerate(rows) if r["split"]=="validation"])
    table=[]
    for mode,cut in [("frozen",31),("last_block",24),("last_two_blocks",17)]:
        seed(42)
        import copy
        tail=copy.deepcopy(features[17:]).to(DEVICE)
        for i,layer in enumerate(tail):
            layer.requires_grad_(i+17>=cut)
        head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.3),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,2)).to(DEVICE)
        opt=torch.optim.AdamW([{"params":list(tail.parameters()),"lr":1e-5},{"params":head.parameters(),"lr":.001}],weight_decay=.0001)
        history=[];best=(-1.,-1.); dest=OUT/"vgg_fine_tuning"/mode;dest.mkdir(parents=True,exist_ok=True)
        def forward(ids):
            x=data[ids].to(DEVICE,dtype=torch.float32)
            return head(tail(x.flatten(0,1))).reshape(len(ids),8,2).mean(1)
        for epoch in range(1,5):
            head.train();tail.train();loss_sum=0.
            for ids in tr[torch.randperm(len(tr))].split(6):
                opt.zero_grad(); logits=forward(ids);loss=nn.functional.cross_entropy(logits,y[ids])
                loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ids)
            head.eval();tail.eval()
            with torch.no_grad():
                prob=torch.cat([forward(b).softmax(1)[:,1] for b in val.split(6)]).cpu().numpy()
            m=metrics(y[val].cpu().numpy(),prob,.5)
            history.append(dict(epoch=epoch,training_loss=loss_sum/len(tr),**m))
            if (m["f1"],m["roc_auc"])>best:
                best=(m["f1"],m["roc_auc"]);bestrow=history[-1]
                torch.save(dict(tail=tail.state_dict(),head=head.state_dict(),mode=mode,frames=8,epoch=epoch),dest/"best_tail.pt")
        save_csv(dest/"validation.csv",history);table.append(dict(mode=mode,**bestrow))
        print("SFD VGG comparison",mode,round(best[0],3),flush=True)
    save_csv(OUT/"vgg_fine_tuning/comparison.csv",table)

if __name__=="__main__":
    main()
