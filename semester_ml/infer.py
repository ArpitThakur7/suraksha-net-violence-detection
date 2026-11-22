"""Score a recorded video with a clean-run checkpoint or the historical checkpoint."""
import argparse
import json
from core import *


def load_model(path):
    item=torch.load(path,map_location="cpu",weights_only=True)
    if "backbone" in item and "head" in item:
        c=item["config"];base,_=backbone(c["backbone"],False); head=Head(c)
        base.load_state_dict(item["backbone"]); head.load_state_dict(item["head"])
        return nn.Sequential(base,nn.Unflatten(0,(1,c["length"])),head).to(DEVICE).eval(),item
    sys.path.insert(0,str(ROOT))
    from suraksha.model import CheckpointCNNLSTM
    model=CheckpointCNNLSTM();model.load_state_dict(item["model_state_dict"])
    return model.to(DEVICE).eval(),item


def release_logits(model,item,array):
    config=item["config"]
    x=normalize(array)
    if config["backbone"]!="resnet18":x=x[indices(32,16)]
    features=model[0](x)
    return model[2](features[indices(len(features),config["length"])].unsqueeze(0))


def score(path,checkpoint):
    model,item=load_model(checkpoint)
    with torch.inference_mode():
        if "head" in item:
            logits=release_logits(model,item,frames(path));threshold=float(item["threshold"])
        else:
            # Reuse the verified original checkpoint's exact clip preprocessing.
            from suraksha.video import window_tensor
            with av.open(str(path)) as c:
                rgb=[(i,np.asarray(frame.to_image().convert("RGB").resize((224,224),Image.Resampling.BILINEAR))) for i,frame in enumerate(c.decode(video=0))]
            if not rgb: raise ValueError("No video frames")
            n=item["config"]["normalization"]
            x=window_tensor(rgb,torch.tensor(n["mean"]).view(1,3,1,1),torch.tensor(n["std"]).view(1,3,1,1))
            logits=model(x.to(DEVICE))
            threshold=json.loads((ROOT/"review/vishnu/recommended_threshold.json").read_text())["recommended_threshold"]
    prob=float(logits.softmax(1)[0,1])
    return dict(video=str(path),violence_score=prob,threshold=threshold,label="Violence" if prob>=threshold else "Non-Violence")

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("video",type=Path)
    args.add_argument("--checkpoint",type=Path,required=True)
    opts=args.parse_args();print(json.dumps(score(opts.video,opts.checkpoint),indent=2))
