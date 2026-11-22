"""Build a reproducible release from a validation-selected experiment."""
import argparse
import csv
import shutil
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from core import *
from infer import load_model,release_logits
from train_suite import metrics


def plot_curves(rows, predictions, threshold, out):
    fig, ax = plt.subplots(2,2, figsize=(10,8), layout="constrained")
    for a,field in zip(ax[0],["accuracy","loss"]):
        for split in ["train","validation"]:
            a.plot([int(r["epoch"]) for r in rows], [float(r[split+"_"+field]) for r in rows], marker="o",label=split)
        a.set(xlabel="Epoch",ylabel=field.capitalize()); a.legend()
    y=[int(r["label"]) for r in predictions]; p=[float(r["prob"]) for r in predictions]
    fpr,tpr,_=roc_curve(y,p); m=metrics(y,p,threshold)
    ax[1,0].plot(fpr,tpr,label=f"Test AUC {m['roc_auc']:.3f}"); ax[1,0].plot([0,1],[0,1],"--",color="grey")
    ax[1,0].set(xlabel="False positive rate",ylabel="Violence recall");ax[1,0].legend()
    matrix=np.array([[m["tn"],m["fp"]],[m["fn"],m["tp"]]])
    a=ax[1,1];a.imshow(matrix,cmap="Blues")
    for (i,j),v in np.ndenumerate(matrix): a.text(j,i,str(v),ha="center",va="center",color="white" if v>matrix.max()*.55 else "black")
    a.set(xticks=[0,1],yticks=[0,1],xticklabels=["Nonviolence","Violence"],yticklabels=["Nonviolence","Violence"],xlabel="Predicted class",ylabel="Actual class")
    fig.savefig(out/"evaluation.png",dpi=145);plt.close(fig)


def cam(model,item,row,kind,out):
    array=frames(ROOT/row["path"])
    images=array if item["config"]["backbone"]=="resnet18" else array[indices(32,16)]
    conv=next(m for m in reversed(list(model[0].modules())) if isinstance(m,nn.Conv2d))
    acts=[]
    hook=conv.register_forward_hook(lambda module,inputs,output: acts.append(output))
    try:
        # cuDNN's eval-mode LSTM cannot backpropagate, so use the native operation here.
        with torch.backends.cudnn.flags(enabled=False):
            logits=release_logits(model,item,array)
            activation=acts[0]
            gradient,=torch.autograd.grad(logits[0,int(row["prediction"])],activation)
            maps=(gradient.mean((2,3),keepdim=True)*activation).sum(1).clamp_min(0).detach().cpu().numpy()
    finally:
        hook.remove()
    i=int(np.argmax(maps.max(axis=(1,2))))
    fig,axes=plt.subplots(1,2,figsize=(6,3),layout="constrained")
    for a in axes: a.imshow(images[i]);a.axis("off")
    axes[1].imshow(maps[i],cmap="inferno",alpha=.48,extent=(-.5,SIZE-.5,SIZE-.5,-.5),interpolation="bilinear")
    axes[0].set_title("Model input");axes[1].set_title("Predicted-class Grad-CAM")
    fig.suptitle(f"{kind}  |  {row['source'].upper()}  |  P(violence) {float(row['prob']):.3f}")
    fig.savefig(out/(kind+".png"),dpi=135);plt.close(fig)
    return dict(kind=kind,backbone_frame_index=i,**row)


def generate(suite):
    s=OUT/suite; release=s/"release";release.mkdir(exist_ok=True)
    selection=json.loads((s/"selection.json").read_text());name=selection["name"]; run=s/name
    model,item=load_model(run/"inference.pth")
    t=item["threshold"];config=item["config"]; rows=read_csv(run/"own_predictions.csv")
    shutil.copy2(run/"inference.pth",release/"best_model.pth")
    write_json(release/"model_config.json",config)
    write_json(release/"preprocessing_config.json",dict(color="RGB",size=[SIZE,SIZE],base_frames=32,length=config["length"],sampling=item["input_contract"],mean=MEAN,std=STD))
    write_json(release/"class_mapping.json",{"0":"Non-Violence","1":"Violence"})
    write_json(release/"recommended_threshold.json",dict(threshold=t,selection=selection,manifest=json.loads((s/"manifest.json").read_text())["digest"]))
    plot_curves(read_csv(run/"epochs.csv"),rows,t,release)
    bysource={}
    for source in sorted({r["source"] for r in rows} | {"all"}):
        sub=[r for r in rows if source=="all" or r["source"]==source]
        bysource[source]=metrics([int(r["label"]) for r in sub],[float(r["prob"]) for r in sub],t)
    write_json(release/"test_by_source.json",bysource)
    labels={}
    for value,file in enumerate(["nonviolent-action-classes.csv","violent-action-classes.csv"]):
        with (ROOT/"datasets/public/avd/violence-detection-dataset"/file).open(encoding="utf-8-sig") as f:
            for r in list(csv.reader(f,delimiter=";"))[1:]:
                if len(r)>=2: labels[value,r[0].strip()]=r[1].strip()
    cases=[]; errors=[]
    for actual,predicted,kind in [(1,1,"true_positive"),(0,0,"true_negative"),(0,1,"false_positive"),(1,0,"false_negative")]:
        sub=[r for r in rows if int(r["label"])==actual and int(r["prediction"])==predicted]
        if sub:cases.append(cam(model,item,sub[0],kind,release))
        if actual!=predicted:
            for r in sub:
                action=labels.get((actual,Path(r["path"]).name),"") if r["source"]=="avd" else ""
                errors.append(dict(kind=kind,annotated_action=action,**r))
    save_csv(release/"gradcam_examples.csv",cases);save_csv(release/"errors.csv",errors)
    # Re-decode every final test clip with the exported checkpoint. Batch-one kernels
    # can differ slightly from batched cached-feature evaluation on the same GPU.
    parity=[]
    reloaded,config=load_model(release/"best_model.pth")
    for r in rows:
        with torch.inference_mode():observed=float(release_logits(reloaded,config,frames(ROOT/r["path"])).softmax(1)[0,1])
        delta=abs(observed-float(r["prob"]))
        parity.append(dict(source=r["source"],path=r["path"],cached=float(r["prob"]),decoded=observed,absolute_difference=delta,
                           decision_changed=int(observed>=t)!=int(r["prediction"])))
    save_csv(release/"checkpoint_parity.csv",parity)
    if max(p["absolute_difference"] for p in parity)>.01 or any(p["decision_changed"] for p in parity):
        raise RuntimeError("Cached test predictions differ materially from deployment; see checkpoint_parity.csv")
    key=bysource["all"]
    metadata=source_rows(suite=="three_sources")
    manifest=[{k:r[k] for k in ("path","dataset","event_group","sha256","label","frames","split")} for r in metadata]
    for split in ["train","validation","test"]:save_csv(release/(split+"_manifest.csv"),[r for r in manifest if r["split"]==split])
    costs=[]
    for result in read_csv(s/"test_results.csv"):
        ck=s/result["run"]/"inference.pth"
        if ck==run/"inference.pth": net=model
        else:net,_=load_model(ck)
        costs.append(dict(run=result["run"],parameters=sum(p.numel() for p in net.parameters()),mib=round(ck.stat().st_size/2**20,2)))
        del net
    save_csv(release/"model_sizes.csv",costs)
    val=json.loads((run/"validation.json").read_text());epochs=read_csv(run/"epochs.csv");best=epochs[val["best_epoch"]-1]
    details=["# Semester model evaluation","",f"Selected **{name}** on validation F1 with threshold **{t:.2f}**. The training suite used one seed (42) and ImageNet backbones. The best epoch was {val['best_epoch']}. Training accuracy at that epoch was {float(best['train_accuracy']):.1%} and validation accuracy at a 0.50 cutoff was {float(best['validation_accuracy']):.1%}. Repeated-seed variability was not measured.","",
    "| Test source | Clips | Accuracy | Violence recall | Precision | F1 | ROC-AUC | False alarms | Misses |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for source,m in bysource.items():
        details.append(f"| {source.upper()} | {m['n']} | {m['accuracy']:.1%} | {m['recall']:.1%} | {m['precision']:.1%} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['fp']} | {m['fn']} |")
    details.extend(["","Each prediction covers one labeled clip. Overlapping CCTV windows and incident notifications require separate measurements. Multiple AIRTLab views of an event stay together. SFD source recording IDs stay together. The test set is shared across prespecified comparisons and the release uses the validation-selected winner.","",
    "See ../validation_comparison.csv for the ablation and hyperparameter table, ../generalization.csv for transfer tests and ../test_results.csv for baselines. evaluation.png plots training, validation, ROC and the test confusion matrix. model_sizes.csv compares artifact sizes and parameters. errors.csv lists every mistake with publisher action labels where available. The four Grad-CAM examples use the predicted class and the frame with highest activation. Scene conditions such as occlusion or camera shake require human annotation.","",
    "Run from the project root with the project Python environment:","",f"`python semester_ml/infer.py review/vishnu/sample_video.mp4 --checkpoint ml_runs/{suite}/release/best_model.pth`",""])
    (release/"REPORT.md").write_text("\n".join(details),encoding="utf-8")
    print(release, key,flush=True)

if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("suite",choices=["two_sources","three_sources"])
    generate(arg.parse_args().suite)

