"""Export Vishnu's exact checkpoint and measure clip decisions on the 97 validation videos."""
import copy
import os
import time
import statistics
import onnxruntime as ort
import psutil
from torch.nn.utils import prune
from core import *
import onnx
from train_suite import metrics
sys.path.insert(0,str(ROOT))
from suraksha.video import detector, window_tensor

DEST=ROOT/"optimization/handoff"
THRESHOLD=.4


def tensor(row,mean,std):
    import cv2
    cap=cv2.VideoCapture(str(ROOT/row["path"]));frames=[]
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            frames.append((len(frames),np.asarray(Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR))))
    finally:
        cap.release()
    if not frames:raise ValueError(row["path"])
    return window_tensor(frames,mean,std)


def run():
    DEST.mkdir(parents=True,exist_ok=True)
    model,mean,std=detector()
    rows=read_csv(ROOT/"handoff/validation.csv")
    example=tensor(rows[0],mean,std)
    # Fixed batch-one streaming input; each call classifies one complete sequence.
    onnx_path=DEST/"vishnu_fp32.onnx"
    torch.onnx.export(model,example,onnx_path,opset_version=17,input_names=["frames"],output_names=["logits"],dynamo=False)
    onnx.checker.check_model(str(onnx_path))
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    ort_session=ort.InferenceSession(str(onnx_path),options,providers=["CPUExecutionProvider"])
    models={"eager_cpu":model,"int8_cpu":torch.jit.load(str(ROOT/"optimization/dynamic_int8.torchscript.pt")).eval()}
    files={"eager_cpu":ROOT/"optimization/fp32.torchscript.pt","int8_cpu":ROOT/"optimization/dynamic_int8.torchscript.pt","onnx_cpu":onnx_path}
    experiment_device=DEVICE
    if experiment_device.type=="cuda":
        models["fp32_gpu"]=copy.deepcopy(model).to(DEVICE);files["fp32_gpu"]=files["eager_cpu"]
        models["fp16_gpu"]=copy.deepcopy(model).to(DEVICE).half(); files["fp16_gpu"]=DEST/"gpu_fp16.pt"
        torch.save(models["fp16_gpu"].state_dict(),files["fp16_gpu"])
    for percent in [10,20,30]:
        candidate=copy.deepcopy(model)
        layers=[(m,"weight") for m in candidate.modules() if isinstance(m,(nn.Linear,nn.Conv2d))]
        prune.global_unstructured(layers,prune.L1Unstructured,amount=percent/100)
        for m,k in layers:prune.remove(m,k)
        name=f"prune_{percent}_{experiment_device.type}"
        models[name]=candidate.to(experiment_device);files[name]=DEST/(name+".pt")
        torch.save(candidate.state_dict(),files[name])
    def execute(name,x):
        if name=="onnx_cpu":return torch.from_numpy(ort_session.run(None,{"frames":x.numpy()})[0])
        if name.endswith("_gpu") or name.endswith("_cuda"):
            x=x.to(DEVICE,dtype=torch.float16 if name=="fp16_gpu" else torch.float32)
        return models[name](x).float().cpu()
    names=list(models)+["onnx_cpu"];timings={n:[] for n in names};recs=[];parity=0.;proc=psutil.Process()
    import random
    rng=random.Random(42);rss=[];prep=[]
    if DEVICE.type=="cuda":torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for n in names:
            for _ in range(3):execute(n,example)
        usage=proc.cpu_times();wall=time.perf_counter()
        for r in rows:
            start=time.perf_counter();x=tensor(r,mean,std);prep.append(1000*(time.perf_counter()-start))
            order=names.copy();rng.shuffle(order);outputs={}
            for n in order:
                start=time.perf_counter();outputs[n]=execute(n,x)
                timings[n].append(1000*(time.perf_counter()-start))
            parity=max(parity,float((outputs["onnx_cpu"]-outputs["eager_cpu"]).abs().max()))
            recs.append(dict(path=r["path"],dataset=r["dataset"],label=int(r["label"]),**{n:float(v.softmax(1)[0,1]) for n,v in outputs.items()}))
            rss.append(proc.memory_info().rss)
    wall=time.perf_counter()-wall;end=proc.cpu_times()
    summary=[]
    for n in names:
        t=np.array(timings[n]);m=metrics([r["label"] for r in recs],[r[n] for r in recs],THRESHOLD)
        summary.append(dict(engine=n,threshold=THRESHOLD,mib=files[n].stat().st_size/2**20,mean_ms=t.mean(),p50_ms=np.median(t),p95_ms=np.percentile(t,95),sequences_per_s=1000/t.mean(),changed_vs_fp32=sum((r[n]>=THRESHOLD)!=(r["eager_cpu"]>=THRESHOLD) for r in recs),**m))
    save_csv(DEST/"engines.csv",summary);save_csv(DEST/"clip_predictions.csv",recs)
    write_json(DEST/"execution.json",dict(clips=len(rows),onnx_max_abs_logit_difference=parity,onnx_opset=17,
       peak_process_rss_mib=max(rss)/2**20,process_average_cpu_cores=((end.user+end.system)-(usage.user+usage.system))/wall,
       mean_decode_resize_normalize_ms=np.mean(prep),gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else None,
       note="Resources are process-wide across co-resident engines. Forward timing includes input/output device transfers on GPU. Pruned tensors use dense storage. Historical training overlap unknown."))
    if parity>1e-4:raise RuntimeError(f"ONNX logit parity failed: {parity}")
    print("Verified ONNX parity",parity,"on",len(rows),"clips.",flush=True)

if __name__=="__main__":run()

