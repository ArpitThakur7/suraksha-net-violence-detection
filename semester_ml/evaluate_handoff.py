"""Measure Vishnu's fixed 0.40 whole-clip classifier on the grouped test manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from core import *
from train_suite import metrics
sys.path.insert(0,str(ROOT))
from suraksha.video import detector

def evaluate(suite):
    exp=OUT/suite;out=exp/"vishnu_handoff";out.mkdir(exist_ok=True)
    model,mean,std=detector();model=model.to(DEVICE)
    rows=[r for r in source_rows(suite=="three_sources") if r["split"]=="test"]
    def decode(r):
        positions=indices(int(r["frames"]),16);needed=set(positions);images={}
        with av.open(str(ROOT/r["path"])) as c:
            s=c.streams.video[0];s.codec_context.thread_count=1
            for i,frame in enumerate(c.decode(s)):
                if i in needed: images[i]=np.asarray(frame.to_image().convert("RGB").resize((224,224),Image.Resampling.BILINEAR))
        if set(images)!=needed:raise ValueError(r["path"])
        return np.stack([images[i] for i in positions])
    predictions=[]
    with ThreadPoolExecutor(3) as pool,torch.inference_mode():
        for start in range(0,len(rows),6):
            batch=rows[start:start+6]
            for r,array in zip(batch,pool.map(decode,batch)):
                x=torch.from_numpy(array).permute(0,3,1,2).float().div_(255)
                score=float(model(((x-mean)/std).unsqueeze(0).to(DEVICE)).softmax(1)[0,1])
                predictions.append(dict(source=r["dataset"],path=r["path"],actual=int(r["label"]),score=score,decision=int(score>=.4)))
    save_csv(out/"predictions.csv",predictions)
    scores=[]
    for source in sorted({r["source"] for r in predictions}|{"all"}):
        sub=[r for r in predictions if source=="all" or r["source"]==source]
        scores.append(dict(source=source,threshold=.4,**metrics([r["actual"] for r in sub],[r["score"] for r in sub],.4)))
    save_csv(out/"fixed_threshold_results.csv",scores);print(scores,flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("suite",choices=["two_sources","three_sources"]);evaluate(a.parse_args().suite)

