"""Train a single model on manifest-locked training and validation clips."""
import argparse
from core import *
from train_suite import DEFAULT, Features, train

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--dataset",required=True,choices=["avd","sfd","rwf","combined"])
    a.add_argument("--model",required=True,choices=["vgg16","efficientnet","cnn_lstm"])
    a.add_argument("--rwf",action="store_true",help="Include RWF in a combined run")
    a.add_argument("--scheduler",choices=["cosine","plateau","step"],default="cosine");a.add_argument("--epochs",type=int,default=12)
    args=a.parse_args()
    if args.epochs<1:raise ValueError("Use one or more epochs.")
    rwf=args.rwf or args.dataset=="rwf"
    sources=(["sfd","avd"]+(["rwf"] if rwf else [])) if args.dataset=="combined" else [args.dataset]
    rows=[r for r in source_rows(rwf) if r["dataset"] in sources and r["split"]!="test"]
    c=DEFAULT|dict(sources=sources,backbone="resnet18" if args.model=="cnn_lstm" else args.model,
                   scheduler=args.scheduler,epochs=args.epochs)
    output=OUT/"individual"/("_".join(sources)+f"_{args.model}_{args.scheduler}_{args.epochs}")
    torch.set_num_threads(3)
    validation=train(c,Features(rows),output)
    print(output,validation["f1"],validation["threshold"])
if __name__=="__main__":main()

