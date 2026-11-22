import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ml_runs"
sys.path.insert(0, str(ROOT / ".ml_deps"))
os.environ["TORCH_HOME"] = str(OUT / "torch_cache")
from torchvision import models

SIZE = 112
MEAN = [0.485,0.456,0.406]
STD = [0.229,0.224,0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf-8")
    tmp.replace(path)


def source_rows(include_rwf):
    rows = read_csv(ROOT / "handoff/dataset_metadata.csv")
    if include_rwf:
        rows += read_csv(OUT / "rwf_manifest.csv")
    return [r for r in rows if not r.get("exclusion")]


def indices(count, length):
    if count <= 0:
        raise ValueError("Empty video")
    if count < length:
        return np.minimum(np.arange(length), count-1)
    return np.linspace(0,count-1,length,dtype=int)


def frames(path, known_count=None):
    # A single reproducible 32-frame base sequence supports all temporal comparisons.
    with av.open(str(path)) as container:
        track = container.streams.video[0]
        track.codec_context.thread_count = 1
        if known_count is not None and known_count <= 0:
            raise ValueError(f"Empty recording: {path}")
        count = int(known_count if known_count is not None else track.frames or 0)
        if count <= 0:
            count = sum(1 for _ in container.decode(track))
            return frames(path,count) if count else (_ for _ in ()).throw(ValueError("Empty recording"))
        selected = indices(count,32)
        need = set(selected.tolist())
        imgs = {}
        actual = 0
        for i,frame in enumerate(container.decode(track)):
            actual += 1
            if i in need:
                imgs[i] = np.asarray(frame.to_image().convert("RGB").resize((SIZE,SIZE),Image.Resampling.BILINEAR))
    if actual != count:
        return frames(path,actual)
    return np.stack([imgs[i] for i in selected])


def normalize(array, device=DEVICE):
    t=torch.as_tensor(array,device=device).permute(0,3,1,2).float().div_(255)
    return (t-t.new_tensor(MEAN)[None,:,None,None])/t.new_tensor(STD)[None,:,None,None]


def backbone(name, pretrained=True):
    if name == "resnet18":
        base=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(*list(base.children())[:-1],nn.Flatten()),512
    if name == "vgg16":
        base=models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(base.features,nn.AdaptiveAvgPool2d(1),nn.Flatten()),512
    base=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    return nn.Sequential(base.features,base.avgpool,nn.Flatten()),1280


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temporal=config["backbone"]=="resnet18"
        dim=1280 if config["backbone"]=="efficientnet" else 512
        if self.temporal:
            self.lstm=nn.LSTM(dim,config["hidden"],config["layers"],batch_first=True,
                              dropout=config["dropout"] if config["layers"]>1 else 0)
            dim=config["hidden"]
        self.output=nn.Sequential(nn.Dropout(config["dropout"]),nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,2))

    def forward(self, x):
        if self.temporal:
            seq,_=self.lstm(x)
            return self.output(seq[:,-1])
        return self.output(x).mean(1)


def cache_path(row):
    return OUT / "features" / (row["sha256"]+".npz")
