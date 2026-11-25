from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import os

import av
import numpy as np
from PIL import Image
import torch

from .model import CheckpointCNNLSTM

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(__file__).resolve().parent / "models"
WINDOW = 4.0
STRIDE = 2.0


@lru_cache(maxsize=1)
def detector():
    weights = torch.load(MODELS / "baseline_resnet18_lstm.pth", map_location="cpu", weights_only=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = CheckpointCNNLSTM()
    model.load_state_dict(weights["model_state_dict"])
    norm = weights["config"]["normalization"]
    return model.eval(), torch.tensor(norm["mean"]).view(1, 3, 1, 1), torch.tensor(norm["std"]).view(1, 3, 1, 1)


@dataclass
class Incident:
    start: float
    end: float
    score: float
    snapshot: bytes


def window_tensor(frames, mean, std):
    n = len(frames)
    indices = np.linspace(0, n-1, 16, dtype=int) if n >= 16 else np.minimum(np.arange(16), n-1)
    array = np.stack([frames[i][1] for i in indices])
    values = torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255)
    return ((values - mean) / std).unsqueeze(0)


@torch.inference_mode()
def score_window(frames):
    model, mean, std = detector()
    logits = model(window_tensor(frames, mean, std))
    return float(logits.softmax(1)[0, 1])


@lru_cache(maxsize=3)
def optimized_engine(name):
    model,mean,std=detector()
    if name=="cuda-fp16":
        if not torch.cuda.is_available():raise ValueError("CUDA is unavailable for the selected engine.")
        import copy
        gpu=copy.deepcopy(model).to("cuda").half().eval()
        def forward(x):return gpu(x.to("cuda",dtype=torch.float16)).float().cpu()
    elif name=="onnx-cpu":
        import onnxruntime as ort
        opts=ort.SessionOptions();opts.intra_op_num_threads=4
        session=ort.InferenceSession(str(MODELS/"baseline_resnet18_lstm.onnx"),opts,providers=["CPUExecutionProvider"])
        def forward(x):return torch.from_numpy(session.run(None,{"frames":x.numpy()})[0])
    elif name=="torch-cpu":forward=model
    else:raise ValueError("Choose torch-cpu, onnx-cpu or cuda-fp16.")
    @torch.inference_mode()
    def score(frames):
        return float(forward(window_tensor(frames,mean,std)).softmax(1)[0,1])
    return score


@lru_cache(maxsize=1)
def profile():
    """The model behind the /review upload screen."""
    return {"name":"Baseline ResNet18-LSTM","size":224,"threshold":0.40}


def active_score(frames):
    return optimized_engine(os.environ.get("SURAKSHA_ENGINE","torch-cpu"))(frames)


def analyze(path, threshold, scorer=None):
    windows, incidents = [], []
    size=224 if scorer is not None else profile()["size"]
    if scorer is None:scorer=active_score
    frames = deque()
    start = 0.0
    last_full_end = 0.0

    def emit(end):
        if not frames:
            return
        score = scorer(frames)
        windows.append({"start": start, "end": end, "score": score})
        if score < threshold:
            return
        previous = incidents[-1] if incidents and incidents[-1].end >= start else None
        if previous:
            previous.end = end
            if score <= previous.score:
                return
        snapshot = BytesIO()
        Image.fromarray(frames[len(frames)//2][1]).save(snapshot, format="JPEG", quality=88)
        if previous:
            previous.score, previous.snapshot = score, snapshot.getvalue()
        else:
            incidents.append(Incident(start, end, score, snapshot.getvalue()))

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("This recording has no video track.")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25)
        origin = None
        end = 0.0
        for index, frame in enumerate(container.decode(stream)):
            raw = float(frame.time) if frame.time is not None else index / fps
            if origin is None:
                origin = raw
            t = max(0.0, raw - origin)
            if t > 900 or index > 120000 or frame.width * frame.height > 16000000:
                raise ValueError("Use a recording up to 15 minutes and 16 megapixels.")
            while t >= start + WINDOW:
                last_full_end = start + WINDOW
                emit(last_full_end)
                start += STRIDE
                while frames and frames[0][0] < start:
                    frames.popleft()
            rgb = frame.to_image().convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            frames.append((t, np.asarray(rgb)))
            end = t + (float(frame.duration * frame.time_base) if frame.duration and frame.time_base else 1 / fps)
        if not frames:
            raise ValueError("No video frames could be decoded.")
        if end > last_full_end + 0.001 or not windows:
            emit(end)
    return end, windows, incidents

