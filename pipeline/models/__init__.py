"""Stage 5 model zoo. Every model maps [B,T,3,H,W] (normalised with MODEL_NORM[name]) to logits
[B,2] in the order [NonViolence, Violence]."""
from __future__ import annotations

from torch import nn

IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# Kinetics-pretrained models were trained with different stats; each must match its own weights.
R2PLUS1D_NORM = ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
MVIT_NORM = ([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])


def _weights_norm(enum_name: str, fallback: tuple[list, list]) -> tuple[list, list]:
    """(mean, std) from the torchvision weights' preset transforms (no download needed);
    the literal fallback is used if torchvision is unavailable or the preset changes shape."""
    try:
        import pipeline.paths  # noqa: F401
        from torchvision.models import video
        t = getattr(video, enum_name).KINETICS400_V1.transforms()
        return [float(v) for v in t.mean], [float(v) for v in t.std]
    except Exception:
        return fallback


MODEL_NORM: dict[str, tuple[list, list]] = {
    "cnn_lstm": IMAGENET_NORM,
    "mnv3_tsm": IMAGENET_NORM,
    "motionlite": IMAGENET_NORM,
    "r2plus1d": _weights_norm("R2Plus1D_18_Weights", R2PLUS1D_NORM),
    "mvit_s": _weights_norm("MViT_V2_S_Weights", MVIT_NORM),
}

# Hard input constraints the trainer validates configs against.
MODEL_INPUT: dict[str, dict] = {"mvit_s": {"T": 16, "crop": 224}}

MODEL_NAMES = tuple(MODEL_NORM)


def build_model(name: str, pretrained: bool = True, T: int = 16, **kw) -> nn.Module:
    """pretrained=True loads torchvision weights (ImageNet for the 2D backbones, Kinetics-400 for
    r2plus1d / mvit_s) into TORCH_HOME. kw go to MotionLite (backbone, diff, tsm, pool, fusion)."""
    if name in MODEL_INPUT and "T" in MODEL_INPUT[name] and T != MODEL_INPUT[name]["T"]:
        raise ValueError(f"{name} requires T={MODEL_INPUT[name]['T']}, got T={T}")
    if name == "cnn_lstm":
        from pipeline.models.cnn_lstm import CNNLSTM
        return CNNLSTM(pretrained=pretrained, **kw)
    if name == "mnv3_tsm":
        from pipeline.models.tsm import MobileNetV3TSM
        return MobileNetV3TSM(T=T, pretrained=pretrained, **kw)
    if name == "r2plus1d":
        from pipeline.models.video3d import R2Plus1D
        return R2Plus1D(pretrained=pretrained, **kw)
    if name == "mvit_s":
        from pipeline.models.video3d import MViTS
        return MViTS(pretrained=pretrained, **kw)
    if name == "motionlite":
        from pipeline.models.motionlite import MotionLite
        return MotionLite(T=T, pretrained=pretrained, **kw)
    raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")
