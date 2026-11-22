import itertools

import pytest
import torch

import pipeline.paths  # noqa: F401
from pipeline.models import MODEL_INPUT, MODEL_NORM, build_model
from pipeline.models.motionlite import FUSION_SPLIT, MotionLite
from pipeline.models.tsm import TemporalShift, add_tsm, temporal_shift

NAMES = ["cnn_lstm", "mnv3_tsm", "r2plus1d", "mvit_s", "motionlite"]
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
R2P1D_NORM = ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
MVIT_NORM = ([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("name", NAMES)
def test_forward_shapes(name):
    torch.manual_seed(0)
    if name == "mvit_s":  # fixed 16 x 224 x 224 input; batch 1 keeps the CPU test affordable
        B, T, S = 1, 16, 224
    else:
        B, T, S = 2, 16, 112
    m = build_model(name, pretrained=False, T=T).eval()
    with torch.no_grad():
        y = m(torch.randn(B, T, 3, S, S))
    assert y.shape == (B, 2)
    assert torch.isfinite(y).all()


def test_model_norm_and_input_constraints():
    assert set(MODEL_NORM) == set(NAMES)
    for n in ("cnn_lstm", "mnv3_tsm", "motionlite"):
        assert MODEL_NORM[n] == IMAGENET
    assert MODEL_NORM["r2plus1d"] == R2P1D_NORM
    assert MODEL_NORM["mvit_s"] == MVIT_NORM
    assert MODEL_INPUT["mvit_s"] == {"T": 16, "crop": 224}
    with pytest.raises(ValueError):
        build_model("mvit_s", pretrained=False, T=8)
    with pytest.raises(ValueError):
        build_model("nope", pretrained=False)


def test_temporal_shift():
    T, C = 4, 16
    x = torch.arange(2 * T * C, dtype=torch.float32).reshape(2 * T, C, 1, 1)
    y = temporal_shift(x, T, fold_div=8)
    assert y.shape == x.shape
    xv, yv = x.view(2, T, C), y.view(2, T, C)
    # channels 0:2 shifted one step back in time: out[t] = in[t+1], zero at the last step
    assert torch.equal(yv[:, :-1, 0:2], xv[:, 1:, 0:2])
    assert (yv[:, -1, 0:2] == 0).all()
    # channels 2:4 shifted one step forward: out[t] = in[t-1], zero at the first step
    assert torch.equal(yv[:, 1:, 2:4], xv[:, :-1, 2:4])
    assert (yv[:, 0, 2:4] == 0).all()
    # the rest unchanged
    assert torch.equal(yv[:, :, 4:], xv[:, :, 4:])
    with pytest.raises(AssertionError):
        temporal_shift(torch.zeros(6, C, 1, 1), T)


def test_tsm_insertion_points():
    m = build_model("mnv3_tsm", pretrained=False, T=4)
    shifts = [mod for mod in m.modules() if isinstance(mod, TemporalShift)]
    # MobileNetV3-Large has 10 inverted-residual blocks with use_res_connect
    assert len(shifts) == 10 and all(s.T == 4 for s in shifts)
    n_before = len(shifts)
    add_tsm(m, 4)  # idempotent
    assert len([mod for mod in m.modules() if isinstance(mod, TemporalShift)]) == n_before
    e = MotionLite(T=4, backbone="effb0", diff=False, pretrained=False)
    assert len([mod for mod in e.modules() if isinstance(mod, TemporalShift)]) == 9
    off = MotionLite(T=4, diff=False, tsm=False, pretrained=False)
    assert not any(isinstance(mod, TemporalShift) for mod in off.modules())


@pytest.mark.parametrize("diff,pool,fusion",
                         list(itertools.product([True, False], ["attn", "avg"], ["early", "mid"])))
def test_motionlite_ablations(diff, pool, fusion):
    torch.manual_seed(0)
    B, T = 2, 8
    m = build_model("motionlite", pretrained=False, T=T, diff=diff, pool=pool, fusion=fusion).eval()
    with torch.no_grad():
        y = m(torch.randn(B, T, 3, 96, 96))
    assert y.shape == (B, 2) and torch.isfinite(y).all()
    assert m.last_attn.shape == (B, T)
    assert torch.allclose(m.last_attn.sum(1), torch.ones(B), atol=1e-5)
    if diff:
        off = build_model("motionlite", pretrained=False, T=T, diff=False, pool=pool, fusion=fusion)
        assert n_params(off) < n_params(m)


def test_motionlite_effb0_mid_forward():
    m = build_model("motionlite", pretrained=False, T=4, backbone="effb0").eval()
    with torch.no_grad():
        y = m(torch.randn(1, 4, 3, 96, 96))
    assert y.shape == (1, 2)
    assert FUSION_SPLIT == {"mnv3": 6, "effb0": 4}


def test_motionlite_early_stem_init():
    m = MotionLite(T=4, diff=True, fusion="early", pretrained=False)
    w = m.early[0][0].weight
    assert w.shape[1] == 6
    for k in range(3, 6):
        assert torch.allclose(w[:, k], w[:, :3].mean(1))


def test_motionlite_wrong_T_raises():
    m = MotionLite(T=8, pretrained=False).eval()
    with pytest.raises(AssertionError):
        m(torch.randn(1, 4, 3, 64, 64))


def test_param_budget():
    m = build_model("motionlite", pretrained=False, backbone="mnv3")
    assert n_params(m) < 8_000_000


def test_motionlite_backward_reaches_both_streams():
    torch.manual_seed(0)
    m = build_model("motionlite", pretrained=False, T=4).train()
    x = torch.randn(2, 4, 3, 64, 64)

    def grads_ok(part):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())

    m(x).sum().backward()
    # zero-init proj: at step 0 only its last conv (and the RGB path) receive gradient
    for part in (m.early, m.proj[-1], m.late, m.attn):
        assert grads_ok(part)
    # once the last proj conv is non-zero, gradient reaches the whole diff stream
    with torch.no_grad():
        m.proj[-1].weight.normal_(0, 0.1)
    m.zero_grad(set_to_none=True)
    m(x).sum().backward()
    for part in (m.early, m.diff_early, m.proj, m.gate, m.late, m.attn):
        assert grads_ok(part)


def test_model_norm_matches_weights_transforms():
    from torchvision.models.video import MViT_V2_S_Weights, R2Plus1D_18_Weights
    for name, w in (("r2plus1d", R2Plus1D_18_Weights.KINETICS400_V1),
                    ("mvit_s", MViT_V2_S_Weights.KINETICS400_V1)):
        t = w.transforms()  # builds the preset only; no download
        mean, std = MODEL_NORM[name]
        assert mean == pytest.approx(list(t.mean)) and std == pytest.approx(list(t.std))


def test_motionlite_mid_fusion_is_identity_at_init():
    torch.manual_seed(0)
    B, T = 2, 4
    m = MotionLite(T=T, diff=True, fusion="mid", pretrained=False).eval()
    x = torch.randn(B, T, 3, 64, 64)
    with torch.no_grad():
        d = torch.cat([torch.zeros_like(x[:, :1]), (x[:, 1:] - x[:, :-1]).abs()], 1)
        assert (m.proj(m.diff_early(d.flatten(0, 1))) == 0).all()
        y = m(x)
        # RGB-only path: early -> late -> attention pool -> head
        f = m.late(m.early(x.flatten(0, 1))).mean((2, 3)).view(B, T, -1)
        w = torch.softmax(m.attn(f).squeeze(-1), 1)
        ref = m.fc((w.unsqueeze(-1) * f).sum(1))
    assert torch.allclose(y, ref, atol=1e-5)
