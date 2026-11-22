"""Inspect the supplied checkpoint without running repository code or training."""
from pathlib import Path
import hashlib
import json
import zipfile
import torch

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'best_model.zip'
TARGET = ROOT / 'checkpoint' / 'best_model.pth'
TARGET.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as z:
    if z.namelist() != ['best_model.pth']:
        raise ValueError('Unexpected archive members')
    with z.open('best_model.pth') as src, TARGET.open('wb') as dst:
        while block := src.read(1024 * 1024):
            dst.write(block)

ckpt = torch.load(TARGET, map_location='cpu', weights_only=True)
if not isinstance(ckpt, dict):
    raise TypeError(f'Unexpected checkpoint type: {type(ckpt).__name__}')

state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
bad = [k for k, v in tensors.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
history = ckpt.get('history', [])
optimizer = ckpt.get('optimizer_state_dict', {})
steps = []
for item in optimizer.get('state', {}).values():
    step = item.get('step')
    if step is not None:
        steps.append(float(step.item() if torch.is_tensor(step) else step))

report = {
    'checkpoint_path': str(TARGET),
    'checkpoint_bytes': TARGET.stat().st_size,
    'checkpoint_sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
    'torch_version': torch.__version__,
    'load_mode': 'weights_only=True, map_location=cpu',
    'top_level_keys': list(ckpt),
    'epoch': ckpt.get('epoch'),
    'best_f1': ckpt.get('best_f1'),
    'config': ckpt.get('config'),
    'history': history,
    'tensor_count': len(tensors),
    'tensor_elements_including_buffers': sum(v.numel() for v in tensors.values()),
    'nonfinite_tensors': bad,
    'tensor_shapes': {k: list(v.shape) for k, v in tensors.items()},
    'batchnorm_batches_tracked': {k: v.item() for k, v in tensors.items() if k.endswith('num_batches_tracked')},
    'optimizer_parameter_states': len(optimizer.get('state', {})),
    'optimizer_step_min': min(steps) if steps else None,
    'optimizer_step_max': max(steps) if steps else None,
    'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.get('param_groups', [])],
    'scheduler_state': ckpt.get('scheduler_state_dict'),
}

repo_config_path = ROOT / 'repo-snapshot/experiments/combined_cnn_lstm_16frames/config.json'
if repo_config_path.exists():
    repo_config = json.loads(repo_config_path.read_text(encoding='utf-8'))
    report['config_matches_committed_combined_experiment'] = ckpt.get('config') == repo_config

(ROOT / 'checkpoint-inspection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
summary = {k: v for k, v in report.items() if k not in {'tensor_shapes', 'batchnorm_batches_tracked', 'optimizer_groups'}}
summary['selected_tensor_shapes'] = {k: list(v.shape) for k, v in tensors.items() if k in {
    'cnn.0.weight', 'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'classifier.1.weight', 'classifier.4.weight'}}
summary['batchnorm_batch_counts'] = sorted(set(report['batchnorm_batches_tracked'].values()))
print(json.dumps(summary, indent=2))
