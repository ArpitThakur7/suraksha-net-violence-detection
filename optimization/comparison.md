# Laptop CPU quantization comparison

97 validation clips, 149 windows, threshold 0.50.
A clip is flagged when any window reaches the threshold.

| Metric | FP32 | Dynamic INT8 |
| --- | ---: | ---: |
| Inference file (MiB) | 45.91 | 43.57 |
| Median forward pass (ms) | 228.5 | 232.0 |
| 95th percentile forward pass (ms) | 255.1 | 256.1 |
| Clip accuracy (%) | 95.88 | 95.88 |
| Violence recall (%) | 96.43 | 96.43 |
| Violence F1 (%) | 96.43 | 96.43 |
| Violent clips missed | 2 | 2 |
| Nonviolent clips flagged | 2 | 2 |

Clip decisions changed: 0. Window decisions changed: 0.

PyTorch 2.6.0+cu124, x86 backend, 4 CPU threads. Timing covers a forward pass on a preprocessed 16-frame input. Five warmups per model preceded paired timings with alternating order. File sizes refer to reloadable TorchScript inference exports.

INT8 applies to the LSTM and two linear layers. The ResNet18 backbone uses FP32. Membership of these videos in the checkpoint's original training split is unknown.

Reproduce from the project root with `python benchmark_quantization.py`.
