# Semester model evaluation

Selected **lstm_layers_2** on validation F1 with threshold **0.30**. The training suite used one seed (42) and ImageNet backbones. The best epoch was 6. Training accuracy at that epoch was 82.4% and validation accuracy at a 0.50 cutoff was 74.2%. Repeated-seed variability was not measured.

| Test source | Clips | Accuracy | Violence recall | Precision | F1 | ROC-AUC | False alarms | Misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 99 | 74.7% | 84.7% | 75.8% | 0.800 | 0.831 | 16 | 9 |
| AVD | 52 | 90.4% | 94.1% | 91.4% | 0.928 | 0.980 | 3 | 2 |
| SFD | 47 | 57.4% | 72.0% | 58.1% | 0.643 | 0.529 | 13 | 7 |

Each prediction covers one labeled clip. Overlapping CCTV windows and incident notifications require separate measurements. Multiple AIRTLab views of an event stay together. SFD source recording IDs stay together. The test set is shared across prespecified comparisons and the release uses the validation-selected winner.

See ../validation_comparison.csv for the ablation and hyperparameter table, ../generalization.csv for transfer tests and ../test_results.csv for baselines. evaluation.png plots training, validation, ROC and the test confusion matrix. model_sizes.csv compares artifact sizes and parameters. errors.csv lists every mistake with publisher action labels where available. The four Grad-CAM examples use the predicted class and the frame with highest activation. Scene conditions such as occlusion or camera shake require human annotation.

Run from the project root with the project Python environment:

`python semester_ml/infer.py review/vishnu/sample_video.mp4 --checkpoint ml_runs/two_sources/release/best_model.pth`
