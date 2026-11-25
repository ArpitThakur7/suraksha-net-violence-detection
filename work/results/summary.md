# Test results (frozen validation threshold)

ROC-AUC with group-bootstrap 95% CI (mean CI bounds over seeds when seeds > 1).

## p1_rwf (n_test = 400)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| videomae_ft | fine-tuned | 3 | 0.978 ± 0.001 [0.964, 0.989] | 0.916 | 0.915 |
| motionlite__train.distill_teacher-videomae | fine-tuned | 2 | 0.910 ± 0.006 [0.873, 0.943] | 0.787 | 0.764 |
| motionlite | fine-tuned | 2 | 0.897 ± 0.006 [0.851, 0.939] | 0.834 | 0.838 |
| mnv3_tsm | fine-tuned | 1 | 0.890 [0.855, 0.926] | 0.812 | 0.815 |
| videomae+clip | frozen probe | 1 | 0.945 [0.921, 0.966] | 0.855 | 0.851 |
| videomae | frozen probe | 1 | 0.945 [0.920, 0.966] | 0.863 | 0.862 |
| r2plus1d | frozen probe | 1 | 0.913 [0.871, 0.946] | 0.802 | 0.828 |
| clip | frozen probe | 1 | 0.895 [0.858, 0.931] | 0.815 | 0.820 |

## p1_sfd (n_test = 47)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| videomae | frozen probe | 1 | 0.869 [0.728, 0.967] | 0.809 | 0.791 |
| videomae+clip | frozen probe | 1 | 0.851 [0.718, 0.948] | 0.745 | 0.760 |
| clip | frozen probe | 1 | 0.769 [0.586, 0.892] | 0.638 | 0.691 |
| r2plus1d | frozen probe | 1 | 0.760 [0.590, 0.882] | 0.638 | 0.691 |

## p1_avd (n_test = 52)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| r2plus1d | frozen probe | 1 | 1.000 [1.000, 1.000] | 0.962 | 0.971 |
| videomae | frozen probe | 1 | 0.997 [0.984, 1.000] | 0.923 | 0.944 |
| videomae+clip | frozen probe | 1 | 0.995 [0.979, 1.000] | 0.962 | 0.970 |
| clip | frozen probe | 1 | 0.984 [0.944, 1.000] | 0.962 | 0.970 |

## p2_all (n_test = 499)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| videomae_ft | fine-tuned | 3 | 0.972 ± 0.004 [0.958, 0.985] | 0.912 | 0.916 |
| motionlite__train.distill_teacher-videomae | fine-tuned | 2 | 0.898 ± 0.007 [0.863, 0.934] | 0.804 | 0.807 |
| mnv3_tsm | fine-tuned | 1 | 0.896 [0.860, 0.931] | 0.808 | 0.827 |
| motionlite | fine-tuned | 2 | 0.895 ± 0.012 [0.859, 0.930] | 0.819 | 0.828 |
| videomae | frozen probe | 1 | 0.949 [0.932, 0.965] | 0.870 | 0.879 |
| videomae+clip | frozen probe | 1 | 0.946 [0.926, 0.965] | 0.868 | 0.875 |
| clip | frozen probe | 1 | 0.900 [0.869, 0.933] | 0.828 | 0.835 |
| r2plus1d | frozen probe | 1 | 0.864 [0.816, 0.907] | 0.768 | 0.761 |

## p3_no_rwf (n_test = 400)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| videomae_ft | fine-tuned | 3 | 0.963 ± 0.001 [0.942, 0.980] | 0.893 | 0.897 |
| mnv3_tsm | fine-tuned | 1 | 0.841 [0.783, 0.897] | 0.777 | 0.746 |
| motionlite__train.distill_teacher-videomae | fine-tuned | 2 | 0.826 ± 0.021 [0.765, 0.882] | 0.745 | 0.720 |
| motionlite | fine-tuned | 2 | 0.826 ± 0.031 [0.769, 0.879] | 0.749 | 0.718 |
| videomae+clip | frozen probe | 1 | 0.948 [0.920, 0.970] | 0.858 | 0.850 |
| clip | frozen probe | 1 | 0.894 [0.851, 0.933] | 0.762 | 0.713 |
| videomae | frozen probe | 1 | 0.888 [0.843, 0.924] | 0.795 | 0.789 |
| r2plus1d | frozen probe | 1 | 0.748 [0.673, 0.819] | 0.672 | 0.677 |

## p3_no_sfd (n_test = 47)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| videomae_ft | fine-tuned | 3 | 0.905 ± 0.014 [0.826, 0.963] | 0.844 | 0.846 |
| mnv3_tsm | fine-tuned | 1 | 0.804 [0.686, 0.950] | 0.723 | 0.772 |
| motionlite__train.distill_teacher-videomae | fine-tuned | 2 | 0.761 ± 0.017 [0.555, 0.869] | 0.617 | 0.673 |
| motionlite | fine-tuned | 2 | 0.757 ± 0.006 [0.624, 0.837] | 0.649 | 0.694 |
| videomae+clip | frozen probe | 1 | 0.885 [0.756, 0.973] | 0.745 | 0.786 |
| videomae | frozen probe | 1 | 0.876 [0.743, 0.971] | 0.766 | 0.800 |
| clip | frozen probe | 1 | 0.824 [0.655, 0.928] | 0.766 | 0.820 |
| r2plus1d | frozen probe | 1 | 0.791 [0.689, 0.888] | 0.702 | 0.759 |

## p3_no_avd (n_test = 52)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| motionlite__train.distill_teacher-videomae | fine-tuned | 2 | 0.829 ± 0.015 [0.640, 0.969] | 0.673 | 0.691 |
| videomae_ft | fine-tuned | 3 | 0.792 ± 0.011 [0.612, 0.947] | 0.718 | 0.762 |
| motionlite | fine-tuned | 2 | 0.762 ± 0.029 [0.585, 0.902] | 0.625 | 0.631 |
| mnv3_tsm | fine-tuned | 1 | 0.721 [0.509, 0.894] | 0.635 | 0.667 |
| videomae+clip | frozen probe | 1 | 0.899 [0.786, 0.979] | 0.673 | 0.667 |
| clip | frozen probe | 1 | 0.863 [0.729, 0.964] | 0.481 | 0.341 |
| videomae | frozen probe | 1 | 0.858 [0.722, 0.963] | 0.712 | 0.737 |
| r2plus1d | frozen probe | 1 | 0.789 [0.611, 0.936] | 0.750 | 0.827 |

## p4_random (n_test = 240)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| mnv3_tsm | fine-tuned | 1 | 0.930 [0.893, 0.963] | 0.867 | 0.868 |
| videomae+clip | frozen probe | 1 | 0.958 [0.935, 0.978] | 0.887 | 0.897 |
| videomae | frozen probe | 1 | 0.958 [0.932, 0.980] | 0.904 | 0.905 |
| r2plus1d | frozen probe | 1 | 0.921 [0.887, 0.952] | 0.821 | 0.807 |
| clip | frozen probe | 1 | 0.916 [0.883, 0.948] | 0.817 | 0.810 |

## p4_grouped (n_test = 207)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| mnv3_tsm | fine-tuned | 1 | 0.917 [0.860, 0.961] | 0.816 | 0.826 |
| videomae+clip | frozen probe | 1 | 0.969 [0.944, 0.991] | 0.918 | 0.914 |
| videomae | frozen probe | 1 | 0.963 [0.933, 0.986] | 0.879 | 0.872 |
| clip | frozen probe | 1 | 0.950 [0.915, 0.979] | 0.855 | 0.860 |
| r2plus1d | frozen probe | 1 | 0.918 [0.865, 0.962] | 0.836 | 0.819 |

## p2_all_leaky (n_test = 499)

| model | kind | seeds | ROC-AUC [95% CI] | accuracy | F1 |
|---|---|---:|---|---:|---:|
| mnv3_tsm | fine-tuned | 1 | 0.885 [0.848, 0.923] | 0.800 | 0.799 |
| videomae | frozen probe | 1 | 0.953 [0.937, 0.969] | 0.868 | 0.877 |
| videomae+clip | frozen probe | 1 | 0.950 [0.930, 0.968] | 0.880 | 0.884 |
| clip | frozen probe | 1 | 0.903 [0.873, 0.936] | 0.806 | 0.804 |
| r2plus1d | frozen probe | 1 | 0.868 [0.820, 0.911] | 0.780 | 0.774 |
