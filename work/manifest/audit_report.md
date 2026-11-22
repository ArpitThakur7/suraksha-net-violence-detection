# Dataset audit report

Total clips: 2650

## Counts by dataset x original_split x label

| dataset | original_split | label | n |
|---|---|---|---|
| avd | test | 0 | 18 |
| avd | test | 1 | 34 |
| avd | train | 0 | 84 |
| avd | train | 1 | 162 |
| avd | validation | 0 | 18 |
| avd | validation | 1 | 34 |
| rwf | train | 0 | 800 |
| rwf | train | 1 | 800 |
| rwf | val | 0 | 200 |
| rwf | val | 1 | 200 |
| sfd | test | 0 | 22 |
| sfd | test | 1 | 25 |
| sfd | train | 0 | 105 |
| sfd | train | 1 | 103 |
| sfd | validation | 0 | 23 |
| sfd | validation | 1 | 22 |

## Exclusions by reason

| dataset | reason | n |
|---|---|---|
| rwf | duplicate_sha | 5 |

## Exact duplicates

Duplicate copies: 11 (within-split 5, cross-split 6)

| first copy | duplicate copy | label | original_split (first / dup) | kind |
|---|---|---|---|---|
| datasets/public/rwf2000/RWF-2000/train/Fight/_______________________________________________________________________ ______________________________________________urlgot_268.avi | datasets/public/rwf2000/RWF-2000/train/Fight/_______________________________________________________________________ ______________________________________________urlgot_279.avi | 1 | train / train | within-split |
| datasets/public/rwf2000/RWF-2000/train/Fight/Q1ycSrzoM2c_1.avi | datasets/public/rwf2000/RWF-2000/train/Fight/mRqg-WXiR7Q_1.avi | 1 | train / train | within-split |
| datasets/public/rwf2000/RWF-2000/train/Fight/Q1ycSrzoM2c_2.avi | datasets/public/rwf2000/RWF-2000/train/Fight/mRqg-WXiR7Q_2.avi | 1 | train / train | within-split |
| datasets/public/rwf2000/RWF-2000/train/NonFight/R2x2rOWh_0.avi | datasets/public/rwf2000/RWF-2000/train/NonFight/XwiwESMn_0.avi | 0 | train / train | within-split |
| datasets/public/rwf2000/RWF-2000/train/NonFight/4Fqxw7Nl_0.avi | datasets/public/rwf2000/RWF-2000/train/NonFight/xYWhwcqL_0.avi | 0 | train / train | within-split |
| datasets/public/rwf2000/RWF-2000/train/NonFight/uz4dXoN5_0.avi | datasets/public/rwf2000/RWF-2000/val/NonFight/7emxm3za_0.avi | 0 | train / val | cross-split |
| datasets/public/rwf2000/RWF-2000/train/NonFight/ndNHBJbS_0.avi | datasets/public/rwf2000/RWF-2000/val/NonFight/E9SvEiNt_0.avi | 0 | train / val | cross-split |
| datasets/public/rwf2000/RWF-2000/train/NonFight/wKMXI2vp_0.avi | datasets/public/rwf2000/RWF-2000/val/NonFight/ZCa5f4yq_0.avi | 0 | train / val | cross-split |
| datasets/public/rwf2000/RWF-2000/train/NonFight/aGfwZ3E4_0.avi | datasets/public/rwf2000/RWF-2000/val/NonFight/aZxUVVqk_0.avi | 0 | train / val | cross-split |
| datasets/public/rwf2000/RWF-2000/train/NonFight/EBwt3Thb_0.avi | datasets/public/rwf2000/RWF-2000/val/NonFight/cw8fPfUL_0.avi | 0 | train / val | cross-split |
| datasets/public/rwf2000/RWF-2000/train/NonFight/XQR0CvyY_1.avi | datasets/public/rwf2000/RWF-2000/val/NonFight/xV69l7Vj_0.avi | 0 | train / val | cross-split |

## Per-dataset summaries (non-excluded clips)

### avd (n=350)

- fps: min 30.00, median 30.00, max 30.00
- duration_s: min 2.47, median 5.30, max 14.47
- frames: min 74, median 159, max 434
- resolutions: 1920x1080 (350)

### rwf (n=1995)

- fps: min 30.00, median 30.00, max 30.00
- duration_s: min 5.00, median 5.00, max 5.00
- frames: min 150, median 150, max 150
- resolutions: 320x240 (698), 1280x720 (626), 640x360 (169), 480x360 (89), 1920x1080 (31), 398x224 (22), 352x288 (16), 406x720 (14)

### sfd (n=300)

- fps: min 10.00, median 29.97, max 30.00
- duration_s: min 1.00, median 2.00, max 4.74
- frames: min 20, median 60, max 142
- resolutions: 640x360 (86), 1280x720 (69), 480x360 (48), 320x240 (15), 704x480 (15), 1920x1080 (11), 626x360 (11), 640x314 (9)
