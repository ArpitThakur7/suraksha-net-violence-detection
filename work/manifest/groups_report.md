# Global groups report

Clips: 2650 (2645 non-excluded)

- global groups: 1129
- groups spanning >1 dataset: 13
- groups spanning >1 original split: 15
- clips with shared_source: 174

## Shared-source clip pairs (same source_id, different datasets)

- shared source_ids: 10; RWF clips with a shared source: 57
- clip pairs: 447
- visual near-duplicates (distance <= 6): 57
- borderline (7-10): 78
- same-source-only (> 10): 312

Minimum pairwise pHash distance per shared source_id:

| source_id | pairs | min distance | datasets (clips) | dataset/split (clips) |
|---|---|---|---|---|
| yt:8VqTmuq4HcI | 8 | 2.25 | rwf:4, sfd:2 | rwf/train:4, sfd/train:2 |
| yt:BpargJW29Wo | 56 | 0.00 | sfd:8, rwf:7 | rwf/train:7, sfd/train:8 |
| yt:FZDDNtcaBcw | 12 | 2.12 | rwf:4, sfd:3 | rwf/train:4, sfd/train:3 |
| yt:PTy2vx8Ejas | 18 | 9.25 | rwf:6, sfd:3 | rwf/train:6, sfd/train:3 |
| yt:V726w0IKCp4 | 30 | 0.50 | sfd:6, rwf:5 | rwf/train:5, sfd/train:6 |
| yt:Wq0BuA8GM84 | 15 | 2.00 | sfd:5, rwf:3 | rwf/val:3, sfd/train:5 |
| yt:lmfaDmCCRoY | 153 | 5.75 | sfd:17, rwf:9 | rwf/train:9, sfd/test:17 |
| yt:s-jW4cyavW0 | 50 | 4.25 | sfd:10, rwf:5 | rwf/train:5, sfd/train:10 |
| yt:tf_X7GtOGyE | 63 | 2.38 | sfd:9, rwf:7 | rwf/val:7, sfd/validation:9 |
| yt:v4dhdnsxiX4 | 42 | 0.25 | rwf:7, sfd:6 | rwf/val:7, sfd/train:6 |

## Near-duplicate pairs across different source_ids

Total pairs <= 10: 814; confirmed (<= 6): 104; borderline unconfirmed: 710. Borderline pairs need human inspection of near_duplicates_sheet.png.

Top 40 by distance (smallest first):

| clip_a | a (dataset/split/label) | clip_b | b (dataset/split/label) | distance | confirmed |
|---|---|---|---|---|---|
| a56abdb0 | rwf/train/0 | 5c0914fb | rwf/train/0 | 0.00 | True |
| 2163b809 | rwf/train/0 | faef73a5 | rwf/val/0 | 0.00 | True |
| 95eb0a11 | rwf/train/0 | 61f406dd | rwf/train/0 | 0.00 | True |
| b375e584 | rwf/train/0 | 95de62bf | rwf/train/0 | 0.00 | True |
| 3eec4b74 | rwf/train/1 | 653eca47 | rwf/train/1 | 1.38 | True |
| 6dcca31f | rwf/train/1 | 3cd49ff4 | rwf/train/1 | 2.12 | True |
| b7f091e1 | avd/train/0 | 247c9617 | avd/train/0 | 2.25 | True |
| b7f091e1 | avd/train/0 | 3c955627 | avd/train/1 | 2.50 | True |
| 247c9617 | avd/train/0 | 3c955627 | avd/train/1 | 2.50 | True |
| 6dcca31f | rwf/train/1 | 7199ab4e | sfd/train/1 | 2.50 | True |
| f5e4b66d | rwf/train/1 | 8817ea84 | rwf/train/1 | 2.50 | True |
| a5af27e8 | rwf/train/1 | ef7759c8 | sfd/train/0 | 2.50 | True |
| b375e584 | rwf/train/0 | 0b317feb | rwf/val/0 | 2.50 | True |
| 95de62bf | rwf/train/0 | 0b317feb | rwf/val/0 | 2.50 | True |
| 3cd49ff4 | rwf/train/1 | 7199ab4e | sfd/train/1 | 2.62 | True |
| 7303e123 | rwf/train/0 | 2b7c5577 | sfd/train/0 | 2.88 | True |
| acd02a7e | rwf/train/1 | 2cf26203 | rwf/train/1 | 3.25 | True |
| 3f9cc7f6 | rwf/train/1 | 7199ab4e | sfd/train/1 | 3.25 | True |
| 3eec4b74 | rwf/train/1 | 8817ea84 | rwf/train/1 | 3.25 | True |
| e9fa4053 | rwf/train/0 | 9201b7e2 | sfd/train/1 | 3.25 | True |
| 6741ac05 | rwf/train/1 | 9f3ab9ec | sfd/validation/1 | 3.38 | True |
| 3fdf090b | rwf/train/1 | 8817ea84 | rwf/train/1 | 3.38 | True |
| 7be24cd4 | rwf/train/1 | 2cf26203 | rwf/train/1 | 3.62 | True |
| 3cd49ff4 | rwf/train/1 | ac57ebaf | sfd/train/0 | 3.62 | True |
| 6dcca31f | rwf/train/1 | ac57ebaf | sfd/train/0 | 3.75 | True |
| 3fdf090b | rwf/train/1 | d7ccc5ad | rwf/train/1 | 3.75 | True |
| 8817ea84 | rwf/train/1 | e06ff3a7 | rwf/train/0 | 3.88 | True |
| eaacb398 | rwf/train/1 | 8c9d7927 | rwf/train/0 | 3.88 | True |
| 37c87f0b | sfd/train/1 | 2b7c5577 | sfd/train/0 | 3.88 | True |
| af8653d3 | rwf/train/1 | 33f4a34a | sfd/train/1 | 4.00 | True |
| 7303e123 | rwf/train/0 | 37c87f0b | sfd/train/1 | 4.00 | True |
| e4724baf | avd/validation/0 | 6596ad19 | avd/validation/0 | 4.12 | True |
| f5e4b66d | rwf/train/1 | 653eca47 | rwf/train/1 | 4.12 | True |
| eaacb398 | rwf/train/1 | 444dd5d5 | rwf/train/0 | 4.12 | True |
| 5a2b6663 | rwf/train/0 | 50b8637e | sfd/validation/0 | 4.12 | True |
| efc8b3c0 | avd/test/1 | e27224ac | avd/test/1 | 4.25 | True |
| f5e4b66d | rwf/train/1 | d7ccc5ad | rwf/train/1 | 4.25 | True |
| d6e21bf5 | avd/train/0 | 6ae7e095 | avd/train/1 | 4.38 | True |
| 469c9fb0 | avd/train/0 | 6596ad19 | avd/validation/0 | 4.38 | True |
| 3eec4b74 | rwf/train/1 | d7ccc5ad | rwf/train/1 | 4.38 | True |

## Near-duplicate pairs by dataset pair

AIRTLab (avd) pairs where both clips are avd are NOT used for grouping (reason avd_same_room): AIRTLab is a single-room staged dataset, so different staged events look alike to pHash; the publisher scene ids (source_id) are authoritative.

| dataset pair | confirmed (<=6) | borderline (7-10) |
|---|---|---|
| avd-avd | 49 | 676 |
| rwf-rwf | 38 | 19 |
| rwf-sfd | 16 | 14 |
| sfd-sfd | 1 | 1 |
| any avd-cross (avd with rwf/sfd) | 0 | 0 |

## RWF-SFD confirmed pairs with different source_ids (cross-source re-uploads)

Count: 16

| clip_a | a (dataset/split/label/source) | clip_b | b (dataset/split/label/source) | distance |
|---|---|---|---|---|
| 6dcca31f | rwf/train/1/yt:Q1ycSrzoM2c | 7199ab4e | sfd/train/1/yt:jQeOqXpcKqg | 2.50 |
| a5af27e8 | rwf/train/1/yt:mVTcBPrjh-E | ef7759c8 | sfd/train/0/yt:IZjLv7alWTk | 2.50 |
| 3cd49ff4 | rwf/train/1/yt:mRqg-WXiR7Q | 7199ab4e | sfd/train/1/yt:jQeOqXpcKqg | 2.62 |
| 7303e123 | rwf/train/0/yt:WHNtrI7MEUg | 2b7c5577 | sfd/train/0/yt:g0x9-0pOuhw | 2.88 |
| 3f9cc7f6 | rwf/train/1/yt:Q1ycSrzoM2c | 7199ab4e | sfd/train/1/yt:jQeOqXpcKqg | 3.25 |
| e9fa4053 | rwf/train/0/yt:mRqg-WXiR7Q | 9201b7e2 | sfd/train/1/yt:jQeOqXpcKqg | 3.25 |
| 6741ac05 | rwf/train/1/yt:C4SQF-UZzkA | 9f3ab9ec | sfd/validation/1/yt:Wwu20e9pfxg | 3.38 |
| 3cd49ff4 | rwf/train/1/yt:mRqg-WXiR7Q | ac57ebaf | sfd/train/0/yt:jQeOqXpcKqg | 3.62 |
| 6dcca31f | rwf/train/1/yt:Q1ycSrzoM2c | ac57ebaf | sfd/train/0/yt:jQeOqXpcKqg | 3.75 |
| af8653d3 | rwf/train/1/yt:mVTcBPrjh-E | 33f4a34a | sfd/train/1/yt:IZjLv7alWTk | 4.00 |
| 7303e123 | rwf/train/0/yt:WHNtrI7MEUg | 37c87f0b | sfd/train/1/yt:IZjLv7alWTk | 4.00 |
| 5a2b6663 | rwf/train/0/yt:C4SQF-UZzkA | 50b8637e | sfd/validation/0/yt:Wwu20e9pfxg | 4.12 |
| 72d9d7e5 | rwf/train/0/yt:WHNtrI7MEUg | 2b7c5577 | sfd/train/0/yt:g0x9-0pOuhw | 4.75 |
| e9fa4053 | rwf/train/0/yt:mRqg-WXiR7Q | fa621bb9 | sfd/train/1/yt:jQeOqXpcKqg | 5.00 |
| 9957356b | rwf/train/0/yt:C4SQF-UZzkA | ff49c710 | sfd/validation/1/yt:Wwu20e9pfxg | 5.88 |
| 3f9cc7f6 | rwf/train/1/yt:Q1ycSrzoM2c | ac57ebaf | sfd/train/0/yt:jQeOqXpcKqg | 6.00 |
