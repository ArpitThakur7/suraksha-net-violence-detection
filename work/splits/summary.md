# Split summary

## Counts per protocol x split x dataset x label

| protocol | split | dataset | label 0 | label 1 | total |
|---|---|---|---|---|---|
| p1_rwf | train | rwf | 650 | 672 | 1322 |
| p1_rwf | val | rwf | 137 | 124 | 261 |
| p1_rwf | test | rwf | 200 | 200 | 400 |
| p1_rwf | **all** | all | 987 | 996 | 1983 |
| p1_sfd | train | sfd | 105 | 103 | 208 |
| p1_sfd | val | sfd | 23 | 22 | 45 |
| p1_sfd | test | sfd | 22 | 25 | 47 |
| p1_sfd | **all** | all | 150 | 150 | 300 |
| p1_avd | train | avd | 84 | 162 | 246 |
| p1_avd | val | avd | 18 | 34 | 52 |
| p1_avd | test | avd | 18 | 34 | 52 |
| p1_avd | **all** | all | 120 | 230 | 350 |
| p2_all | train | rwf | 646 | 663 | 1309 |
| p2_all | train | sfd | 92 | 81 | 173 |
| p2_all | train | avd | 84 | 162 | 246 |
| p2_all | val | rwf | 137 | 124 | 261 |
| p2_all | val | sfd | 19 | 17 | 36 |
| p2_all | val | avd | 18 | 34 | 52 |
| p2_all | test | rwf | 200 | 200 | 400 |
| p2_all | test | sfd | 22 | 25 | 47 |
| p2_all | test | avd | 18 | 34 | 52 |
| p2_all | **all** | all | 1236 | 1340 | 2576 |
| p3_no_rwf | train | sfd | 99 | 98 | 197 |
| p3_no_rwf | train | avd | 84 | 162 | 246 |
| p3_no_rwf | val | sfd | 19 | 17 | 36 |
| p3_no_rwf | val | avd | 18 | 34 | 52 |
| p3_no_rwf | test | rwf | 200 | 200 | 400 |
| p3_no_rwf | **all** | all | 420 | 511 | 931 |
| p3_no_sfd | train | rwf | 660 | 665 | 1325 |
| p3_no_sfd | train | avd | 84 | 162 | 246 |
| p3_no_sfd | val | rwf | 137 | 124 | 261 |
| p3_no_sfd | val | avd | 18 | 34 | 52 |
| p3_no_sfd | test | sfd | 22 | 25 | 47 |
| p3_no_sfd | **all** | all | 921 | 1010 | 1931 |
| p3_no_avd | train | rwf | 658 | 672 | 1330 |
| p3_no_avd | train | sfd | 98 | 86 | 184 |
| p3_no_avd | val | rwf | 137 | 124 | 261 |
| p3_no_avd | val | sfd | 23 | 22 | 45 |
| p3_no_avd | test | avd | 18 | 34 | 52 |
| p3_no_avd | **all** | all | 934 | 938 | 1872 |
| p4_grouped | train | rwf | 574 | 555 | 1129 |
| p4_grouped | val | rwf | 122 | 137 | 259 |
| p4_grouped | test | rwf | 102 | 105 | 207 |
| p4_grouped | **all** | all | 798 | 797 | 1595 |
| p4_random | train | rwf | 558 | 557 | 1115 |
| p4_random | val | rwf | 120 | 120 | 240 |
| p4_random | test | rwf | 120 | 120 | 240 |
| p4_random | **all** | all | 798 | 797 | 1595 |
| p2_all_leaky | train | rwf | 655 | 673 | 1328 |
| p2_all_leaky | train | sfd | 105 | 103 | 208 |
| p2_all_leaky | train | avd | 84 | 162 | 246 |
| p2_all_leaky | val | rwf | 137 | 124 | 261 |
| p2_all_leaky | val | sfd | 23 | 22 | 45 |
| p2_all_leaky | val | avd | 18 | 34 | 52 |
| p2_all_leaky | test | rwf | 200 | 200 | 400 |
| p2_all_leaky | test | sfd | 22 | 25 | 47 |
| p2_all_leaky | test | avd | 18 | 34 | 52 |
| p2_all_leaky | **all** | all | 1262 | 1377 | 2639 |

## Dropped by precedence rule

| protocol | dataset | split dropped | reason | n |
|---|---|---|---|---|
| p1_rwf | rwf | train | identical_clip | 6 |
| p1_rwf | rwf | train | group_in_test | 6 |
| p2_all | sfd | val | group_in_test | 9 |
| p2_all | rwf | train | identical_clip | 6 |
| p2_all | rwf | train | group_in_val | 4 |
| p2_all | sfd | train | group_in_val | 24 |
| p2_all | rwf | train | group_in_test | 15 |
| p2_all | sfd | train | group_in_test | 11 |
| p3_no_rwf | sfd | val | group_in_test | 9 |
| p3_no_rwf | sfd | train | group_in_test | 11 |
| p3_no_sfd | rwf | train | group_in_test | 9 |
| p3_no_avd | rwf | train | group_in_val | 4 |
| p3_no_avd | sfd | train | group_in_val | 24 |
| p2_all_leaky | rwf | train | identical_clip | 6 |

Total dropped rows: 144

## Test clips with shared_source True

| protocol | test clips | shared_source True |
|---|---|---|
| p1_rwf | 400 | 17 |
| p1_sfd | 47 | 17 |
| p1_avd | 52 | 0 |
| p2_all | 499 | 34 |
| p3_no_rwf | 400 | 17 |
| p3_no_sfd | 47 | 17 |
| p3_no_avd | 52 | 0 |
| p4_grouped | 207 | 6 |
| p4_random | 240 | 4 |
| p2_all_leaky | 499 | 34 |
