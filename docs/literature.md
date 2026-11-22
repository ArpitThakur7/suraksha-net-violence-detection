# Literature check and positioning (Task 2)

Date: 2026-11-23. Scope: RWF-2000, SFD (Surveillance Fight Dataset, called SCFD or SURV in some papers), AIRTLab. Every number below comes from a source that was opened (full text, arXiv page, or an API record of the paper). Where a paper was blocked (MDPI, IEEE Xplore and PMC returned 403 or a bot check) the row says so, and numbers quoted from another paper are labelled "quoted in X". "n/s" means not stated in what we read. Facts marked "our audit" are our own measurements, not literature.

## Impact on paper claims (read first)

The honest claim is narrower than spec section 2 implies. The spec text is mostly right ("the claim is the combination ... not the individual blocks"), but four things must change before drafting.

1. **MotionLite is not a novel architecture.** Every ingredient has been published for violence detection on 2D efficient backbones: TSM with MobileNetV2 (Zhang 2022, `zhang2022mobilenettsm`, 87.75% on RWF), frame-difference streams with MobileNetV2 (Islam 2021, `islam2021sepconvlstm`, up to 89.75% on RWF), MobileNetV3 / EfficientNet-B0 with a frame-difference spatial attention plus a temporal attention block (Kang 2021, `kang2021efficient`), EfficientNet-B0 with TSM plus bidirectional motion attention (Wang 2024, `wang2024bimotion`), and gated fusion of two branches (FuseMamba-VD, `senadeera2025fusemamba`; Flow Gated Network, `cheng2020rwf`). The closest prior work is Kang 2021. We found no paper that combines TSM, an RGB-difference stream, learned-gate mid-fusion and temporal attention pooling on a MobileNetV3/EfficientNet-B0 backbone, but that gap is a recombination. Call MotionLite a "strong, fully ablated lightweight model evaluated under a leakage-controlled protocol", not "a new architecture". Contribution (3) is the weakest.
2. **Contribution (1), the leakage audit, is only partly new.** Source-ID grouping is already done by the RWF authors (`cheng2020rwf`). Same-source leakage in RWF and SFD splits is discussed in `veltmeijer2024subgroup`. A cross-dataset near-duplicate check exists in `senadeera2025fusemamba`. What we found no trace of is a **source-ID-level RWF to SFD overlap** (our audit: 57 RWF clips, 40 train and 17 val, in 10 shared YouTube source groups), or a grouped protocol spanning RWF, SFD and AIRTLab (including AIRTLab's two-camera pairs). Claim the cross-dataset overlap finding and the released manifests, not "leakage exists".
3. **Contribution (2), quantified evaluation bias, is the most novel.** No paper located quantifies random-clip versus grouped split inflation, or test-tuned versus validation-tuned thresholds, for violence detection. Papers state the problem qualitatively (`veltmeijer2024subgroup`) or show it implicitly (AIRTLab results use random 16-frame-chunk splits, `contardo2023combining`). Our searches are limited to what the web tools surfaced, so say "to our knowledge".
4. **Contribution (4) is partly published.** Leave-one-dataset-out or train-on-A-test-on-B experiments exist (`coskun2026privacy`, `rendon2024crimenet`, `tan2026gvd`, `veltmeijer2024subgroup`, `jung2025shortwindow`). The new part is covering SFD and AIRTLab, doing it with grouped and overlap-removed splits, and adding calibration, false alarms per hour and measured edge latency, none of which we found reported in RWF-based violence work. SFD appears only in the SFD-to-RWF transfer of `veltmeijer2024subgroup`; AIRTLab appears in no cross-dataset study we found.

Two more points affect positioning. (a) Published RWF-2000 numbers span 77% to 99.98% under mixed protocols; lightweight models under stated protocols sit around 82 to 90%, so the spec's "about 90%+" holds only for large models (CUE-Net 94.00% at 354M parameters) or runs with unclear protocol. (b) Our audit (57 RWF clips share source groups with SFD) means every published RWF/SFD transfer number we found (for example `veltmeijer2024subgroup`, about 83% on RWF when trained on SFD) may be contaminated. `senadeera2025fusemamba` reports no SFD-test versus RWF/RLVS-train near-duplicates (VideoMAE embedding cosine similarity at a 0.75 threshold). That differs from our audit and must be reconciled: their check covers only SFD test clips against RWF and RLVS training clips, and uses their own SFD 80/20 split.

## 1. RWF-2000 results, 2021 to 2026

Reference anchor: Cheng et al., ICPR 2020 (published 2021), Flow Gated Network, 87.25% on the official test split, 0.27M parameters. The paper states it records source-video IDs so clips of one source do not fall on both sides of the split. https://doi.org/10.1109/ICPR48806.2021.9412502 , https://arxiv.org/abs/1911.05913

"Official" means the 1600/400 train/val split shipped with RWF-2000. "Flag" is our reading of the paper, not an accusation.

| # | Citation | Model / approach | Pretraining | Params / GFLOPs | Protocol | RWF-2000 result | Flag (tuning / leakage) | Link |
|---|---|---|---|---|---|---|---|---|
| 1 | Islam et al., IJCNN 2021 | Two-stream MobileNetV2(0.35) + SepConvLSTM; one stream background-suppressed frames, one stream frame differences | ImageNet (MobileNet) | 0.333M (best, M fusion); 0.371M (C fusion); FLOPs column prints 2.0M for 2 streams (units as printed) | "20% of the dataset" for test (official split not named) | Acc 87.25 / 87.75 / 89.25 for fusion variants in Table I as extracted; 89.75 for M fusion in Table III | Three fusion variants compared on the test split, no validation set described: selection on test possible | arXiv 2102.10590, doi 10.1109/IJCNN52387.2021.9534280 |
| 2 | Kang, Park, Park, IEEE Access 2021 | MobileNetV3 / EfficientNet-B0 with frame-grouping, Motion Saliency Map (frame-difference spatial attention) and Temporal Squeeze-and-Excitation | n/s (abstract only) | n/s | n/s (full text blocked) | 92.0, quoted in `rendon2024crimenet` and `veltmeijer2024subgroup` tables; abstract claims state of the art on six datasets | Cannot assess; primary text not read | doi 10.1109/ACCESS.2021.3083273 |
| 3 | Vijeikis et al., Sensors 2022 | MobileNetV2 U-Net-like encoder + LSTM | n/s | n/s ("computationally light") | "A dataset based on RWF-2000" (derived set), protocol n/s | Acc 0.82 +/- 2%, precision 0.81 +/- 3% | Derived dataset, not comparable to the official split | doi 10.3390/s22062216 |
| 4 | Zhang, Li, Guo, PLOS ONE 2022 | MobileNet-TSM (MobileNetV2 + modified temporal shift) | n/s | 8.49 MB parameters, 175.86 MB estimated size (units as printed) | n/s | Acc 87.75 (Baseline+TSM-2); also 97.5 Hockey, 97.959 Crowd | Text says "the highest accuracy we trained" and the comparison table is titled "optimal accuracy": suggests best-of-runs on the evaluation split | doi 10.1371/journal.pone.0276939 |
| 5 | Ghosh and Chakrabarty, arXiv 2022 | 2s-MDCN, RGB + optical flow, 1D/2D/3D convs | none (from scratch) | 0.94M, 8.16 GFLOPs (table; text says 0.92M and 8.97) | 1600 train / 400 val, no separate test described | Acc 89.7 (fusion), 87.5 RGB-only, 80.5 flow-only | Only a val split exists, so the number is a val number used as test | arXiv 2211.04255 |
| 6 | Wang, Yang, Kasabov, Sensors 2023 | Deep spiking neural network | n/s | n/s | Random 70/10/20 split per run, 5 runs | Acc 88.35 (mean), variance 1.77 | Random clip-level split, not source-grouped | doi 10.3390/s23094532 |
| 7 | Veltmeijer, Franken, Gerritsen, MTAP 2024 | Full-video model plus subgroup tracking module | n/s | n/s | Official RWF split, no cross-validation | Weighted acc 87.5 baseline, 87.2 with subgroups; F1 0.87 / 0.86 | Clean: states the official split and discusses leakage | doi 10.1007/s11042-024-19144-5 |
| 8 | Senadeera et al., CVPRW 2024 (CUE-Net) | Spatial cropping + UniformerV2 + Modified Efficient Additive Attention | CLIP-ViT-initialised blocks; Kinetics-400 per `senadeera2025fusemamba` | 354M parameters; 5,826 GFLOPs in its ablation (`senadeera2025fusemamba` lists 5,906) | Predefined 80/20 split | Acc 94.00 (RLVS 99.50) | No tuning described; very large model; competitor numbers are quoted | arXiv 2404.18952 |
| 9 | Pathak et al., arXiv 2024 | X3D with person-tube cropping | n/s | n/s | 70-10-20 split of non-overlapping 128-frame clips (not official) | Acc 84.02 (VioPeru 62.96, own enactments 95.6) | Clips cut from videos then split; grouping not described | arXiv 2412.02127 |
| 10 | Huang and Jiang, Sensors 2025 (IDG-ViolenceNet) | YOLOv11x tracking graph + R3D-18 | Kinetics-400 (R3D-18) | YOLO11x 56.9M params, 194.9 GFLOPs (detector only) | Own 6:2:2 video-level class-stratified split (not official) | Acc 89.40 | Own split; says near-duplicate frames do not cross partitions | doi 10.3390/s25206272 |
| 11 | Chang et al., arXiv 2025 (Vi-SAFE) | GE-YOLOv8 person detector + TSN | n/s | Detector only: 3.53M, 9.2 GFLOPs (pruned) | 8:2 split, 5 runs | Acc 0.880 +/- 0.007 | Margin over OpenPose+ST-GCN (0.878, quoted) lies inside its own std | arXiv 2509.13210 |
| 12 | Jung et al., arXiv 2025 | InternVL3 on short sliding windows, LLM-labelled extra data | n/s | n/s | Split n/s; training mixes RWF, AI Hub CCTV, SCVD | Acc 95.25 (RWF-only training 91.78) | Single run; no duplicate check between mixed sources described | arXiv 2511.10866 |
| 13 | Tao et al., arXiv 2025 (FedVideoMAE) | Frozen VideoMAE + LoRA adapters, federated, differential privacy | VideoMAE (self-supervised) | 5.52M trainable, 156.4M total | Standard split: 1,600 development, 400 test | Test acc 77.25 (no DP); 65.25 to 66.00 (DP + secure aggregation) | Clean split; not a performance paper | arXiv 2512.18809 |
| 14 | Benschop et al., arXiv 2025 | Zero-shot Vision-LLMs (Gemma-3 4B, NVILA-8B, Qwen2.5-VL-7B, VideoLLaMA-3 7B) | Foundation models, no training | 4B to 8B | "Canonical splits"; eval subset size n/s | Top-1 82.25 to 86.25; false-positive rate 8.5 to 24.5% | Few-shot exemplars come from the training split | arXiv 2510.23190 |
| 15 | Rendon-Segador et al., Sensors 2024 (CrimeNet) | ViT with structured learning and adversarial regularisation | n/s | n/s | **Frame-level** labelling, five-fold cross-validation | AUC-ROC 99.98 +/- 0.02 (accuracy not extracted) | Frame-level folds: frames of one video can sit in train and test; see section 4 | doi 10.3390/s24165429 |
| 16 | Senadeera et al., arXiv 2025 (FuseMamba-VD) | Dual-branch VideoMamba with gated class-token fusion | Kinetics-400 | 154.3M, 1,912 GFLOPs | Combined RWF+RLVS+SURV+VioPeru, official splits, embedding dedupe; 915 test clips | Acc 95.85 on the combined test set (RWF-only n/s) | Duplicate check done; not an RWF-only number | arXiv 2506.03162 |
| 17 | Coskun et al., arXiv 2026 | MobileNetV2 + LSTM / BiLSTM / temporal CNN | ImageNet | n/s | Own 6,105-clip manifest (SCVD, RWF, RLVS); in-domain RWF slice | Trained on RWF: acc 0.785 to 0.828, AUC 0.890 to 0.897 in-domain | Own manifest; source grouping n/s | arXiv 2606.01225 |
| 18 | Ullah et al., 2024 IEEE (Dual Deep Learning Network) | ConvNet (EfficientNet-style) + LiteFlowNet features, two GRUs, fused | n/s | n/s | Random-style 70/10/20 split (stated in text) | Acc 90.10 (Table 3) and 94.50 (Table 5) on RWF; SFD 96.69 | The two tables disagree on RWF; random split | clok.uclan.ac.uk/53131 |
| 19 | Tan et al., Sci Rep 2026 (GVD) | Vision-language model with semantic differentiation and orthogonal supervision | n/s | 58.98M, 42.74 GFLOPs (its own table) | Leave-RWF-out: train RLVS+OWF-B+HF, test RWF | 86.36 (metric label n/s); baselines 71.00 to 81.45 | Baseline numbers are round and unusually regular; treat with caution | doi 10.1038/s41598-026-50832-4 |
| 20 | Chen et al., Appl Sci 2025 (DualCascadeTSF-MobileNetV2) | Two cascaded temporal shift modules on MobileNetV2 | n/s | n/s | n/s | **RWF accuracy not obtained** (MDPI returned 403) | Listed because it is direct prior art for TSM on MobileNet | doi 10.3390/app15073862 |

Numbers quoted from other papers' comparison tables (pointers only, not primary): Video Swin Transformer 91.25 and ACTION-VST 93.59 (quoted in `senadeera2024cuenet`); Structured Keypoint Pooling 93.40 (`hachiuma2023keypoint`, quoted in both `senadeera2024cuenet` and `rendon2024crimenet`); Huszar et al. 2023 transfer-learning 3D CNN 0.850 (`huszar2023toward`, quoted in `chang2025visafe`); SPIL 89.30 (quoted in `ullah2024dual`); Su et al. 94.0 (quoted in `veltmeijer2024subgroup`; reference not resolved, so it is not in the bib block).

Reading the table: 18 rows carry a number from a source we opened. Lightweight 2D models under stated protocols cluster at 82 to 90%; 91 to 95% appears with very large models, VLM-trained mixes, or unclear splits. Only rows 7, 8 and 13 state the official or predefined split with no obvious tuning on it.

### SFD and AIRTLab

**SFD** (Akti et al., IPTA 2019, 300 clips). `veltmeijer2024subgroup` states that Su et al. were the first to mention split leakage for SFD and used five leakage-free folds. Quoted in its Table 3: Akti et al. 72.0; Ullah et al. 74.0 and 75.9; Kang et al. 92.0; Tan and Liu 95.6; Su et al. 88.7 (leakage-free folds); Veltmeijer baseline 90.0 and subgroup model 91.3 (same folds). `ullah2024dual` reports 96.69 on SFD and quotes the same 72.0 and 92.0 figures. Only the Su and Veltmeijer rows are stated to be leakage-controlled. https://doi.org/10.1109/IPTA.2019.8936070 , https://arxiv.org/abs/2002.04355

**AIRTLab** (Bianculli et al., Data in Brief 2020). 350 clips (230 violent, 120 non-violent), 1920x1080, 30 fps; every action is recorded by two cameras from two spots (`bianculli2020dataset`, read via Europe PMC). Results read in `contardo2023combining`: Sernani 2021 C3D+SVM accuracy 96.10 +/- 0.71 (AUC 99.30) and C3D+FC 95.62 +/- 0.42; Contardo 2023 MobileNetV2+ConvLSTM 94.10 +/- 0.91 (AUC 98.26) and MobileNetV2+BiLSTM 85.08 +/- 9.18; a 2D-CNN frame model quoted at 99.4. **Protocol flag:** the five splits are random stratified 80/20 over **16-frame chunks** (3,537 chunks, 2,829 train and 708 test) with no grouping by action or camera, so chunks of one clip, and the same action seen by the other camera, can land on both sides. Treat all published AIRTLab numbers as leakage-prone. https://doi.org/10.1109/ACCESS.2021.3131315 (the Sernani 2021 text itself was blocked; its numbers come from Contardo).

## 2. Cross-dataset generalisation studies

| Study | Setup | What dropped | Caveats |
|---|---|---|---|
| Coskun et al. 2026 (`coskun2026privacy`) | MobileNetV2 + LSTM/BiLSTM/TCN on SCVD, RWF-2000, RLVS; seven in-domain and cross-dataset scenarios; selection by validation AUC, cross-domain sets touched only after selection | Train RWF: in-domain acc 0.785 to 0.828 versus SCVD 0.660 to 0.679 and RLVS 0.657 to 0.720. Train SCVD to RWF 0.636 to 0.670; train RLVS to RWF 0.596 to 0.620; train SCVD+RLVS to RWF 0.667 to 0.677 (in-domain 0.98). RWF stays hard even with two other sources | No source-overlap audit described; closest to our LODO but without SFD or AIRTLab |
| Rendon-Segador et al. 2024 (`rendon2024crimenet`) | CrimeNet trained on one dataset, tested on others (UCF-Crime, XD-Violence, NTU CCTV, UBI, RLVS, RWF, ...) | 20 to 30% decrease versus in-domain; UCF-Crime to XD-Violence AUC-ROC 70.20; sliding window with adaptive threshold recovers 10 to 15% | In-domain frame-level numbers near 100% (leakage risk), so the drop is partly an inflated baseline |
| Veltmeijer et al. 2024 (`veltmeijer2024subgroup`) | Train on whole SFD, test on whole RWF, and the reverse | Trained on SFD, RWF about 83% (83.3 baseline, 82.4 subgroup) versus 87.5 in-domain; the other direction in Table 3 reads 86.3 / 88.3 (direction inferred from the text) | Authors note these runs do not meet their own leakage criteria (whole dataset to whole dataset); see our overlap finding |
| Tan et al. 2026 (`tan2026gvd`) | Leave-one-out over RLVS, RWF, HF and a new open-world set | Leave-RWF-out: baselines 71.00 to 81.45, GVD 86.36 | Metric label not recovered; baselines suspiciously regular |
| Jung et al. 2025 (`jung2025shortwindow`) | Train on different sources, evaluate on RWF | UCF-Crime long 55.75, UCF-Crime short 83.25, SCVD 88.75; mixed 95.25 | Table caption and text disagree on the evaluation target |
| Pathak et al. 2024 (`pathak2024streamlining`) | One model, three test sets | RWF 84.02, VioPeru 62.96 | Cross-dataset protocol not described |
| Kollias et al. 2025 (`kollias2025dvd`) | New DVD dataset | No numeric cross-dataset results; says RWF is action-only and rarely has night scenes | Dataset paper |

Gap: none uses AIRTLab, none builds an RWF + SFD + AIRTLab LODO matrix, and none reports calibration, false alarms per hour or latency alongside transfer numbers.

## 3. Lightweight and efficient models, and the overlap with MotionLite

| Work | Contribution (from the abstract) | Overlap with MotionLite |
|---|---|---|
| TSM, Lin et al. (`lin2019tsm`) | Shifts part of the channels along time at near-zero extra FLOPs; 2D CNN approaches 3D | Direct building block (M2 and M5); already used with MobileNetV2 for violence (`zhang2022mobilenettsm`, `chen2025dualcascade`) |
| TDN, Wang et al. (`wang2021tdn`) | Temporal difference module: local difference over consecutive frames feeds 2D CNNs; cross-segment difference excites features | The RGB-difference idea; TDN is a general video architecture with no violence focus |
| TEA (`li2020tea`), TEINet (`liu2020teinet`) | Feature-level temporal-difference motion excitation (TEA); motion-enhanced plus temporal-interaction modules (TEINet) | Close to a learned, gated use of differences inside the backbone; MotionLite's gated mid-fusion overlaps in spirit |
| MoViNet (`kondratyuk2021movinet`) | Searched efficient 3D CNNs with causal, streaming inference and small memory | Alternative edge baseline; streaming inference is a deployment feature we do not claim |
| X3D (`feichtenhofer2020x3d`) | Progressive expansion of a tiny 2D net across space, time, width and depth | Efficiency baseline; used for violence in `pathak2024streamlining` |
| MobileNetV3 (`howard2019mobilenetv3`), EfficientNet (`tan2019efficientnet`) | Mobile backbones | Our backbones; Kang 2021 already used both for violence |
| Kang et al. 2021 (`kang2021efficient`) | MobileNetV3 / EfficientNet-B0, frame-grouping, difference-based spatial attention (MSM), temporal attention (T-SE) | **Closest prior art**: lightweight 2D backbone, difference-derived motion cue, temporal attention |
| Wang et al. 2024 (`wang2024bimotion`) | EfficientNet-B0 with Bi-LTMA motion attention plus TSM (title and search snippet only; full text blocked) | Overlaps backbone, TSM and motion attention |
| Islam et al. 2021 (`islam2021sepconvlstm`) | MobileNetV2 two-stream (differences + background-suppressed frames), SepConvLSTM, 89.75% | Overlaps the difference stream and two-stream fusion (three fixed fusion variants, not a learned gate) |
| Flow Gated Network (`cheng2020rwf`) | A flow branch gates the RGB branch's features | The gating idea; MotionLite gates mid-network with an RGB-difference input rather than optical flow |
| 2s-MDCN (`ghosh2022twostream`) | 0.94M parameters, 8.16 GFLOPs, RGB + optical flow | Efficiency reference point for the latency table |
| FuseMamba-VD (`senadeera2025fusemamba`) | Gated class-token fusion between two branches | Gated fusion concept in a heavier SSM model |
| MobileNetV2 + LSTM family (`vijeikis2022efficient`, `contardo2023combining`) | Lightweight CNN-LSTM baselines | Our M1/M2-style baselines; the AIRTLab numbers here are the ones to re-run under grouping |

Summary: TSM, difference streams, gated fusion and temporal attention each exist; the closest published lightweight combinations are `kang2021efficient` and `islam2021sepconvlstm`. We found no MoViNet-based violence paper with verified RWF numbers (a MoViNet violence model card exists on Hugging Face; no numbers were verified).

## 4. Leakage and evaluation-bias work

- **Within-dataset source grouping.** RWF-2000 recorded source IDs to keep clips of one source on one side (`cheng2020rwf`). `veltmeijer2024subgroup` states that clips in both RWF and SFD come from longer source videos, that a random split leaks, and that for SFD only Su et al. had split without leakage; they ask others to share splits.
- **Cross-dataset duplicates.** `senadeera2025fusemamba` embeds clips with VideoMAE, inspects train/test pairs above cosine similarity 0.75, and removes one RLVS test clip duplicated in training; it reports no SFD-test similarity to RWF/RLVS training clips. This is the closest existing cross-dataset audit, and it is clip-level, not source-ID-level.
- **Frame-derived data.** `glazner2025leak` clusters visually similar frames before splitting video-derived datasets (UCF101-based experiments).
- **General video benchmarks.** HACS removed videos overlapping the validation or test sets of Kinetics, ActivityNet, UCF-101 and HMDB-51 (`zhao2017hacs`); Kinetics kept one clip per YouTube link and used a classifier ranking to catch remaining duplicates (`kay2017kinetics`); `gowda2021truezeroshot` shows Kinetics-400 pretraining overlaps UCF101/HMDB51 classes, inflating zero-shot results (up to 17.1% differences in few-shot splits).
- **Random chunk or frame splits in violence work.** AIRTLab in `contardo2023combining`; frame-level CrimeNet in `rendon2024crimenet`; random 70/10/20 per run in `wang2023spiking`.
- **Threshold tuning on test.** We found no violence-detection paper that studies it. Related symptoms: best-of-runs reported as the number in `zhang2022mobilenettsm`; three fusion variants shown on one test split in `islam2021sepconvlstm`; only a validation split in `ghosh2022twostream`. `coskun2026privacy` selects on validation AUC and touches cross-domain sets only afterwards, which is rare.
- **Pretraining leakage.** Kinetics-400 pretraining versus YouTube-sourced violence clips is unexamined in violence papers; `gowda2021truezeroshot` is the nearest analogue. Mention it in limitations.

## 5. Has the RWF-2000 and SFD source overlap been reported?

Searched: web queries combining RWF-2000, SFD / SCFD / "Surveillance Camera Fight" / SURV, "overlap", "duplicate", "same YouTube source", "data leakage", "cross-dataset"; the arXiv API for "RWF-2000" (10 hits, all read or checked); the Semantic Scholar citation list of the RWF-2000 paper (251 citing records, titles scanned for audit or leakage language); Europe PMC for "RWF-2000" (36 hits, open full texts read). Full texts read for the central papers: `senadeera2025fusemamba`, `veltmeijer2024subgroup`, `coskun2026privacy`, `cheng2020rwf`. The Papers with Code RWF page was not opened.

Result: **no paper reports a source-level overlap between RWF-2000 and SFD.** What exists: (i) `senadeera2025fusemamba` finds none between SFD test and RWF/RLVS training by embedding similarity, and one RLVS-test versus RWF-train duplicate; (ii) `veltmeijer2024subgroup` notes both datasets are mostly YouTube-sourced and runs whole-dataset transfer with no overlap check; (iii) the RWF paper claims no leakage inside RWF. Our audit (57 RWF clips, 40 train and 17 val, sharing 10 YouTube source groups with SFD) is therefore unreported to our knowledge; web search cannot prove absence. SFD clips are 1.0 to 4.7 s and RWF clips 5 s, so overlap would be partial footage rather than identical files, which a clip-level embedding check at 0.75 may miss. That is a testable explanation for the difference with (i).

## 6. Positioning and honest novelty claim

MotionLite can claim: a lightweight 2D model (MobileNetV3-L or EfficientNet-B0) with temporal shift, an on-GPU RGB-difference stream, learned-gate mid-fusion and temporal attention pooling, **with each part ablated under grouped splits and several seeds**, evaluated on laptop CPU/GPU and quantised formats, with calibration and false alarms per hour. It cannot claim a new architecture, a new motion cue, a new fusion principle or state-of-the-art accuracy: Kang 2021, Islam 2021, Wang 2024 and the MobileNet-TSM papers cover the pieces, and large models report 94 to 95% on RWF under protocols that differ from ours. The defensible statement is "a carefully ablated recombination whose value is measured under a leakage-controlled protocol", with Kang 2021 named as the closest prior work. If MotionLite does not beat Kang-style or MobileNet-TSM baselines in matched runs, say so; the paper still stands on contributions (1), (2) and (4).

Novelty ranking of the four planned contributions: (2) quantified evaluation bias is the most novel (not found quantified anywhere); (1) the cross-dataset source-ID audit and shared manifests across RWF, SFD and AIRTLab is novel in its cross-dataset, source-level form, but grouping and duplicate checks as ideas are established; (4) LODO plus calibration, false alarms per hour and edge latency is novel in coverage, not in the idea of cross-dataset testing (already in `coskun2026privacy`, `rendon2024crimenet`, `veltmeijer2024subgroup`, `tan2026gvd`); (3) MotionLite is the least novel. **Flag:** contribution (3) as written in spec section 2 overlaps already-published work, and "leakage matters" in contribution (1) is already published.

## 7. BibTeX (refs.bib-ready)

Authors and venues come from Crossref (DOI works) and the arXiv API (preprints). `ullah2024dual` and `contardo2023combining` were entered by hand from the manuscripts read; check them before submission. Entries with `eprint` only are preprints.

```bibtex
@inproceedings{cheng2020rwf,
  author = {Cheng, Ming and Cai, Kunjing and Li, Ming},
  title = {RWF-2000: An Open Large Scale Video Database for Violence Detection},
  booktitle = {2020 25th International Conference on Pattern Recognition (ICPR)},
  pages = {4183-4190},
  year = {2021},
  doi = {10.1109/ICPR48806.2021.9412502}
}

@inproceedings{akti2019vision,
  author = {Akti, Seymanur and Tataroglu, Gozde Ayse and Ekenel, Hazim Kemal},
  title = {Vision-based Fight Detection from Surveillance Cameras},
  booktitle = {2019 Ninth International Conference on Image Processing Theory, Tools and Applications (IPTA)},
  pages = {1-6},
  year = {2019},
  doi = {10.1109/IPTA.2019.8936070}
}

@article{bianculli2020dataset,
  author = {Bianculli, Miriana and Falcionelli, Nicola and Sernani, Paolo and Tomassini, Selene and Contardo, Paolo and Lombardi, Mara and Dragoni, Aldo Franco},
  title = {A dataset for automatic violence detection in videos},
  journal = {Data in Brief},
  volume = {33},
  pages = {106587},
  year = {2020},
  doi = {10.1016/j.dib.2020.106587}
}

@article{sernani2021airtlab,
  author = {Sernani, Paolo and Falcionelli, Nicola and Tomassini, Selene and Contardo, Paolo and Dragoni, Aldo Franco},
  title = {Deep Learning for Automatic Violence Detection: Tests on the AIRTLab Dataset},
  journal = {IEEE Access},
  volume = {9},
  pages = {160580-160595},
  year = {2021},
  doi = {10.1109/ACCESS.2021.3131315}
}

@inproceedings{islam2021sepconvlstm,
  author = {Islam, Zahidul and Rukonuzzaman, Mohammad and Ahmed, Raiyan and Kabir, Md. Hasanul and Farazi, Moshiur},
  title = {Efficient Two-Stream Network for Violence Detection Using Separable Convolutional LSTM},
  booktitle = {2021 International Joint Conference on Neural Networks (IJCNN)},
  pages = {1-8},
  year = {2021},
  doi = {10.1109/ijcnn52387.2021.9534280}
}

@article{kang2021efficient,
  author = {Kang, Min-Seok and Park, Rae-Hong and Park, Hyung-Min},
  title = {Efficient Spatio-Temporal Modeling Methods for Real-Time Violence Recognition},
  journal = {IEEE Access},
  volume = {9},
  pages = {76270-76285},
  year = {2021},
  doi = {10.1109/access.2021.3083273}
}

@article{vijeikis2022efficient,
  author = {Vijeikis, Romas and Raudonis, Vidas and Dervinis, Gintaras},
  title = {Efficient Violence Detection in Surveillance},
  journal = {Sensors},
  volume = {22},
  number = {6},
  pages = {2216},
  year = {2022},
  doi = {10.3390/s22062216}
}

@article{zhang2022mobilenettsm,
  author = {Zhang, Youshan and Li, Yong and Guo, Shaozhe},
  title = {Lightweight mobile network for real-time violence recognition},
  journal = {PLOS ONE},
  volume = {17},
  number = {10},
  pages = {e0276939},
  year = {2022},
  doi = {10.1371/journal.pone.0276939}
}

@article{wang2023spiking,
  author = {Wang, Xiang and Yang, Jie and Kasabov, Nikola K.},
  title = {Integrating Spatial and Temporal Information for Violent Activity Detection from Video Using Deep Spiking Neural Networks},
  journal = {Sensors},
  volume = {23},
  number = {9},
  pages = {4532},
  year = {2023},
  doi = {10.3390/s23094532}
}

@article{huszar2023toward,
  author = {Huszár, Viktor Dènes and Adhikarla, Vamsi Kiran and Négyesi, Imre and Krasznay, Csaba},
  title = {Toward Fast and Accurate Violence Detection for Automated Video Surveillance Applications},
  journal = {IEEE Access},
  volume = {11},
  pages = {18772-18793},
  year = {2023},
  doi = {10.1109/ACCESS.2023.3245521}
}

@inproceedings{hachiuma2023keypoint,
  author = {Hachiuma, Ryo and Sato, Fumiaki and Sekii, Taiki},
  title = {Unified Keypoint-Based Action Recognition Framework via Structured Keypoint Pooling},
  booktitle = {2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {22962-22971},
  year = {2023},
  doi = {10.1109/CVPR52729.2023.02199}
}

@article{veltmeijer2024subgroup,
  author = {Veltmeijer, Emmeke and Franken, Morris and Gerritsen, Charlotte},
  title = {Real-time violence detection and localization through subgroup analysis},
  journal = {Multimedia Tools and Applications},
  volume = {84},
  number = {7},
  pages = {3793-3807},
  year = {2024},
  doi = {10.1007/s11042-024-19144-5}
}

@article{rendon2024crimenet,
  author = {Rendón-Segador, Fernando J. and Álvarez-García, Juan A. and Soria-Morillo, Luis M.},
  title = {Transformer and Adaptive Threshold Sliding Window for Improving Violence Detection in Videos},
  journal = {Sensors},
  volume = {24},
  number = {16},
  pages = {5429},
  year = {2024},
  doi = {10.3390/s24165429}
}

@article{wang2024bimotion,
  author = {Wang, Jingwen and Zhao, Daqi and Li, Haoming and Wang, Deqiang},
  title = {Lightweight Violence Detection Model Based on 2D CNN with Bi-Directional Motion Attention},
  journal = {Applied Sciences},
  volume = {14},
  number = {11},
  pages = {4895},
  year = {2024},
  doi = {10.3390/app14114895}
}

@article{chen2025dualcascade,
  author = {Chen, Yuang and Li, Yong and Li, Shaohua and Lv, Shuhan and Lin, Fang},
  title = {DualCascadeTSF-MobileNetV2: A Lightweight Violence Behavior Recognition Model},
  journal = {Applied Sciences},
  volume = {15},
  number = {7},
  pages = {3862},
  year = {2025},
  doi = {10.3390/app15073862}
}

@article{huang2025idg,
  author = {Huang, Hong and Jiang, Qingping},
  title = {IDG-ViolenceNet: A Video Violence Detection Model Integrating Identity-Aware Graphs and 3D-CNN},
  journal = {Sensors},
  volume = {25},
  number = {20},
  pages = {6272},
  year = {2025},
  doi = {10.3390/s25206272}
}

@article{tan2026gvd,
  author = {Tan, Meng and Liu, Rongqin and Zhou, Shengkai and Zheng, Qingxiao and Lin, Ling},
  title = {GVD: general violence detection via semantic differentiation and orthogonal supervision},
  journal = {Scientific Reports},
  volume = {16},
  number = {1},
  pages = {19510},
  year = {2026},
  doi = {10.1038/s41598-026-50832-4}
}

@inproceedings{liu2020teinet,
  author = {Liu, Zhaoyang and Luo, Donghao and Wang, Yabiao and Wang, Limin and Tai, Ying and Wang, Chengjie and Li, Jilin and Huang, Feiyue and Lu, Tong},
  title = {TEINet: Towards an Efficient Architecture for Video Recognition},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {34},
  number = {07},
  pages = {11669-11676},
  year = {2020},
  doi = {10.1609/aaai.v34i07.6836}
}

@misc{senadeera2024cuenet,
  author = {Senadeera, Damith Chamalke and Yang, Xiaoyun and Kollias, Dimitrios and Slabaugh, Gregory},
  title = {CUE-Net: Violence Detection Video Analytics with Spatial Cropping, Enhanced UniformerV2 and Modified Efficient Additive Attention},
  year = {2024},
  eprint = {2404.18952},
  archiveprefix = {arXiv}
}

@misc{jung2025shortwindow,
  author = {Jung, Seoik and Song, Taekyung and Lee, Yangro and Lee, Sungjun},
  title = {Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling},
  year = {2025},
  eprint = {2511.10866},
  archiveprefix = {arXiv}
}

@misc{pathak2024streamlining,
  author = {Pathak, Gourang and Kumar, Abhay and Rawat, Sannidhya and Gupta, Shikha},
  title = {Streamlining Video Analysis for Efficient Violence Detection},
  year = {2024},
  eprint = {2412.02127},
  archiveprefix = {arXiv}
}

@misc{chang2025visafe,
  author = {Chang, Ligang and Xu, Shengkai and Shen, Liangchang and Xu, Binhan and Wang, Junqiao and Shi, Tianyu and Du, Yanhui},
  title = {Vi-SAFE: A Spatial-Temporal Framework for Efficient Violence Detection in Public Surveillance},
  year = {2025},
  eprint = {2509.13210},
  archiveprefix = {arXiv}
}

@misc{senadeera2025fusemamba,
  author = {Senadeera, Damith Chamalke and Awais, Muhammad and Li, Shibo and Kollias, Dimitrios and Slabaugh, Gregory},
  title = {FuseMamba-VD: Dual Branch VideoMamba with Gated Class Token Fusion for Violence Detection},
  year = {2025},
  eprint = {2506.03162},
  archiveprefix = {arXiv}
}

@misc{kollias2025dvd,
  author = {Kollias, Dimitrios and Senadeera, Damith C. and Zheng, Jianian and Yadav, Kaushal K. K. and Slabaugh, Greg and Awais, Muhammad and Yang, Xiaoyun},
  title = {DVD: A Comprehensive Dataset for Advancing Violence Detection in Real-World Scenarios},
  year = {2025},
  eprint = {2506.05372},
  archiveprefix = {arXiv}
}

@misc{thuau2025federated,
  author = {Thuau, Sébastien and Haidar, Siba and Chelouah, Rachid},
  title = {Federated Learning for Video Violence Detection: Complementary Roles of Lightweight CNNs and Vision-Language Models for Energy-Efficient Use},
  year = {2025},
  eprint = {2511.07171},
  archiveprefix = {arXiv}
}

@misc{tao2025fedvideomae,
  author = {Tao, Ziyuan and Xu, Chuanzhi and Jayawardana, Sandaru and Mahmood, Adnan and Bao, Wei and Thilakarathna, Kanchana and Lim, Teng Joon},
  title = {FedVideoMAE: Efficient Federated Video Moderation with Differential Privacy and Secure Aggregation},
  year = {2025},
  eprint = {2512.18809},
  archiveprefix = {arXiv}
}

@misc{benschop2025vllm,
  author = {Benschop, Pascal and Meo, Cristian and Dauwels, Justin and Mense, Jelte P.},
  title = {Evaluation of Vision-LLMs in Surveillance Video},
  year = {2025},
  eprint = {2510.23190},
  archiveprefix = {arXiv}
}

@misc{coskun2026privacy,
  author = {Coşkun, Hasan and Çolhak, Furkan and Kulakov, Andrea and Dimitrova, Vesna},
  title = {Privacy-Preserving Smart Surveillance with Cross-Dataset Violence Detection and Decentralized Evidence Governance},
  year = {2026},
  eprint = {2606.01225},
  archiveprefix = {arXiv}
}

@misc{ghosh2022twostream,
  author = {Ghosh, Dipon Kumar and Chakrabarty, Amitabha},
  title = {Two-stream Multi-dimensional Convolutional Network for Real-time Violence Detection},
  year = {2022},
  eprint = {2211.04255},
  archiveprefix = {arXiv}
}

@misc{glazner2025leak,
  author = {Glazner, Noam and Tsfaty, Noam and Shalev, Sharon and Weizman, Avishai},
  title = {Find the Leak, Fix the Split: Cluster-Based Method to Prevent Leakage in Video-Derived Datasets},
  year = {2025},
  eprint = {2511.13944},
  archiveprefix = {arXiv}
}

@misc{gowda2021truezeroshot,
  author = {Gowda, Shreyank N and Sevilla-Lara, Laura and Kim, Kiyoon and Keller, Frank and Rohrbach, Marcus},
  title = {A New Split for Evaluating True Zero-Shot Action Recognition},
  year = {2021},
  eprint = {2107.13029},
  archiveprefix = {arXiv}
}

@misc{lin2019tsm,
  author = {Lin, Ji and Gan, Chuang and Han, Song},
  title = {TSM: Temporal Shift Module for Efficient Video Understanding},
  year = {2018},
  eprint = {1811.08383},
  archiveprefix = {arXiv}
}

@misc{wang2021tdn,
  author = {Wang, Limin and Tong, Zhan and Ji, Bin and Wu, Gangshan},
  title = {TDN: Temporal Difference Networks for Efficient Action Recognition},
  year = {2020},
  eprint = {2012.10071},
  archiveprefix = {arXiv}
}

@misc{kondratyuk2021movinet,
  author = {Kondratyuk, Dan and Yuan, Liangzhe and Li, Yandong and Zhang, Li and Tan, Mingxing and Brown, Matthew and Gong, Boqing},
  title = {MoViNets: Mobile Video Networks for Efficient Video Recognition},
  year = {2021},
  eprint = {2103.11511},
  archiveprefix = {arXiv}
}

@misc{feichtenhofer2020x3d,
  author = {Feichtenhofer, Christoph},
  title = {X3D: Expanding Architectures for Efficient Video Recognition},
  year = {2020},
  eprint = {2004.04730},
  archiveprefix = {arXiv}
}

@misc{li2020tea,
  author = {Li, Yan and Ji, Bin and Shi, Xintian and Zhang, Jianguo and Kang, Bin and Wang, Limin},
  title = {TEA: Temporal Excitation and Aggregation for Action Recognition},
  year = {2020},
  eprint = {2004.01398},
  archiveprefix = {arXiv}
}

@misc{howard2019mobilenetv3,
  author = {Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
  title = {Searching for MobileNetV3},
  year = {2019},
  eprint = {1905.02244},
  archiveprefix = {arXiv}
}

@misc{tan2019efficientnet,
  author = {Tan, Mingxing and Le, Quoc V.},
  title = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  year = {2019},
  eprint = {1905.11946},
  archiveprefix = {arXiv}
}

@misc{zhao2017hacs,
  author = {Zhao, Hang and Torralba, Antonio and Torresani, Lorenzo and Yan, Zhicheng},
  title = {HACS: Human Action Clips and Segments Dataset for Recognition and Temporal Localization},
  year = {2017},
  eprint = {1712.09374},
  archiveprefix = {arXiv}
}

@misc{kay2017kinetics,
  author = {Kay, Will and Carreira, Joao and Simonyan, Karen and Zhang, Brian and Hillier, Chloe and Vijayanarasimhan, Sudheendra and Viola, Fabio and Green, Tim and Back, Trevor and Natsev, Paul and Suleyman, Mustafa and Zisserman, Andrew},
  title = {The Kinetics Human Action Video Dataset},
  year = {2017},
  eprint = {1705.06950},
  archiveprefix = {arXiv}
}

@inproceedings{contardo2023combining,
  author = {Contardo, Paolo and Tomassini, Selene and Falcionelli, Nicola and Dragoni, Aldo Franco and Sernani, Paolo},
  title = {Combining a mobile deep neural network and a recurrent layer for violence detection in videos},
  booktitle = {Proc. 5th International Conference on Recent Trends and Applications in Computer Science and Information Technology (RTA-CSIT 2023), CEUR Workshop Proceedings vol. 3402},
  year = {2023},
  url = {https://ceur-ws.org/Vol-3402/paper05.pdf}
}

@misc{ullah2024dual,
  author = {Ullah, Fath U Min and Khan, Zulfiqar Ahmad and Baik, Sung Wook and Talavera, Estefania and Anwar, Saeed and Muhammad, Khan},
  title = {Dual Deep Learning Network for Abnormal Action Detection},
  year = {2024},
  note = {Accepted manuscript, 2024 IEEE; proceedings title not stated in the manuscript we read},
  url = {https://clok.uclan.ac.uk/53131/}
}
```
