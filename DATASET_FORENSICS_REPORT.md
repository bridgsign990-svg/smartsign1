# Static A-Z Dataset Forensics Report

Audit date: 2026-10-04

This is a read-only forensic analysis of the existing static landmark dataset. No production dataset rows, model weights, feature extractor, thresholds, or live recognition code were changed for this forensic task. The previously supplied capture-block metrics were recomputed after discovering that the earlier diagnostic clone omitted the trainer's augmentation. The corrected results below replay `model_trainer.py`'s existing noise/scale augmentation and mirror-view augmentation on training blocks only; held-out blocks remain untouched.

## 1. Executive Summary

- Active dataset: `data/dataset.csv`, 14,950 samples, 26 labels A-Z, 42 float features per row.
- Counts range from 500 to 650 samples/class; exact duplicate feature vectors are zero, both within and across labels. At Euclidean distance thresholds 1e-4 and 0.01, no block-2 or block-3 examples were nearest-neighbor duplicates of round-1 training vectors.
- The CSV has three contiguous class-ordered row blocks. These are inferred recording rounds, not verified sessions: there are no signer, timestamp, recording, camera-side, handedness, or lighting fields.
- Under a capture-block split with the current training augmentations replayed, round 1 to round 2 yields **80.65% accuracy and 0.762 macro F1**. Training on rounds 1-2 and testing the partial round 3 yields **76.82% accuracy** on its 13 represented classes and **0.739 macro F1** on those supported classes (0.369 macro F1 over all 26 labels, with 13 labels absent).
- The previous claims K->J 300/300, Y->J 300/300, and P->T 152/300 are **not reproduced** under this faithful recipe. K and Y score 299/300 and 300/300 respectively in block 2. P is instead predicted Q for 300/300 block-2 samples and Q for 121/150 block-3 samples.
- The most concerning reproducible block shifts are P->Q and Z->B, each 300/300 in block 2. M, S, V and U also have substantial, varied errors. Without source visuals and metadata, this cannot establish mislabeled signs versus capture-domain/pose changes.
- Two JPEGs are present in `data/`, but neither is linked to CSV rows; `dataset/` is empty. They were not opened or associated with samples, and no contact sheet was made. The feature CSV cannot show the original hand images.
- Final decision: **multiple problems, led by unverified capture-block/domain shift and likely data-label review needs**. Do not relabel or retrain automatically.

## 2. Dataset Inventory

| File | Samples | Classes | Width | Class-count range | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `data/dataset.csv` | 14,950 | 26 | 43 columns | 500-650 | Active static data: label + 42 features |
| `data/dataset_backup.csv` | 11,070 | 26 | 43 columns | 300-650 | Backup |
| `data/dataset_before_recollect.csv` | 10,380 | 26 | 43 columns | 300-600 | Backup |
| `data/dataset_from_images.csv` | 0 | 0 | empty | n/a | No image-derived samples |
| `data/dataset_legacy_42_features_backup.csv` | 7,430 | 26 | 43 columns | 120-650 | Backup |
| `data/dataset_old_backup.csv` | 5,200 | 26 | 43 columns | 200 | Backup |
| `data/dataset_pre_retrain.csv` | 13,000 | 26 | 43 columns | 500 | Backup |
| `data/gesture_dataset.csv` | 3,558 | 22 | 631 columns | 36-245 | Separate temporal sequence dataset; excluded from static analysis |

For the active `dataset.csv`:

- Missing labels, malformed rows, non-finite values, wrong feature width: **0 detected**.
- Exact duplicate vectors under the same label: **0**.
- Exact duplicate vectors shared across different labels: **0**.
- Unique vectors: **14,950 / 14,950**.
- Near-duplicate check: nearest round-1 training-row Euclidean distance for round 2 had median 0.3941 (p05 0.1904, p95 1.6746); round 3 median 0.3671 (p05 0.2031, p95 1.8051). No nearest distances were <=1e-4 or <=0.01. These are numeric thresholds, not perceptual definitions of duplicate signs.
- Nearest round-1 sample label matched the CSV label for 81.56% of block-2 rows and 74.26% of block-3 rows. Mismatches alone do not indicate bad labels.

Class counts:

| Class | Samples | Class | Samples | Class | Samples | Class | Samples |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| A | 650 | B | 500 | C | 500 | D | 500 |
| E | 500 | F | 500 | G | 650 | H | 650 |
| I | 500 | J | 650 | K | 650 | L | 500 |
| M | 650 | N | 650 | O | 500 | P | 650 |
| Q | 500 | R | 650 | S | 650 | T | 650 |
| U | 650 | V | 650 | W | 500 | X | 500 |
| Y | 500 | Z | 500 |  |  |  |  |

The 1.30:1 max/min class-count ratio is modest and does not explain the observed complete class-block failures by itself.

## 3. Capture / Session Analysis

The active CSV has 65 contiguous same-label runs, grouped into three repeated blocks:

| Inferred block | Rows, 1-based inclusive | Samples | Labels |
| --- | --- | ---: | ---: |
| 1 | 1-5,200 | 5,200 | A-Z, 200 each |
| 2 | 5,201-13,000 | 7,800 | A-Z, 300 each |
| 3 | 13,001-14,950 | 1,950 | J, A, S, T, M, N, U, V, R, K, P, G, H, 150 each |

The block boundaries are **inferred** from ordered repeated class runs and the collector implementation. The dataset does not prove that these are separate sessions, signers, days, or conditions. `data_collector.py` writes a sample every 0.05 seconds while recording and stores no metadata to independently confirm the boundary.

For block 1, pooled across classes, the median distance between sequential neighbors was 0.1047, versus 0.3512 for randomly paired samples from the same class. Across each label's ordered rows, the adjacent-vs-random same-class medians were K: 0.0999 vs 1.5553, Y: 0.0801 vs 0.4780, P: 0.1583 vs 1.1711, S: 0.1033 vs 0.5755, and Z: 0.1161 vs 3.4664. These comparisons are consistent with temporal correlation between nearby frames and support the risk of random-row leakage. They do not identify a signer or source video.

The CSV contains no filename/index/session information beyond its physical row order and label. No class can be tied to verified backgrounds, lighting, camera angle, signer's left/right hand, or a particular physical recording.

## 4. Confusion Analysis and Corrected Evaluation

The current trainer first splits rows, then augments the training split with three noise and three scale-jitter copies per original, then adds mirrored copies except labels J/P/Q/Z. The capture-block experiment follows that same recipe, with augmentation confined to training data. Results use the deployed raw-vs-mirrored inference choice. The exact training seed and classifier parameters are retained.

| Evaluation | Training data | Test data | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Block 2 validation | Block 1: 5,200 real; 67,200 augmented | Block 2: 7,800, all 26 labels | 80.65% | 76.46% | 80.65% | 76.20% |
| Block 3 partial test | Blocks 1-2: 13,000 real; 168,000 augmented | Block 3: 1,950, 13 labels | 76.82% | 42.46% all-26 | 38.41% all-26 | 36.94% all-26; 73.88% over supported classes |

Block 3's 76.82% is not an A-Z score; half the labels are absent. The all-26 macro metrics count the unsupported labels as zero for visibility. The supported-class F1 is the useful class-average for that partial test.

The saved metric JSON reports 99.9257% on a stratified random-row holdout. `model_trainer.py` currently uses an 18% split, while the metrics JSON says 20%; the source's `cv_scores` is a one-element array containing that holdout score, not cross-validation. Samples are collected at short intervals and same-capture neighboring feature vectors are correlated, so this random-row score is not independent-recording or real-world performance.

### Block 2 per-class results

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| A | 0.674 | 0.943 | 0.786 | 300 |
| B | 0.486 | 0.990 | 0.652 | 300 |
| C | 1.000 | 0.990 | 0.995 | 300 |
| D | 0.974 | 0.757 | 0.852 | 300 |
| E | 0.966 | 0.950 | 0.958 | 300 |
| F | 1.000 | 1.000 | 1.000 | 300 |
| G | 0.997 | 0.997 | 0.997 | 300 |
| H | 0.997 | 1.000 | 0.998 | 300 |
| I | 0.926 | 1.000 | 0.962 | 300 |
| J | 0.960 | 0.790 | 0.867 | 300 |
| K | 0.987 | 0.997 | 0.992 | 300 |
| L | 1.000 | 1.000 | 1.000 | 300 |
| M | 0.661 | 0.540 | 0.594 | 300 |
| N | 0.718 | 1.000 | 0.836 | 300 |
| O | 0.987 | 1.000 | 0.993 | 300 |
| P | 0.000 | 0.000 | 0.000 | 300 |
| Q | 0.500 | 1.000 | 0.667 | 300 |
| R | 0.539 | 0.990 | 0.698 | 300 |
| S | 0.062 | 0.017 | 0.026 | 300 |
| T | 0.867 | 1.000 | 0.929 | 300 |
| U | 0.717 | 0.910 | 0.802 | 300 |
| V | 0.867 | 0.130 | 0.226 | 300 |
| W | 1.000 | 0.983 | 0.992 | 300 |
| X | 1.000 | 0.987 | 0.993 | 300 |
| Y | 0.997 | 1.000 | 0.998 | 300 |
| Z | 0.000 | 0.000 | 0.000 | 300 |

### Block 3 per-class results

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| A | 0.613 | 0.973 | 0.753 | 150 |
| G | 1.000 | 1.000 | 1.000 | 150 |
| H | 1.000 | 1.000 | 1.000 | 150 |
| J | 1.000 | 1.000 | 1.000 | 150 |
| K | 1.000 | 1.000 | 1.000 | 150 |
| M | 0.540 | 0.633 | 0.583 | 150 |
| N | 0.843 | 1.000 | 0.915 | 150 |
| P | 1.000 | 0.193 | 0.324 | 150 |
| R | 0.543 | 1.000 | 0.704 | 150 |
| S | 0.500 | 0.027 | 0.051 | 150 |
| T | 1.000 | 1.000 | 1.000 | 150 |
| U | 1.000 | 0.160 | 0.276 | 150 |
| V | 1.000 | 1.000 | 1.000 | 150 |

## 5. K vs J Investigation

- The earlier K->J 300/300 report came from an unaugmented diagnostic clone and is not reproduced when the existing trainer augmentation is replayed.
- Block 2: K predicts K for 299/300; one sample predicts Z. Raw and mirrored views each are 99.67% correct. Block 3 K is 150/150 correct.
- Block 2 K's nearest round-1 training row is labeled R for all 300 examples (median distance 1.2605), while the RF still classifies 299 correctly. This indicates an out-of-prototype feature block, but not a K-vs-J feature overlap.
- Block-2 K centroid is closer to the round-1 K centroid (1.902) than J (3.014); round-1 K/J centroid separation is 3.766, about 27.84 pooled coordinate SD RMS units.
- Conclusion: no current evidence that K is actually being confused with J under the faithful trainer recipe. The K block is feature-shifted toward a third class in nearest-neighbor analysis; cause requires visual/session evidence.

## 6. Y vs J Investigation

- The earlier Y->J 300/300 result is not reproduced under the trainer augmentation.
- Block 2 Y predicts Y for 300/300; block-3 does not contain Y.
- All 300 block-2 Y vectors have a nearest round-1 training row labeled Y. The block centroid is closer to the Y training centroid (1.227) than J (3.077); Y/J training centroid separation is 3.287.
- Conclusion: no evidence that block-2 Y data is mislabeled or confused with J in this evaluation. Visual signs remain unavailable for independent content review.

## 7. P vs T Investigation

- The earlier P->T 152/300 confusion is not reproduced under the trainer augmentation. Block 2 P predicts Q for 300/300, and block 3 P predicts Q for 121/150, P for 29/150.
- In block 2, P raw predictions are Q across the whole class. The mirrored predictions disagree on 291/300, but the mirrored view never wins; the dual rule still outputs Q. Thus the P failure is not explained by the dual inference selecting the mirrored view.
- 268/300 block-2 P samples have a nearest round-1 training row labeled Q; only 28/300 are nearest P. The P block centroid is nearly equidistant from the round-1 P and Q centroids (2.088 vs 2.099). P/T training centroid separation is relatively large (2.157; 24.37 pooled coordinate SD RMS).
- Conclusion: P's block-2 features shifted toward Q, not T. This could reflect changed hand pose/recording condition or a label problem, but the CSV cannot prove which. Review P rows 9,701-10,000 and their source captures if recoverable.

## 8. M / S / Z Investigation

- **M:** block 2 predicts M 162/300, N 75, S 52, E 9, A 2. Block 3 improves to M 95/150 but predicts A 38 and N 17. Block-2 M centroid is closer to round-1 S (0.414) than M (0.501); nearest round-1 labels are N 161, S 83, M 56. This is strong evidence of block-dependent feature shift/overlap, not proof of a mislabeled M recording.
- **S:** block 2 predicts S only 5/300, with A 135, M 70, N 43, T 43, R 4. Block 3 predicts S 4/150, with M 81, A 54, N 11. Block-2 nearest-round-1 labels are M for 216/300 and S for 41/300. Block-3 nearest-round-1 labels are S for 145/150 even though the RF predicts S only 4/150. This disagreement between nearest-neighbor and RF evidence suggests a complex class-boundary or distribution issue; cause requires visual review.
- **Z:** block 2 predicts B for 300/300; block 3 does not include Z. Both raw and mirrored model views predict B for every block-2 Z sample, so orientation selection does not cause this error. The block centroid is closer to the round-1 B centroid (3.057) than its Z centroid (3.785), while nearest round-1 rows are J for all 300; this inconsistency indicates an out-of-distribution feature block rather than simple verified Z/B visual similarity.
- No class-level source labels can be approved from classifier output alone.

## 9. Capture-Block Confusion Tables

Sparse rows below list `predicted:count`; omitted predictions are zero.

Block 2, 300 samples per class:

| Actual | Predictions |
| --- | --- |
| A | A:283, M:13, T:3, Z:1 |
| B | B:297, D:1, K:1, V:1 |
| C | C:297, Z:3 |
| D | D:227, R:63, U:5, V:5 |
| E | E:285, J:10, B:2, O:3 |
| F | F:300 |
| G | G:299, H:1 |
| H | H:300 |
| I | I:300 |
| J | J:237, I:24, S:24, B:7, D:5, E:1, O:1, Y:1 |
| K | K:299, Z:1 |
| L | L:300 |
| M | M:162, N:75, S:52, E:9, A:2 |
| N | N:300 |
| O | O:300 |
| P | Q:300 |
| Q | Q:300 |
| R | R:297, U:2, K:1 |
| S | A:135, M:70, N:43, T:43, R:4, S:5 |
| T | T:300 |
| U | U:273, R:27 |
| V | R:160, U:101, V:39 |
| W | W:295, B:5 |
| X | X:296, G:1, K:2, Z:1 |
| Y | Y:300 |
| Z | B:300 |

Block 3, 150 samples for 13 labels:

| Actual | Predictions |
| --- | --- |
| A | A:146, S:4 |
| G | G:150 |
| H | H:150 |
| J | J:150 |
| K | K:150 |
| M | M:95, A:38, N:17 |
| N | N:150 |
| P | Q:121, P:29 |
| R | R:150 |
| S | M:81, A:54, N:11, S:4 |
| T | T:150 |
| U | R:126, U:24 |
| V | V:150 |

## 10. Mirroring / Handedness

- The 42 values are 21 MediaPipe landmarks in index order, interleaved `(x,y)`; z is not used. Features are wrist-anchored at landmark 0 and divided by the wrist-to-middle-MCP (landmark 9) distance. There is no rotation normalization.
- `data_collector.py` horizontally flips the camera frame before extracting each training vector. The active CSV stores only the resulting features, not a mirror flag or hand-side label.
- The trainer adds mirrored feature copies except J/P/Q/Z. Inference compares raw and X-mirrored feature vectors. Image upload can additionally pixel-flip the original photo during its medium-confidence retry.
- The feature vector has no handedness field and no source-media metadata. Left-vs-right signer hand effects cannot be isolated.
- The collector flips the camera frame horizontally before MediaPipe extraction. Trainer augmentation adds mirrored feature copies except J/P/Q/Z. Live and image inference score raw and X-mirrored features; image upload can additionally pixel-flip the source image.
- Block-2 dual orientation accuracy is 80.65%, raw-only 81.03%, mirror-only 79.73%. Dual disagrees between views on 660/7,800 rows; the mirror view wins on 3,389. Round-3 dual is 76.82%, raw-only 77.08%, mirror-only 72.51%.
- The most severe errors are not fixed by orientation selection: P is Q under both views in the winning raw result, and Z is B under both views. K/Y are strong under both. S and M are poor under both; mirror/raw differences exist but do not isolate handedness.
- Conclusion: orientation/mirroring affects a subset of predictions, but evidence does not support it as the primary cause of the P/Z/S/M block failures. Left/right hand causality remains untestable without metadata and visuals.

## 11. Feature-Space Findings

Feature rows are normalized wrist-anchored XY coordinates; distances below are Euclidean in that 42-dimensional space. The centroid z-distance is descriptive, not a calibrated hypothesis test. Nearest-neighbor labels use raw round-1 training features.

| Round-2 actual | Compared round-1 class | Train centroid distance | Pooled coordinate SD RMS | Standardized separation | Block centroid to own / comparison centroid | Nearest train label distribution |
| --- | --- | ---: | ---: | ---: | --- | --- |
| K | J | 3.766 | 0.135 | 27.84 | K:1.902 / J:3.014 | R:300 |
| Y | J | 3.287 | 0.128 | 25.63 | Y:1.227 / J:3.077 | Y:300 |
| P | Q | 2.718 | 0.171 | 15.90 | P:2.088 / Q:2.099 | Q:268, P:28, T:3, L:1 |
| P | T | 2.157 | 0.088 | 24.37 | P:2.088 / T:2.435 | Same P nearest-label distribution above |
| M | S | 0.470 | 0.058 | 8.16 | M:0.501 / S:0.414 | M block: N:161, S:83, M:56 |
| M | N | 0.963 | 0.078 | 12.37 | M:0.501 / N:0.728 | Same M nearest-label distribution above |
| S | M | 0.470 | 0.058 | 8.16 | S:0.714 / M:0.821 | S block: M:216, S:41, N:43 |
| Z | B | 2.975 | 0.189 | 15.77 | Z:3.785 / B:3.057 | Z block: J:300 |
| Z | W | 3.483 | 0.193 | 18.06 | Z:3.785 / W:3.435 | Same Z nearest-label distribution above |

The round-2 P centroid is almost equally distant from P and Q training centroids, corroborating the P->Q model/nearest-neighbor shift. Z shows contradictory directional evidence (RF predicts B while nearest rows are J); it is far from its own training centroid. This suggests distribution shift, but cannot name its visual cause. K/Y are near their own expected class centroids and correctly classified in this faithful evaluation. No raw image source exists to judge actual handshape similarity.

## 12. Suspected Samples and Label Quality

No label is changed or proposed for automatic correction.

`dataset_review_candidates.json` contains all 1,961 row-indexed model disagreements from the two evaluated capture blocks: 1,509 in block 2 and 452 in block 3. Of these, 1,958 are `POSSIBLE_SUSPECT` and three meet the high-vote review flag. Correct model/CSV agreements are summarized as `NORMAL` in the manifest, but that does not prove the labels are true. Sample IDs are 1-based CSV line numbers because the file stores no sample IDs.

Three candidate errors meet the report's review-priority threshold of forest vote fraction >=0.80; this vote fraction is not calibrated confidence:

| Row | Current label | Dual-view prediction | Vote fraction | Raw / mirror prediction | Nearest round-1 label | Recommendation |
| ---: | --- | --- | ---: | --- | --- | --- |
| 8,145 | J | B | 0.8138 | B / B | J | HIGH_CONFIDENCE_SUSPECT; human source review, no proposed relabel |
| 8,866 | M | S | 0.8250 | S / S | S | HIGH_CONFIDENCE_SUSPECT; human source review, no proposed relabel |
| 8,867 | M | S | 0.8375 | S / S | S | HIGH_CONFIDENCE_SUSPECT; human source review, no proposed relabel |

These samples are consecutive in the inferred round-2 M block. They may be mislabeled, may depict an atypical/capture-shifted M, or may expose a classifier boundary problem. The feature CSV cannot adjudicate this. All other flagged rows remain possible suspects with no proposed label.

## 13. Recording Conditions and Visual Inspection

- Dataset-directory image source: none; `dataset/` is empty.
- The image-derived CSV is empty.
- Two unlinked JPEG files exist in `data/`; no dataset row has a media path or stable ID mapping to them. They were not opened or used for contact sheets.
- No original video, frame sequence, sample timestamps, signer IDs, background/lighting tags, orientation flags, or hand-side labels are in the active dataset.
- Therefore visual K/J, Y/J, P/T, M/S/Z comparisons and lighting/background checks are unavailable. No visual sign-shape conclusions are made.
- Inferred capture blocks are ordered class runs. The 50 ms collector interval and feature adjacency support temporal sampling, but do not prove the actual recording/session identity or condition change.

## 14. Recommendations and What Not to Change Yet

1. Human-review round-2 P rows 9,701-10,000 and Z rows 12,701-13,000 first, using original source video if recoverable. Verify the target label before any reannotation.
2. Review round-2 S/M/V and round-3 S/U/P blocks. Their error signatures differ by block, so review source conditions and labels together rather than treating each class as globally bad.
3. Keep the original CSV immutable. Use the row-indexed review manifest to approve/reject individual evidence; do not infer relabels from nearest neighbors or model votes.
4. For future genuine captures, record signer ID, session ID, timestamp/order, hand side, camera facing/mirrored state, and available lighting/background/pose notes. Keep people-specific information access-controlled.
5. After reviewing existing labels, reserve a signer-separated held-out set. Only then tune the current model or evaluate feature alternatives.

Do not yet change the Random Forest, 42-feature representation, classifier thresholds, or class labels based on this forensic report. Do not synthesize samples or import external sign datasets.

## 15. Final Decision

**Decision: multiple issues; prioritize capture/label verification and metadata.** The current faithful evaluation does not reproduce K->J, Y->J, or P->T. It instead shows a block-2 P->Q and Z->B collapse, broad S/M/V errors, and additional block-3 P/U/S errors. Feature-space evidence supports strong block shifts for P, M, S, and Z, but does not establish incorrect labels. K and Y should remain unchanged based on current evidence.

- Which classes are problematic: block-2 P/S/Z/M/V; block-3 P/S/U/M; plus D and R in block 2.
- Why K->J and Y->J: not reproduced under the trainer-faithful augmentation; round-2 K/Y perform well. Do not treat earlier unaugmented-clone confusions as current evidence.
- Why P->T: not reproduced; faithful run shows P->Q. P block 2 aligns feature-wise with Q, requiring source review.
- Is the dataset mislabeled? Two high-vote M->S sample candidates merit human review; no label error is proven.
- Does mirroring explain the failures? No; raw-only is slightly better overall, and P/Z errors persist across orientation handling. Handedness causality remains untestable.
- Is there visual evidence? No row-linked source imagery/video is available.

### NEXT ACTION

1. Verify the original source labels/media for block-2 P and Z before changing any rows.
2. Human-review the two M->S rows and the block-2/3 S/M/V/U/P disagreement groups.
3. Add signer/session/orientation metadata to future genuine recordings.
4. Preserve the current dataset/model until review decisions are recorded.
5. Retrain only after label review and a signer-separated evaluation split exist.
