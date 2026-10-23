# Static Recognition Performance Report

Audit date: 2026-10-04

> **Superseded evaluation note:** The capture-block scores and confusion results below were produced by a diagnostic clone that omitted the trainer's noise/scale/mirror augmentation. They are retained as historical output only and should not be used for decisions. The trainer-faithful forensic evaluation, which applies the existing augmentation to training blocks only, is in [DATASET_FORENSICS_REPORT.md](DATASET_FORENSICS_REPORT.md).

This report evaluates the existing static A-Z classifier and the evidence available in the repository. The capture-round splits below are row-order block proxies, not verified signer/session splits. No real-world or signer-independent accuracy is claimed.

## Dataset

| Measure | Result |
| --- | --- |
| Active dataset | `data/dataset.csv` |
| Total static samples | 14,950 |
| Classes | 26 (A-Z) |
| Samples/class | 500-650; mean 575 |
| Class imbalance | 1.30:1 maximum/minimum |
| Malformed/missing-label/non-finite/wrong-width rows | 0 detected |
| Exact duplicate same-label feature rows | 0 detected |
| Signer count | Unknown; no signer metadata |
| Capture conditions | Unknown; no background/lighting/camera-condition metadata |
| Independent recordings | Not identifiable by ID; 3 ordered sample-count blocks are used only as capture-round proxies |
| Signer-independent evaluation | **No. Impossible with current data because signer identity is not recorded.** |
| Multiple hands / handedness | Static detector supports one hand; handedness is not stored or used |

The primary dataset has three ordered blocks: 200 samples/class for A-Z (5,200 rows), 300/class for A-Z (7,800 rows), and 150 rows for 13 classes (1,950 rows). The collector saves at 50 ms intervals and does not include timestamps/session/signer IDs. Therefore the training script's random row split leaks highly correlated neighboring samples across train and validation. No independent test set is currently designated in the repository.

Other CSV files are listed in `STATIC_RECOGNITION_AUDIT.md`; `dataset_from_images.csv` and the image directory contain no training data. The separate 3,558-row gesture-sequence CSV is out of scope.

## Model

| Measure | Result |
| --- | --- |
| Active artifact | `models/sign_model_light.pkl` |
| Algorithm | `StandardScaler` + `RandomForestClassifier` |
| Hyperparameters | 80 trees, max depth 18, min leaf 2, balanced class weights, seed 42, `n_jobs=1` |
| Features | 42 wrist-anchored normalized XY values from 21 landmarks |
| Model labels | Integer classes 0-25 map to A-Z; saved order matches classifier class IDs |
| Confidence | Maximum `predict_proba` forest vote fraction; not calibrated probability |
| Training method in current source | Stratified 18% random-row holdout; augment training partition only |
| Saved metrics metadata | Says 20% holdout and labels a single holdout as CV; metadata does not match current trainer source |

No model architecture or weights were changed. The preferred light model loads and passes its four startup sanity samples. `sign_model.pkl`, `sign_model_legacy_backup.pkl`, and `sign_model_broken.pkl` in this workspace are Git LFS pointer text files, not usable pickle artifacts; classifier fallback loads the active light model.

## Evaluation Method

The primary evaluation retrains clones of the saved pipeline on ordered capture blocks and scores the next block. This avoids row-random mixing but is not signer-independent, and the block boundaries cannot be proven to be separate signers or independently recorded sessions.

The deployed inference rule is used in reported results: score raw and horizontally mirrored features, select the orientation with the larger maximum class vote, then compare against the actual label. Round 2 is the full 26-class validation set. Round 3 is a partial 13-class test; its full-26 macro average assigns zero support/classes as zero for the all-class summary and is shown alongside the supported-class macro F1.

### Full-class capture-round validation

Train: round 1, 5,200 rows. Validation: round 2, 7,800 rows, 300 per class.

| Metric | Result |
| --- | ---: |
| Accuracy | 67.46% |
| Macro precision | 67.10% |
| Macro recall | 67.46% |
| Macro F1 | 63.53% |

### Partial independent-block test proxy

Train: rounds 1 and 2, 13,000 rows. Test: round 3, 1,950 rows, 150 each for A, G, H, J, K, M, N, P, R, S, T, U, V.

| Metric | Result |
| --- | ---: |
| Accuracy on the 13 represented classes | 81.90% |
| Macro precision, all 26 labels | 44.47% |
| Macro recall, all 26 labels | 40.95% |
| Macro F1, all 26 labels | 40.15% |
| Macro F1, 13 represented classes only | 80.31% |

The 13-class accuracy is not an A-Z score. The all-26 macro result is shown to make the missing half of the test support explicit; the supported-class macro result is the meaningful class-average for this partial split.

### Per-class results: round 2 validation

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| A | 1.000 | 0.873 | 0.932 | 300 |
| B | 0.996 | 0.857 | 0.921 | 300 |
| C | 1.000 | 0.940 | 0.969 | 300 |
| D | 1.000 | 1.000 | 1.000 | 300 |
| E | 1.000 | 0.557 | 0.715 | 300 |
| F | 1.000 | 1.000 | 1.000 | 300 |
| G | 0.997 | 1.000 | 0.998 | 300 |
| H | 1.000 | 1.000 | 1.000 | 300 |
| I | 1.000 | 0.953 | 0.976 | 300 |
| J | 0.159 | 0.997 | 0.275 | 300 |
| K | 0.000 | 0.000 | 0.000 | 300 |
| L | 1.000 | 1.000 | 1.000 | 300 |
| M | 0.000 | 0.000 | 0.000 | 300 |
| N | 0.993 | 0.997 | 0.995 | 300 |
| O | 1.000 | 1.000 | 1.000 | 300 |
| P | 0.000 | 0.000 | 0.000 | 300 |
| Q | 0.850 | 1.000 | 0.919 | 300 |
| R | 0.482 | 1.000 | 0.650 | 300 |
| S | 0.000 | 0.000 | 0.000 | 300 |
| T | 0.439 | 0.903 | 0.591 | 300 |
| U | 0.964 | 0.623 | 0.757 | 300 |
| V | 0.950 | 0.127 | 0.224 | 300 |
| W | 0.615 | 1.000 | 0.761 | 300 |
| X | 1.000 | 0.713 | 0.833 | 300 |
| Y | 0.000 | 0.000 | 0.000 | 300 |
| Z | 0.000 | 0.000 | 0.000 | 300 |

### Per-class results: round 3 partial test

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| A | 1.000 | 1.000 | 1.000 | 150 |
| B | n/a | n/a | n/a | 0 |
| C | n/a | n/a | n/a | 0 |
| D | n/a | n/a | n/a | 0 |
| E | n/a | n/a | n/a | 0 |
| F | n/a | n/a | n/a | 0 |
| G | 1.000 | 1.000 | 1.000 | 150 |
| H | 1.000 | 1.000 | 1.000 | 150 |
| I | n/a | n/a | n/a | 0 |
| J | 0.439 | 1.000 | 0.610 | 150 |
| K | 1.000 | 0.993 | 0.997 | 150 |
| L | n/a | n/a | n/a | 0 |
| M | 0.667 | 0.173 | 0.275 | 150 |
| N | 0.785 | 1.000 | 0.880 | 150 |
| O | n/a | n/a | n/a | 0 |
| P | 1.000 | 0.773 | 0.872 | 150 |
| Q | n/a | n/a | n/a | 0 |
| R | 0.673 | 1.000 | 0.804 | 150 |
| S | 1.000 | 0.193 | 0.324 | 150 |
| T | 1.000 | 1.000 | 1.000 | 150 |
| U | 1.000 | 0.513 | 0.678 | 150 |
| V | 1.000 | 1.000 | 1.000 | 150 |
| W | n/a | n/a | n/a | 0 |
| X | n/a | n/a | n/a | 0 |
| Y | n/a | n/a | n/a | 0 |
| Z | n/a | n/a | n/a | 0 |

### Sparse confusion matrix: round 2 validation

Rows are actual classes; entries show non-zero predicted-class counts. Omitted cells are zero. Each actual class has 300 samples.

| Actual | Non-zero predictions (predicted:count) |
| --- | --- |
| A | A:262, J:37, M:1 |
| B | B:257, J:3, W:40 |
| C | C:282, J:18 |
| D | D:300 |
| E | E:167, J:133 |
| F | F:300 |
| G | G:300 |
| H | H:300 |
| I | I:286, J:14 |
| J | B:1, J:299 |
| K | J:300 |
| L | L:300 |
| M | J:260, S:38, T:2 |
| N | J:1, N:299 |
| O | O:300 |
| P | J:104, Q:44, T:152 |
| Q | Q:300 |
| R | R:300 |
| S | J:108, T:192 |
| T | J:18, N:2, Q:9, T:271 |
| U | J:8, R:103, U:187, V:2 |
| V | J:35, R:220, U:7, V:38 |
| W | W:300 |
| X | G:1, J:85, X:214 |
| Y | J:300 |
| Z | J:152, W:148 |

### Sparse confusion matrix: round 3 test proxy

Rows are actual classes; entries show non-zero predicted-class counts. Omitted cells are zero. Each represented actual class has 150 samples; all other classes have zero support.

| Actual | Non-zero predictions (predicted:count) |
| --- | --- |
| A | A:150 |
| G | G:150 |
| H | H:150 |
| J | J:150 |
| K | J:1, K:149 |
| M | J:85, M:26, N:39 |
| N | N:150 |
| P | P:116, Q:34 |
| R | R:150 |
| S | J:106, M:13, N:2, S:29 |
| T | T:150 |
| U | R:73, U:77 |
| V | V:150 |

## Live Recognition and Selective Confirmation

The camera path requests 640x480 video and downsizes server-bound frames to 320 pixels. Browser configuration schedules frames at a nominal 70 ms interval. Physical-camera latency and stability could not be measured in this environment.

The live UI distinguishes `no_hand`, `low_confidence`, `pending`, `recognised`, and `detecting`. Before the fix, the candidate path allowed a 4%+ label to enter the consecutive counter, and a 3-frame browser setting could confirm a 75% read. This is now corrected:

- Candidate floor remains 0.10 (and the existing classifier's low-confidence fallback may keep a >=0.04 guess for display).
- Confirmed-output floor: 0.80, configurable with `BRIDGESIGN_MIN_CONFIRMED_STATIC_CONFIDENCE`.
- Stability rule: five consecutive same-label frames, configurable with `BRIDGESIGN_CONSECUTIVE_THRESHOLD`; browser sends five for both modes.
- A below-floor prediction resets the stability count, remains low-confidence feedback, and cannot be emitted as a confirmed sign.
- No hand clears candidate and confirmation state. No-hand remains an abstention, not a class prediction.
- The Image endpoint and image-array helper use the same 0.80 confirmed-output floor.
- The existing Learning Module A-C camera/quiz grader remains at its existing 0.60 question threshold; this phase does not alter it.

The floor/stability selection was replayed on ordered CSV capture blocks. With existing effective low candidate floor and three frames, round 2 produced 150 correct and 98 wrong commits; round 3 produced 30 correct and 15 wrong commits. At 0.80 and five frames, round 2 produced 86 correct and 0 wrong commits across 17/26 blocks with correct commits; round 3 produced 21 correct and 0 wrong commits across 8/13 blocks. These are offline capture replay observations, not webcam rates or proof of real-world precision.

Frame-level selective operating points (dual-orientation capture-round validation):

| Confidence floor | Accepted coverage | Accuracy among accepted |
| ---: | ---: | ---: |
| 0.60 | 62.5% | 89.9% |
| 0.70 | 51.2% | 94.7% |
| 0.80 | 33.4% | 99.7% |
| 0.90 | 17.4% | 99.9% |

This is an explicit coverage/precision trade-off. The 0.80 floor is a conservative operational abstention point based on the available captures; the forest vote fraction is not calibrated probability.

## Image Recognition

The upload API uses the same hand detector, feature extractor, active label map, raw/mirrored feature comparison, no-hand rejection, and now the same 0.80 display/acceptance floor. Photos may use an additional pixel-flipped detection pass at medium confidence; live video does not. No image-derived dataset examples exist in the checked-in image dataset file. No physical camera/image generalization claim is made.

## Changes and Before/After

No model training, dataset rewriting, synthetic samples, classifier replacement, or feature-representation changes were made.

| Behavior | Before | After |
| --- | --- | --- |
| Classifier weights/features | Existing 80-tree RF, 42 features | Unchanged |
| Live configured stability | 3 browser frames; server fallback 1 | 5 browser frames; server fallback 5 |
| Live confirmed confidence | Could confirm a low candidate after enough frames | Requires >=0.80 and five same-label frames |
| Image accepted confidence | 0.10 | 0.80 |
| Round-2 5-frame capture replay | 103 correct / 42 wrong commits at existing effective candidate floor | 86 correct / 0 wrong at 0.80 |
| Round-3 5-frame capture replay | 21 correct / 7 wrong at existing effective candidate floor | 21 correct / 0 wrong at 0.80 |
| Model-only two-orientation latency | 4.36 ms median / 7.36 ms p95 | Same model and inference; no model change |

The after result reduces coverage and false commits on these capture blocks. It is not a before/after claim about live camera accuracy.

## Regressions and Remaining Weaknesses

- The stricter live/image floor will increase abstentions, especially for classes whose capture-round scores are systematically low. Round-2 frame-level B coverage at an 0.80 floor was only 2.3%; this is a real usability trade-off, and an A-C live camera assessment was not changed.
- Round-2 worst classes include K, M, P, S, Y, and Z with F1 of 0. J has 0.275 F1 and V 0.224. Round 3 also performs poorly on M (0.275), S (0.324), and U (0.678).
- Confusions include K->J (300/300), Y->J (300/300), P->T (152/300), E->J (133/300), S->T (192/300), V->R (220/300) in round 2; S->J (106/150), U->R (73/150), and M->J/N (124/150 combined) in round 3.
- **Cause requires further data investigation.** The files do not establish whether errors arise from class labeling, signer/capture variation, orientation, or visually similar poses.
- Signer identity, multiple signers, conditions, and verified recording independence are unknown. The system is not ready for broad A-Z generalization claims.
- The browser camera permission flow opened, but no camera video track was available. No physical-camera recognition, flicker, handedness, or end-to-end camera latency result is claimed.

## Final Decision

1. **Is the current model good enough?** Not for reliable general A-Z recognition claims. It performs strongly on some held capture blocks and fails completely on others.
2. **Strongest weaknesses:** random-row leakage in the historical reported score; class-specific capture shift/confusions; no signer or session metadata; uncalibrated confidence.
3. **Weakest classes:** round-2 K, M, P, S, Y, Z (F1 0); then V (0.224), J (0.275), T (0.591), R (0.650).
4. **More data needed?** Yes, after verifying labels and adding capture metadata. Existing quantity is substantial, but capture diversity and reliability are not demonstrated.
5. **More signers needed?** Yes for generalization evaluation; signer count is unknown and no signer-independent test can be constructed.
6. **Feature pipeline changes?** No mismatch found. Keep current features until independent data can test an alternative.
7. **Classifier replacement?** No evidence justifies it. Keep the Random Forest while correcting evaluation and collecting traceable data.
8. **Smoothing?** Yes, the false confirmation path justified a deterministic 0.80/five-frame rule; live camera behavior remains to be physically verified.
9. **Honest accuracy claim:** no real-world or signer-independent accuracy can be claimed. The best available capture-round metrics are 67.46%/0.635 macro F1 on full-class validation and 81.90% accuracy/0.803 supported-class macro F1 on a 13-class partial test proxy.
10. **Next:** validate existing labels/capture blocks with the data owner, add signer/session/condition metadata to future genuine recordings, then reserve a signer-independent test set before model tuning.
