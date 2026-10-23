# Static Sign Recognition Audit

Audit date: 2026-10-04

Scope: the existing static A-Z classifier only. Dynamic gesture recognition, sequence models, and sign-language translation beyond the current static labels were not modified or evaluated here.

## Executive Findings

- The active model is a 42-feature scikit-learn Random Forest stored in `models/sign_model_light.pkl`; it loads successfully and maps integer outputs 0-25 to A-Z in the same order as the model classes.
- The dataset contains 14,950 clean feature rows, 500-650 per class, with no exact duplicate feature rows detected.
- The primary CSV has three contiguous collection rounds: 200 samples/class for A-Z, 300/class for A-Z, and 150 samples for a subset of 13 classes. Rows contain no signer, capture-session, date, lighting, or background identifiers.
- The current saved random-row holdout score is not an independent-session or signer-independent estimate. It must not be presented as real-world recognition accuracy.
- A trainer-faithful capture-round evaluation shows substantial class-specific failures. With existing noise/scale/mirror augmentation applied only to training blocks and the deployed dual-orientation inference rule, round 1 to round 2 is 80.65% accuracy and 0.762 macro F1. The partial round-3 test is 76.82% accuracy and 0.739 macro F1 over its 13 supported classes. Detailed corrected forensic results are in `DATASET_FORENSICS_REPORT.md`.
- The feature extractor is shared by training collection, live inference, and image inference. No 42-feature preprocessing mismatch was found.
- A safety defect was verified: the live candidate fallback allowed a 75% prediction to be confirmed after stable frames despite a much higher confidence operating point being required by validation evidence. Live confirmation now requires 80% confidence and five same-label frames. Lower-confidence labels remain available as explicitly unconfirmed best guesses.

## Dataset Inventory

| File | Rows | Classes | Row width | Samples/class range | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `data/dataset.csv` | 14,950 | 26 | 43 values | 500-650 | Active static dataset: label + 42 floats |
| `data/dataset_backup.csv` | 11,070 | 26 | 43 values | 300-650 | Backup |
| `data/dataset_before_recollect.csv` | 10,380 | 26 | 43 values | 300-600 | Backup |
| `data/dataset_from_images.csv` | 0 | 0 | empty | n/a | No image-derived training rows |
| `data/dataset_legacy_42_features_backup.csv` | 7,430 | 26 | 43 values | 120-650 | Backup |
| `data/dataset_old_backup.csv` | 5,200 | 26 | 43 values | 200 | Backup |
| `data/dataset_pre_retrain.csv` | 13,000 | 26 | 43 values | 500 | Backup |
| `data/gesture_dataset.csv` | 3,558 | 22 | 631 values | 36-245 | Separate 15-frame sequence data; excluded from this static audit |

Primary dataset class counts:

| Class | Count | Class | Count | Class | Count | Class | Count |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| A | 650 | B | 500 | C | 500 | D | 500 |
| E | 500 | F | 500 | G | 650 | H | 650 |
| I | 500 | J | 650 | K | 650 | L | 500 |
| M | 650 | N | 650 | O | 500 | P | 650 |
| Q | 500 | R | 650 | S | 650 | T | 650 |
| U | 650 | V | 650 | W | 500 | X | 500 |
| Y | 500 | Z | 500 |  |  |  |  |

Summary:

- Total samples: 14,950.
- Classes: 26, all A-Z present.
- Minimum/maximum/average class count: 500 / 650 / 575.
- Maximum-to-minimum imbalance ratio: 1.30:1; no severe class imbalance.
- Missing labels, malformed rows, non-finite features, and wrong-width rows: 0 detected in the primary file.
- Exact duplicate `(label, feature-vector)` rows: 0 detected. This check does not rule out near-duplicates.
- The `dataset/` image directory is empty. The image dataset CSV is also empty.

## Recording and Leakage Audit

The active CSV's row order contains these contiguous blocks:

1. A-Z in order, 200 rows/class (5,200 rows total).
2. A-Z in order, 300 rows/class (7,800 rows total).
3. J, A, S, T, M, N, U, V, R, K, P, G, H, 150 rows/class (1,950 rows total).

`data_collector.py` records a sample every 0.05 seconds while a sign is held. It writes only `label,feature_1,...,feature_42`; it does not store timestamps, signer IDs, capture/session IDs, camera orientation, or conditions. Neighboring feature vectors are measurably closer than random same-capture pairs in rounds 1 and 2, consistent with temporal correlation.

The current trainer uses a stratified random row split (`test_size=0.18`, `random_state=42`) and augments only after that split. Augmentation itself therefore does not leak into validation, but the row split mixes frames from the same contiguous capture block across train and validation. This is capture/session leakage for the intended generalization question. It makes the saved 99.9257% validation score unsuitable as an independent-recording or real-world estimate.

The metrics JSON says its holdout was 20%, while current `model_trainer.py` uses 18%. It labels a single holdout score as `cv_accuracy_mean`; `cv_scores` contains one value, so no cross-validation was run by the current trainer code. The saved metric file is not evidence that the active artifact used the current code revision or current split.

A capture-round split is possible. It does not establish signer independence: signer identity is absent from every row. **Signer-independent evaluation is currently impossible with the available dataset.** Number of signers, backgrounds, lighting conditions, and actual session boundaries beyond the visible row blocks are unknown.

## Feature Pipeline

- MediaPipe hand detector: 21 landmarks, one hand maximum.
- Coordinates used: x and y only; z is discarded by `HandDetector.get_landmarks`.
- Feature count/order: 21 landmarks in MediaPipe index order, interleaved `(x0,y0,x1,y1,...,x20,y20)`, yielding 42 values.
- Normalization: subtract wrist landmark 0 from each coordinate, then divide by the wrist-to-middle-MCP (landmark 9) Euclidean distance. Near-zero scale falls back to 1.
- Rotation normalization: none.
- Handedness: not captured or used in the 42-vector.
- Missing landmarks: fewer than 21 returns `None`; classifier rejects wrong feature length.
- Multi-hand: not supported by the static detector path (`MAX_NUM_HANDS=1`); only hand index 0 is used.
- Training collection, live inference, and image inference import the same `FeatureExtractor`. Training loads the already-extracted 42 values; inference applies the same wrist anchoring and scale normalization before the saved scaler. No feature-dimension or normalization mismatch was found.
- Orientation handling is not identical: live inference scores raw and feature-mirrored vectors. Photo inference also scores raw and feature-mirrored vectors and, at moderate confidence, may run a second pixel-flipped image pass. The pixel-level retry is image-specific; it does not change the 42-value extractor.

## Model and Label Audit

- Active algorithm: `StandardScaler` followed by `RandomForestClassifier`.
- Parameters: 80 trees, max depth 18, minimum leaf size 2, `class_weight="balanced"`, `random_state=42`, one inference worker.
- Training augmentation: noise and scale jitter, plus mirrored feature views excluding J, P, Q, and Z.
- Model confidence: maximum `predict_proba` value, which is the fraction of forest tree votes in the class; it is not calibrated probability.
- Active artifact: `models/sign_model_light.pkl`, 42 inputs, pipeline classes 0-25, labels A-Z in matching order.
- `models/sign_model.pkl`, `sign_model_legacy_backup.pkl`, and `sign_model_broken.pkl` are 133-134 byte Git LFS pointer text files in this workspace, not loadable pickle payloads. `Classifier` correctly prefers and loads the valid light artifact.
- No architecture change or model retraining was performed.

## Live and Image Paths

Live path:

- Browser requests 640x480 video, downsamples transmitted frames to a 320-pixel maximum dimension at JPEG quality 0.55, and schedules inference at a nominal 70 ms interval.
- Server MediaPipe detection uses one hand, detection/tracking confidence 0.45, with a still-image detection fallback if video tracking returns no hand.
- Hand plausibility gate rejects tiny or implausibly thin boxes.
- No-hand frames clear the candidate/stability state and return `hand_state=no_hand`.
- Before this change, `config.MIN_PREDICTION_CONFIDENCE=0.10` acted as a candidate floor, but `_predict_static_sign` could return a 0.04+ candidate and the confirmation helper counted any non-empty label. The frontend requested three consecutive frames, so low-confidence candidates could become `recognised`.
- Now only candidates at or above `MIN_CONFIRMED_STATIC_CONFIDENCE=0.80` can advance confirmation. Five consecutive same-label frames are required by both the browser settings and server fallback. Below-threshold guesses remain visible as low-confidence feedback but cannot be committed.

Image path:

- `/api/translate_image` uses the same `FeatureExtractor`, classifier labels, raw/mirrored feature scoring, no-hand check, and now the same 0.80 confirmed-output floor.
- The array helper `translate_from_array` now also returns an empty label for sub-threshold predictions while preserving no-hand status.
- The image path may retry a pixel-flipped photo; live path does not do this extra image-level retry. No external callsites to `translate_from_array` were found.

Latency:

- Local model-only benchmark for `predict_proba` on both orientations: median 4.36 ms, p95 7.36 ms, max 35.58 ms across 200 calls on this machine.
- This excludes MediaPipe, feature extraction, HTTP, browser scheduling, and camera latency. No physical-camera latency measurement is claimed.

## Severity and Decisions

| Severity | Finding | Decision |
| --- | --- | --- |
| HIGH | Random row holdout mixes frames from the same capture blocks | Do not use the saved 99.9257% as real-world accuracy; retain capture-block evaluation |
| HIGH | Round-2 failures include several classes with zero F1 | Review labels/captures and gather signer-separated data before model retuning |
| HIGH | No signer identity or recording metadata | Add capture metadata to future collection; independent-signer testing cannot be done yet |
| HIGH | Low-confidence candidates could previously become confirmed | Fixed with 0.80 confirmation floor and five stable frames |
| MEDIUM | One-frame XY representation omits depth, rotation normalization, handedness, and motion paths | Keep representation for now; investigate only after labeled multi-signer captures |
| MEDIUM | Photo path has an extra pixel-flip retry | Documented; shared feature extraction remains consistent |
| LOW | Backup model artifacts are LFS pointers | Active model fallback works; no replacement artifact downloaded |
| NO ACTION | Feature preprocessing and label order | Verified consistent; unchanged |

## Browser Verification Limits

The browser session was not used to claim physical-camera recognition. Automated tests exercise the live endpoint with mocked landmarks/model output; the existing A-C assessment test also uses a mocked prediction. A physical camera check is reported separately in `STATIC_RECOGNITION_REPORT.md`.