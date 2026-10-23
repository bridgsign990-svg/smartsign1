# Phase 0 current-state audit

**Audited:** 2026-10-23  
**Branch:** `version-2`  
**Scope:** checked-in/local model and dataset artifacts. This is an engineering
inventory, not a live-camera or user validation report.

## Model artifact inventory

SHA-256 values below were computed from the files in `models/` during this
audit. Runtime status is only asserted where the application `/ready` endpoint
was actually observed; file presence alone does not mean a module is usable.

| Module/artifact | Local status | SHA-256 | Labels / shape | Evidence and boundary |
|---|---|---|---|---|
| Static letters: `models/sign_model_v2.pkl` | Candidate; not the production path | `656a5e53f528c12ff8b8b5bdaa76dbb055e6c6065ef9c6d3fccf4e123055efc9` | 24 static classes, 42 features | One signer/session, 1,720 training rows, 310 reserved test rows; target counts are 20% complete. Exploratory only. |
| Gestures: `models/gesture_lstm_v2.onnx` | ONNX artifact loads locally | `cfcdbca3611139a2d3cb90bd56429e35a0ebd4dca9d2938256f61f996ca58fd1` | 22 classes; input `[batch, sequence, 42]`; output `[batch, 22]` | Metadata reports 97.9% validation accuracy with exact-duplicate-only split, 306 duplicate extra rows, and no signer independence. Not live-camera accuracy. |
| Gestures: `models/gesture_lstm.onnx` | Legacy fallback artifact | `825ba6376c65f376944c92856b837f33853fa8abac3fdb97ce8425bb2a23fd0e` | 13 classes; input `[batch, sequence, 42]`; output `[batch, 13]` | Metadata reports 99.4%; split basis is undocumented. |
| Hand detector: `models/hand_landmarker.task` | File present; task asset | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | MediaPipe task asset | Checksum verified; runtime readiness requires `/ready`. |
| Pose detector: `models/pose_landmarker_lite.task` | File present; task asset | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` | MediaPipe task asset | Checksum verified; runtime readiness requires `/ready`. |
| Experimental video: `models/openhands_wlasl/epoch=109-step=49059.ckpt` | Checkpoint present | `c91551063db37f018f8246c64f1d2bfe53364d7c976377379e4d81c2d031179c` | OpenHands/WLASL label metadata is separate | Dataset/checkpoint rights and benchmark provenance still need review before redistribution or public claims. |

No Git LFS pointer stubs were found among the six checked binary weight/task
artifacts above. The older `models/holistic_landmarker.task.metadata.json` refers to a
`holistic_landmarker.task` file that is absent from this checkout; that
metadata is stale and must not be used to claim holistic readiness. The current
runtime configuration does not reference that asset.

The available metadata still does not provide every desired manifest field.
In particular, source-data hashes and documented group-split evaluation are
missing for the gesture exports, and authorization/license evidence for the
experimental video checkpoint needs separate review. These gaps are recorded
instead of being inferred from filenames or embedded class lists.

## Dataset and report consistency

- `data/dataset_v2.csv`, `reports/static_a_to_z/dataset_v2_quality.json`, and
  `reports/static_a_to_z/sign_model_v2_training.json` agree on dataset SHA-256
  `71252198f1d1f953ade37ea67ce523efb61ca1821b79df6ed11fec1b0f970420`.
- Current QC reports 1,781 valid rows: 1,720 static rows and 61 preserved
  motion rows (60 J, 1 Z), with no structural invalids or exact duplicate
  vectors. All rows are from signer `S01`, session
  `S01-20261023-morning`.
- Static classes have 60 samples each, or 100 for the priority classes. This
  is 20% of the 300/500 per-class targets. The dataset is not ready for the
  planned group-separated evaluation.
- `reports/static_a_to_z/sign_model_v2_evaluation.json` is stale for the
  current dataset: it names a different dataset hash, 1,880 rows, and absolute
  model paths under `C:\Users\USER\Desktop\bridgesign-main`. Do not cite its
  metrics as results for the current artifact/data pair. Regenerate evaluation
  only after the collection and split are suitable; the existing one-group
  data cannot produce independent-signer evidence.
- The checked-in legacy static metrics report near-perfect performance from a
  random row holdout. It does not establish signer/session generalization and
  must not be presented as independent accuracy.

## Readiness and next gates

Artifact checks establish that the files are readable binary assets and that
the two ONNX tensor shapes match the feature interface. They do not verify
camera capture, browser timing, detection thresholds, gesture onset/offset,
or end-to-end gesture recognition. A live `/ready` response was not captured
as part of this audit; use the endpoint after startup for current module
readiness.

Phase 0 remains open until the project owner collects the planned additional
sessions, quality-reviews the resulting data, obtains a current evaluation
report tied to its dataset hash, and confirms the application reports honest
per-module runtime readiness.
