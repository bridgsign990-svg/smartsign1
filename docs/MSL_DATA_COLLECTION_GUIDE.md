# MSL Data Collection Guide

This workflow collects dynamic Malaysian Sign Language (MSL) recordings and prepares landmark sequences for later review and research. It does not train a model. Keep MSL files under `data/msl_dataset/`; this workflow does not use or modify the static A–Z dataset or model, and it must not be used to mix in MS-ASL/ASL media.

## 1. Register a signer and obtain consent

Use a non-identifying signer code. Store signed consent forms separately in the approved secure location; put only a non-identifying reference in the manifest.

```powershell
python -m tools.msl_dataset.register_signer --signer S01
python -m tools.msl_dataset.update_consent --signer S01 --consent-status GRANTED --consent-reference consent-record-001
```

Do not begin recording unless permission has been granted. `REVOKED` immediately blocks further recording and preprocessing.

## 1a. Audit existing videos without moving them

Before registering or processing older material, index it in place:

```powershell
python -m tools.msl_dataset.import_existing_recordings --root data/msl_dataset/raw --config config/msl_dataset.json
```

The command writes `existing_recordings.csv`, `existing_recordings_summary.json`, and
`existing_recordings_report.md` under the configured manifests directory, plus
`recording_review.csv` under the review directory. It hashes and decodes source videos
without changing them. Folder-derived signer, sign, and session values remain explicitly
unconfirmed; absent fields stay `UNKNOWN`. Every imported recording starts `PENDING`
human review. The import does not grant consent or authorize preprocessing.

Where the configured MediaPipe task model is available, the audit uses the existing
Holistic extractor to measure hand/pose/face availability. If the model asset is absent,
landmark QC is reported as unavailable rather than treating readable video as landmark-
validated.

For a sequential human review, first print the full QC queue, then review each path
in a video player and record explicit decisions. Enter on a decision prompt preserves
the existing value:

```powershell
python -m tools.msl_dataset.review_existing_recording --list
python -m tools.msl_dataset.review_existing_recording --interactive --reviewer reviewer-01
```

The interactive review displays the recording ID/path, folder-derived sign candidate,
signer/session candidates, duration, FPS, resolution, frame count, hand/left/right-hand
detection percentages, technical QC, and current human decisions. It asks separately
for signer (`CONFIRMED`, `REJECTED`, `UNKNOWN`), session (same), label
(`CONFIRMED`, `REJECTED`, `NEEDS_REVIEW`), language (`MSL`, `NOT_MSL`, `UNKNOWN`),
recording quality (`APPROVED`, `REJECTED`, `NEEDS_RETAKE`), and human review
(`APPROVED`, `REJECTED`, `NEEDS_REVIEW`). The CLI presents the local video path for
manual playback; it does not embed a browser player.

Folder labels remain candidates until a reviewer confirms them. A confirmed label
must already exist as an active, human-validated entry in `sign_vocabulary.json`.
If it does not, the label decision is retained but reports
`VOCABULARY_VALIDATION_REQUIRED`; register and validate the vocabulary entry using
the existing `register_sign` and `validate_sign` CLIs. Do not create vocabulary
entries merely to match folder names.

Consent is read only from the existing signer registry. If unknown, it remains
`UNKNOWN` and blocks processing; record consent only from actual consent evidence
with `update_consent`. Session confirmation likewise requires an already registered
active session; when its details are known, create it with `create_session` rather
than fabricating metadata. The review report and manifest keep technical QC,
signer, consent, session, label, language, quality, human review, and processing
eligibility as separate fields.

Only when every gate is satisfied does the existing workflow mark a recording
`READY_FOR_PROCESSING`. The normal `prepare_dataset` command then rechecks eligibility
before processing. Reviewing alone does not create training sequences; original
videos are never moved or modified.

## 2. Register a session

Create one session for each collection visit. Session IDs are unique per signer.

```powershell
python -m tools.msl_dataset.create_session --signer S01 --session session_001
```

Session metadata can be recorded with CLI options when known:

```powershell
python -m tools.msl_dataset.create_session --signer S01 --session session_002 --date 2026-02-16 --camera webcam --resolution 1280x720 --fps 30 --lighting-condition indoor --background-condition plain --orientation front
```

Do not put names or other identifying details in those fields.

## 3. Define the vocabulary

Use the controlled vocabulary for agreed MSL labels. Each label should be reviewed by an MSL-competent human and have a clear canonical meaning before it is treated as validated.

```powershell
python -m tools.msl_dataset.register_sign --sign-id WATER --canonical-name water --display-name WATER --description "Agreed local definition"
python -m tools.msl_dataset.validate_sign --sign-id WATER --status VALIDATED --reviewer reviewer-01 --evidence-note "Checked against the approved vocabulary"
```

## 4. Record videos

```powershell
python -m tools.msl_dataset.record_video --signer S01 --sign WATER --session session_001
```

The recorder requires an active signer with granted consent and an active registered session. It writes new recordings to:

```text
data/msl_dataset/raw/<signer>/<sign>/<session>/video_NNN.mp4
```

It advances to the next unused number, refuses to overwrite a video, checks that the finished file can be decoded, and records its SHA-256 and metadata. Press `s` to start and `q` to stop. Raw recordings are immutable: do not edit or replace them. If one needs correction, preserve the original and create a new recording number.

## 5. Validate and plan preprocessing

Validate consent, session, directory layout, duplicate content, and complete video decoding:

```powershell
python -m tools.msl_dataset.validate_recordings --raw-dir data/msl_dataset/raw --manifests-dir data/msl_dataset/manifests
```

Inspect a no-write plan before running extraction:

```powershell
python -m tools.msl_dataset.prepare_dataset --config config/msl_dataset.json --dry-run
```

Any `RAW_HASH_CHANGED`, consent, session, duplicate, layout, or decode failure must be resolved by human review; do not re-register changed bytes as if they were the original recording.

## 6. Extract and review sequences

MediaPipe Tasks Holistic requires the local model asset configured by `holistic_model_path` (default `models/holistic_landmarker.task`).
The official Holistic Landmarker bundle is stored at that path. Its Google-hosted
source URL, installed MediaPipe version, file size, and SHA-256 are recorded in
`models/holistic_landmarker.task.metadata.json`. The import/audit command can run
technical landmark QC without publishing training sequences:

```powershell
python -m tools.msl_dataset.import_existing_recordings --root data/msl_dataset/raw --config config/msl_dataset.json
```

Technical results are stored separately from human confirmation in
`existing_recordings.csv`, `existing_recordings_summary.json`, and
`existing_recordings_report.md`. `READY_FOR_REVIEW` means the video decoded and
usable hand landmarks were detected; it does not mean its signer, session, MSL
label, language, or review was validated. `READABLE_BUT_LOW_HAND_DETECTION`
distinguishes readable clips with no hand detections from `VIDEO_INVALID`.
This audit reports the number of possible windows but does not write sequence
files or bypass the consent and metadata gates in dataset preparation.

```powershell
python -m tools.msl_dataset.prepare_dataset --config config/msl_dataset.json
python -m tools.msl_dataset.collection_status --manifests-dir data/msl_dataset/manifests --review-dir data/msl_dataset/review
```

The processor writes temporal `.npz` sequences and CSV manifests under `data/msl_dataset/processed/` and `data/msl_dataset/manifests/`. Inputs are re-hashed after extraction so a recording changed during processing is not published as a valid sequence. The `sequence` array is finite model input; missing landmark groups are zero-filled and accompanied by a presence mask. `raw_sequence` can contain NaN values to explicitly represent missing raw landmarks.

Inspect generated samples, then record a human decision:

```powershell
python -m tools.msl_dataset.review_sequence --sequence-id <sequence-id> --status APPROVED --reviewer reviewer-01 --note "Quality checked" --manifests-dir data/msl_dataset/manifests
```

Possible decisions are `PENDING`, `APPROVED`, `REJECTED`, and `NEEDS_RETAKE`. Reviewers should flag framing, visibility, sign execution, label correctness, and unusable frames.

## 7. Create and validate signer-separated splits

```powershell
python -m tools.msl_dataset.create_splits --config config/msl_dataset.json --manifests-dir data/msl_dataset/manifests
python -m tools.msl_dataset.validate_dataset --manifests-dir data/msl_dataset/manifests
```

The configured signer groups must be disjoint and cover all signers in the sequence manifest. The pipeline additionally rejects cross-split reuse of sequence IDs, recording IDs, source paths, or source hashes. Never split clips or temporal windows from one signer across train, validation, and test.

## 8. Report collection balance and limits

```powershell
python -m tools.msl_dataset.report_dataset --manifests-dir data/msl_dataset/manifests
```

`collection_targets` in `config/msl_dataset.json` controls the minimum signer diversity per sign and recordings per signer/sign. The report prioritizes adding new signers before collecting more repetitions from existing signers. Treat the report as an operational aid, not a substitute for consent, vocabulary review, recording review, or protocol approval.

All artifacts stay in `data/msl_dataset/`. Do not copy MS-ASL files into this tree, change `data/dataset.csv` or `models/sign_model_light.pkl`, or train a neural network as part of collection or preprocessing.
