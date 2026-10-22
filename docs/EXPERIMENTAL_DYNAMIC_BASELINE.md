# Experimental Dynamic Baseline

This workflow is isolated from production MSL data and the static A-Z system.
It uses the existing Holistic extractor and normalization, and is intended only
for architecture validation. An ASL experiment is **not validated MSL** and its
metrics must not be represented as MSL accuracy.

## Current availability

The local MS-ASL import has metadata but no downloaded videos or prepared
sequences. The first source returned HTML, and the remaining 99 download
attempts were not made. The 13 existing MSL videos remain pending human review
and are not eligible training input. See
[`data/experimental_dynamic/reports/baseline_report.md`](../data/experimental_dynamic/reports/baseline_report.md).

## Prepare an eligible experimental dataset

Obtain authorized ASL clips from a valid source, then process the selected
metadata using the existing importer (do not run an unrestricted download):

```powershell
.\.venv\Scripts\python.exe -m tools.msasl_dataset.import_msasl `
  --metadata "C:\path\to\authorized\MS-ASL-metadata.zip" `
  --limit 100 --download --process --split --report --validate
```

The existing importer writes temporal sequences and `sequences.csv` under
`data/external_datasets/msasl/`. Select classes represented in train,
validation, and test before training. Its official split metadata is retained;
the baseline rejects signer or source-video leakage and refuses evaluation on
classes absent from training.

## Train, evaluate, and predict

```powershell
.\.venv\Scripts\python.exe -m tools.experimental_dynamic.baseline status
.\.venv\Scripts\python.exe -m tools.experimental_dynamic.baseline train `
  --manifest data/external_datasets/msasl/manifests/sequences.csv `
  --dataset MS-ASL --language ASL --epochs 10 --batch-size 8 --seed 42
.\.venv\Scripts\python.exe -m tools.experimental_dynamic.baseline evaluate `
  --manifest data/external_datasets/msasl/manifests/sequences.csv `
  --checkpoint data/experimental_dynamic/runs/dynamic_baseline/best_model.pt
.\.venv\Scripts\python.exe -m tools.experimental_dynamic.baseline predict-video `
  --video data/experimental_dynamic/raw/example.mp4
```

Checkpoints, configuration, metrics, and confusion matrices are written only
under `data/experimental_dynamic/runs/`. Review
`data/experimental_dynamic/reports/baseline_report.md` before sharing results.
Inference on a video has no actual class unless the caller separately knows and
supplies the ground truth; the command does not infer one.

## Feature and split behavior

- The current MSL configuration is reused for feature extraction:
  30-frame windows, 15-frame stride, wrist-relative normalization, and the
  existing feature vector and landmark-presence channels (currently 562
  features per frame).
- MediaPipe timestamps use the source FPS; frames are not resampled.
- Missing landmark groups use the extractor's finite zero-filled model
  features and presence masks. The loader rejects NaN/Inf model inputs.
- Official source splits are preserved when present. Otherwise, splitting is
  grouped by signer; if signer IDs are absent, videos/samples are grouped and
  the report states `SIGNER_SEPARATION_UNAVAILABLE`.
- The model is a small bidirectional LSTM with dropout and a linear classifier.
  Test metrics are experimental and only meaningful for an appropriate,
  class-covered, held-out dataset.

For MSL experiments, the tool accepts only the existing
`data/msl_dataset/manifests/sequences.csv` and rechecks every source against
`existing_recordings.csv`, current consent/session/vocabulary registries,
human-review decisions, technical QC, and the unchanged source hash. It refuses
to train unless each source is already `READY_FOR_PROCESSING` and `PROCESSED`.
It does not alter those gates or production manifests. ASL experiments remain
tagged `EXPERIMENTAL_ASL_ARCHITECTURE_VALIDATION`; a gated MSL run is separately
tagged `EXPERIMENTAL_MSL_ARCHITECTURE_VALIDATION`.

After the community-provided MSL data is registered, reviewed, and processed
through the existing gates, a separate experimental run can use:

```powershell
.\.venv\Scripts\python.exe -m tools.experimental_dynamic.baseline train `
  --manifest data/msl_dataset/manifests/sequences.csv `
  --dataset MANAD --language MSL --epochs 10 --batch-size 8 --seed 42
```

This will still write its checkpoint and reports only under
`data/experimental_dynamic/`; it will not change the production MSL model.
