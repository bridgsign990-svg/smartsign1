# BridgeSign Repository Forensic Audit

Audit date: 2026-10-21

## Summary

The protected static A-Z recognizer is operational with the local
`models/sign_model_light.pkl`. Live motion gestures are opt-in and the current
13-class ONNX export is the only usable dynamic model in this checkout. Three
other model paths are unresolved Git LFS pointer files. The isolated OpenHands
video pipeline completed inference on one local WLASL clip but predicted
`business` for a clip labeled `computer`; this is not an accuracy estimate.
Checkpoint and dataset permissions remain unresolved.

No model weights were fabricated, retrained, deleted, or replaced. The existing
project commit remains untouched and unpushed. A safe publication requires a
new branch whose history does not contain unavailable LFS objects or the
checkpoint with unresolved redistribution rights.

## Recovery Work Completed

The starting tree already contained the six runtime/documentation/test changes
and two untracked files from the earlier repair. Those were preserved. This
continuation additionally modified `gesture_collector.py`,
`gesture_trainer.py`, and `lstm_gesture_trainer.py`; added
`gesture_data_utils.py` and `tests/test_gesture_dataset.py`; and updated this
report and `README.md`. No files were removed or quarantined. All work remains
unstaged and uncommitted.

The gesture collector now has an opt-in V2 CSV with signer, session, timestamp,
device, capture dimensions, and observed sequence frame rate. Both trainers
read legacy or V2 files, keep exact duplicates together, prefer signer-based
splits and otherwise use session groups when available, and fail clearly when
a grouped split omits a class. Trainer defaults write separate candidate
artifacts and refuse to overwrite existing outputs. No training was run and no
dataset or model file was changed.

## Git And Repository State

- `main` is at `ce6bf79` (`Add BridgeSign project`), one commit ahead of
  `origin/main` at `2e97d05` (`first commit`). The worktree was clean before
  this audit; audit fixes are now local changes only.
- The current commit adds 288 files. Do not push it as-is: its tree contains
  pointer stubs for three missing LFS objects.
- The missing cache objects are for `models/sign_model.pkl`,
  `models/gesture_model.pkl`, and `models/gesture_lstm.pt`. Their working files
  are 133, 133, and 131 bytes respectively and begin with the Git LFS pointer
  header. Their expected payload sizes are approximately 92 MB, 50 MB, and
  270 KB. Their object IDs are absent from `.git/lfs/objects`; attempts to
  fetch them previously returned 404.
- Usable local LFS objects exist for `models/sign_model_light.pkl`,
  `models/sign_model_v2.pkl`, and the three MediaPipe `.task` models. They are
  28,573,851 bytes, 18,107,649 bytes, 7,819,105 bytes, 13,683,609 bytes, and
  5,777,746 bytes respectively. The `.gitattributes` rules for `*.pkl`,
  `*.pt`, and `*.task` are effective.
- `.git/lfs/bad` exists but is empty. It was not modified. No `git lfs fsck`,
  forced push, reset, or history rewrite was performed.
- The ignored local gesture dataset contains 3,558 valid 15-frame samples
  across 22 labels. A read-only sequence comparison found 327 exact duplicate
  rows in 198 groups, all with matching labels; no source/session metadata
  establishes whether these arose from repeated captures or dataset merging.
  Both gesture trainers originally used random row splits, allowing exact
  duplicates into both partitions.
- The local `.venv-2`, SQLite database, static dataset CSV, and 23 MB gesture
  CSV are ignored. No `.env` is present. A scan of committed text found no
  common token/private-key patterns.
- `static/chime.mp3` is zero bytes and has no source reference. The UI uses
  `softChime()` implemented with Web Audio in `static/app.js`; the empty file
  was left in place because it is inert. `git lfs status` lists its empty-file
  SHA-256 even though `git check-attr` reports no LFS attributes,
  `git lfs ls-files` omits it, and its Git blob is empty. This appears to be an
  LFS status/reporting anomaly, not a missing required model object.
- `README.md` had a stray `# smartsign` heading and a NUL-filled UTF-16 tail.
  That malformed suffix was removed; the remaining file is valid UTF-8 with no
  NUL bytes.

## Model Inventory

| Artifact | Finding | Use / action |
|---|---|---|
| `models/sign_model_light.pkl` | Verified local scikit-learn pipeline; 42 features, 26 A-Z labels | Active static model; preserve. Real dataset row labeled `A` predicted `A` at 0.9948 confidence. |
| `models/sign_model_v2.pkl` | Verified local scikit-learn pipeline; 42 features, 26 A-Z labels | Offline interim candidate only; not selected by the app. Preserve separately from the active baseline. |
| `models/sign_model.pkl` | 133-byte LFS pointer; payload unavailable | Legacy fallback/trainer output path. Do not claim the baseline/V2 files are byte-equivalent. Keep the code fallback; omit this pointer from any publication branch until recovered. |
| `models/gesture_lstm.onnx` and `gesture_lstm_meta.json` | Real ONNX, 270,587 bytes; dynamic sequence input `[batch, sequence, 42]`; output 13; metadata agrees | Usable dynamic inference artifact, but limited to 12 signs plus `STATIC`. No other labels should be added to its metadata. |
| `models/gesture_lstm.pt` | 131-byte LFS pointer; payload unavailable | PyTorch checkpoint cannot be used. ONNX remains a compatible inference route for its 13 trained classes. Preserve the pointer on the local archival commit only. |
| `models/gesture_model.pkl` | 133-byte LFS pointer; payload unavailable | The intended sklearn fallback for 22 labels is unusable. Its 22-label JSON/metrics files are not weights and cannot restore inference. |
| `models/hand_landmarker.task` | Real local file; LFS object cached | Required by the current static detector; preserve. |
| `models/holistic_landmarker.task` | Real local file; LFS object cached | Used by dataset tooling; preserve. |
| `models/pose_landmarker_lite.task` | Real local file; LFS object cached | Used by pose tooling; preserve. |
| `models/openhands_wlasl/epoch=109-step=49059.ckpt` | Real 15,550,367-byte checkpoint; SHA-256 matches the worker constant | OpenHands WLASL-2000 pose BiLSTM; strict checkpoint load succeeded. Preserve locally. Do not redistribute until checkpoint rights are confirmed. |
| `static/avatar/model/avatar.vrm` | Valid GLB container; metadata names VirtualCast, Inc. and enables redistribution/modification and corporation use, with credit required | Preserve metadata and attribute VirtualCast, Inc.; link the VRM Public License 1.0 when redistributing. |

The current ONNX labels are `GOODBYE`, `HELLO`, `HELP`, `J`, `NO`, `PLEASE`,
`SORRY`, `STATIC`, `STOP`, `THANK_YOU`, `WATER`, `YES`, and `Z`. The local
`data/gesture_dataset.csv` has 3,558 structurally valid 15-frame rows across 22
labels. The nine labels absent from the ONNX model are `AGAIN`, `AMBULANCE`,
`ANGRY`, `DANGER`, `DOCTOR`, `EMERGENCY`, `FIRE`, `FOOD`, and `HOSPITAL`.
The ONNX metadata must not be expanded to include them.

## Recognition Findings And Repairs

- The static classifier searches for `sign_model_light.pkl` before the missing
  legacy `sign_model.pkl`. The active artifact loads 26 classes and accepts the
  expected 42-feature vector. Static feature extraction remains wrist-centered
  and scale-normalized; it was not changed.
- Both static and dynamic loaders previously treated any existing path as a
  candidate. An LFS pointer is a regular file, so this produced misleading
  pickle/PyTorch errors. `model_artifacts.py` now detects the LFS pointer
  header; the static loader records a clear missing-payload error, and the
  dynamic loader skips pointer candidates and can select ONNX.
- `BRIDGESIGN_LIVE_GESTURES` defaults to `0`, and app startup logs that live
  gestures are disabled. This is intentional for the stable alphabet path.
  `/ready` now reports `live_gestures_enabled`,
  `live_gestures_available`, and the loaded label list without making the
  optional feature a requirement for static readiness.
- With the feature enabled, the old motion threshold of 30 rejected 201/207
  recorded `J` sequences and 158/200 recorded `Z` sequences. The default is
  now 6.0, configurable by `BRIDGESIGN_GESTURE_MOTION_THRESHOLD`; the feature
  remains disabled unless `BRIDGESIGN_LIVE_GESTURES=1` is set before startup.
- A fresh diagnostic on the current CSV at threshold 6 passed 356/360 sampled
  supported-class sequences and gated 177/200 `STATIC` sequences. None of the
  gated static sequences produced a non-`STATIC` prediction at confidence
  0.78 or above. This dataset may overlap training and cannot validate live
  camera performance or independently justify a threshold.
- In that same 360-sequence probe, ONNX predictions were correct on 350/360
  supported-class rows before the gate and 346/356 rows that passed the gate.
  These are recorded-data diagnostics only, not held-out accuracy.
- Gesture training now groups exact duplicate sequences before splitting and,
  for the new V2 collector format, groups by signer when multiple signer IDs
  exist or otherwise by session. Historical CSV rows remain unchanged. New
  trainer outputs use separate candidate artifact names and refuse to
  overwrite existing files; they do not replace the active ONNX export.
- The ONNX was run on deterministic samples from the local recorded dataset.
  The model emitted several confident confusions, including `PLEASE` samples.
  Dataset overlap with model training is unknown, so this is a diagnostic, not
  an accuracy estimate. The 99.42% validation figure in the metadata is not
  independently reproducible from the checked-in metadata alone.
- In a deterministic 30-sample-per-class probe on the local gesture CSV, the
  ONNX correctly classified 360/390 samples from its supported labels before
  the motion gate; there were 25 confidently wrong samples at the app's 0.78
  cutoff. These rows may overlap training, so this is not a held-out score.
- The collector records up to 30 fps; browser inference is configured at
  roughly 70 ms per frame (about 14 fps). The 15-frame ONNX sequence therefore
  has a different wall-clock duration in live use than in collection. This
  timing mismatch is not repaired or validated by the threshold change and
  remains a likely practical limitation.

To opt in locally, set these process environment variables before launching
the app (the project does not automatically load `.env` files):

```powershell
$env:BRIDGESIGN_LIVE_GESTURES = "1"
$env:BRIDGESIGN_GESTURE_MOTION_THRESHOLD = "6.0"
.\.venv-2\Scripts\python.exe app.py
```

## Video Recognition

The webcam motion pipeline and uploaded-video pipeline are separate. The
checkpoint is an OpenHands `Network` configured as a pose-flattener encoder
and bidirectional LSTM decoder with attention, 1,294,288 parameters. The worker
uses the 2,000-item WLASL vocabulary, MediaPipe Holistic pose generation,
`mediapipe_holistic_minimal_27` selection, and shoulder-based center/scale
normalization. It loads strictly with the checked-in config and the pinned
OpenHands source revision `b9ccf9eaf2a71301fc4601a64a6af06fdef04c9b` in the
isolated Python 3.12.7 / CPU PyTorch 2.14.1 environment.

The OpenHands source is Apache-2.0. WLASL requires the Computational Use of
Data Agreement and states academic/computational use only; the OpenHands
checkpoint release does not provide a separate checkpoint license. The
checkpoint's redistribution status is unresolved. The workspace contains
experimental WLASL clips. One local WLASL V2 clip labeled `computer`
(`data/experimental/wlasl_v2/resampled/class_0002/12312_25fps.mp4`) completed
the upload, preprocessing, checkpoint inference, and output path: 84 frames
were processed, but the top predictions were `business` (0.486199), `busy`
(0.348537), and `science` (0.105093), not `computer`. This is one diagnostic
example, not an accuracy estimate; dataset authorization and checkpoint
redistribution rights still need review. No physical-camera video test was run.

The CWASA release page identifies the example pages as CC BY-SA 4.0 and says
the underlying CWASA software is copyright UEA and must be used unmodified on
terms equivalent to CC BY-ND 4.0. The vendored `allcsa.js`, CSS, XSL, shaders,
and Anna/COMMON avatar JARs contain no bundled license files or JAR manifest
license marker. The checked-in `static/vendor/cwasa/README.md` records their
source URLs. Keep the upstream runtime files unchanged, include the upstream
attribution/license notices and link to the UEA conditions of use, and verify
the exact Anna avatar's standard-avatar terms before a public redistribution.
Other bundled third-party code includes Three.js (MIT), three-vrm (MIT), and
MediaPipe browser helpers (Apache-2.0); the repository currently has no root
LICENSE/NOTICE file aggregating those notices.

## Dependencies And Deployment

- The primary `requirements.txt` pins Flask 3.1.3, MediaPipe 0.10.33, NumPy
  2.4.4, scikit-learn 1.8.0, ONNX Runtime 1.26.0, OpenCV 4.13.0.92, and
  CPU PyTorch 2.12.0. The selected Python 3.12.7 environment reports
  `pip check: No broken requirements found`.
- Video recognition is isolated from that environment. Its separate Python
  3.12.7 runtime uses CPU PyTorch 2.14.1, MediaPipe 0.10.21, OpenCV 4.11.0,
  and the pinned OpenHands source commit above.
- `config.py` requires an explicit `SECRET_KEY` in production and generates an
  ephemeral key only for local development. `.env.example` is a template; the
  app does not load `.env` automatically. Runtime state defaults to local
  `data/` and is excluded by `.gitignore`.
- Docker uses Python 3.11 slim, installs the primary requirements, sets
  `BRIDGESIGN_WARM_INFERENCE=0`, and launches Gunicorn. Render/Railway
  definitions provide production settings. The Docker image was not built in
  this audit; the deployed environment is therefore not runtime-certified.

`static/chime.mp3` and model sidecars are not model replacements. The
`gesture_model_classes.json` lists 22 labels for the missing sklearn model;
the ONNX's own 13-class output and metadata are authoritative for current live
inference. Existing class/metrics sidecars are retained because training,
learning, and audit tooling still references them.

## Validation

- Before this continuation, `.\.venv-2\Scripts\python.exe -m pytest -q`
  passed: 179 passed in 74.70s.
- `.\.venv-2\Scripts\python.exe -m pytest tests\test_gesture_dataset.py tests\test_gesture_live_routing.py tests\test_classifier_validation.py tests\test_video_recognition.py -q`
  passed: 29 passed in 9.94s.
- After the current changes,
  `.\.venv-2\Scripts\python.exe -m pytest -q` passed: 187 passed in 55.90s,
  with no skipped or failed tests.
- `.\.venv-2\Scripts\python.exe -m compileall -q gesture_data_utils.py gesture_collector.py gesture_trainer.py lstm_gesture_trainer.py tests\test_gesture_dataset.py`
  and `git diff --check` completed successfully. Pylance reported no syntax
  errors in the five modified/added Python files.
- Applying the new grouped-split helper to the local legacy gesture CSV
  produced 2,843 training and 715 validation rows, included all 22 labels,
  and kept all 327 duplicate rows with their matching sequences. The split
  basis is `exact-duplicate-only`, not signer/session independent.
- Static model load: 26 labels, 42 input features; an actual saved `A` feature
  row predicted `A` (0.9948 model confidence).
- `/health` and `/ready` returned 200 with live gestures disabled. With
  `BRIDGESIGN_LIVE_GESTURES=1`, `/ready` returned 200 and reported the ONNX
  model available with its 13 labels.
- The configured isolated runtime reports Python 3.12.7, CPU PyTorch 2.14.1,
  OpenCV 4.11.0, MediaPipe 0.10.21, and OmegaConf 2.3.0.
- `.\.venv-2\Scripts\python.exe -m pip check` reported
  `No broken requirements found`; `git diff --check` passed (Git emitted only
  line-ending conversion warnings).
- A local WLASL V2 `computer` video reached model inference (84 frames);
  top-1 was `business` at 0.486199. No physical camera session was available.
  No Docker build or training run was performed.

## Safe GitHub Recovery Plan

Do not push `ce6bf79` as-is. The current branch history includes pointer blobs
for which the local LFS objects are missing; deleting those files in a later
commit will not remove their earlier pointers from the history being pushed.

Recommended non-force-push approach, after review and after authenticating
through VS Code or Git Credential Manager (do not reuse the credential
previously pasted into chat):

1. Keep `ce6bf79` intact on local `main`; it is the archival copy of the
  submitted project and must not be reset or discarded. Do not cherry-pick it
  onto the publication branch, because that would retain its unavailable LFS
  pointer objects in the new branch history.
2. Create a separate worktree/branch from `origin/main`. Populate that branch
  with a project tree exported from `ce6bf79`, apply the audit fixes, and
  remove only `models/sign_model.pkl`, `models/gesture_model.pkl`, and
  `models/gesture_lstm.pt` from that publication tree. Do not alter the
  archival `main` worktree or recoverable local models.
3. Exclude `models/openhands_wlasl/epoch=109-step=49059.ckpt` from the public
  branch until its redistribution rights are confirmed. Keep the local copy;
  arrange user-side provisioning only after license review.
4. Preserve the CWASA bundle only after the unchanged upstream files,
  attribution, CC BY-SA / CC BY-ND conditions, and Anna avatar terms are
  reviewed and represented in the public repository. If that review cannot
  be completed, publish a clear asset-install step and keep the required
  runtime files out of the public branch.
5. Run the full test suite and inspect the new tree and history. Confirm the
  branch is based directly on `origin/main`, contains no ancestor `ce6bf79`,
  and has no pointer OIDs for the excluded artifacts. Keep the existing valid
  LFS objects for the active static model, V2 model, and MediaPipe assets.
6. Commit the reviewed snapshot on that branch, then push only that branch (no
  force push) and review it as a pull request. Do not push the archival
  `main` branch. If the three original models are recovered with permission,
  add them in a later reviewed commit/branch.

This plan avoids rewriting shared remote history and keeps the existing
project commit reachable. No branch, tag, commit, or remote ref was created or
changed as part of this audit. GitHub authentication is still required before
any remote publication.