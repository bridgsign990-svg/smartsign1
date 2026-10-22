# BridgeSign Project and Technical Defense Guide

This guide explains how BridgeSign is built, how its static sign recognizer
works, what was actually trained and measured, and how to answer common
project-review questions. It is based on the current source code and saved
reports; it distinguishes implemented features from research goals and
limitations.

## 1. Short project description

BridgeSign is a Flask-based assistive communication and sign-learning
prototype. It brings several capabilities together:

- static hand-sign recognition for an alphabet/sign vocabulary
- camera and still-image recognition interfaces
- speech/text interaction features
- guided lessons, quizzes, and learning progress
- an avatar demonstration layer
- emergency phrase support
- experimental video, motion, and call functionality

It is a research and demonstration prototype. It is **not** a validated
Malawian Sign Language (MSL) translator, a production emergency-response
service, or evidence that participants learned more effectively. The app
contains several components, but they do not all have the same validation
status.

## 2. How to run the application

From the project folder in PowerShell:

```powershell
.\.venv-2\Scripts\python.exe app.py
```

Then open <http://127.0.0.1:5000>. Sign in or register. Allow camera access
when the browser asks. The browser generally permits camera access on localhost
or HTTPS, not an arbitrary insecure network address.

Use the project's virtual environment rather than the system `python.exe`.
The system Python may not have Flask, SQLAlchemy, MediaPipe, or the other
project dependencies installed.

To install dependencies into a selected environment, use the project
requirements file:

```powershell
.\.venv-2\Scripts\python.exe -m pip install -r requirements.txt
```

The application starts on port 5000 by default. `PORT` and `BRIDGESIGN_HOST`
can override the port and host. For local browser testing, keep the host at
its default (`127.0.0.1`).

## 3. System architecture

```text
Browser UI (HTML/CSS/JavaScript)
  ├─ Login, tabs, camera, image, learning, speech, avatar, and call interfaces
  ├─ Browser obtains camera permission and captures video
  └─ Sends frames or landmarks to the Flask API
                   │
                   ▼
Flask application (app.py)
  ├─ Authentication and HTML page routes
  ├─ JSON APIs for inference and app features
  ├─ Per-user live recognition state
  └─ Lazy initialization of inference modules
       ├─ MediaPipe hand detector
       ├─ 42-value feature extractor
       ├─ Saved scikit-learn classifier
       └─ Word assembler / optional motion classifier
                   │
                   ▼
Persistence (database.py)
  ├─ SQLite by default for local development
  └─ PostgreSQL when DATABASE_URL is configured
```

### Main files to navigate

| Area | Files |
|---|---|
| Flask app, routes, API behavior | [`app.py`](./app.py) |
| Shared camera/model/runtime settings | [`config.py`](./config.py) |
| Main page layout | [`templates/index.html`](./templates/index.html) |
| Main browser behavior | [`static/app.js`](./static/app.js) |
| Styling | [`static/style.css`](./static/style.css) |
| Learning interface | [`static/learning.js`](./static/learning.js), [`learning_mode.py`](./learning_mode.py) |
| Browser video import UI | [`static/video_ingest.js`](./static/video_ingest.js) |
| Experimental prerecorded-video recognition | [`video_recognition.py`](./video_recognition.py), [`video_recognition_worker.py`](./video_recognition_worker.py) |
| Hand landmark detection | [`hand_detector.py`](./hand_detector.py) |
| Static 42-feature representation | [`feature_extractor.py`](./feature_extractor.py) |
| Model loading and prediction | [`classifier.py`](./classifier.py) |
| Model artifact validation | [`model_artifacts.py`](./model_artifacts.py) |
| Word composition | [`word_assembler.py`](./word_assembler.py) |
| Persistence and ORM models | [`database.py`](./database.py) |
| Legacy static model trainer | [`model_trainer.py`](./model_trainer.py) |
| V2 static collection/training/evaluation | [`data_collector.py`](./data_collector.py), [`static_model_v2.py`](./static_model_v2.py) |
| Project positioning and claims | [`docs/PROJECT_POSITIONING.md`](./docs/PROJECT_POSITIONING.md) |
| V2 workflow instructions | [`docs/STATIC_A_TO_Z_V2.md`](./docs/STATIC_A_TO_Z_V2.md) |
| Repository artifact and Git/LFS audit | [`REPOSITORY_FORENSIC_AUDIT.md`](./REPOSITORY_FORENSIC_AUDIT.md) |

## 4. What happens when the app runs

1. Python executes `app.py`.
2. The database is initialized. The default local database is
   `data/bridgesign.sqlite3`; a configured `DATABASE_URL` can select
   PostgreSQL.
3. The Flask app registers page routes and APIs, and the learning module
   prepares its starter course/catalog data.
4. Flask serves the login page. After authentication, `/` renders
   `templates/index.html`.
5. The browser loads JavaScript, styles, and local MediaPipe/avatar assets.
6. The user starts the camera. The browser obtains a camera stream with
   `getUserMedia`.
7. The live loop sends camera frames or hand landmarks to Flask.
8. Flask runs recognition and returns JSON; JavaScript updates the live result
   and the other UI indicators.

Inference objects are initialized lazily when needed. This keeps the web
application import separate from the more expensive MediaPipe/model startup.
The `/health` and `/ready` endpoints are useful for checking app and inference
readiness.

## Experimental prerecorded-video sign recognition

The separate **Video Sign Recognition** page accepts authenticated uploads of
short `.mp4`, `.avi`, `.mov`, and `.m4v` clips. It runs the OpenHands
AI4Bharat WLASL BiLSTM checkpoint in a separate Python environment and process;
the existing static and dynamic recognizers and their saved models are not
changed. The request limit defaults to 25 MB, the clip limit to 10 seconds,
and the worker timeout to 90 seconds. These limits can be overridden with
`BRIDGESIGN_VIDEO_RECOGNITION_MAX_UPLOAD_BYTES`,
`BRIDGESIGN_VIDEO_RECOGNITION_MAX_SECONDS`, and
`BRIDGESIGN_VIDEO_RECOGNITION_TIMEOUT_SECONDS`.

### Install and launch

In PowerShell, install the pinned OpenHands source and CPU-only dependencies in
the separate `%LOCALAPPDATA%\BridgeSign\OpenHandsRuntime\venv` environment:

```powershell
.\tools\install_video_recognition.ps1
```

Restart BridgeSign so it reads the user-level runtime settings, then launch the
main app as usual:

```powershell
.\.venv-2\Scripts\python.exe app.py
```

Sign in and open <http://127.0.0.1:5000/video-recognition>. The page displays
the 2,000 supported labels and reports missing runtime configuration rather
than silently falling back to another model. The service also honors
`BRIDGESIGN_VIDEO_RECOGNITION_PYTHON` and
`BRIDGESIGN_OPENHANDS_SOURCE_DIR` if the isolated runtime is installed
elsewhere.

### Model scope and limitations

The bundled checkpoint is the OpenHands `WLASL-2000` BiLSTM checkpoint
(`epoch=109-step=49059.ckpt`), with its model configuration and class mapping.
It is listed in the
[OpenHands pretrained-model documentation](https://openhands.ai4bharat.org/en/latest/instructions/models.html)
and was tested from source revision
[`b9ccf9e`](https://github.com/AI4Bharat/OpenHands/tree/b9ccf9eaf2a71301fc4601a64a6af06fdef04c9b).
The checkpoint bundle is the
[`wlasl_lstm.zip`](https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/wlasl_lstm.zip)
release. The OpenHands source project is Apache-2.0 licensed. Its README says
to check the licensing terms of the training dataset; WLASL's repository
requires the
[Computational Use of Data Agreement](https://github.com/dxli94/WLASL/blob/master/C-UDA-1.0.pdf).
The checkpoint release does not state a separate checkpoint license, so do
not infer that the WLASL dataset or derived checkpoint has independent
commercial-use permission.

This is **isolated sign recognition**, not continuous sign-language-to-English
translation. The input is a time-ordered video converted to MediaPipe Holistic
body/hand pose and passed through the checkpoint's pose-selection and
normalization transforms. The returned top-five labels and scores are model
outputs, not calibrated probabilities or guaranteed correct meanings. The
model is trained for American Sign Language (ASL) WLASL labels; it has not
been validated for Malawian Sign Language (MSL), and the application does not
compose the guesses into sentences.

Inference is CPU-only and does not require CUDA or a dedicated GPU. A
standalone test was run on Windows with Python 3.12, 16 GB RAM, and Intel Iris
Xe integrated graphics using a separate environment with CPU PyTorch,
torchvision, PyTorch Lightning, MediaPipe, OpenCV, NumPy, OmegaConf, tqdm,
and natsort. The tested WLASL clip produced the
model's `drink` top label, but this single example is not an accuracy
evaluation. The isolated dependencies are pinned in
[`video_recognition_requirements.txt`](./video_recognition_requirements.txt);
the installer does not add them to BridgeSign's primary environment.

Run the module's focused and existing recognition regression tests with:

```powershell
.\.venv-2\Scripts\python.exe -m pytest tests\test_video_recognition.py tests\test_static_video_dataset.py tests\test_static_recognition.py -q
```

## 5. Static recognition: from image to letter

### 5.1 Hand detection: what MediaPipe does

MediaPipe is used as a **pre-trained hand detector/landmark estimator**. It
finds a hand and returns 21 landmark points such as the wrist, finger joints,
and fingertips. BridgeSign does not train MediaPipe from scratch and does not
fine-tune its landmark detector on the BridgeSign CSV.

The server implementation in [`hand_detector.py`](./hand_detector.py) uses
MediaPipe Tasks `HandLandmarker`. In the current live performance settings,
browser-side hand detection is disabled by default and the browser sends a
resized JPEG frame to the server endpoint `/api/infer_frame`. The server
detects the hand, extracts landmarks, and runs the classifier. There is also
browser-side MediaPipe code that can send pre-extracted landmarks to
`/api/infer_landmarks` when enabled.

This design separates two jobs:

1. **Landmark estimation:** MediaPipe predicts where the hand joints are.
2. **Sign classification:** BridgeSign's scikit-learn model predicts a label
   from the landmark geometry.

MediaPipe is not the A-Z classifier. It supplies the points used by the
classifier.

### 5.2 The 42 features

[`feature_extractor.py`](./feature_extractor.py) takes the 21 hand landmarks
and:

1. uses each point's X and Y pixel coordinates;
2. subtracts the wrist coordinates from all points, making the representation
   wrist-centred (translation normalization);
3. divides coordinates by the wrist-to-middle-MCP distance (scale
   normalization);
4. flattens 21 normalized `(x, y)` pairs into **42 numeric features**.

The current model does not use signer ID, session ID, timestamp, camera device,
or image pixels as classifier features. The V2 CSV records provenance beside
the 42 features so evaluation can group samples; only the 42 feature columns
are supplied to the model.

This representation is designed for static hand shapes. It does not by itself
represent a full temporal trajectory, facial expression, body posture, or
linguistic context.

### 5.3 Prediction and stability

[`classifier.py`](./classifier.py) loads a saved pickle payload containing a
scikit-learn pipeline and class labels. It returns a label and confidence.
The live recognition code may predict both the supplied feature vector and its
horizontal mirror, then use the more confident result. This helps with front-
versus-back-camera orientation differences.

The backend maintains per-user recognition state across frames. It requires a
label to be stable over consecutive frames before confirming it; the default
threshold is five frames. Static confirmations also have a confidence floor.
These rules reduce flicker and accidental one-frame outputs, but they do not
make an incorrect model prediction correct.

In Word mode, confirmed letters/signs are passed to
[`word_assembler.py`](./word_assembler.py) to form text. This is not
continuous sign-language translation or a language model.

### Experimental live motion gestures

The separate webcam motion-gesture classifier is disabled by default. To
enable its current experimental ONNX model in PowerShell, set the environment
before starting BridgeSign:

```powershell
$env:BRIDGESIGN_LIVE_GESTURES = "1"
$env:BRIDGESIGN_GESTURE_MOTION_THRESHOLD = "6.0"
.\.venv-2\Scripts\python.exe app.py
```

The model accepts 15 frames of 42 hand features and currently predicts only
`GOODBYE`, `HELLO`, `HELP`, `J`, `NO`, `PLEASE`, `SORRY`, `STATIC`, `STOP`,
`THANK_YOU`, `WATER`, `YES`, and `Z`. Nine labels in the local 22-label
gesture dataset were not trained into this model and are not supported by it.
The browser sends inference frames about every 70 ms, while the collector
records camera frames at up to 30 fps; sequence timing has not been validated
across those rates. The feature is experimental and its recorded-data probes
are not an independent accuracy evaluation. `/ready` reports whether the
optional model is enabled and actually available.

#### Gesture data collection and candidate training

The legacy `data/gesture_dataset.csv` has no signer or session metadata. In the
current local copy, 327 rows are exact duplicates of other sequences; the
original capture/merge cause cannot be established. The trainers now keep
exact duplicates together during validation and prefer signer-separated
splits, then session-separated splits when provenance is present. Duplicate-
only splits do not establish signer/session independence.

For new captures, write to the separate versioned CSV; this leaves the legacy
dataset unchanged:

```powershell
.\.venv-2\Scripts\python.exe gesture_collector.py --sign HELP --samples 200 --v2 --signer-id signer-01
```

Each run creates a session ID unless `--session-id` is supplied, and records
the device, capture timestamp, frame size, and observed sequence frame rate.
Use distinct signer IDs across signers and distinct session IDs across
recording sessions. Keep signer identifiers pseudonymous and access-controlled.

After collecting reviewed, sufficiently representative data, training can be
run against that versioned file:

```powershell
.\.venv-2\Scripts\python.exe gesture_trainer.py --dataset .\data\gesture_dataset_v2.csv
.\.venv-2\Scripts\python.exe lstm_gesture_trainer.py --dataset .\data\gesture_dataset_v2.csv
```

These commands save separate `*_candidate` artifacts and refuse to overwrite
existing outputs. Candidate models are not loaded by the application and must
not replace the active ONNX model without independent evaluation. The old
22-class data has no signer/session provenance, so its validation remains
limited to duplicate-group separation.

## 6. What model is used?

The existing static classifier is a **Random Forest classifier** from
scikit-learn, inside a pipeline with `StandardScaler`.

Random Forest is an ensemble: it fits many decision trees on the training
data and combines their predictions. A decision tree makes a sequence of
feature-based decisions and ends in a predicted class. Combining trees can
make the final prediction less dependent on one tree's quirks.

The V2 configuration recorded in the training report is:

| Setting | Value |
|---|---:|
| Classifier | `RandomForestClassifier` |
| Trees (`n_estimators`) | 80 |
| Maximum tree depth | 18 |
| Minimum samples per leaf | 2 |
| Class weighting | `balanced` |
| Random seed | 42 |
| Parallel jobs | 1 |
| Input scaler | `StandardScaler` |

The model predicts among the 26 labels A-Z. The exact tree count, depth, leaf
size, and seed are recorded in
[`reports/static_a_to_z/sign_model_v2_training.json`](./reports/static_a_to_z/sign_model_v2_training.json).

### “How many epochs did you train?”

**There are no epochs in this training process.** Epochs describe repeated
passes through a dataset in iterative neural-network training. This static
classifier is a Random Forest, not a neural network trained by epochs.

The comparable model setting is **80 trees** (`n_estimators=80`). The reported
V2 fit took approximately **26.9 seconds** on the recorded run, but elapsed
time varies with machine and environment. Do not call the 80 trees “80
epochs.”

MediaPipe's landmark model is already pretrained. BridgeSign uses it for
landmark inference; this project does not report training epochs for that
pretrained model either.

## 7. Training data, augmentation, and model versions

### Protected baseline

- Dataset: `data/dataset.csv`
- Model: `models/sign_model_light.pkl`
- This is the existing model used by the live app by default.
- It is separate from the V2 model and was not overwritten by V2 training.

The historical trainer-faithful reference supplied for the baseline is about
**80.65% accuracy / 0.762 macro F1**. The checked-in legacy random-holdout
metrics report a different, much higher result. Those results use different
evaluation setups and should not be presented as interchangeable.

### V2 interim model

- Dataset: `data/dataset_v2.csv`
- Model: `models/sign_model_v2.pkl`
- Training report: `reports/static_a_to_z/sign_model_v2_training.json`
- Evaluation report: `reports/static_a_to_z/sign_model_v2_evaluation.json`
- Quality report and review candidates:
  `reports/static_a_to_z/dataset_v2_quality.json` and
  `reports/static_a_to_z/dataset_v2_review.csv`

Current measured data status:

- 1,880 structurally valid rows; zero structurally invalid rows.
- 60 samples for each normal class and 100 for each priority class.
- Normal classes: A, C-L, N-O, R, T, W-Y (60 each).
- Priority classes: B, M, P, Q, S, U, V, Z (100 each).
- One signer (`S01`) and two collection sessions.
- Official targets remain 300 per normal class and 500 per priority class.
- The current overall count is 20% of the 9,400-sample target.
- One identical feature vector appears with labels C and D. The two rows are
  flagged for human review; they have not been automatically deleted or
  relabelled.

### How V2 augmentation works

Augmentation happens **after splitting and only on training samples**:

- three rounds of noise and scale-jitter augmentation are created by the
  existing trainer helper;
- safe horizontal-mirror views are added for classes not in the exclusion
  list (`J`, `P`, `Q`, `Z`);
- validation/test rows are never augmented.

Augmentation increases training variety for fitting; it does **not** create
new real signers, new independent camera sessions, or genuine new collected
examples. The dataset sample count remains 1,880.

## 8. Evaluation: what the current numbers mean

To evaluate two models fairly on a shared test set, the same raw test rows are
sent to the protected baseline and the V2 model. The evaluation report includes
accuracy, macro precision/recall/F1, per-class metrics, confusion matrix,
confused pairs, and overlap checks.

For the current interim run:

| Model | Shared test accuracy | Shared test macro F1 |
|---|---:|---:|
| Protected baseline | 64.50% | 0.6414 |
| Interim V2 | 96.45% | 0.9628 |

The shared test block contained 338 raw rows. These are **not** training
accuracy figures.
The V2 report's largest single confused pair was D/E with two errors; several
other pairs had one error each. These small counts should not be interpreted
as a stable ranking of real-world failure modes.

## 8.1 Useful API route groups

The frontend calls Flask endpoints rather than reading model files itself.
Some useful routes when tracing the code are:

| Route | Purpose |
|---|---|
| `/` | Render the signed-in main interface |
| `/login`, `/register`, `/logout` | Authentication pages/actions |
| `/api/camera/start`, `/api/camera/stop` | Reset per-user live recognition state |
| `/api/infer_frame` | Accept a browser JPEG frame; server detects hand and predicts |
| `/api/infer_landmarks` | Accept browser-extracted hand landmarks and predict |
| `/api/translate_image` | Process a still-image upload |
| `/api/learning/...` | Course, lesson, quiz, practice, and progress operations |
| `/api/emergency/...` | List and manage emergency phrase choices |
| `/call/...` | Render call pages; a Flask-Sock endpoint handles room signalling |
| `/health`, `/ready` | Basic application and inference readiness checks |

For live recognition, begin with the camera callbacks in `static/app.js`, then
follow the selected inference endpoint in `app.py`. The route is protected by
the login session, returns JSON, and keeps confirmation/word assembly state
per user.

### Important limitation on those scores

The dataset has only one signer and two sessions. The sessions share only
class K, so a session-disjoint split cannot put all 26 classes into training
and test. For the explicit interim run, the pipeline used a seeded,
class-stratified 82/18 row holdout before augmentation. It kept the exact
duplicate C/D rows together, and there were no exact duplicate vectors crossing
the resulting train/test boundary.

This is an untouched row holdout, but it is **not signer/session-disjoint**:
frames from the same signer and sessions can appear on both sides. Similar
frames within a session may therefore make the test score optimistic. There
was no separate validation split, and group-bootstrap confidence intervals
were not estimable with only two independent collection groups. The report
therefore does **not** recommend replacing the active baseline.

Do not say “V2 is 96.45% accurate in general” or claim proven improvement based
on this small, non-grouped interim evaluation. Explain the split and its
limitations whenever presenting the figures.

The 80.65% / 0.762 baseline reference is a historical result from another
evaluation setup. It is not the 64.50% baseline score measured on the V2 shared
test rows. The shared-test comparison is the direct side-by-side result for
this interim experiment.

## 9. Questions reviewers may ask

### What is the research problem?

Communication barriers can make everyday interaction, learning, and access to
information harder for Deaf and hard-of-hearing people. BridgeSign explores a
modular assistive communication and learning prototype, while recognizing that
locally validated MSL data and user evaluation are still required.

### What is the system's input and output?

For static recognition, the input is a camera frame containing a visible hand.
MediaPipe estimates 21 hand landmarks; the feature extractor turns them into
42 normalized numbers; the classifier outputs a candidate label and
confidence. The app stabilizes predictions across frames before confirming
them.

### Why use MediaPipe?

It provides a practical pretrained hand-landmark detector, so this project
can focus its static classifier on hand geometry instead of training a
landmark detector from scratch. It is a dependency and landmark-estimation
stage, not the classifier that decides A-Z.

### Why Random Forest?

The feature vector is compact structured numeric data, and scikit-learn's
Random Forest provides a fast, CPU-friendly baseline with interpretable
hyperparameters and straightforward evaluation. A previous SVM/voting
approach was removed from the legacy trainer because it took much longer
without a meaningful gain in that experiment. A future model change should be
supported by a controlled comparison, not assumed to be better.

### Why StandardScaler for trees?

It is retained as part of the project's existing scikit-learn pipeline for
compatibility and reproducibility. Tree splits are generally less sensitive
to feature scaling than distance- or gradient-based models, but the reported
pipeline applies this scaler consistently.

### How do you handle overfitting?

The intended controls are a held-out raw-data split before augmentation,
fixed model complexity (80 trees, depth 18, minimum leaf size 2), exact
duplicate checks, and evaluation separate from the fit. The current interim
dataset does not support signer/session-disjoint evaluation, so generalization
is not established; that limitation is more important than the high row-holdout
score.

### Why isn't there an epoch count?

Random Forest is not trained in epochs. It fits 80 trees. Epoch count would be
relevant only if describing a neural network trained over repeated passes
through data.

### What is class imbalance, and what did you do?

Normal classes have 60 examples and the priority classes have 100 each.
Random Forest uses `class_weight="balanced"`, and evaluation reports
macro-averaged and per-class scores. Class weighting does not replace the
collection targets.

### What does “macro F1” mean?

F1 is the harmonic mean of precision and recall. Macro F1 computes F1 for
each class and averages those class scores equally, so a class with fewer
examples still matters as much as a class with more examples. Accuracy is
the fraction of all predictions that were correct.

### What is the difference between validation and test data?

Training data are used to fit the model. Validation data can support choices
made during development. Test data should remain untouched until final
evaluation. In the current interim case there is no separate validation
partition; the held-out 338 rows are reserved for the baseline-vs-V2
evaluation.

### Is this a Malawian Sign Language translator?

Not yet. The static alphabet workflow is a limited handshape-recognition
prototype, and the project does not have a sufficiently large, signer-diverse,
human-validated MSL dataset to support that claim. Do not equate a working
alphabet demo, ASL-related asset, or experimental ASL dataset with validated
MSL translation.

### Does the app use the V2 model now?

No. The app's active classifier searches for `sign_model_light.pkl` first.
The separately saved `sign_model_v2.pkl` is used by the offline V2 evaluation
pipeline and has not been activated as the production/default model.

### How will you improve the evidence?

Continue collecting real samples toward at least 300 per normal letter and 500
per priority letter, with balanced collection across multiple controlled
sessions and preferably multiple signers. Review flagged rows manually. Then
use signer-disjoint evaluation when there are enough signers, or
session-disjoint evaluation otherwise. Only consider changing the active model
after the held-out evaluation supports it.

### What are the biggest limitations?

- one signer and two collection sessions in the current V2 data;
- far fewer than the 300/500 collection targets;
- a suspicious C/D pair requiring human review;
- no signer/session-disjoint A-Z evaluation from current data;
- static 42-value handshape features do not represent full sign-language
  grammar, facial expression, or temporal motion;
- no demonstrated participant study, educational outcome, or production
  emergency-response validation.

## 10. Commands for the static A-Z workflow

Run these from the project root with the project environment.

Inspect dataset quality:

```powershell
.\.venv-2\Scripts\python.exe static_model_v2.py quality
```

Train only when the official minimums are met:

```powershell
.\.venv-2\Scripts\python.exe static_model_v2.py train --seed 42
```

Explicitly train an interim model below the official targets:

```powershell
.\.venv-2\Scripts\python.exe static_model_v2.py train --allow-under-target --seed 42
```

Compare the saved V2 candidate with the protected baseline on the reserved
test rows:

```powershell
.\.venv-2\Scripts\python.exe static_model_v2.py evaluate --bootstrap-iterations 1000
```

The `--allow-under-target` switch is explicit and recorded in the model and
report metadata. It does not lower or remove the 300/500 targets and does not
skip structural quality checks.

## 11. Research-safe summary to use in a presentation

> BridgeSign is a Flask-based assistive communication and learning prototype.
> Its static recognizer uses MediaPipe to estimate hand landmarks, converts
> those landmarks to a fixed 42-value representation, and classifies the
> representation with a scikit-learn Random Forest. The interim V2 model was
> trained on 1,880 real collected samples, below the official per-class
> collection targets, using an explicit under-target override. Its row-based
> held-out score is promising for exploration but is not evidence of
> signer-independent generalization because the data currently come from one
> signer and two sessions. The protected baseline remains the active app model.
> Further multi-signer collection, human review, and group-disjoint evaluation
> are needed before claiming robust MSL recognition.#   s m a r t s i g n .  
 