# Gesture Module Investigation Handoff

**Snapshot:** 2026-10-23  
**Repository branch:** `version-2`

This file records the current implementation and observations from the live
gesture investigation. It is context for a follow-up review, not a set of
recommended fixes.

## Runtime path observed

- The browser uses the bundled MediaPipe Hands tracker to extract 21 hand
  landmarks and sends landmark features to the Flask `/api/infer_landmarks`
  endpoint.
- The server runs the separate temporal gesture classifier. Startup output in
  this workspace showed `models/gesture_lstm_v2.onnx` loaded with 22 labels.
- The static letter classifier did not load in the same startup; its startup
  output said no valid trained static model was found. The live gesture path
  can still run separately.
- Word-class suggestions are filtered by the selected mode. In Letter mode,
  a word-class result is reported as a possible word and asks for Word mode.
  The motion letters J and Z are part of the gesture model's label set.

## Changes present in this branch

- `gesture_classifier.py` selects the named V2 ONNX model before legacy
  fallbacks instead of choosing a model by file modification time. It records
  the loaded artifact/backend/error and validates ONNX input/output shapes and
  finite prediction values.
- `app.py` initializes recognition modules independently, reports readiness
  per module at `/ready`, and lets the gesture recognizer run without a loaded
  static letter classifier. Live inference responses include tentative
  gesture candidate, confidence, and status fields.
- `static/performance-config.js` enables client-side hand tracking, uses an
  80 ms inference interval, and increases the request timeout to 1.8 seconds.
- `static/app.js` displays tentative candidates and separate feedback for
  Word mode, low confidence, warming up, waiting for motion, no gesture, and
  a rejected J/Z path check. `templates/index.html` uses updated script cache
  keys.
- `PROJECT_IMPROVEMENT_ROADMAP.md` and
  `reports/PHASE0_CURRENT_STATE_AUDIT.md` record model/data provenance and
  validation boundaries.

## Live observations

During the user-directed camera check on 2026-10-23:

- The browser reported 21 hand landmarks and initially displayed “Gesture
  path was unclear” while Letter mode was selected.
- The model later offered FOOD at 99% as a possible word in Letter mode and
  prompted for Word mode.
- A later window offered YES at 72% as a tentative candidate. The page then
  displayed Z in Recent Signs.
- The intended sign and exact movement timing were not recorded as ground
  truth. These observations show that the live path returned candidates and a
  committed label; they do not establish whether any prediction was correct
  or establish live recognition accuracy.

The V2 metadata's 97.9% validation result uses an exact-duplicate-only split,
contains 306 duplicate rows, and is not signer-independent. It is not a
live-camera accuracy result. The older 13-label export reports 99.4%, but its
metadata does not document the split basis.

## Checks performed

- `python -m py_compile app.py gesture_classifier.py` completed successfully.
- `git diff --check` completed successfully.
- The app was launched with `.venv\Scripts\python.exe app.py`. Startup logs
  showed the V2 gesture model and 22 labels loaded. The browser camera was
  restarted and returned live recognition output.
- A PowerShell request to `http://127.0.0.1:5000/ready` from the command
  environment could not connect, so this session did not capture a fresh
  `/ready` JSON response.
- No full test suite or independent-sign evaluation was run in this session.

## Files touched by the current work

- `app.py`
- `gesture_classifier.py`
- `static/app.js`
- `static/performance-config.js`
- `templates/index.html`
- `PROJECT_IMPROVEMENT_ROADMAP.md`
- `reports/PHASE0_CURRENT_STATE_AUDIT.md`
- `GESTURE_MODULE_HANDOFF.md`
