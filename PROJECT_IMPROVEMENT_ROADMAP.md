# BridgeSign Project Improvement Roadmap

**Purpose:** one project-wide plan for making BridgeSign more reliable,
demonstrable, and defensible. This is a roadmap, not a claim that the proposed
work has already been completed or validated.

**Current snapshot:** 2026-10-23. Recheck the live code, model artifacts, and
reports before presenting this snapshot as current.

## Project position

BridgeSign is an assistive communication and sign-learning prototype. It
combines camera recognition, gesture recognition, image/video experiments,
learning activities, avatar demonstrations, emergency phrases, and calls.
These modules have different levels of maturity. A feature existing in the UI
does not prove that its recognition, language content, accessibility, or
real-world benefit has been validated.

The goal is to improve each feature with evidence and visible status—not to
promise that a model or module has "no limitations." In particular, do not
describe BridgeSign as a validated MSL translator or emergency-response
service unless appropriate independent evidence and review exist.

## Flagship use case: supported clinic communication

### The idea

Design a **Clinic Communication Assistant** for a Deaf patient and a clinician
who do not share a fluent communication method. The clinician opens a
patient-controlled session, chooses the agreed language and communication
approach, and both people use a shared screen to exchange information. The
assistant helps each person understand and confirm the other; it does not
replace a qualified sign-language interpreter, diagnose the patient, or make
clinical decisions.

The desired experience is a clear, respectful conversation—not a camera that
silently guesses at every sign. The patient should control when the camera or
microphone is active, what is shown or spoken, and whether anything is saved.
The clinician should be able to see when content is a verbatim transcript,
patient-approved rephrasing, model guess, or human-reviewed sign.

### Example visit flow

1. **Start together:** The clinician opens a clinic session. The patient sees
   the language, privacy, and interpreter options and explicitly consents
   before camera/microphone use. Declining AI must not block access to care.
2. **Choose a communication mode:** Offer a qualified interpreter when
   available and appropriate; otherwise let the patient choose supported
   options such as typed chat, large-text phrase board, or speech-to-text and
   text-to-speech. Do not presume a preferred language or dialect.
3. **Patient communicates:** The patient signs or types. Initially, typed
   messages and an accessible phrase board provide the dependable path. Sign
   recognition may suggest only labels and phrases supported by a reviewed,
   evaluated model. Show uncertainty and ask the patient to confirm before
   speaking or sharing a model interpretation.
4. **Clinician responds:** The clinician speaks or types. Speech-to-text shows
   the transcript for correction; the patient can read it, replay speech, or
   request plain-language wording. Preserve the original wording alongside
   any suggested simplification so important meaning is not silently changed.
5. **Clarify, do not guess:** An AI assistant may offer a neutral clarification
   such as “Would you like to tell the clinician where the pain is?” It must
   not infer symptoms, fill gaps, select a diagnosis, recommend treatment, or
   turn an uncertain sign prediction into a medical fact.
6. **Confirm the plan:** The clinician types or dictates next steps. The
   patient reviews the displayed plan and confirms, edits, or asks for a
   repeat. Clearly distinguish patient-confirmed information from
   machine-generated suggestions.
7. **End privately:** The session ends visibly. By default, do not retain raw
   audio/video or transcripts. Any optional note export or saving requires
   separate, informed consent and an explicit retention policy.

### What “AI in the module” should mean

Do not train one giant model on every project file, user conversation, or
clinic recording. That does not make a system reliably multilingual or
medically accurate, and it can expose sensitive data or amplify unreviewed
sign errors. Build a **bounded communication assistant** that coordinates
separate, replaceable capabilities:

- **Input adapters:** reviewed sign-recognition models, typed text, and
  speech-to-text. Each reports its source, supported language, confidence,
  and errors.
- **Conversation coordinator:** manages turn-taking and offers short,
  neutral clarification prompts. It cannot diagnose, prescribe, or claim to
  interpret unsupported signs.
- **Patient-controlled language helper:** optionally rewrites clinician text
  into simpler wording while keeping the original visible and asking the
  patient whether the rewrite is clear.
- **Output adapters:** editable text, large text, text-to-speech, and—only
  after linguistic review—sign/avatar demonstrations for supported content.
- **Safety and consent layer:** enforces activation, confirmation, privacy,
  retention, and fallback behavior independently of the generative model.

The coordinator may use a carefully selected general language model for
limited plain-language and dialogue-support tasks, but it must not be treated
as a sign-language translator just because it can generate fluent text. Medical
meaning must stay traceable to what the patient or clinician actually
confirmed. If a supported sign is not recognized, the system should say so and
offer typing, a phrase board, repetition, or a qualified interpreter—not
invent a likely message.

### Delivery plan

**Stage A — Accessible communication MVP (before AI sign translation)**

- Build a patient-controlled clinic session with typed chat, large-text
  display, turn-taking, editable phrase board, text-to-speech, and speech
  transcription shown for correction.
- Make privacy, consent, language selection, cancel/end, and no-network
  behavior explicit.
- Test the end-to-end conversation with Deaf users and clinicians, including
  misunderstanding, correction, and interpreter handoff.

**Stage B — Bounded AI assistance**

- Add opt-in plain-language rephrasing and neutral clarification suggestions.
- Always show original and suggested text together; require user acceptance
  before sharing or speaking a suggestion.
- Evaluate factual preservation, omissions, added meaning, reading ease, and
  response time using consented, non-sensitive test scenarios reviewed by
  Deaf participants and clinical staff.
- Keep diagnosis, triage, treatment advice, and autonomous note-writing out of
  scope.

**Stage C — Reviewed sign input/output**

- Agree with Deaf signers, MSL experts, clinicians, and accessibility
  specialists on language/dialect, sign variants, vocabulary, and consent.
- Collect sign data with explicit consent and signer/session provenance; do
  not repurpose clinic footage for training by default.
- Validate recognition on held-out signers and realistic clinic conditions.
  Show confidence and require patient confirmation for every interpretation.
- Validate generated or avatar sign output with qualified reviewers and user
  testing before presenting it as a communication option.
- Keep a qualified-interpreter route and a non-AI route available throughout.

### Safety and acceptance gates

Do not pilot this as a substitute for interpretation or medical advice. Before
any clinic pilot:

- obtain privacy, security, consent, accessibility, clinical-governance, and
  applicable legal review for the deployment location;
- define who can access the session, whether data are stored, retention and
  deletion behavior, breach handling, and vendor/model data-processing terms;
- test false interpretations, negation, numbers, medication names, allergies,
  pain location/severity, names, and corrections; uncertain or unsupported
  output must be visibly uncertain and easy to reject;
- establish a stop/fallback procedure and never claim a message was understood
  until the receiver confirms it;
- run usability and safety evaluation with Deaf participants and clinicians.
  Report misunderstandings and correction burden, not just model accuracy.

**Pilot gate:** No unresolved critical privacy/security or clinical-safety
issue; patients can reject/correct AI output; the communication flow has a
tested interpreter/non-AI fallback; and Deaf participants and clinicians
approve the tested experience for a limited, supervised pilot.

## Current evidence and boundaries

- The webcam capture session collected 1,781 valid feature rows: 1,720 for 24
  held static-letter classes and 61 motion-letter rows (60 J, 1 Z). The static
  V2 workflow excludes J and Z; the rows remain preserved in the dataset.
- `models/sign_model_v2.pkl` is an **interim** 24-class candidate trained from
  one signer/session. It has not been activated in the live app and is not an
  independent-session accuracy result.
- The current gesture ONNX model is separate from the static classifier.
  Check the model's loaded labels and readiness endpoint rather than inferring
  availability from class metadata or filenames.
- The pose and hand detector task assets are local model artifacts. At the
  time of the snapshot, stale Git LFS pointer stubs had been removed from the
  current model directory. Keep artifact verification in startup diagnostics.
- Existing audits document important prior findings, but their dates and
  artifact assumptions differ from the current workspace. Treat them as
  historical evidence, not as a replacement for rerunning checks:
  [static recognition audit](./STATIC_RECOGNITION_AUDIT.md),
  [repository forensic audit](./REPOSITORY_FORENSIC_AUDIT.md), and
  [avatar/content audit](./AVATAR_CONTENT_AUDIT.md).

## Improvement principles

1. **Correctness before feature count.** Prefer a smaller set that works
   consistently over a larger unsupported vocabulary.
2. **Separate static and motion recognition.** A still pose and a temporal
   gesture need different examples, model paths, and evaluation.
3. **Keep evidence with every model.** Record its source data hash, labels,
   feature format, training configuration, evaluation split, metrics, and
   artifact checksum.
4. **No misleading success states.** Show which module is ready, what labels it
   supports, and whether output is a tentative candidate or a confirmed result.
5. **Human review for language content.** Generated animations, valid XML, and
   model predictions do not establish that a sign is linguistically correct.
6. **Protect users and their data.** Collect only necessary data, use
   pseudonymous signer IDs, explain camera/data use, and review retention and
   consent before broader collection or publication.
7. **Ship in gates.** Do not activate a candidate merely because training
   completed or an internal score is high.

## Workstreams

### 1. Recognition and model lifecycle

**Current issue:** Static letters, live gestures, pose tracking, and uploaded
video recognition have different artifacts and readiness requirements. A
single "inference ready" indicator can hide a missing classifier. Stale LFS
pointer files can also look like ordinary files.

**Plan**

- Keep the static classifier restricted to the 24 held-letter classes (A-Z
  excluding J and Z); handle J and Z through motion recognition.
- Maintain separate candidate and production paths. Promote a candidate only
  through a reviewed command or explicit configuration change.
- Add a model manifest for every artifact: model family, artifact path and
  SHA-256, labels, feature shape/version, training-data hash, training date,
  status (candidate/production), and evaluation report.
- At startup, validate artifacts before loading. Report missing files, LFS
  pointers, invalid payloads, label mismatches, and feature-shape mismatches as
  module-specific errors.
- Make `/ready` report each module independently: hand detector, pose detector,
  static classifier, gesture model, and video worker. Keep health/liveness
  separate from full recognition readiness.
- Calibrate confidence thresholds using held-out data. Distinguish "candidate"
  from "confirmed" in the UI; do not let a low-confidence guess become a
  committed letter just because it repeats.
- Benchmark end-to-end camera-to-result latency and failure rates, not only
  classifier inference time.

**Acceptance evidence**

- Automated tests cover missing/corrupt/pointer artifacts, label sets,
  feature dimensions, fallback selection, and per-module readiness.
- Release report includes per-class precision/recall/F1, confusion pairs,
  calibration/coverage tradeoffs, held-out split details, and measured
  end-to-end latency on documented hardware.

### 2. Static letters: collection, training, and promotion

**Current issue:** The current V2 training sample is from one
signer/session. It can train an exploratory candidate, but it cannot support a
claim of generalization to new sessions or people.

**Plan**

- Keep using the provenance-aware V2 dataset; preserve the legacy dataset and
  model paths.
- Collect multiple balanced sessions. Use a new session ID for each session
  and a new pseudonymous signer ID for each signer. Vary lighting, camera
  distance, orientation, background, and natural pose while keeping signs
  legible.
- The documented collection targets are 300 samples per ordinary static class
  and 500 for priority classes B, M, P, Q, S, U, and V. Current one-session
  counts are lower. These are collection targets, not a guarantee of accuracy.
- Keep J and Z out of static capture. If earlier J/Z rows remain in the CSV,
  retain them for auditability and report them separately.
- Review the dataset QC and human-review list before training. Do not
  auto-delete, relabel, or hide suspicious examples.
- Use signer-disjoint evaluation when enough signers are available; otherwise
  use session-disjoint evaluation. If only row-level splitting is possible,
  label results exploratory and do not use them as a promotion gate.
- Compare candidate and baseline on the same untouched test examples. Record
  all confusion and class-specific misses, particularly visually confusable
  handshapes.
- Run an informal live-camera pilot separately from the offline test report.
- Promote only after the project owner reviews the evaluation and confirms
  expected real-use behavior.

**Next action:** collect the planned evening and next-morning sessions using
fresh session IDs, then run quality review, training, and group-separated
evaluation. The detailed collector commands and safeguards are in
[the static-letter V2 guide](./docs/STATIC_A_TO_Z_V2.md).

### 3. Motion gestures and temporal recognition

**Current issue:** A gesture is a sequence, not a still image. Collection
rate, browser inference rate, sequence duration, and motion thresholds affect
what the model sees. Historical recorded-data scores may include training
examples and are not independent accuracy.

**Plan**

- Keep gesture vocabulary limited to labels actually present in the loaded
  model. Never add a class by editing metadata alone.
- Collect temporal sequences with signer/session/device/timestamp and observed
  frame-rate metadata. Record complete gesture start, movement, and end rather
  than repeated static frames.
- Align sequence resampling and live inference timing. Measure sequence
  duration and dropped-frame behavior under actual browser capture.
- Test STATIC/no-motion rejection, gesture onset/offset, J/Z motion paths,
  one-hand/two-hand behavior, mirrored camera input, and static-to-gesture
  crossover.
- Report per-label results on a held-out signer/session split. Separate
  recorded-dataset diagnostics from live-camera results.
- Keep threshold changes tied to validation reports; test false activation as
  well as missed gestures.

**Acceptance evidence**

- Each class has a defined collection protocol and enough independent groups
  for the planned evaluation.
- The live report includes per-class recall, false activations per minute,
  end-to-end response time, and behavior when no recognized gesture is present.

### 4. Uploaded video and broader vocabulary

**Current issue:** The experimental video pipeline is separate from the webcam
letter/gesture pipeline. A successful inference request does not prove video
translation is accurate. Dataset and checkpoint terms also need review before
redistribution or public deployment.

**Plan**

- Treat video recognition as experimental until evaluated on authorized,
  held-out clips with labels and a clearly defined metric.
- Establish a small reproducible benchmark with class coverage, clip-level
  splits, top-k accuracy, confusion analysis, and failure cases.
- Verify data/checkpoint licenses, consent, and redistribution terms before
  publishing weights, example clips, or derived datasets.
- Clearly label output as a model suggestion, not a sentence translation,
  unless language modeling and linguistic review support that claim.
- Do not merge video labels into live gesture or static models without an
  explicit cross-module evaluation.

### 5. Learning experience and progress

**Current issue:** Lessons and quizzes exist, but the presence of lesson
content does not demonstrate learning effectiveness or curricular correctness.

**Plan**

- Define learning outcomes and prerequisites for each course/module/lesson.
- Add a content review state and reviewer attribution; distinguish
  "technically playable" from "linguistically reviewed."
- Test progress persistence, retries, interrupted sessions, quiz scoring,
  accessibility, keyboard use, screen readers, and small-screen layouts.
- Run usability sessions with intended learners and educators. Measure task
  completion, errors, time, and learner feedback; do not claim learning gains
  without an appropriate study.
- Make camera practice feedback explain uncertainty and allow replay/practice
  without penalizing the learner for detector failures.
- Align lesson vocabulary with the actual supported recognition labels and
  available, reviewed demonstrations.

**Acceptance evidence**

- Every published lesson has an outcome, reviewed content, accessible
  interactions, passing persistence tests, and a clear completion rule.
- Any educational-effectiveness claim cites a study design and results.

### 6. Avatar, signs, and language content

**Current issue:** Asset validity is not sign validity. Existing content audits
record large review gaps, including signs without lessons or complete usage and
movement notes, and no human-validated signs at audit time.

**Plan**

- Recruit or partner with qualified MSL signers/reviewers; agree on dialect,
  regional variation, and acceptable sources before writing definitions.
- Review each sign's meaning, handshape, movement, non-manual markers, usage,
  and regional variants. Record reviewer, date, evidence/source, and status.
- Label imported or generated avatars as demonstrations until a reviewer
  validates their motion. Do not treat SiGML/XML parse success or VRM
  playback as linguistic validation.
- Prioritize signs already used in recognition, emergency phrases, and
  lessons. Fill missing meanings and movement notes only from reviewed
  sources—not guesswork.
- Add fallback explanations when no reviewed avatar or asset exists.
- Retain license, attribution, and asset provenance alongside each asset.

**Acceptance evidence**

- Publicly presented signs have a reviewer and review date, or are explicitly
  marked unverified.
- Asset integrity, renderer compatibility, content status, and linguistic
  review are tracked as separate checks.

### 7. Emergency communication, speech, and calls

**Goal:** Build a Deaf-first emergency support tool that helps a person in
Malawi quickly communicate what is happening and reach a trusted person or
verified care provider. This should be meaningfully different from a generic
phrase list or chatbot: fast to activate, usable without signing fluency from
the other person, localized with community input, and honest about whether a
call or message actually connected.

This is an emergency communication aid, not an emergency dispatch service,
medical triage system, or guarantee of rescue. It must not advertise
unverified phone numbers, claim to contact a clinic when it only opened the
phone dialer, or delay the user's own attempt to call for help.

**Example emergency flow**

1. **One clear entry point:** A prominent, accessible SOS action is reachable
   quickly without navigating through lessons or account settings. Provide
   vibration/visual feedback, large controls, and a way to cancel an accidental
   activation.
2. **Choose or confirm the need:** Offer a small, reviewed set of urgent needs
   such as “I need medical help,” “I am injured,” “I cannot hear/speak,”
   “please call my trusted contact,” and “I need transport.” Let the person
   select, edit, or decline; do not infer the emergency from a camera feed.
3. **Show a shareable emergency card:** Present large text and optional
   text-to-speech for a bystander, with locally reviewed wording and
   language choices. Include only user-approved details, such as preferred
   name, communication needs, trusted contact, allergies or critical medical
   information if the user has deliberately added them, and current location
   only after permission.
4. **Reach help through verified routes:** Offer explicit actions for a
   configured trusted contact and for locally verified clinic/emergency
   contacts. Where supported and tested, provide call, SMS, or another
   locally usable route. Show what the operating system actually opened and
   whether delivery/connection is confirmed; distinguish “dialer opened,”
   “message prepared,” “sent,” and “recipient confirmed.”
5. **Work when connectivity is poor:** Keep the emergency card and user-picked
   contacts available offline. If a network route fails, explain that clearly
   and show alternatives, such as displaying the card to a nearby person,
   retrying, or using the phone's native emergency call feature. Do not imply
   that offline mode can send a message.
6. **Keep the person in control:** Allow a trusted contact to be called only
   from a configured and user-approved list. Do not silently share precise
   location, health details, audio, video, or transcripts. Provide an obvious
   stop/end control and a plain-language explanation of each share action.

**Malawi localization and validation**

- Co-design with Deaf Malawians, local sign-language communities, disability
  organizations, clinicians, clinics, and emergency-service stakeholders.
  Confirm local languages, MSL varieties, preferred phrasing, literacy needs,
  power/connectivity constraints, and what responders can realistically
  receive.
- Verify each clinic and emergency contact with the organization responsible,
  record the source and date of verification, assign an owner and review
  interval, and remove or flag contacts that cannot be confirmed. Do not
  guess hotline numbers or assume that a national service supports SMS,
  video, or text relay.
- Test on representative low-cost Android devices and real mobile networks,
  including no-data, weak-signal, low-battery, denied-location, and
  accessibility settings. Confirm behavior with local carriers and providers
  before promising call/SMS delivery.
- Evaluate task success and time to initiate contact, accidental activation,
  failed/ambiguous delivery states, comprehension by bystanders, and user
  confidence. Conduct supervised drills with Deaf participants and local
  partners; never use a real emergency as a test.
- Publish a small, explicit supported-area/service list and last-verified
  dates. If a route is unavailable, say so and provide the safe alternatives
  that are actually available on that device.

**Plan**

- Replace the generic phrase-list-only framing with the emergency card,
  trusted-contact flow, and verified local routes described above.
- Make offline/unsupported speech, location, camera, and network states
  visible; keep the emergency card usable without those services.
- Test phrase creation, limits, deletion, persistence, and accessibility, and
  let the user pre-configure emergency details before they are needed.
- Test call invite expiry, malformed/expired tokens, disconnect/rejoin,
  signaling loss, and TURN configuration in a controlled environment.
- Review privacy/data retention for emergency contacts, health details,
  location, transcripts, camera frames, call metadata, and user-created
  phrases. Store the minimum necessary, encrypt sensitive data, and provide
  user-visible edit/delete controls.
- Add clear consent and contact status indicators; do not log sensitive
  emergency content.
- Treat the clinic communication assistant as a separate, consented workflow
  with its own privacy, clinical-safety, accessibility, and acceptance gates;
  emergency phrases and video calls are not a substitute for that workflow.

**Acceptance evidence**

- Deaf Malawian users and local clinical/emergency partners have reviewed the
  wording, contact routes, and drills.
- Every published contact has a named verification source, last-verified
  date, responsible owner, and working test outcome.
- Supervised drills cover online/offline, weak signal, denied permissions,
  cancel/retry, accidental activation, and call/message delivery states.
- No screen suggests that a call/message was delivered unless the device,
  provider, or recipient confirms it; opening the dialer or composing an SMS
  is not delivery confirmation.
- The emergency path remains useful if sign recognition, generative AI,
  location, or data service is unavailable.

### 8. Reliability, accessibility, and maintainability

**Plan**

- Add a CI gate for focused tests, full tests, syntax/type checks where
  configured, and template/static asset checks.
- Make startup diagnostics concise and actionable; avoid marking the whole
  system ready because only one inference module loaded.
- Add structured, privacy-safe logs for model load state, request failures,
  latency, and version IDs. Never log raw camera images or sensitive phrases.
- Add configuration validation and document which environment variables are
  required for local, staging, and production use.
- Run accessibility review for keyboard navigation, focus handling, contrast,
  captions/text equivalents, and screen-reader names.
- Maintain a dependency update and vulnerability process with tests before
  upgrades.
- Add data/model backup and restore instructions; preserve raw capture and
  trained artifacts with checksums and provenance.

## Delivery phases and gates

### Phase 0 — Make the current state trustworthy

- Publish this roadmap and link current audit reports.
- Verify all model files and report readiness per module.
- Keep the interim static V2 model separate from production.
- Finish the planned additional capture sessions and regenerate quality
  reports.
- Run the applicable tests and store the exact command/results.

**Exit gate:** collection data is intact, all artifacts are identified, and
the app does not claim unsupported module readiness.

### Phase 1 — Validate core recognition

- Train a candidate only after data review.
- Evaluate on signer/session-disjoint data when possible.
- Run a documented browser-camera pilot with supported letters and gestures.
- Fix the highest-impact confusion and timing failures based on evidence.

**Exit gate:** written per-class results, failure analysis, measured latency,
and owner review. A high random-row score alone does not pass.

### Phase 2 — Review content and learner workflows

- Obtain qualified sign-language content review.
- Close priority meaning/movement/lesson gaps.
- Complete accessibility and persistence/usability checks.

**Exit gate:** reviewed content status is visible, unreviewed content is
clearly labeled, and learner-facing flows pass acceptance tests.

### Phase 3 — Prototype supported clinic communication

- Build Stage A using editable text, phrase board, speech-to-text,
  text-to-speech, visible turn-taking, consent, and interpreter fallback.
- Conduct supervised co-design and usability sessions with Deaf participants
  and clinicians before adding generative assistance.
- Add Stage B plain-language help only after factual-preservation tests and
  privacy review.

**Exit gate:** users can inspect, correct, decline, and end every AI-assisted
turn; fallback routes work; no unsupported sign interpretation is represented
as fact.

### Phase 4 — Co-design Malawi emergency support

- Map trusted-contact and verified clinic/emergency routes with Deaf
  Malawians and local service providers.
- Prototype the offline emergency card and contact flow before connecting any
  external service.
- Run supervised drills on representative devices and networks; publish
  supported routes and verification dates.

**Exit gate:** local partners verify the enabled routes, Deaf users complete
the drills safely, delivery states are truthful, and offline/failure paths are
understood.

### Phase 5 — Expand carefully

- Evaluate broader gesture/video vocabularies using authorized, independent
  data.
- Add reviewed sign input/output to the clinic workflow only after held-out
  signer evaluation and human language review.
- Improve calls and emergency communication with explicit failure handling
  and privacy review.
- Reassess deployment, licenses, monitoring, data retention, and support needs
  before any public or clinical/safety-related claim.

**Exit gate:** each newly enabled feature has its own evidence, rights review,
support plan, and rollback path.

## Project-review questions to be ready for

1. **What does the system recognize today?** Answer from loaded model labels
   and module readiness, not the course catalog or asset folder.
2. **How was it evaluated?** State the exact dataset, split unit, sample
   counts, metrics, and whether the test data were independent.
3. **What does it not do?** Explain unsupported labels, uncertainty, dialect
   variation, and that recognition output is not validated translation.
4. **How do you know avatar signs are correct?** Show reviewer records—or
   state clearly that the asset is only technically loadable and remains
   unverified.
5. **What happens when recognition fails?** Demonstrate no-hand/uncertain
   states, retry, text fallback, and module-specific readiness.
6. **How is user data handled?** Explain consent, pseudonymous collection IDs,
   storage, retention, deletion, and model/data access.
7. **How will the clinic assistant avoid inventing or changing meaning?**
   Explain source-visible transcripts, user confirmation, uncertainty,
   correction, and the interpreter/non-AI fallback.
8. **How is emergency support different from a phrase list?** Demonstrate the
   user-controlled emergency card, trusted/verified contact paths, offline
   fallback, and truthful call/message status.
9. **What is next and how will success be measured?** Use the phases and
   acceptance gates above; do not promise unvalidated accuracy or impact.

## Supporting documentation map

- [Project defense and run guide](./README.md)
- [Static-letter collection/training/evaluation guide](./docs/STATIC_A_TO_Z_V2.md)
- [Static recognition audit](./STATIC_RECOGNITION_AUDIT.md)
- [Repository/model/deployment audit](./REPOSITORY_FORENSIC_AUDIT.md)
- [Avatar and sign-content audit](./AVATAR_CONTENT_AUDIT.md)
- [Dataset forensic report](./DATASET_FORENSICS_REPORT.md)
- [Learning mode implementation](./learning_mode.py)

Re-run the relevant audits after material changes. Update the current snapshot,
acceptance evidence, and unresolved risks when new data or independent
evaluation results are available.
