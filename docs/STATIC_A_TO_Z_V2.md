# Static-letter collection and V2 model workflow

This workflow extends the existing webcam collector while keeping its original
`data/dataset.csv` output and `models/sign_model_light.pkl` model protected.
V2 data and artifacts are separate:

- `data/dataset_v2.csv`
- `models/sign_model_v2.pkl`
- `reports/static_a_to_z/dataset_v2_quality.json`
- `reports/static_a_to_z/dataset_v2_review.csv`
- `reports/static_a_to_z/dataset_v2_split.json`
- `reports/static_a_to_z/sign_model_v2_training.json`
- `reports/static_a_to_z/sign_model_v2_evaluation.json`

The existing 42 features are unchanged: 21 MediaPipe landmarks, wrist-centred
and normalized by wrist-to-middle-MCP distance, using flattened XY coordinates.
Signer and session metadata are not model features.

The static classifier covers 24 held handshapes: A-Z excluding J and Z. J and
Z require movement and are handled by the gesture-sequence model; do
not collect them as static poses or include them in static-letter training.
Existing V2 rows labeled J or Z remain in the CSV for traceability and quality
reports, but are excluded from static training and evaluation.

## Collect across controlled sessions

Run from the project root. For the initial target of 300 samples per static
letter and 500 for priority letters B, M, P, Q, S, U, and V, five collection
groups can each contribute 60 samples per normal letter and 100 per priority letter. These
may be five signers (enabling signer-disjoint evaluation) or five distinct
sessions from one or more signers (session-disjoint evaluation).

```powershell
python data_collector.py --v2 --batch A B C D E F G H I K L M N O P Q R S T U V W X Y --signer-id S01 --session-id S01-01 --samples 300 --priority-extra 200 --session-samples 60 --session-priority-extra 40
```

Repeat with a unique `--session-id` for each controlled session. Use a new
`--signer-id` when collecting from another signer. Keep each session balanced
by collecting all 24 static classes. V2 collection rejects J and Z so motion
signs cannot accidentally be recorded as held poses. Press **S once per accepted
sample**;
the collector rejects missing/ambiguous hands, incomplete/non-finite landmarks,
small or edge-cropped hands, invalid features, and exact duplicate vectors
within that collection session. The on-screen prompts encourage variation
without asking for unusable framing. If resuming a session, pass its original
session ID; the collector will count existing rows for that session.

Each accepted row records label, signer, session, UTC timestamp, source/device,
frame size, and the unchanged 42 feature values. Use stable, non-identifying
signer codes (for example `S01`); do not put names or other personal data into
the dataset.

## Quality control, training, and evaluation

```powershell
python static_model_v2.py quality
python static_model_v2.py train --seed 42
python static_model_v2.py evaluate --bootstrap-iterations 1000
```

Training first writes the quality and human-review reports. Without an override,
it stops if any rows are structurally invalid, any normal letter has fewer than
300 valid samples, any priority letter has fewer than 500, or there are fewer
than five independent split groups. With five or more signers,
the split is signer-disjoint; otherwise it is signer/session-disjoint. Groups
are seeded into approximately 60/20/20 train/validation/test partitions, and
every class must occur in every split. Exact feature vectors that cross split
boundaries stop training and are listed for human review. Augmentation is
applied only to training rows.

V2 uses the existing trainer's StandardScaler + 80-tree Random Forest
configuration and training augmentation, with seed 42 by default. Training
reports validation metrics but does not use test rows. The separate evaluation
command compares the protected baseline and V2 on the exact same untouched V2
test rows for the 24 static classes. Baseline predictions of J, Z, or another
non-static label count as `NON_STATIC_OUTPUT`, an incorrect static-letter result.
It reports accuracy, macro F1, precision, recall, per-class scores,
confusion matrix and confusion pairs, plus paired signer/session cluster-bootstrap
intervals when at least five independent groups are present in the test split.
It also checks whether any exact V2 test feature vectors occur in the configured
legacy baseline training CSV.
Neither command changes the production model path or activates V2.

With fewer than five independent groups in test, the report still gives the
observed same-test metrics but marks uncertainty as not estimable and will not
recommend V2. The evaluation report's candidate rule requires positive 95%
group-bootstrap intervals for both accuracy and macro-F1 differences and no
significantly negative priority-class accuracy difference, plus zero exact
feature-vector overlap with the legacy training CSV. This exact-match check
cannot establish the historical model's actual training rows or detect similar
poses/signer overlap because legacy provenance was not recorded. It is decision
support only; any production model change requires a separate explicit review.

### Explicit interim training below target

The official collection targets remain 300 samples for normal classes and 500
for B, M, P, Q, S, U, and V. The default training command still rejects data
below either target. To explicitly train an exploratory interim model:

```powershell
python static_model_v2.py train --allow-under-target --seed 42
python static_model_v2.py evaluate --bootstrap-iterations 1000
```

The override does not bypass structural quality checks: invalid rows stop
training, the quality and review artifacts are written first, and suspicious
rows remain flagged for human review rather than being removed or relabeled.
When signer/session-disjoint splits cannot place every static class in every
partition, the override uses a seeded stratified 82/18 raw-row holdout before
augmentation, matching the legacy trainer's pre-augmentation holdout principle.
The test rows are not augmented, used for fitting, or used for model selection.
Exact duplicate feature vectors are kept together in one split; any remaining
cross-split exact duplicate still stops training. Collection sessions may occur
in both partitions, so this fallback is not signer/session-disjoint
evidence and may retain within-session similarity risk.

Interim training uses the documented row holdout when there are too few groups
to put every static class in each split; it has no separate validation
partition. Cluster-bootstrap confidence intervals are not estimable from fewer
than five independent test groups. Treat these results as exploratory, not as
a generalization claim or production-readiness signal. Interim metadata is
saved inside `models/sign_model_v2.pkl` and in
`reports/static_a_to_z/sign_model_v2_training.json`; the 300/500 targets remain
recorded in both.

## Baseline and current evidence

The historical baseline supplied for this work is approximately 80.65%
accuracy / 0.762 macro F1. The checked-in `models/sign_model_light_metrics.json`
instead records about 99.93% accuracy from a random stratified holdout. These
figures conflict and must not be presented as the same evaluation. The
historical legacy dataset/model may include motion labels and has no signer or
session provenance. V2 static comparisons use only the 24 held-letter classes
and a shared provenance-aware V2 test set; the historical figure is shown only
as a user-provided reference.

No V2 rows, model metrics, or live-test results are claimed until real samples
have been collected and evaluated. Automated confusion reports include sample
row IDs and class-specific training/group counts, but do not guess whether
errors are caused by sign similarity, inconsistent labels, signer variation,
the feature representation, or the classifier; inspect the flagged samples
before drawing that conclusion. A future informal webcam check should be
reported separately from offline held-out accuracy; it is not a substitute for
the independent test set.
