"""Shared loading and leakage-resistant splitting for gesture sequences."""

import csv
from itertools import chain
from pathlib import Path

import numpy as np

from gesture_feature_extractor import GESTURE_WINDOW, STATIC_FEATURES


GESTURE_V2_METADATA_FIELDS = [
    "sample_id",
    "label",
    "signer_id",
    "session_id",
    "captured_at_utc",
    "device",
    "frame_width",
    "frame_height",
    "capture_fps",
    "capture_duration_ms",
]
GESTURE_V2_FEATURE_FIELDS = [
    f"feature_{index:03d}"
    for index in range(GESTURE_WINDOW * STATIC_FEATURES)
]
GESTURE_V2_FIELDS = GESTURE_V2_METADATA_FIELDS + GESTURE_V2_FEATURE_FIELDS


def load_gesture_sequences(path):
    """Load legacy rows or the versioned CSV into sequences and provenance."""
    dataset_path = Path(path)
    expected_features = GESTURE_WINDOW * STATIC_FEATURES
    sequences = []
    labels = []
    signer_ids = []
    session_ids = []

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Gesture dataset not found: {dataset_path}. "
            "Collect samples with gesture_collector.py first."
        )

    with dataset_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"Gesture dataset is empty: {dataset_path}")

        versioned = first_row == GESTURE_V2_FIELDS
        if (
            not versioned
            and first_row
            and first_row[0].strip().lower() == "sample_id"
        ):
            raise ValueError(f"Gesture dataset has an unexpected header: {dataset_path}")

        rows = enumerate(reader, start=2) if versioned else enumerate(
            chain([first_row], reader),
            start=1,
        )
        feature_start = len(GESTURE_V2_METADATA_FIELDS) if versioned else 1

        for line_number, row in rows:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != feature_start + expected_features:
                raise ValueError(
                    f"Line {line_number} in {dataset_path} has {len(row)} columns; "
                    f"expected {feature_start + expected_features}."
                )

            label_index = 1 if versioned else 0
            label = row[label_index].strip()
            if not label:
                raise ValueError(f"Line {line_number} in {dataset_path} has an empty label.")
            if versioned:
                required_metadata = {
                    "sample_id": row[0].strip(),
                    "signer_id": row[2].strip(),
                    "session_id": row[3].strip(),
                    "captured_at_utc": row[4].strip(),
                    "device": row[5].strip(),
                }
                missing_metadata = [
                    name for name, value in required_metadata.items() if not value
                ]
                if missing_metadata:
                    raise ValueError(
                        f"Line {line_number} in {dataset_path} has empty provenance "
                        f"fields: {missing_metadata}."
                    )
                try:
                    frame_width = int(row[6])
                    frame_height = int(row[7])
                    capture_fps = float(row[8])
                    capture_duration_ms = float(row[9])
                except ValueError as error:
                    raise ValueError(
                        f"Line {line_number} in {dataset_path} has invalid capture metadata."
                    ) from error
                if (
                    frame_width <= 0
                    or frame_height <= 0
                    or not np.isfinite(capture_fps)
                    or capture_fps <= 0
                    or not np.isfinite(capture_duration_ms)
                    or capture_duration_ms < 0
                ):
                    raise ValueError(
                        f"Line {line_number} in {dataset_path} has invalid capture metadata."
                    )
            try:
                features = np.asarray(
                    row[feature_start:],
                    dtype=np.float32,
                ).reshape(GESTURE_WINDOW, STATIC_FEATURES)
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number} in {dataset_path} contains a non-numeric feature."
                ) from error
            if not np.isfinite(features).all():
                raise ValueError(
                    f"Line {line_number} in {dataset_path} contains a non-finite feature."
                )

            sequences.append(features)
            labels.append(label)
            if versioned:
                signer_ids.append(row[2].strip())
                session_ids.append(row[3].strip())

    if not sequences:
        raise ValueError(f"Gesture dataset contains no valid samples: {dataset_path}")
    return (
        np.asarray(sequences, dtype=np.float32),
        labels,
        signer_ids if versioned else None,
        session_ids if versioned else None,
    )


def grouped_train_validation_split(
    sequences,
    labels,
    signer_ids=None,
    session_ids=None,
    test_size=0.2,
    random_state=42,
):
    """Split related samples together, preferring signer/session groups."""
    from sklearn.model_selection import GroupShuffleSplit

    features = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(labels)
    if features.ndim != 3 or features.shape[1:] != (GESTURE_WINDOW, STATIC_FEATURES):
        raise ValueError(
            f"Expected sequences shaped (samples, {GESTURE_WINDOW}, {STATIC_FEATURES})."
        )
    if len(features) != len(labels) or not len(labels):
        raise ValueError("Gesture sequences and labels must have the same non-zero length.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not np.isfinite(features).all():
        raise ValueError("Gesture sequences contain non-finite features.")

    signer_ids = _validated_metadata(signer_ids, len(labels), "signer_ids")
    session_ids = _validated_metadata(session_ids, len(labels), "session_ids")
    signer_groups = _complete_nonempty_groups(signer_ids)
    session_groups = _complete_nonempty_groups(session_ids)

    if signer_groups is not None and len(set(signer_groups)) > 1:
        source_groups = signer_groups
        split_basis = "signer"
    elif session_groups is not None and len(set(session_groups)) > 1:
        source_groups = session_groups
        split_basis = "session"
    else:
        source_groups = None
        split_basis = "exact-duplicate-only"

    parents = list(range(len(labels)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    if source_groups is not None:
        first_by_group = {}
        for index, group in enumerate(source_groups):
            if group in first_by_group:
                union(index, first_by_group[group])
            else:
                first_by_group[group] = index

    first_by_sequence = {}
    duplicate_extra_rows = 0
    for index, sequence in enumerate(features):
        key = np.ascontiguousarray(sequence, dtype=np.float32).tobytes()
        if key in first_by_sequence:
            previous_index = first_by_sequence[key]
            if labels[index] != labels[previous_index]:
                raise ValueError(
                    "Identical gesture sequences have conflicting labels; "
                    "review the source data before training."
                )
            union(index, previous_index)
            duplicate_extra_rows += 1
        else:
            first_by_sequence[key] = index

    groups = np.asarray([find(index) for index in range(len(labels))])
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two independent gesture groups are required for validation.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, validation_indices = next(
        splitter.split(features, labels, groups=groups)
    )
    all_labels = set(labels.tolist())
    missing_train = sorted(all_labels - set(labels[train_indices].tolist()))
    missing_validation = sorted(all_labels - set(labels[validation_indices].tolist()))
    if missing_train or missing_validation:
        raise ValueError(
            "Grouped gesture split does not represent every class in both partitions "
            f"(training missing: {missing_train}; validation missing: {missing_validation}). "
            "Collect more independent signers or sessions."
        )

    return train_indices, validation_indices, {
        "split_basis": split_basis,
        "duplicate_extra_rows": duplicate_extra_rows,
        "group_count": int(len(np.unique(groups))),
    }


def _validated_metadata(values, sample_count, name):
    if values is None:
        return None
    if len(values) != sample_count:
        raise ValueError(f"{name} must contain one value per gesture sequence.")
    return [str(value).strip() if value is not None else "" for value in values]


def _complete_nonempty_groups(values):
    if values is None or not values or any(not value for value in values):
        return None
    return values
