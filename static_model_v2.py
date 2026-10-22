"""Quality-control, train, and evaluate the provenance-aware static A-Z V2 model."""

import argparse
import csv
import hashlib
import json
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from data_collector import V2_DATASET_FIELDS, V2_FEATURE_FIELDS, V2_PRIORITY_LETTERS
from feature_extractor import FEATURE_VECTOR_SIZE
from model_trainer import AUG_COPIES, AUG_NOISE_STD, MIRROR_EXCLUDE


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
RANDOM_SEED = 42
MIN_SAMPLES_PER_CLASS = 300
MIN_PRIORITY_SAMPLES_PER_CLASS = 500
DEFAULT_DATASET = Path(config.DATA_DIR) / "dataset_v2.csv"
DEFAULT_MODEL = Path(config.MODELS_DIR) / "sign_model_v2.pkl"
DEFAULT_REPORT_DIR = Path(config.BASE_DIR) / "reports" / "static_a_to_z"
DEFAULT_SPLIT = DEFAULT_REPORT_DIR / "dataset_v2_split.json"
DEFAULT_QUALITY = DEFAULT_REPORT_DIR / "dataset_v2_quality.json"
DEFAULT_REVIEW = DEFAULT_REPORT_DIR / "dataset_v2_review.csv"
DEFAULT_TRAINING = DEFAULT_REPORT_DIR / "sign_model_v2_training.json"
DEFAULT_EVALUATION = DEFAULT_REPORT_DIR / "sign_model_v2_evaluation.json"


def _json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v2_dataset(path: Path):
    """Read V2 rows and retain invalid row details for the QC report."""
    if not path.is_file():
        raise FileNotFoundError(f"V2 dataset not found: {path}. Collect into dataset_v2.csv first.")

    records = []
    invalid = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != V2_DATASET_FIELDS:
            raise ValueError(
                f"Unexpected V2 CSV header in {path}; expected the collector's "
                "metadata fields followed by the unchanged 42-feature representation."
            )
        for row_id, row in enumerate(reader, start=2):
            reasons = []
            if None in row:
                reasons.append("row_has_extra_columns")
            label = (row.get("label") or "").strip().upper()
            if label not in LABELS:
                reasons.append("invalid_label")
            for field in ("signer_id", "session_id", "timestamp_utc"):
                if not (row.get(field) or "").strip():
                    reasons.append(f"missing_{field}")
            timestamp = (row.get("timestamp_utc") or "").strip()
            if timestamp:
                try:
                    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    reasons.append("invalid_timestamp_utc")
            try:
                frame_width = int(row["frame_width"])
                frame_height = int(row["frame_height"])
                if frame_width <= 0 or frame_height <= 0:
                    reasons.append("invalid_frame_dimensions")
            except (TypeError, ValueError, KeyError):
                reasons.append("invalid_frame_dimensions")
            try:
                features = [float(row[field]) for field in V2_FEATURE_FIELDS]
            except (TypeError, ValueError, KeyError):
                features = []
                reasons.append("unparseable_features")
            if len(features) != FEATURE_VECTOR_SIZE:
                if "unparseable_features" not in reasons:
                    reasons.append("wrong_feature_count")
            elif not np.isfinite(np.asarray(features, dtype=np.float64)).all():
                reasons.append("non_finite_features")

            if reasons:
                invalid.append({"row_id": row_id, "label": label, "reasons": reasons})
                continue
            records.append({
                "row_id": row_id,
                "label": label,
                "signer_id": row["signer_id"].strip(),
                "session_id": row["session_id"].strip(),
                "timestamp_utc": row["timestamp_utc"].strip(),
                "source": (row.get("source") or "").strip(),
                "device": (row.get("device") or "").strip(),
                "features": features,
            })
    return records, invalid


def build_quality_report(records, invalid, dataset_path: Path):
    counts = Counter(record["label"] for record in records)
    per_class = {label: counts.get(label, 0) for label in LABELS}
    values = list(per_class.values())
    vector_rows = defaultdict(list)
    for record in records:
        vector_rows[tuple(record["features"])].append(record)

    suspicious = []
    duplicate_excess = 0
    for entries in vector_rows.values():
        if len(entries) > 1:
            duplicate_excess += len(entries) - 1
            labels = sorted({entry["label"] for entry in entries})
            reasons = ["exact_duplicate_feature_vector"]
            if len(labels) > 1:
                reasons.append("same_feature_vector_has_different_labels")
            for entry in entries:
                suspicious.append({
                    "row_id": entry["row_id"],
                    "label": entry["label"],
                    "signer_id": entry["signer_id"],
                    "session_id": entry["session_id"],
                    "duplicate_row_ids": [other["row_id"] for other in entries
                                          if other["row_id"] != entry["row_id"]],
                    "reasons": reasons,
                })

    signer_counts = Counter(record["signer_id"] for record in records)
    session_counts = Counter(
        f"{record['signer_id']}::{record['session_id']}" for record in records
    )
    report = {
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _dataset_hash(dataset_path),
        "valid_total_samples": len(records),
        "invalid_sample_count": len(invalid),
        "invalid_samples": invalid,
        "samples_per_class": per_class,
        "class_count_summary": {
            "minimum": min(values) if values else 0,
            "maximum": max(values) if values else 0,
            "average": (sum(values) / len(values)) if values else 0,
            "imbalance_ratio_max_to_min": (
                max(values) / min(values) if values and min(values) > 0 else None
            ),
        },
        "duplicate_feature_vectors": {
            "unique_vectors_repeated": sum(1 for rows in vector_rows.values() if len(rows) > 1),
            "excess_rows": duplicate_excess,
        },
        "suspicious_samples": suspicious,
        "signer_distribution": dict(sorted(signer_counts.items())),
        "collection_session_distribution": dict(sorted(session_counts.items())),
        "label_review_note": (
            "The report flags structural and exact-duplicate issues only. It does not "
            "infer or change labels; sign correctness requires human review."
        ),
    }
    return report


def write_quality_artifacts(dataset_path: Path, quality_path: Path, review_path: Path):
    records, invalid = load_v2_dataset(dataset_path)
    report = build_quality_report(records, invalid, dataset_path)
    _json_write(quality_path, report)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["row_id", "label", "signer_id", "session_id", "reason", "related_row_ids"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in invalid:
            writer.writerow({
                "row_id": entry["row_id"],
                "label": entry["label"],
                "reason": ";".join(entry["reasons"]),
                "related_row_ids": "",
            })
        for entry in report["suspicious_samples"]:
            writer.writerow({
                "row_id": entry["row_id"],
                "label": entry["label"],
                "signer_id": entry["signer_id"],
                "session_id": entry["session_id"],
                "reason": ";".join(entry["reasons"]),
                "related_row_ids": ";".join(map(str, entry["duplicate_row_ids"])),
            })
    print(f"Quality report: {quality_path}")
    print(f"Human-review list: {review_path}")
    return records, invalid, report


def _row_stratified_split(records, seed: int):
    from sklearn.model_selection import train_test_split

    counts = Counter(record["label"] for record in records)
    missing = sorted(set(LABELS) - set(counts))
    if missing:
        raise ValueError(
            "A stratified interim holdout requires at least one valid sample for "
            f"every A-Z class; missing: {missing}"
        )
    too_small = {label: count for label, count in counts.items() if count < 2}
    if too_small:
        raise ValueError(
            "A stratified interim holdout requires at least two valid samples per "
            f"class; below minimum: {too_small}"
        )

    row_ids = [record["row_id"] for record in records]
    labels = [record["label"] for record in records]
    train_ids, test_ids = train_test_split(
        row_ids, test_size=0.18, random_state=seed, stratify=labels,
    )
    assignments = {row_id: "train" for row_id in train_ids}
    assignments.update({row_id: "test" for row_id in test_ids})
    rows_by_vector = defaultdict(list)
    for record in records:
        rows_by_vector[tuple(record["features"])].append(record["row_id"])
    for duplicate_row_ids in rows_by_vector.values():
        duplicate_splits = Counter(assignments[row_id] for row_id in duplicate_row_ids)
        if len(duplicate_splits) > 1:
            destination = (
                "train" if duplicate_splits["train"] >= duplicate_splits["test"]
                else "test"
            )
            for row_id in duplicate_row_ids:
                assignments[row_id] = destination

    missing_by_split = {
        split: sorted(
            set(LABELS) - {
                record["label"] for record in records
                if assignments[record["row_id"]] == split
            }
        )
        for split in ("train", "test")
    }
    if any(missing_by_split.values()):
        raise ValueError(
            "The stratified interim holdout cannot retain every A-Z class in both "
            f"train and test after keeping exact duplicates together: {missing_by_split}"
        )

    group_for_row = {
        record["row_id"]: f"{record['signer_id']}::{record['session_id']}"
        for record in records
    }
    return (
        assignments,
        "trainer_faithful_stratified_row_holdout_pre_augmentation",
        group_for_row,
        _cross_split_duplicates(records, assignments),
    )


def _cross_split_duplicates(records, assignments):
    vector_splits = defaultdict(set)
    vector_rows = defaultdict(list)
    for record in records:
        vector = tuple(record["features"])
        split = assignments[record["row_id"]]
        vector_splits[vector].add(split)
        vector_rows[vector].append(record["row_id"])
    return [
        {"row_ids": row_ids, "splits": sorted(vector_splits[vector])}
        for vector, row_ids in vector_rows.items()
        if len(vector_splits[vector]) > 1
    ]


def _split_records(records, seed: int, allow_row_stratified: bool = False):
    signers = sorted({record["signer_id"] for record in records})
    use_signer_groups = len(signers) >= 5
    strategy = "signer_disjoint" if use_signer_groups else "collection_session_disjoint"
    group_for_row = {}
    for record in records:
        group = (record["signer_id"] if use_signer_groups else
                 f"{record['signer_id']}::{record['session_id']}")
        group_for_row[record["row_id"]] = group
    groups = sorted(set(group_for_row.values()))
    if len(groups) < 5:
        if not allow_row_stratified:
            raise ValueError(
                f"At least 5 independent {('signers' if use_signer_groups else 'signer/session groups')} "
                f"are required for a 60/20/20 group holdout; found {len(groups)}."
            )
        return _row_stratified_split(records, seed)

    rng = np.random.default_rng(seed)
    shuffled = [groups[index] for index in rng.permutation(len(groups))]
    test_count = max(1, int(round(len(groups) * 0.20)))
    validation_count = max(1, int(round(len(groups) * 0.20)))
    train_count = len(groups) - test_count - validation_count
    if train_count < 3:
        raise ValueError("The split would not leave a majority of independent groups for training.")
    assignments = {}
    test_groups = set(shuffled[:test_count])
    validation_groups = set(shuffled[test_count:test_count + validation_count])
    for row_id, group in group_for_row.items():
        assignments[row_id] = (
            "test" if group in test_groups else
            "validation" if group in validation_groups else
            "train"
        )

    split_classes = {
        split: {record["label"] for record in records
                if assignments[record["row_id"]] == split}
        for split in ("train", "validation", "test")
    }
    missing = {split: sorted(set(LABELS) - classes)
               for split, classes in split_classes.items()}
    if any(missing.values()):
        if allow_row_stratified:
            return _row_stratified_split(records, seed)
        raise ValueError(
            "Every split must contain every A-Z class. Collect all classes in more "
            f"independent signers/sessions; missing by split: {missing}"
        )

    cross_split_duplicates = _cross_split_duplicates(records, assignments)
    return assignments, strategy, group_for_row, cross_split_duplicates


def _split_summary(records, assignments, group_for_row):
    result = {}
    for split in ("train", "validation", "test"):
        selected = [record for record in records
                    if assignments[record["row_id"]] == split]
        result[split] = {
            "samples": len(selected),
            "class_counts": {label: sum(record["label"] == label for record in selected)
                             for label in LABELS},
            "independent_groups": len({group_for_row[record["row_id"]]
                                       for record in selected}),
        }
    return result


def _metric_summary(y_true, y_pred, class_names):
    from sklearn.metrics import classification_report, confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    top_pairs = []
    for left in range(len(class_names)):
        for right in range(left + 1, len(class_names)):
            count = int(matrix[left, right] + matrix[right, left])
            if count:
                top_pairs.append({
                    "classes": [class_names[left], class_names[right]],
                    "total_misclassifications": count,
                    "directions": {
                        f"{class_names[left]}_as_{class_names[right]}": int(matrix[left, right]),
                        f"{class_names[right]}_as_{class_names[left]}": int(matrix[right, left]),
                    },
                })
    top_pairs.sort(key=lambda pair: (-pair["total_misclassifications"], pair["classes"]))
    per_class = {}
    for index, label in enumerate(class_names):
        support = int(matrix[index].sum())
        per_class[label] = {
            "support": support,
            "accuracy": float(matrix[index, index] / support) if support else 0.0,
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
        }
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "classification_report": report,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": class_names,
        "top_confused_class_pairs": top_pairs[:15],
    }


def train_v2(dataset_path: Path, model_path: Path, split_path: Path,
             quality_path: Path, review_path: Path, training_path: Path, seed: int,
             allow_under_target: bool = False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from model_trainer import augment_dataset, augment_mirror_views

    records, invalid, quality = write_quality_artifacts(
        dataset_path, quality_path, review_path
    )
    if invalid:
        raise ValueError(
            f"Training stopped: {len(invalid)} invalid rows are listed in "
            f"{review_path}. Review and correct the source data first."
        )
    counts = quality["samples_per_class"]
    low_counts = {
        label: {"observed": count, "required": (
            MIN_PRIORITY_SAMPLES_PER_CLASS if label in V2_PRIORITY_LETTERS
            else MIN_SAMPLES_PER_CLASS
        )}
        for label, count in counts.items()
        if count < (
            MIN_PRIORITY_SAMPLES_PER_CLASS if label in V2_PRIORITY_LETTERS
            else MIN_SAMPLES_PER_CLASS
        )
    }
    if low_counts:
        if not allow_under_target:
            raise ValueError(
                "Training requires at least 300 valid samples per letter and 500 for "
                f"priority letters {sorted(V2_PRIORITY_LETTERS)}; below target: {low_counts}"
            )

    assignments, strategy, group_for_row, cross_duplicates = _split_records(
        records, seed, allow_row_stratified=allow_under_target
    )
    split_summary = _split_summary(records, assignments, group_for_row)
    group_splits = defaultdict(set)
    for row_id, split in assignments.items():
        group_splits[group_for_row[row_id]].add(split)
    manifest = {
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": quality["dataset_sha256"],
        "random_seed": seed,
        "split_strategy": strategy,
        "group_split_rule": (
            "Signer-disjoint when at least five unique signers exist; otherwise "
            "signer/session-disjoint. Groups are shuffled with the recorded seed, "
            "then assigned approximately 60/20/20 to train/validation/test."
            if strategy != "trainer_faithful_stratified_row_holdout_pre_augmentation"
            else (
                "The available signer/session groups cannot populate every A-Z "
                "class in every split. A seeded, class-stratified 82/18 row holdout "
                "was made on raw samples before augmentation; test rows are never "
                "augmented or used for fitting. Exact duplicate feature vectors are "
                "kept together in one split, preferring train on a tie. Collection "
                "sessions may cross splits."
            )
        ),
        "collection_groups_disjoint": (
            strategy != "trainer_faithful_stratified_row_holdout_pre_augmentation"
        ),
        "validation_available": any(
            split == "validation" for split in assignments.values()
        ),
        "split_counts": split_summary,
        "row_assignments": {str(row_id): split for row_id, split in assignments.items()},
        "group_assignments": {
            group: next(iter(splits)) if len(splits) == 1 else "multiple_splits"
            for group, splits in sorted(group_splits.items())
        },
        "cross_split_exact_duplicate_vectors": cross_duplicates,
    }
    _json_write(split_path, manifest)
    if cross_duplicates:
        raise ValueError(
            "Training stopped: exact feature vectors cross split boundaries. Review "
            f"the flagged rows in {review_path} and the split manifest {split_path}; "
            "do not delete or relabel samples automatically."
        )

    class_names = list(LABELS)
    label_to_id = {label: index for index, label in enumerate(class_names)}
    x_by_row = {record["row_id"]: record["features"] for record in records}
    y_by_row = {record["row_id"]: label_to_id[record["label"]] for record in records}
    split_rows = {
        split: [int(row_id) for row_id, name in assignments.items() if name == split]
        for split in ("train", "validation", "test")
    }
    x_train_raw = np.asarray([x_by_row[row_id] for row_id in split_rows["train"]], dtype=np.float32)
    y_train_raw = np.asarray([y_by_row[row_id] for row_id in split_rows["train"]], dtype=np.int32)
    x_validation = np.asarray([x_by_row[row_id] for row_id in split_rows["validation"]], dtype=np.float32)
    y_validation = np.asarray([y_by_row[row_id] for row_id in split_rows["validation"]], dtype=np.int32)

    x_train, y_train = augment_dataset(
        x_train_raw, y_train_raw, n_copies=AUG_COPIES, noise_std=AUG_NOISE_STD,
        rng=np.random.default_rng(seed),
    )
    x_train, y_train = augment_mirror_views(
        x_train, y_train, class_names, exclude=MIRROR_EXCLUDE
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )),
    ])
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    validation_metrics = None
    if len(y_validation):
        validation_predictions = pipeline.predict(x_validation)
        validation_metrics = _metric_summary(y_validation, validation_predictions, class_names)

    training_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    targets = {
        label: (MIN_PRIORITY_SAMPLES_PER_CLASS if label in V2_PRIORITY_LETTERS
                else MIN_SAMPLES_PER_CLASS)
        for label in LABELS
    }
    target_achievement = {
        label: (counts[label] / targets[label] * 100.0)
        for label in LABELS
    }
    target_summary = {
        "normal_class_target": MIN_SAMPLES_PER_CLASS,
        "priority_class_target": MIN_PRIORITY_SAMPLES_PER_CLASS,
        "priority_classes": sorted(V2_PRIORITY_LETTERS),
        "total_valid_samples": len(records),
        "samples_per_class": counts,
        "minimum_class_count": min(counts.values(), default=0),
        "classes_below_target": {
            label: {
                "observed": counts[label],
                "required": targets[label],
                "target_achieved_percent": target_achievement[label],
            }
            for label in LABELS if counts[label] < targets[label]
        },
        "target_achievement_percent_per_class": target_achievement,
        "overall_target_achieved_percent": (
            len(records) / sum(targets.values()) * 100.0
        ),
    }
    training_status = "interim_under_target" if allow_under_target else "target_met"
    model_metadata = {
        "status": training_status,
        "target_override": "--allow-under-target" if allow_under_target else None,
        "training_timestamp": training_timestamp,
        "dataset_sha256": quality["dataset_sha256"],
        **target_summary,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump({
            "pipeline": pipeline,
            "classes": class_names,
            "metadata": model_metadata,
        }, stream)
    training_report = {
        "model": str(model_path.resolve()),
        "status": training_status,
        "target_override": "--allow-under-target" if allow_under_target else None,
        "training_timestamp": training_timestamp,
        "normal_class_target": MIN_SAMPLES_PER_CLASS,
        "priority_class_target": MIN_PRIORITY_SAMPLES_PER_CLASS,
        "total_valid_samples": len(records),
        "samples_per_class": counts,
        "minimum_class_count": min(counts.values(), default=0),
        "classes_below_target": target_summary["classes_below_target"],
        "target_achievement_percent_per_class": target_achievement,
        "overall_target_achieved_percent": target_summary["overall_target_achieved_percent"],
        "dataset_sha256": quality["dataset_sha256"],
        "split_manifest": str(split_path.resolve()),
        "split_strategy": strategy,
        "random_seed": seed,
        "feature_count": FEATURE_VECTOR_SIZE,
        "feature_representation": "existing wrist-normalized 21-point XY, flattened to 42 values",
        "hyperparameters": {
            "classifier": "RandomForestClassifier",
            "n_estimators": 80,
            "max_depth": 18,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": seed,
            "scaler": "StandardScaler",
        },
        "augmentation": {
            "n_copies": AUG_COPIES,
            "noise_std": AUG_NOISE_STD,
            "mirror_excluded_labels": sorted(MIRROR_EXCLUDE),
            "applied_to": "training split only",
        },
        "sample_counts": {
            "raw_training": len(x_train_raw),
            "augmented_training": len(x_train),
            "validation": len(x_validation),
            "test_reserved": len(split_rows["test"]),
        },
        "training_class_distribution": split_summary["train"]["class_counts"],
        "validation_class_distribution": split_summary["validation"]["class_counts"],
        "test_class_distribution": split_summary["test"]["class_counts"],
        "training_time_seconds": training_seconds,
        "validation_metrics": validation_metrics,
        "validation_note": (
            None if validation_metrics is not None else
            "No separate validation split was available. The held-out test set "
            "remains untouched and is used only by the evaluate command."
        ),
    }
    _json_write(training_path, training_report)
    print(f"V2 model saved: {model_path}")
    print(f"Training report: {training_path}")
    print(f"TRAINING STATUS: {'INTERIM' if allow_under_target else 'TARGET MET'}")
    if allow_under_target:
        print("TARGET OVERRIDE: --allow-under-target")
    print(f"NORMAL CLASS TARGET: {MIN_SAMPLES_PER_CLASS}")
    print(f"PRIORITY CLASS TARGET: {MIN_PRIORITY_SAMPLES_PER_CLASS}")
    print(f"VALID SAMPLES: {len(records)}; minimum class count: {model_metadata['minimum_class_count']}")
    if strategy == "trainer_faithful_stratified_row_holdout_pre_augmentation":
        print(
            "WARNING: signer/session-disjoint A-Z evaluation was not possible; "
            "using a seeded pre-augmentation stratified row holdout."
        )
    print("Test rows remain untouched; run the evaluate command for final comparison.")
    return training_report


def _load_model_with_metadata(path: Path):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError(f"Model has an unsupported payload format: {path}")
    classes = payload.get("classes")
    if not isinstance(classes, list):
        classes_path = path.with_name(path.stem + "_classes.json")
        if not classes_path.is_file():
            raise ValueError(f"No class-label mapping found for model: {path}")
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Model metadata has an unsupported format: {path}")
    return payload["pipeline"], list(classes), metadata


def _load_model(path: Path):
    pipeline, classes, _ = _load_model_with_metadata(path)
    return pipeline, classes


def _predict_as_labels(pipeline, model_classes, features):
    predictions = np.asarray(pipeline.predict(features))
    if np.issubdtype(predictions.dtype, np.integer):
        return [model_classes[int(value)] for value in predictions]
    return [str(value) for value in predictions]


def _paired_bootstrap(y_true, baseline, v2, group_ids, class_names,
                      iterations=1000, seed=2026):
    group_names = sorted(set(group_ids))
    if len(group_names) < 5:
        return {
            "independent_test_groups": len(group_names),
            "confidence_intervals_estimable": False,
            "reason": (
                "Fewer than five independent signer/session groups are in the test "
                "split; sample-level resampling would overstate independent evidence."
            ),
            "accuracy_delta_v2_minus_baseline": None,
            "macro_f1_delta_v2_minus_baseline": None,
            "priority_class_accuracy_deltas": None,
            "method": "Group bootstrap not run because the test split has too few groups.",
        }
    rng = np.random.default_rng(seed)
    by_group = {
        group: np.flatnonzero(np.asarray(group_ids) == group)
        for group in group_names
    }
    accuracy_deltas = []
    f1_deltas = []
    priority_deltas = {label: [] for label in V2_PRIORITY_LETTERS}
    for _ in range(iterations):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        chosen = np.concatenate([by_group[group] for group in sampled_groups])
        true = y_true[chosen]
        base_matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        v2_matrix = np.zeros_like(base_matrix)
        np.add.at(base_matrix, (true, baseline[chosen]), 1)
        np.add.at(v2_matrix, (true, v2[chosen]), 1)

        def summary(matrix):
            tp = np.diag(matrix).astype(float)
            support = matrix.sum(axis=1).astype(float)
            predicted = matrix.sum(axis=0).astype(float)
            f1 = np.divide(2 * tp, support + predicted,
                           out=np.zeros_like(tp), where=(support + predicted) > 0)
            return float(tp.sum() / matrix.sum()), float(f1.mean()), np.divide(
                tp, support, out=np.zeros_like(tp), where=support > 0
            )

        base_accuracy, base_f1, base_per_class = summary(base_matrix)
        v2_accuracy, v2_f1, v2_per_class = summary(v2_matrix)
        accuracy_deltas.append(v2_accuracy - base_accuracy)
        f1_deltas.append(v2_f1 - base_f1)
        for label in V2_PRIORITY_LETTERS:
            index = class_names.index(label)
            priority_deltas[label].append(v2_per_class[index] - base_per_class[index])

    def interval(values):
        low, high = np.percentile(values, [2.5, 97.5])
        return {"lower_95_percent": float(low), "upper_95_percent": float(high)}

    return {
        "independent_test_groups": len(group_names),
        "confidence_intervals_estimable": True,
        "accuracy_delta_v2_minus_baseline": interval(accuracy_deltas),
        "macro_f1_delta_v2_minus_baseline": interval(f1_deltas),
        "priority_class_accuracy_deltas": {
            label: interval(priority_deltas[label]) for label in sorted(priority_deltas)
        },
        "method": (
            f"{iterations} paired cluster-bootstrap replicates over the shared test "
            "signer/session groups; the same resampled groups are used for both models."
        ),
    }


def _targeted_error_analysis(records, test_records, assignments, manifest, class_names,
                             baseline_predictions, v2_predictions, v2_top_pairs):
    pair_set = [("P", "Q"), ("Z", "B"), ("M", "S"), ("U", "V")]
    for pair in v2_top_pairs:
        labels = tuple(pair["classes"])
        if labels not in pair_set and labels[::-1] not in pair_set:
            pair_set.append(labels)

    index_to_label = {index: label for index, label in enumerate(class_names)}
    group_by_row = {}
    for record in records:
        if manifest["split_strategy"] == "signer_disjoint":
            group_by_row[record["row_id"]] = record["signer_id"]
        else:
            group_by_row[record["row_id"]] = (
                f"{record['signer_id']}::{record['session_id']}"
            )

    analysis = []
    for left, right in pair_set:
        pair_test_rows = []
        train_counts = {left: 0, right: 0}
        train_groups = {left: set(), right: set()}
        for record in records:
            row_id = record["row_id"]
            split = assignments.get(row_id)
            if split == "train" and record["label"] in train_counts:
                train_counts[record["label"]] += 1
                train_groups[record["label"]].add(group_by_row[row_id])

        for index, record in enumerate(test_records):
            true_label = record["label"]
            base_label = index_to_label[int(baseline_predictions[index])]
            v2_label = index_to_label[int(v2_predictions[index])]
            if (true_label in {left, right}
                    and (base_label in {left, right} or v2_label in {left, right})
                    and (base_label != true_label or v2_label != true_label)):
                pair_test_rows.append({
                    "row_id": record["row_id"],
                    "true_label": true_label,
                    "baseline_prediction": base_label,
                    "v2_prediction": v2_label,
                })
        analysis.append({
            "pair": [left, right],
            "training_samples": train_counts,
            "training_independent_groups": {
                label: len(groups) for label, groups in train_groups.items()
            },
            "test_pair_error_rows": pair_test_rows,
            "causal_assessment": (
                "Not determined by aggregate confusion metrics. Review the listed "
                "held-out samples and collection provenance before attributing errors "
                "to sample quantity, hand-shape similarity, signer variation, labels, "
                "features, or classifier limitations."
            ),
        })
    return analysis


def _legacy_dataset_overlap(test_records, legacy_dataset_path: Path):
    from model_trainer import load_dataset

    if not legacy_dataset_path.is_file():
        return {
            "checked": False,
            "dataset": str(legacy_dataset_path.resolve()),
            "exact_feature_overlap_test_rows": None,
            "overlap_rows": [],
            "note": "The configured baseline training dataset was not available for overlap checks.",
        }
    legacy_features, legacy_labels = load_dataset(str(legacy_dataset_path))
    if legacy_features.ndim != 2 or legacy_features.shape[1] != FEATURE_VECTOR_SIZE:
        raise ValueError(
            "The configured baseline training dataset does not use the expected 42 features."
        )
    legacy_vectors = defaultdict(set)
    for features, label in zip(legacy_features, legacy_labels):
        legacy_vectors[tuple(np.asarray(features, dtype=np.float32))].add(label)

    overlaps = []
    for record in test_records:
        vector = tuple(np.asarray(record["features"], dtype=np.float32))
        labels = legacy_vectors.get(vector)
        if labels:
            overlaps.append({
                "v2_test_row_id": record["row_id"],
                "v2_test_label": record["label"],
                "legacy_dataset_labels": sorted(labels),
            })
    return {
        "checked": True,
        "dataset": str(legacy_dataset_path.resolve()),
        "dataset_sha256": _dataset_hash(legacy_dataset_path),
        "exact_feature_overlap_test_rows": len(overlaps),
        "overlap_rows": overlaps,
        "note": (
            "Exact-vector matches are checked against the configured legacy training "
            "input. This cannot establish which rows were used to fit the saved baseline "
            "or rule out near-duplicates and signer overlap because legacy provenance "
            "was not recorded."
        ),
    }


def evaluate_v2(dataset_path: Path, model_path: Path, split_path: Path,
                baseline_path: Path, baseline_training_data: Path,
                output_path: Path, bootstrap_iterations: int):
    records, invalid = load_v2_dataset(dataset_path)
    if invalid:
        raise ValueError("Evaluation stopped because the V2 dataset now has invalid rows.")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if _dataset_hash(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError(
            "Dataset changed after training/splitting. Re-run quality, train, and "
            "evaluation so the held-out test selection remains reproducible."
        )
    assignments = {int(row_id): split for row_id, split in manifest["row_assignments"].items()}
    test_records = [record for record in records
                    if assignments.get(record["row_id"]) == "test"]
    class_names = list(LABELS)
    label_to_id = {label: index for index, label in enumerate(class_names)}
    x_test = np.asarray([record["features"] for record in test_records], dtype=np.float32)
    y_true = np.asarray([label_to_id[record["label"]] for record in test_records], dtype=np.int32)
    v2_pipeline, v2_classes, v2_metadata = _load_model_with_metadata(model_path)
    baseline_pipeline, baseline_classes, _ = _load_model_with_metadata(baseline_path)
    if set(v2_classes) != set(class_names) or set(baseline_classes) != set(class_names):
        raise ValueError("Both models must support all 26 A-Z classes for a fair comparison.")
    if x_test.shape[1] != FEATURE_VECTOR_SIZE:
        raise ValueError("The V2 test features do not match the saved 42-feature representation.")

    baseline_labels = _predict_as_labels(baseline_pipeline, baseline_classes, x_test)
    v2_labels = _predict_as_labels(v2_pipeline, v2_classes, x_test)
    baseline_predictions = np.asarray([label_to_id[label] for label in baseline_labels], dtype=np.int32)
    v2_predictions = np.asarray([label_to_id[label] for label in v2_labels], dtype=np.int32)
    baseline_metrics = _metric_summary(y_true, baseline_predictions, class_names)
    v2_metrics = _metric_summary(y_true, v2_predictions, class_names)
    baseline_overlap = _legacy_dataset_overlap(test_records, baseline_training_data)
    assignments = {
        int(row_id): split for row_id, split in manifest["row_assignments"].items()
    }
    error_analysis = _targeted_error_analysis(
        records, test_records, assignments, manifest, class_names,
        baseline_predictions, v2_predictions,
        v2_metrics["top_confused_class_pairs"],
    )
    if manifest["split_strategy"] == "signer_disjoint":
        test_group_ids = [record["signer_id"] for record in test_records]
    else:
        test_group_ids = [
            f"{record['signer_id']}::{record['session_id']}" for record in test_records
        ]
    confidence_intervals = _paired_bootstrap(
        y_true, baseline_predictions, v2_predictions, test_group_ids, class_names,
        iterations=bootstrap_iterations,
    )
    accuracy_ci = confidence_intervals["accuracy_delta_v2_minus_baseline"]
    f1_ci = confidence_intervals["macro_f1_delta_v2_minus_baseline"]
    priority_ci = confidence_intervals["priority_class_accuracy_deltas"]
    no_significant_priority_regression = (
        priority_ci is not None
        and not any(bounds["upper_95_percent"] < 0 for bounds in priority_ci.values())
    )
    candidate = bool(
        accuracy_ci is not None
        and f1_ci is not None
        and accuracy_ci["lower_95_percent"] > 0
        and f1_ci["lower_95_percent"] > 0
        and no_significant_priority_regression
        and baseline_overlap["checked"]
        and baseline_overlap["exact_feature_overlap_test_rows"] == 0
    )
    report = {
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_valid_sample_count": len(records),
        "dataset_class_distribution": dict(sorted(Counter(
            record["label"] for record in records
        ).items())),
        "test_split_strategy": manifest["split_strategy"],
        "test_collection_groups_disjoint": manifest.get("collection_groups_disjoint", True),
        "test_sample_count": len(test_records),
        "test_class_distribution": manifest["split_counts"]["test"]["class_counts"],
        "baseline_model": str(baseline_path.resolve()),
        "v2_model": str(model_path.resolve()),
        "model_comparison": {
            "baseline": {
                "label": "Protected baseline",
                "status": "baseline_protected",
                "path": str(baseline_path.resolve()),
                "test_metrics": baseline_metrics,
            },
            "v2": {
                "label": (
                    "Interim V2 model"
                    if v2_metadata.get("status") == "interim_under_target"
                    else "V2 model"
                ),
                "status": v2_metadata.get("status", "metadata_unavailable"),
                "path": str(model_path.resolve()),
                "training_metadata": v2_metadata,
                "test_metrics": v2_metrics,
            },
        },
        "baseline_test_metrics": baseline_metrics,
        "v2_test_metrics": v2_metrics,
        "baseline_training_data_overlap_check": baseline_overlap,
        "targeted_error_analysis": error_analysis,
        "paired_bootstrap": confidence_intervals,
        "selection_guidance": {
            "v2_meets_predefined_candidate_rule": candidate,
            "candidate_rule": (
                "Both paired-bootstrap 95% intervals for accuracy and macro-F1 deltas "
                "must be wholly above zero, no priority class may show a 95% interval "
                "wholly below zero, and the shared test rows must have no exact feature "
                "matches in the configured legacy training input."
            ),
            "production_model_changed": False,
            "recommendation": (
                "V2 has evidence to be considered for a separately approved rollout; "
                "this workflow never activates or replaces the baseline."
                if candidate else
                "V2 has not demonstrated the required improvement; the protected "
                "baseline remains the active model."
            ),
        },
        "historical_baseline_reference": {
            "accuracy": 0.8065,
            "macro_f1": 0.762,
            "source": "User-provided approximate historical trainer-faithful baseline.",
            "note": (
                "Compared models above are evaluated on the same V2 held-out test "
                "rows. The existing stored random-holdout metrics differ and are not "
                "substituted for the historical reference."
            ),
        },
    }
    _json_write(output_path, report)
    print(f"Evaluation report: {output_path}")
    print(f"Baseline test: accuracy={baseline_metrics['accuracy']:.4f}, "
          f"macro_f1={baseline_metrics['macro_f1']:.4f}")
    v2_name = (
        "Interim V2 test" if v2_metadata.get("status") == "interim_under_target"
        else "V2 test"
    )
    print(f"{v2_name}: accuracy={v2_metrics['accuracy']:.4f}, "
          f"macro_f1={v2_metrics['macro_f1']:.4f}")
    print(report["selection_guidance"]["recommendation"])
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_dataset_argument(subparser):
        subparser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)

    quality_parser = subparsers.add_parser("quality", help="Report V2 dataset quality.")
    add_dataset_argument(quality_parser)
    quality_parser.add_argument("--report", type=Path, default=DEFAULT_QUALITY)
    quality_parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW)

    train_parser = subparsers.add_parser("train", help="Train the separate V2 model.")
    add_dataset_argument(train_parser)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    train_parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY)
    train_parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW)
    train_parser.add_argument("--report", type=Path, default=DEFAULT_TRAINING)
    train_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    train_parser.add_argument(
        "--allow-under-target",
        action="store_true",
        help=(
            "Explicitly train an INTERIM model below the official 300/500 targets; "
            "uses a pre-augmentation stratified row holdout if grouped A-Z splits "
            "cannot be formed."
        ),
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Compare V2 and baseline on the held-out V2 test.")
    add_dataset_argument(evaluate_parser)
    evaluate_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate_parser.add_argument("--baseline", type=Path,
                                 default=Path(config.MODELS_DIR) / "sign_model_light.pkl")
    evaluate_parser.add_argument(
        "--baseline-training-data", type=Path,
        default=Path(config.DATA_DIR) / "dataset.csv",
    )
    evaluate_parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    evaluate_parser.add_argument("--report", type=Path, default=DEFAULT_EVALUATION)
    evaluate_parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "quality":
        write_quality_artifacts(args.dataset, args.report, args.review_csv)
    elif args.command == "train":
        train_v2(
            args.dataset, args.model, args.split, args.quality_report,
            args.review_csv, args.report, args.seed,
            allow_under_target=args.allow_under_target,
        )
    elif args.command == "evaluate":
        if args.bootstrap_iterations < 100:
            raise SystemExit("--bootstrap-iterations must be at least 100.")
        evaluate_v2(
            args.dataset, args.model, args.split, args.baseline,
            args.baseline_training_data,
            args.report, args.bootstrap_iterations,
        )


if __name__ == "__main__":
    main()
