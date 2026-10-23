import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data_collector import (
    V2_DATASET_FIELDS,
    _append_v2_sample,
    _v2_capture_problem,
    count_v2_samples,
)
import data_collector
from static_model_v2 import (
    ALL_LETTERS,
    LABELS,
    MOTION_LABELS,
    _load_model_with_metadata,
    _paired_bootstrap,
    _split_records,
    build_quality_report,
    train_v2,
)


class FakeDetector:
    def __init__(self, hands):
        self.results = SimpleNamespace(hand_landmarks=hands)

    def get_landmarks(self, frame, hand_no=0):
        return [[index, index + 1, index + 2] for index in range(21)]


class FakeExtractor:
    def extract_features(self, landmarks):
        return [float(index) for index in range(42)]


class TestStaticModelV2(unittest.TestCase):
    def test_static_training_labels_exclude_motion_letters(self):
        self.assertEqual(set(ALL_LETTERS) - set(LABELS), set(MOTION_LABELS))
        self.assertNotIn("J", LABELS)
        self.assertNotIn("Z", LABELS)
        self.assertEqual(len(LABELS), 24)

    def test_v2_collector_rejects_motion_letters_before_camera_start(self):
        with patch.object(
            sys,
            "argv",
            ["data_collector.py", "--v2", "--signer-id", "S01", "--sign", "Z"],
        ):
            with self.assertRaisesRegex(SystemExit, "excludes motion letters Z"):
                data_collector.main()

    def hand(self):
        return [
            SimpleNamespace(x=0.3 + (index % 5) * 0.08, y=0.2 + (index // 5) * 0.12)
            for index in range(21)
        ]

    def test_collector_rejects_no_hand_and_ambiguous_hands(self):
        frame = SimpleNamespace(shape=(480, 640, 3))
        reason, features = _v2_capture_problem(FakeDetector([]), frame, FakeExtractor())
        self.assertIn("No hand", reason)
        self.assertIsNone(features)

        reason, features = _v2_capture_problem(
            FakeDetector([self.hand(), self.hand()]), frame, FakeExtractor()
        )
        self.assertIn("Exactly one", reason)
        self.assertIsNone(features)

    def test_collector_accepts_complete_finite_well_framed_hand(self):
        frame = SimpleNamespace(shape=(480, 640, 3))
        reason, features = _v2_capture_problem(
            FakeDetector([self.hand()]), frame, FakeExtractor()
        )
        self.assertIsNone(reason)
        self.assertEqual(len(features), 42)

    def test_v2_append_records_metadata_and_preserves_legacy_file_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset_v2.csv"
            metadata = {
                "signer_id": "S01",
                "session_id": "S01-01",
                "source": "webcam",
                "device": "camera-0",
            }
            _append_v2_sample(str(path), "A", [0.0] * 42, metadata, (480, 640, 3))
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["signer_id"], "S01")
            self.assertEqual(rows[0]["session_id"], "S01-01")
            self.assertEqual(len(V2_DATASET_FIELDS), 50)
            self.assertEqual(count_v2_samples(str(path)), {"A": 1})
            self.assertEqual(count_v2_samples(str(path), "S01-01"), {"A": 1})

    def test_quality_report_flags_duplicates_without_relabeling_or_dropping(self):
        features = [float(index) for index in range(42)]
        records = [
            {"row_id": 2, "label": "A", "signer_id": "S01", "session_id": "01",
             "features": features},
            {"row_id": 3, "label": "B", "signer_id": "S01", "session_id": "01",
             "features": features},
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset_v2.csv"
            dataset_path.write_text("test-only fixture\n", encoding="utf-8")
            report = build_quality_report(records, [], dataset_path)
        self.assertEqual(report["valid_total_samples"], 2)
        self.assertEqual(report["invalid_sample_count"], 0)
        self.assertEqual(report["duplicate_feature_vectors"]["excess_rows"], 1)
        self.assertEqual(len(report["suspicious_samples"]), 2)
        self.assertIn("does not infer or change labels", report["label_review_note"])

    def test_quality_report_tracks_motion_rows_but_excludes_them_from_static_classes(self):
        records = [
            {"row_id": 2, "label": "A", "signer_id": "S01", "session_id": "01",
             "features": [float(index) for index in range(42)]},
            {"row_id": 3, "label": "J", "signer_id": "S01", "session_id": "01",
             "features": [float(index + 1) for index in range(42)]},
            {"row_id": 4, "label": "Z", "signer_id": "S01", "session_id": "01",
             "features": [float(index + 2) for index in range(42)]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset_v2.csv"
            dataset_path.write_text("test-only fixture\n", encoding="utf-8")
            report = build_quality_report(records, [], dataset_path)

        self.assertEqual(report["samples_per_class"]["A"], 1)
        self.assertNotIn("J", report["samples_per_class"])
        self.assertNotIn("Z", report["samples_per_class"])
        self.assertEqual(report["excluded_motion_samples_per_class"], {"J": 1, "Z": 1})
        self.assertEqual(report["static_training_sample_count"], 1)
        self.assertEqual(report["excluded_motion_sample_count"], 2)

    def test_split_keeps_each_collection_session_wholly_in_one_split(self):
        records = []
        for session_index in range(5):
            for class_index, label in enumerate(LABELS):
                features = [float(class_index), float(session_index)] + [0.0] * 40
                records.append({
                    "row_id": 2 + session_index * len(LABELS) + class_index,
                    "label": label,
                    "signer_id": "S01",
                    "session_id": f"session-{session_index}",
                    "features": features,
                })
        assignments, strategy, groups, cross_duplicates = _split_records(records, 42)
        self.assertEqual(strategy, "collection_session_disjoint")
        self.assertFalse(cross_duplicates)
        session_splits = {}
        for record in records:
            split = assignments[record["row_id"]]
            session_splits.setdefault(record["session_id"], set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in session_splits.values()))
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_split_refuses_too_few_independent_collection_groups(self):
        records = []
        for session_index in range(4):
            for class_index, label in enumerate(LABELS):
                records.append({
                    "row_id": 2 + session_index * len(LABELS) + class_index,
                    "label": label,
                    "signer_id": "S01",
                    "session_id": f"session-{session_index}",
                    "features": [float(class_index), float(session_index)] + [0.0] * 40,
                })
        with self.assertRaisesRegex(ValueError, "At least 5 independent"):
            _split_records(records, 42)

    def test_bootstrap_refuses_to_overstate_single_group_uncertainty(self):
        result = _paired_bootstrap(
            y_true=[0, 0, 0],
            baseline=[0, 0, 0],
            v2=[0, 0, 1],
            group_ids=["one-session"] * 3,
            class_names=["A"],
            iterations=100,
        )
        self.assertFalse(result["confidence_intervals_estimable"])
        self.assertIsNone(result["accuracy_delta_v2_minus_baseline"])

    def training_records(self):
        records = []
        for class_index, label in enumerate(LABELS):
            for sample_index in range(6):
                features = [
                    float(class_index),
                    float(sample_index),
                    *[
                        float((class_index * 31 + sample_index * 7 + index) % 53)
                        for index in range(40)
                    ],
                ]
                records.append({
                    "row_id": len(records) + 2,
                    "label": label,
                    "signer_id": "S01",
                    "session_id": f"session-{sample_index % 2}",
                    "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    "features": features,
                })
        return records

    def test_interim_holdout_keeps_duplicate_feature_vectors_together(self):
        records = self.training_records()
        c_record = next(row for row in records if row["label"] == "C")
        d_record = next(row for row in records if row["label"] == "D")
        d_record["features"] = c_record["features"].copy()

        assignments, strategy, _, cross_duplicates = _split_records(
            records, 42, allow_row_stratified=True
        )

        self.assertEqual(
            strategy, "trainer_faithful_stratified_row_holdout_pre_augmentation"
        )
        self.assertEqual(assignments[c_record["row_id"]], assignments[d_record["row_id"]])
        self.assertFalse(cross_duplicates)

    def test_training_without_override_still_rejects_under_target_data(self):
        records = self.training_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset_v2.csv"
            dataset.write_text("test-only training fixture\n", encoding="utf-8")
            quality = build_quality_report(records, [], dataset)
            paths = [root / f"{name}.json" for name in ("split", "quality", "training")]
            review = root / "review.csv"
            with patch(
                "static_model_v2.write_quality_artifacts",
                return_value=(records, [], quality),
            ) as quality_check:
                with self.assertRaisesRegex(ValueError, "at least 300 valid samples"):
                    train_v2(
                        dataset, root / "sign_model_v2.pkl", paths[0], paths[1],
                        review, paths[2], 42,
                    )
            quality_check.assert_called_once()
            self.assertFalse((root / "sign_model_v2.pkl").exists())

    def test_under_target_override_saves_interim_model_and_separate_report(self):
        records = self.training_records()
        for motion_index, label in enumerate(sorted(MOTION_LABELS)):
            records.append({
                "row_id": len(records) + 2,
                "label": label,
                "signer_id": "S01",
                "session_id": "session-motion",
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "features": [float(1000 + motion_index * 42 + index) for index in range(42)],
            })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset_v2.csv"
            dataset.write_text("test-only training fixture\n", encoding="utf-8")
            quality = build_quality_report(records, [], dataset)
            baseline = root / "sign_model_light.pkl"
            baseline_bytes = b"protected test baseline artifact"
            baseline.write_bytes(baseline_bytes)
            model = root / "sign_model_v2.pkl"
            split = root / "split.json"
            quality_path = root / "quality.json"
            review = root / "review.csv"
            report_path = root / "training.json"
            with patch(
                "static_model_v2.write_quality_artifacts",
                return_value=(records, [], quality),
            ):
                report = train_v2(
                    dataset, model, split, quality_path, review, report_path, 42,
                    allow_under_target=True,
                )

            self.assertTrue(model.is_file())
            self.assertNotEqual(model.resolve(), baseline.resolve())
            self.assertEqual(baseline.read_bytes(), baseline_bytes)
            self.assertEqual(report["status"], "interim_under_target")
            self.assertEqual(report["target_override"], "--allow-under-target")
            self.assertEqual(report["total_valid_samples"], len(LABELS) * 6)
            self.assertEqual(report["excluded_motion_sample_count"], 2)
            self.assertEqual(
                report["excluded_motion_samples_per_class"], {"J": 1, "Z": 1}
            )
            self.assertEqual(report["minimum_class_count"], 6)
            self.assertEqual(set(report["classes_below_target"]), set(LABELS))
            self.assertEqual(report["dataset_sha256"], hashlib.sha256(dataset.read_bytes()).hexdigest())
            self.assertIsNotNone(report["training_timestamp"])

            _, model_classes, metadata = _load_model_with_metadata(model)
            self.assertEqual(set(model_classes), set(LABELS))
            self.assertFalse(set(model_classes) & set(MOTION_LABELS))
            self.assertEqual(metadata["status"], "interim_under_target")
            self.assertEqual(metadata["target_override"], "--allow-under-target")
            self.assertEqual(metadata["normal_class_target"], 300)
            self.assertEqual(metadata["priority_class_target"], 500)

            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIsNone(saved_report["validation_metrics"])
            self.assertEqual(
                saved_report["split_strategy"],
                "trainer_faithful_stratified_row_holdout_pre_augmentation",
            )
            manifest = json.loads(split.read_text(encoding="utf-8"))
            self.assertFalse(manifest["collection_groups_disjoint"])
            self.assertEqual(manifest["split_counts"]["validation"]["samples"], 0)
            self.assertGreater(manifest["split_counts"]["test"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
