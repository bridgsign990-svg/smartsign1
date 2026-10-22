import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import gesture_trainer
import lstm_gesture_trainer
from gesture_data_utils import (
    GESTURE_V2_FIELDS,
    grouped_train_validation_split,
    load_gesture_sequences,
)
from gesture_collector import _append_v2_sample


class GestureDatasetTests(unittest.TestCase):
    def test_legacy_rows_load_without_invented_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            sequence = np.arange(15 * 42, dtype=np.float32).reshape(15, 42)
            path.write_text(
                "HELP," + ",".join(map(str, sequence.flatten())) + "\n",
                encoding="utf-8",
            )

            loaded, labels, signers, sessions = load_gesture_sequences(path)

        self.assertEqual(loaded.shape, (1, 15, 42))
        np.testing.assert_array_equal(loaded[0], sequence)
        self.assertEqual(labels, ["HELP"])
        self.assertIsNone(signers)
        self.assertIsNone(sessions)

    def test_versioned_collector_rows_round_trip_provenance_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gesture_dataset_v2.csv"
            sequence = np.arange(15 * 42, dtype=np.float32).reshape(15, 42)
            _append_v2_sample(
                str(path),
                "HELP",
                sequence,
                {
                    "signer_id": "signer-1",
                    "session_id": "session-1",
                    "device": "camera-test",
                },
                (480, 640, 3),
                "2026-10-21T12:00:00.000+00:00",
                [float(index) / 14 for index in range(15)],
            )

            with path.open("r", newline="", encoding="utf-8") as source:
                self.assertEqual(next(csv.reader(source)), GESTURE_V2_FIELDS)
            loaded, labels, signers, sessions = load_gesture_sequences(path)
            with path.open("r", newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))

        np.testing.assert_array_equal(loaded[0], sequence)
        self.assertEqual(labels, ["HELP"])
        self.assertEqual(signers, ["signer-1"])
        self.assertEqual(sessions, ["session-1"])
        self.assertEqual(row["device"], "camera-test")
        self.assertEqual(row["frame_width"], "640")
        self.assertEqual(row["frame_height"], "480")
        self.assertEqual(row["capture_fps"], "14.0")

    def test_exact_duplicates_are_kept_in_one_partition(self):
        sequences = []
        labels = []
        for label_index, label in enumerate(("HELP", "WATER")):
            for sample_index in range(10):
                sequence = np.full((15, 42), label_index * 100 + sample_index, dtype=np.float32)
                sequences.append(sequence)
                labels.append(label)
                if sample_index == 0:
                    sequences.append(sequence.copy())
                    labels.append(label)

        train, validation, report = grouped_train_validation_split(
            sequences,
            labels,
            test_size=0.25,
            random_state=42,
        )
        train_keys = {sequences[index].tobytes() for index in train}
        validation_keys = {sequences[index].tobytes() for index in validation}

        self.assertFalse(train_keys & validation_keys)
        self.assertEqual(report["split_basis"], "exact-duplicate-only")
        self.assertEqual(report["duplicate_extra_rows"], 2)
        self.assertEqual(set(labels[index] for index in train), {"HELP", "WATER"})
        self.assertEqual(set(labels[index] for index in validation), {"HELP", "WATER"})

    def test_signers_are_kept_in_one_partition(self):
        sequences = []
        labels = []
        signers = []
        sessions = []
        for signer_index in range(6):
            for label_index, label in enumerate(("HELP", "WATER")):
                sequences.append(
                    np.full((15, 42), signer_index * 10 + label_index, dtype=np.float32)
                )
                labels.append(label)
                signers.append(f"signer-{signer_index}")
                sessions.append(f"session-{signer_index}")

        train, validation, report = grouped_train_validation_split(
            sequences,
            labels,
            signer_ids=signers,
            session_ids=sessions,
            test_size=0.25,
            random_state=42,
        )

        train_signers = {signers[index] for index in train}
        validation_signers = {signers[index] for index in validation}
        self.assertFalse(train_signers & validation_signers)
        self.assertEqual(report["split_basis"], "signer")

    def test_single_signer_uses_independent_sessions(self):
        sequences = []
        labels = []
        signers = []
        sessions = []
        for session_index in range(6):
            for label_index, label in enumerate(("HELP", "WATER")):
                sequences.append(
                    np.full((15, 42), session_index * 10 + label_index, dtype=np.float32)
                )
                labels.append(label)
                signers.append("signer-1")
                sessions.append(f"session-{session_index}")

        train, validation, report = grouped_train_validation_split(
            sequences,
            labels,
            signer_ids=signers,
            session_ids=sessions,
            test_size=0.25,
            random_state=42,
        )

        train_sessions = {sessions[index] for index in train}
        validation_sessions = {sessions[index] for index in validation}
        self.assertFalse(train_sessions & validation_sessions)
        self.assertEqual(report["split_basis"], "session")

    def test_conflicting_duplicate_labels_fail_before_training(self):
        sequence = np.zeros((15, 42), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            grouped_train_validation_split(
                [sequence, sequence.copy()],
                ["HELP", "WATER"],
            )

    def test_malformed_legacy_rows_fail_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.csv"
            path.write_text("HELP,1,2,3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Line 1"):
                load_gesture_sequences(path)

    def test_trainers_refuse_to_overwrite_existing_candidate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            random_forest_output = Path(directory) / "gesture_candidate.pkl"
            lstm_output = Path(directory) / "gesture_candidate.pt"
            random_forest_output.write_bytes(b"preserve-pickle")
            lstm_output.write_bytes(b"preserve-checkpoint")

            with patch.object(
                sys,
                "argv",
                ["gesture_trainer.py", "--output", str(random_forest_output)],
            ):
                with self.assertRaises(FileExistsError):
                    gesture_trainer.main()
            with patch.object(
                sys,
                "argv",
                ["lstm_gesture_trainer.py", "--output", str(lstm_output)],
            ):
                with self.assertRaises(FileExistsError):
                    lstm_gesture_trainer.main()

            self.assertEqual(random_forest_output.read_bytes(), b"preserve-pickle")
            self.assertEqual(lstm_output.read_bytes(), b"preserve-checkpoint")


if __name__ == "__main__":
    unittest.main()
