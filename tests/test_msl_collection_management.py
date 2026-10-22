import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.msl_dataset import record_video as recorder
from tools.msl_dataset import msl_dataset_pipeline as pipeline
from tools.msl_dataset.collection_management import (
    collection_priorities,
    create_session,
    register_recording,
    register_signer,
    set_sign_validation,
    upsert_vocabulary_sign,
    update_consent,
    validate_recordings,
)
from tools.msl_dataset.msl_dataset_pipeline import MSLSignDataset, default_config


class ReadableCapture:
    def __init__(self, _path):
        self.frames_read = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_COUNT: 12,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)

    def read(self):
        if self.frames_read == 12:
            return False, None
        self.frames_read += 1
        return True, object()

    def release(self):
        pass


def test_recording_requires_granted_consent_and_matching_metadata(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    raw = tmp_path / "raw"
    recording = raw / "S01" / "WATER" / "session_001" / "video_001.mp4"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"fixture-video")
    monkeypatch.setattr(cv2, "VideoCapture", ReadableCapture)

    register_signer("S01", manifests_dir=manifests)
    with pytest.raises(ValueError, match="granted consent"):
        create_session("S01", "session_001", manifests)

    update_consent("S01", "GRANTED", "consent-record-1", manifests)
    create_session("S01", "session_001", manifests)
    with pytest.raises(ValueError, match="must follow"):
        register_recording(recording, "S01", "FOOD", "session_001", manifests)

    registered = register_recording(recording, "S01", "WATER", "session_001", manifests)
    assert registered["validation_status"] == "PENDING_VALIDATION"


def test_validator_detects_changed_registered_source_without_accepting_it(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    raw = tmp_path / "raw"
    recording = raw / "S02" / "HELLO" / "session_001" / "video_001.mp4"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"before")
    monkeypatch.setattr(cv2, "VideoCapture", ReadableCapture)

    register_signer("S02", "GRANTED", "consent-record-2", manifests)
    create_session("S02", "session_001", manifests)
    register_recording(recording, "S02", "HELLO", "session_001", manifests)
    recording.write_bytes(b"after")

    result = validate_recordings(raw, manifests)

    assert result["invalid"][0]["status"] == "RAW_HASH_CHANGED"
    registrations = json.loads((manifests / "recordings.json").read_text(encoding="utf-8"))
    assert len(registrations) == 1
    assert registrations[0]["sha256"] != result["records"][0]["sha256"]


def test_collection_priorities_prefer_new_signer_diversity():
    priorities = collection_priorities(
        [
            {"sign": "WATER", "signer_id": "S01"},
            {"sign": "WATER", "signer_id": "S01"},
        ],
        {"WATER": {}},
        {"signers_per_sign": 3, "recordings_per_signer_sign": 3},
    )

    assert priorities == [{
        "sign": "WATER",
        "action": "ADD_NEW_SIGNER",
        "current_signers": 1,
        "target_signers": 3,
        "gap": 2,
    }]


def test_recorder_advances_number_and_registers_new_video(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    output = tmp_path / "raw"
    session_dir = output / "S03" / "WATER" / "session_001"
    session_dir.mkdir(parents=True)
    (session_dir / "video_007.mp4").write_bytes(b"existing")
    register_signer("S03", "GRANTED", "consent-record-3", manifests)
    create_session("S03", "session_001", manifests)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    class Capture:
        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def get(self, _prop):
            return 3

        def release(self):
            pass

    class Writer:
        def __init__(self, path, _fourcc, _fps, _size):
            self.path = path
            Path(path).write_bytes(b"new-recording")

        def isOpened(self):
            return True

        def write(self, _frame):
            pass

        def release(self):
            pass

    keys = iter((ord("s"), ord("q")))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _source: Capture())
    monkeypatch.setattr(cv2, "VideoWriter", Writer)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *_args: 0)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(recorder, "_validate_finished_video", lambda *_args: {"frames": 12})

    result = recorder.record_video(
        "S03", "WATER", "session_001", output, recording_number=1,
        manifests_dir=manifests,
    )

    assert result == session_dir / "video_008.mp4"
    assert len(json.loads((manifests / "recordings.json").read_text(encoding="utf-8"))) == 1


def test_registered_video_prepares_sequences_and_loads_in_pytorch(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    manifests = tmp_path / "manifests"
    processed = tmp_path / "processed"
    review = tmp_path / "review"
    video = raw / "S04" / "WATER" / "session_001" / "video_001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"test-video")
    register_signer("S04", "GRANTED", "consent-record-4", manifests)
    create_session("S04", "session_001", manifests)

    class Capture(ReadableCapture):
        def read(self):
            if self.frames_read == 12:
                return False, None
            self.frames_read += 1
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_COUNT: 12,
                cv2.CAP_PROP_FPS: 30,
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
            }.get(prop, 0)

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    register_recording(video, "S04", "WATER", "session_001", manifests)
    assert validate_recordings(raw, manifests)["valid"] == 1

    config = default_config()
    config.update({
        "raw_dataset_path": str(raw),
        "processed_dataset_path": str(processed),
        "manifests_path": str(manifests),
        "review_path": str(review),
        "sequence_length": 12,
        "stride": 12,
        "min_frames": 12,
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def extract(_path, _face_indices, _config):
        values = np.ones((12, 8), dtype=np.float32)
        presence = np.ones((12, 4), dtype=np.float32)
        stats = {
            "detected_frames": 12,
            "left_hand_detected": 12,
            "right_hand_detected": 12,
            "pose_detected": 12,
            "face_detected": 12,
        }
        return values, values.copy(), presence, stats

    monkeypatch.setattr(pipeline, "_extract_landmarks_for_video", extract)
    result = pipeline.prepare_msl_dataset(config_path=config_path)

    assert len(result["sequence_rows"]) == 1
    dataset = MSLSignDataset(manifests / "sequences.csv")
    features, label, metadata = dataset[0]
    assert tuple(features.shape) == (12, 8)
    assert label == 0
    assert metadata == {
        "signer_id": "S04",
        "video_id": "video_001",
        "session_id": "session_001",
    }
    monkeypatch.setattr(
        pipeline,
        "process_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume should reuse cached sequences")),
    )
    resumed = pipeline.prepare_msl_dataset(config_path=config_path)
    assert resumed["sequence_rows"][0]["sequence_id"] == result["sequence_rows"][0]["sequence_id"]


def test_confirmed_import_uses_existing_preprocessor_without_moving_source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    manifests = tmp_path / "manifests"
    processed = tmp_path / "processed"
    review = tmp_path / "review"
    raw.mkdir()
    source = tmp_path / "legacy" / "S05" / "WATER" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"legacy-video")
    register_signer("S05", "GRANTED", "consent-record-5", manifests)
    create_session("S05", "session_001", manifests)
    upsert_vocabulary_sign("WATER", "water", "WATER", "Confirmed MSL label", manifests)
    set_sign_validation("WATER", "VALIDATED", "reviewer-01", "Vocabulary evidence", manifests)
    imported_manifest = manifests / "existing_recordings.csv"
    imported_manifest.parent.mkdir(parents=True, exist_ok=True)
    imported_fields = [
        "recording_id", "original_path", "signer_id", "sign", "session_id", "sha256",
        "validation_status", "metadata_confirmation_status", "review_status",
        "signer_identity_status", "language", "language_validation_state",
        "label_validation_state", "source_hash_changed", "processing_status",
        "processing_notes", "already_processed",
        "signer_review_status", "session_review_status", "label_review_status",
        "language_review_status", "recording_quality_status", "human_review_status",
        "technical_qc_status", "finite_model_input",
    ]
    imported_row = {
        "recording_id": "imported-001",
        "original_path": str(source.resolve()),
        "signer_id": "S05",
        "sign": "WATER",
        "session_id": "session_001",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "validation_status": "READY_FOR_PROCESSING",
        "metadata_confirmation_status": "CONFIRMED",
        "review_status": "APPROVED",
        "signer_identity_status": "CONFIRMED",
        "signer_review_status": "CONFIRMED",
        "session_review_status": "CONFIRMED",
        "label_review_status": "CONFIRMED",
        "language_review_status": "MSL",
        "recording_quality_status": "APPROVED",
        "human_review_status": "APPROVED",
        "language": "MSL",
        "language_validation_state": "VALIDATED",
        "label_validation_state": "VALIDATED",
        "source_hash_changed": "false",
        "processing_status": "NOT_PROCESSED",
        "processing_notes": "",
        "already_processed": "false",
        "technical_qc_status": "READY_FOR_REVIEW",
        "finite_model_input": "true",
    }
    with imported_manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=imported_fields)
        writer.writeheader()
        writer.writerow(imported_row)
    config = default_config()
    config.update({
        "raw_dataset_path": str(raw),
        "processed_dataset_path": str(processed),
        "manifests_path": str(manifests),
        "review_path": str(review),
        "sequence_length": 4,
        "stride": 2,
        "min_frames": 1,
    })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def extract(_path, _face_indices, _config):
        features = np.ones((5, 8), dtype=np.float32)
        presence = np.ones((5, 4), dtype=np.float32)
        stats = {
            "detected_frames": 5,
            "left_hand_detected": 5,
            "right_hand_detected": 5,
            "pose_detected": 5,
            "face_detected": 5,
        }
        return features, features.copy(), presence, stats

    monkeypatch.setattr(pipeline, "_read_video_metadata", lambda _path: (30.0, 5, 1 / 6, 640, 480))
    monkeypatch.setattr(pipeline, "_extract_landmarks_for_video", extract)
    result = pipeline.prepare_msl_dataset(config_path=config_path)

    assert len(result["sequence_rows"]) == 1
    assert result["sequence_rows"][0]["session_id"] == "session_001"
    with np.load(result["sequence_rows"][0]["sequence_path"], allow_pickle=False) as sample:
        assert sample["sequence"].shape == (4, 8)
        assert sample["session_id"].item() == "session_001"
    assert source.read_bytes() == b"legacy-video"
    with imported_manifest.open("r", newline="", encoding="utf-8") as source_csv:
        saved_import = next(csv.DictReader(source_csv))
    assert saved_import["processing_status"] == "PROCESSED"
    assert saved_import["already_processed"] == "true"
    monkeypatch.setattr(
        pipeline,
        "process_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("import resume should reuse cached output")),
    )
    resumed = pipeline.prepare_msl_dataset(config_path=config_path)
    assert resumed["sequence_rows"][0]["sequence_id"] == result["sequence_rows"][0]["sequence_id"]
