import hashlib
import json

import cv2
import numpy as np
import pytest
from pathlib import Path

from tools.msl_dataset import import_existing_recordings as importer
from tools.msl_dataset.collection_management import (
    collection_status,
    create_session,
    register_signer,
    set_sign_validation,
    upsert_vocabulary_sign,
)
from tools.msl_dataset.msl_dataset_pipeline import (
    _create_holistic_detector,
    default_config,
)


class FakeVideoCapture:
    def __init__(self, path):
        self.path = path
        self.index = 0

    def isOpened(self):
        return "unreadable" not in str(self.path)

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_COUNT: 15,
            cv2.CAP_PROP_FPS: 25,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)

    def read(self):
        if self.index >= 15:
            return False, None
        self.index += 1
        return True, object()

    def release(self):
        pass


def _config_file(tmp_path):
    config_path = tmp_path / "config.json"
    config = default_config()
    config["holistic_model_path"] = str(tmp_path / "missing.task")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_path_inference_recognizes_controlled_legacy_and_unknown_paths():
    controlled = importer._infer_path_metadata(importer.Path("S01/WATER/session_001/video.mp4"))
    legacy = importer._infer_path_metadata(importer.Path("S01/WATER/video.mp4"))
    sign_only = importer._infer_path_metadata(importer.Path("WATER/video.mp4"))
    unknown = importer._infer_path_metadata(importer.Path("misc/a bad/b/video.mp4"))

    assert (controlled["signer_id"], controlled["sign"], controlled["session_id"]) == (
        "S01", "WATER", "session_001"
    )
    assert legacy["legacy_flat_layout"] == "true"
    assert legacy["session_id"] == "UNKNOWN"
    assert sign_only["signer_id"] == "UNKNOWN"
    assert unknown["sign"] == "UNKNOWN"


def test_import_manifest_records_unknowns_duplicates_and_pending_review(tmp_path, monkeypatch):
    source = tmp_path / "source"
    manifests = tmp_path / "manifests"
    review = tmp_path / "review"
    first = source / "S01" / "WATER" / "video.mp4"
    duplicate = source / "S01" / "WATER" / "copy.mp4"
    unreadable = source / "S01" / "FOOD" / "unreadable.mp4"
    for path in (first, duplicate, unreadable):
        path.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"same-video-content")
    duplicate.write_bytes(b"same-video-content")
    unreadable.write_bytes(b"bad")
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (_ for _ in ()).throw(
        FileNotFoundError("Holistic task asset missing")
    ))

    result = importer.import_existing_recordings(
        [source], manifests, review, _config_file(tmp_path)
    )
    summary = result["summary"]
    rows = {row["filename"]: row for row in result["records"]}

    assert summary["total_videos"] == 3
    assert summary["readable_videos"] == 2
    assert summary["unreadable_videos"] == 1
    assert summary["duplicate_content_files"] == 2
    assert summary["identifiable_label_candidates"] == 3
    assert summary["videos_with_unknown_session"] == 3
    assert summary["videos_pending_review"] == 3
    assert rows["video.mp4"]["signer_id"] == "S01"
    assert rows["video.mp4"]["signer_identity_status"] == "INFERRED_UNCONFIRMED"
    assert rows["video.mp4"]["session_id"] == "UNKNOWN"
    assert rows["video.mp4"]["validation_status"] == "METADATA_INCOMPLETE"
    assert rows["video.mp4"]["language"] == "UNKNOWN"
    assert rows["video.mp4"]["review_status"] == "PENDING"
    assert rows["unreadable.mp4"]["validation_status"] == "VIDEO_INVALID"
    assert (manifests / "existing_recordings.csv").is_file()
    assert (review / "recording_review.csv").is_file()


def test_missing_holistic_model_fails_with_configured_path(tmp_path):
    missing_path = tmp_path / "not-present.task"

    with pytest.raises(FileNotFoundError) as exc_info:
        _create_holistic_detector({"holistic_model_path": str(missing_path)})
    assert str(missing_path) in str(exc_info.value)


def test_technical_qc_records_hand_metrics_without_approving_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    video = source / "S01" / "WATER" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"technical-qc-fixture")
    config = _config_file(tmp_path)
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (type(
        "Detector", (), {"close": lambda self: None}
    )(), "tasks"))
    features = np.ones((15, 562), dtype=np.float32)
    presence = np.zeros((15, 4), dtype=np.float32)
    presence[:9, 0] = 1
    presence[5:12, 1] = 1
    stats = {
        "frame_count": 15,
        "left_hand_detected": 9,
        "right_hand_detected": 7,
        "pose_detected": 14,
        "face_detected": 12,
    }
    monkeypatch.setattr(
        importer,
        "_extract_landmarks_for_video",
        lambda *_args: ([row for row in features], [], [row for row in presence], stats),
    )

    result = importer.import_existing_recordings(
        [source], tmp_path / "manifests", tmp_path / "review", config
    )
    row = result["records"][0]

    assert row["frames_processed"] == 15
    assert row["hand_detected_frames"] == 12
    assert row["hand_detected_percentage"] == 80.0
    assert row["left_hand_detected_percentage"] == 60.0
    assert row["right_hand_detected_percentage"] == pytest.approx(46.667)
    assert row["pose_detected_percentage"] == pytest.approx(93.333)
    assert row["face_detected_percentage"] == 80.0
    assert row["sequence_window_count"] == 1
    assert row["model_feature_dimension"] == 562
    assert row["finite_model_input"] == "true"
    assert row["technical_qc_status"] == "READY_FOR_REVIEW"
    assert row["validation_status"] == "METADATA_INCOMPLETE"
    assert row["signer_validation_status"] == "UNKNOWN"
    assert row["label_validation_state"] == "UNVALIDATED"
    assert row["language_validation_state"] == "UNVERIFIED"
    assert row["consent_status"] == "UNKNOWN"
    assert row["review_status"] == "PENDING"
    assert row["processing_status"] == "NOT_PROCESSED"
    assert result["summary"]["videos_technically_ready"] == 1


def test_non_finite_model_features_are_qc_errors(tmp_path, monkeypatch):
    source = tmp_path / "source"
    video = source / "S01" / "WATER" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"non-finite-qc-fixture")
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (type(
        "Detector", (), {"close": lambda self: None}
    )(), "tasks"))
    features = np.ones((15, 562), dtype=np.float32)
    features[0, 0] = np.inf
    presence = np.ones((15, 4), dtype=np.float32)
    stats = {
        "frame_count": 15,
        "left_hand_detected": 15,
        "right_hand_detected": 15,
        "pose_detected": 15,
        "face_detected": 15,
    }
    monkeypatch.setattr(
        importer,
        "_extract_landmarks_for_video",
        lambda *_args: ([row for row in features], [], [row for row in presence], stats),
    )

    result = importer.import_existing_recordings(
        [source], tmp_path / "manifests", tmp_path / "review", _config_file(tmp_path)
    )
    row = result["records"][0]

    assert row["finite_model_input"] == "false"
    assert row["technical_qc_status"] == "QC_ERROR"
    assert "NaN/Inf" in row["landmark_validation_notes"]


def test_readable_video_without_hands_is_not_marked_video_invalid(tmp_path, monkeypatch):
    source = tmp_path / "source"
    video = source / "S01" / "WATER" / "no_hands.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"no-hands-fixture")
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (type(
        "Detector", (), {"close": lambda self: None}
    )(), "tasks"))
    features = np.ones((15, 562), dtype=np.float32)
    presence = np.zeros((15, 4), dtype=np.float32)
    stats = {
        "frame_count": 15,
        "left_hand_detected": 0,
        "right_hand_detected": 0,
        "pose_detected": 15,
        "face_detected": 12,
    }
    monkeypatch.setattr(
        importer,
        "_extract_landmarks_for_video",
        lambda *_args: ([row for row in features], [], [row for row in presence], stats),
    )

    result = importer.import_existing_recordings(
        [source], tmp_path / "manifests", tmp_path / "review", _config_file(tmp_path)
    )
    row = result["records"][0]

    assert row["video_validation_status"] == "READABLE"
    assert row["technical_qc_status"] == "READABLE_BUT_LOW_HAND_DETECTION"
    assert row["landmark_availability"] == "NO_HAND_LANDMARKS"
    assert row["validation_status"] == "LANDMARK_QUALITY_LOW"


def test_import_is_idempotent_and_preserves_recording_review(tmp_path, monkeypatch):
    source = tmp_path / "source"
    manifests = tmp_path / "manifests"
    review = tmp_path / "review"
    video = source / "S01" / "HELLO" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (_ for _ in ()).throw(
        RuntimeError("asset unavailable")
    ))
    config = _config_file(tmp_path)
    first_result = importer.import_existing_recordings([source], manifests, review, config)
    recording_id = first_result["records"][0]["recording_id"]
    importer.update_existing_recording_review(
        recording_id, "APPROVED", "reviewer-01", "Checked locally", manifests, review
    )
    second_result = importer.import_existing_recordings([source], manifests, review, config)

    assert len(second_result["records"]) == 1
    assert second_result["records"][0]["recording_id"] == recording_id
    assert second_result["records"][0]["review_status"] == "APPROVED"
    status = collection_status(manifests, review)
    assert status["imported_recordings"] == 1
    assert status["imported_review_status_counts"] == {"APPROVED": 1}
    assert status["pending_review_recordings"] == 0
    assert status["sign_status"][0]["ready_for_training"] is False


def test_import_recognizes_controlled_layout_and_unknown_signer_layout(tmp_path, monkeypatch):
    source = tmp_path / "source"
    controlled = source / "S08" / "FOOD" / "session_002" / "controlled.mp4"
    ambiguous = source / "misc" / "unknown folder" / "clip.mp4"
    controlled.parent.mkdir(parents=True)
    ambiguous.parent.mkdir(parents=True)
    controlled.write_bytes(b"controlled fixture")
    ambiguous.write_bytes(b"unknown fixture")
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (_ for _ in ()).throw(
        RuntimeError("asset unavailable")
    ))

    result = importer.import_existing_recordings(
        [source], tmp_path / "manifests", tmp_path / "review", _config_file(tmp_path)
    )
    rows = {row["filename"]: row for row in result["records"]}

    assert rows["controlled.mp4"]["source_type"] == "controlled_layout"
    assert rows["controlled.mp4"]["session_id"] == "session_002"
    assert rows["controlled.mp4"]["signer_id"] == "S08"
    assert rows["clip.mp4"]["signer_id"] == "UNKNOWN"
    assert rows["clip.mp4"]["sign"] == "UNKNOWN"
    assert rows["clip.mp4"]["session_id"] == "UNKNOWN"


def test_import_detects_recording_already_in_registry_and_keeps_static_files(tmp_path, monkeypatch):
    source = tmp_path / "source"
    manifests = tmp_path / "manifests"
    review = tmp_path / "review"
    video = source / "S01" / "WATER" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"registered-content")
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    manifests.mkdir()
    (manifests / "recordings.json").write_text(json.dumps([{
        "recording_id": "registered-1",
        "file_path": str(video.resolve()),
        "sha256": digest,
    }]), encoding="utf-8")
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (_ for _ in ()).throw(
        RuntimeError("asset unavailable")
    ))
    protected = [tmp_path / "data" / "dataset.csv", tmp_path / "models" / "sign_model_light.pkl"]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"static asset fixture")
    hashes_before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in protected]

    result = importer.import_existing_recordings(
        [source], manifests, review, _config_file(tmp_path)
    )

    imported = result["records"][0]
    assert imported["represented_in_recordings_csv"] == "false"
    assert imported["content_already_in_recordings_manifest"] == "true"
    assert imported["duplicate_content"] == "true"
    assert imported["duplicate_of_recording_ids"] == "registered-1"
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in protected] == hashes_before


def test_confirmed_metadata_requires_evidence_and_enables_processing(tmp_path, monkeypatch):
    source = tmp_path / "source"
    video = source / "S01" / "WATER" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"reviewed fixture")
    manifests = tmp_path / "manifests"
    review = tmp_path / "review"
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (_ for _ in ()).throw(
        RuntimeError("asset unavailable")
    ))
    result = importer.import_existing_recordings(
        [source], manifests, review, _config_file(tmp_path), check_landmarks=False
    )
    recording_id = result["records"][0]["recording_id"]
    rows = importer._read_csv(manifests / importer.IMPORT_FILENAME)
    rows[0]["landmark_status"] = "CHECKED"
    rows[0]["hand_detected_frames"] = "15"
    rows[0]["video_validation_status"] = "READABLE"
    rows[0]["technical_qc_status"] = "READY_FOR_REVIEW"
    rows[0]["finite_model_input"] = "true"
    importer._write_csv(manifests / importer.IMPORT_FILENAME, list(rows[0]), rows)

    with pytest.raises(ValueError, match="APPROVED"):
        importer.confirm_existing_recording_metadata(
            recording_id, "S01", "WATER", "session_001", "MSL",
            "consent/session record", "review evidence", "language evidence", manifests,
            session_evidence="session evidence",
        )

    register_signer("S01", "GRANTED", "consent-record", manifests)
    create_session("S01", "session_001", manifests)
    upsert_vocabulary_sign("WATER", "water", "WATER", "Verified local label", manifests)
    set_sign_validation("WATER", "VALIDATED", "reviewer-01", "Vocabulary evidence", manifests)
    confirmed = importer.update_existing_recording_review(
        recording_id,
        None,
        "reviewer-01",
        "Human-verified identity, session, label, and MSL language from source records.",
        manifests,
        review,
        signer_review_status="CONFIRMED",
        confirmed_signer_id="S01",
        session_review_status="CONFIRMED",
        confirmed_session_id="session_001",
        label_review_status="CONFIRMED",
        confirmed_sign="WATER",
        language_review_status="MSL",
        recording_quality_status="APPROVED",
        human_review_status="APPROVED",
    )

    assert confirmed["signer_identity_status"] == "CONFIRMED"
    assert confirmed["label_validation_state"] == "VALIDATED"
    assert confirmed["metadata_confirmation_status"] == "CONFIRMED"
    assert confirmed["validation_status"] == "READY_FOR_PROCESSING"
    assert video.read_bytes() == b"reviewed fixture"
    assert importer._final_status(
        confirmed,
        {"S01": {"status": "ACTIVE", "consent_status": "GRANTED"}},
        {"S01/session_001": {"status": "ACTIVE"}},
        {"WATER": {"language": "MSL", "status": "ACTIVE", "human_validation": "VALIDATED"}},
    ) == "READY_FOR_PROCESSING"


def _eligible_import_row(video: Path) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"eligible-video")
    row = {
        "recording_id": "recording-1",
        "original_path": str(video.resolve()),
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "source_hash_changed": "false",
        "video_validation_status": "READABLE",
        "technical_qc_status": "READY_FOR_REVIEW",
        "landmark_status": "CHECKED",
        "hand_detected_frames": "12",
        "finite_model_input": "true",
        "signer_id": "S01",
        "signer_identity_status": "CONFIRMED",
        "signer_review_status": "CONFIRMED",
        "sign": "WATER",
        "sign_candidate": "WATER",
        "label_review_status": "CONFIRMED",
        "label_validation_state": "VALIDATED",
        "language": "MSL",
        "language_review_status": "MSL",
        "language_validation_state": "VALIDATED",
        "session_id": "session_001",
        "session_review_status": "CONFIRMED",
        "recording_quality_status": "APPROVED",
        "human_review_status": "APPROVED",
        "review_status": "APPROVED",
        "metadata_confirmation_status": "CONFIRMED",
        "signer_confirmation_evidence": "verified",
        "session_confirmation_evidence": "verified",
        "label_confirmation_evidence": "verified",
        "language_confirmation_evidence": "verified",
    }
    signers = {"S01": {"status": "ACTIVE", "consent_status": "GRANTED"}}
    sessions = {"S01/session_001": {"status": "ACTIVE"}}
    vocabulary = {
        "WATER": {"language": "MSL", "status": "ACTIVE", "human_validation": "VALIDATED"}
    }
    return row, signers, sessions, vocabulary


@pytest.mark.parametrize(
    ("change", "expected_blocker"),
    [
        ({"signer_review_status": "UNKNOWN", "signer_identity_status": "INFERRED_UNCONFIRMED"},
         "SIGNER_CONFIRMATION_REQUIRED"),
        ({"consent": "UNKNOWN"}, "CONSENT_NOT_GRANTED"),
        ({"session": "UNKNOWN"}, "ACTIVE_SESSION_CONFIRMATION_REQUIRED"),
        ({"language_review_status": "UNKNOWN", "language": "UNKNOWN"},
         "MSL_LANGUAGE_CONFIRMATION_REQUIRED"),
        ({"label_review_status": "NEEDS_REVIEW", "label_validation_state": "UNVALIDATED"},
         "VOCABULARY_VALIDATION_REQUIRED"),
        ({"recording_quality_status": "REJECTED"}, "RECORDING_QUALITY_APPROVAL_REQUIRED"),
        ({"human_review_status": "REJECTED", "review_status": "REJECTED"},
         "HUMAN_REVIEW_APPROVAL_REQUIRED"),
        ({"source_hash_changed": "true"}, "SOURCE_HASH_MISMATCH"),
        ({"technical_qc_status": "VIDEO_INVALID"}, "TECHNICAL_QC_FAILED"),
    ],
)
def test_every_processing_gate_blocks_when_missing_or_rejected(tmp_path, change, expected_blocker):
    row, signers, sessions, vocabulary = _eligible_import_row(tmp_path / "S01" / "WATER" / "v.mp4")
    overrides = dict(change)
    if "consent" in overrides:
        signers["S01"]["consent_status"] = overrides.pop("consent")
    if "session" in overrides:
        sessions["S01/session_001"]["status"] = overrides.pop("session")
    row.update(overrides)

    blockers = importer._processing_blockers(row, signers, sessions, vocabulary)

    assert expected_blocker in blockers
    assert importer._final_status(row, signers, sessions, vocabulary) != "READY_FOR_PROCESSING"


def test_all_processing_gates_pass_only_with_validated_vocabulary(tmp_path):
    row, signers, sessions, vocabulary = _eligible_import_row(
        tmp_path / "S01" / "WATER" / "v.mp4"
    )

    assert importer._processing_blockers(row, signers, sessions, vocabulary) == []
    assert importer._final_status(row, signers, sessions, vocabulary) == "READY_FOR_PROCESSING"


def test_label_confirmation_without_vocabulary_validation_remains_pending(tmp_path, monkeypatch):
    source = tmp_path / "source"
    video = source / "S01" / "WATER" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"unvalidated vocabulary")
    manifests = tmp_path / "manifests"
    review = tmp_path / "review"
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (_ for _ in ()).throw(
        RuntimeError("asset unavailable")
    ))
    importer.import_existing_recordings(
        [source], manifests, review, _config_file(tmp_path), check_landmarks=False
    )
    rows = importer._read_csv(manifests / importer.IMPORT_FILENAME)
    rows[0].update({
        "landmark_status": "CHECKED",
        "hand_detected_frames": "15",
        "video_validation_status": "READABLE",
        "technical_qc_status": "READY_FOR_REVIEW",
        "finite_model_input": "true",
    })
    importer._write_csv(manifests / importer.IMPORT_FILENAME, list(rows[0]), rows)

    result = importer.update_existing_recording_review(
        rows[0]["recording_id"],
        None,
        "reviewer-01",
        "Candidate label was observed but vocabulary is not yet validated.",
        manifests,
        review,
        label_review_status="CONFIRMED",
        confirmed_sign="WATER",
    )
    saved = importer._read_csv(manifests / importer.IMPORT_FILENAME)[0]

    assert saved["sign_candidate"] == "WATER"
    assert saved["sign"] == "WATER"
    assert saved["label_review_status"] == "CONFIRMED"
    assert saved["label_validation_state"] == "VOCABULARY_VALIDATION_REQUIRED"
    assert saved["validation_status"] == "METADATA_INCOMPLETE"
    assert "VOCABULARY_VALIDATION_REQUIRED" in result["processing_blockers"]
    assert not (manifests / "sign_vocabulary.json").exists()


def test_reimport_preserves_explicit_review_decisions(tmp_path, monkeypatch):
    source = tmp_path / "source"
    video = source / "S01" / "WATER" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"review preservation fixture")
    manifests = tmp_path / "manifests"
    review = tmp_path / "review"
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(importer, "_create_holistic_detector", lambda _config: (_ for _ in ()).throw(
        RuntimeError("asset unavailable")
    ))
    config = _config_file(tmp_path)
    importer.import_existing_recordings(
        [source], manifests, review, config, check_landmarks=False
    )
    recording_id = importer._read_csv(manifests / importer.IMPORT_FILENAME)[0]["recording_id"]
    importer.update_existing_recording_review(
        recording_id,
        None,
        "reviewer-01",
        "Reviewed but needs another label review.",
        manifests,
        review,
        signer_review_status="UNKNOWN",
        label_review_status="NEEDS_REVIEW",
        language_review_status="UNKNOWN",
        recording_quality_status="NEEDS_RETAKE",
        human_review_status="NEEDS_REVIEW",
    )

    importer.import_existing_recordings(
        [source], manifests, review, config, check_landmarks=False
    )
    saved = importer._read_csv(manifests / importer.IMPORT_FILENAME)[0]

    assert saved["signer_review_status"] == "UNKNOWN"
    assert saved["label_review_status"] == "NEEDS_REVIEW"
    assert saved["language_review_status"] == "UNKNOWN"
    assert saved["recording_quality_status"] == "NEEDS_RETAKE"
    assert saved["human_review_status"] == "NEEDS_REVIEW"
    assert saved["review_status"] == "NEEDS_RETAKE"
