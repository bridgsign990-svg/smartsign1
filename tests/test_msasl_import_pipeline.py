import csv
import hashlib
import json
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from tools.msasl_dataset.import_msasl import (
    DATASET_ROOT,
    MANIFEST_FIELDS,
    MSASLDataset,
    SEQUENCE_FIELDS,
    build_split_manifest,
    download_selected,
    extract_temporal_clip,
    generate_reports,
    import_metadata,
    process_selected,
    select_records,
    validate_processed_dataset,
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_row(**changes):
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update({
        "dataset": "MS-ASL",
        "source_split": "train",
        "source_record_id": "train_0000000_abcd1234",
        "video_id": "video001",
        "source_url": "https://example.invalid/video001.mp4",
        "label_id": "0",
        "label": "HELLO",
        "signer_id": "S01",
        "start_time": "0.0",
        "end_time": "1.0",
        "download_status": "PENDING",
        "processing_status": "PENDING",
    })
    row.update(changes)
    return row


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_sample(path: Path, sequence: np.ndarray, signer: str = "S01", label: str = "HELLO") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sequence=sequence,
        raw_sequence=np.zeros((sequence.shape[0], 8), dtype=np.float32),
        landmark_presence=np.ones((sequence.shape[0], 2), dtype=np.float32),
        feature_group_names=np.asarray(["left_hand", "pose"], dtype="<U16"),
        dataset=np.asarray("MS-ASL", dtype="<U16"),
        language=np.asarray("American Sign Language (ASL)", dtype="<U64"),
        label=np.asarray(label, dtype="<U256"),
        label_id=np.asarray("0", dtype="<U64"),
        signer_id=np.asarray(signer, dtype="<U128"),
        video_id=np.asarray("video001", dtype="<U256"),
        source_split=np.asarray("train", dtype="<U16"),
        source_record_id=np.asarray("train_0000000_abcd1234", dtype="<U64"),
        start_time=np.asarray(0, dtype=np.float32),
        end_time=np.asarray(1, dtype=np.float32),
    )


def test_metadata_zip_parsing_and_split_normalization(tmp_path):
    archive_path = tmp_path / "msasl_metadata.zip"
    train = [{
        "url": "https://www.youtube.com/watch?v=vid001",
        "label": 0,
        "clean_text": "HELLO",
        "signer_id": 4,
        "start_time": 1.25,
        "end_time": 2.5,
        "width": 640,
        "height": 360,
        "fps": 25,
    }]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("MS-ASL/MSASL_train.json", json.dumps(train))
        archive.writestr("MS-ASL/MSASL_val.json", json.dumps([]))
        archive.writestr("MS-ASL/MSASL_test.json", json.dumps([]))
        archive.writestr("MS-ASL/MSASL_classes.json", json.dumps(["HELLO"]))
        archive.writestr("MS-ASL/LICENSE.txt", "Example license information")

    result = import_metadata(archive_path, limit=10, dataset_root=tmp_path / "external")
    rows = list(csv.DictReader((tmp_path / "external" / "manifests" / "import_manifest.csv").open(encoding="utf-8")))

    assert result["metadata_records"] == result["selected_records"] == 1
    assert rows[0]["dataset"] == "MS-ASL"
    assert rows[0]["source_split"] == "train"
    assert rows[0]["video_id"] == "vid001"
    assert rows[0]["label"] == "HELLO"
    assert rows[0]["signer_id"] == "4"
    assert rows[0]["start_time"] == "1.25"
    assert (tmp_path / "external" / "metadata" / "MSASL_train.json").is_file()
    assert "Example license" in result["license_information"]


def test_metadata_missing_fields_are_safe(tmp_path):
    source = tmp_path / "metadata"
    source.mkdir()
    (source / "MSASL_train.json").write_text(json.dumps([{}]), encoding="utf-8")

    import_metadata(source, dataset_root=tmp_path / "external")
    row = next(csv.DictReader((tmp_path / "external" / "manifests" / "import_manifest.csv").open(encoding="utf-8")))

    for key in ("video_id", "source_url", "label_id", "label", "signer_id", "start_time", "end_time", "width", "height", "fps"):
        assert row[key] == ""
    assert row["download_status"] == "PENDING"


def test_duplicate_records_remain_distinguishable(tmp_path):
    source = tmp_path / "metadata"
    source.mkdir()
    record = {"url": "https://youtu.be/repeated", "label": 0, "text": "HELLO"}
    (source / "MSASL_train.json").write_text(json.dumps([record, record]), encoding="utf-8")

    import_metadata(source, dataset_root=tmp_path / "external")
    rows = list(csv.DictReader((tmp_path / "external" / "manifests" / "import_manifest.csv").open(encoding="utf-8")))

    assert [row["video_id"] for row in rows] == ["repeated", "repeated"]
    assert len({row["source_record_id"] for row in rows}) == 2


def test_sample_limit_is_deterministic_and_respects_limit():
    rows = [
        {"source_split": split, "source_record_id": f"{split}_{index}"}
        for split, count in (("train", 20), ("val", 5), ("test", 5))
        for index in range(count)
    ]

    first = select_records(rows, 10)
    second = select_records(rows, 10)

    assert len(first) == 10
    assert [row["source_record_id"] for row in first] == [row["source_record_id"] for row in second]
    assert {row["source_split"] for row in first} == {"train", "val", "test"}


def test_download_resume_uses_local_file_without_request(tmp_path):
    root = tmp_path / "external"
    local = root / "raw" / "existing.mp4"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"existing")
    manifest = root / "manifests" / "import_manifest.csv"
    row = _manifest_row(local_video_path=str(local), download_status="DOWNLOADED")
    _write_csv(manifest, MANIFEST_FIELDS, [row])

    with patch("tools.msasl_dataset.import_msasl.urllib.request.urlopen") as opener:
        result = download_selected(manifest, retries=0)

    assert result == {"ALREADY_EXISTS": 1}
    opener.assert_not_called()
    updated = next(csv.DictReader(manifest.open(encoding="utf-8")))
    assert updated["download_status"] == "ALREADY_EXISTS"


def test_failed_download_is_recorded_and_does_not_raise(tmp_path):
    root = tmp_path / "external"
    manifest = root / "manifests" / "import_manifest.csv"
    _write_csv(manifest, MANIFEST_FIELDS, [_manifest_row()])
    error = urllib.error.HTTPError("https://example.invalid/video.mp4", 404, "Not Found", {}, None)

    with patch("tools.msasl_dataset.import_msasl.urllib.request.urlopen", side_effect=error):
        counts = download_selected(manifest, retries=1)

    row = next(csv.DictReader(manifest.open(encoding="utf-8")))
    assert counts == {"UNAVAILABLE": 1}
    assert row["download_status"] == "UNAVAILABLE"
    assert row["download_error"].startswith("HTTP 404")


def test_temporal_clip_extracts_requested_segment(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((24, 32, 3), index * 5, dtype=np.uint8))
    writer.release()

    result = extract_temporal_clip(source, tmp_path / "clips" / "clip.mp4", 1.0, 2.0)

    assert result["status"] == "EXTRACTED"
    assert 9 <= result["frames"] <= 12
    assert Path(result["clip_path"]).is_file()
    assert result["duration"] <= 1.2


def test_invalid_temporal_clip_fails_without_writing_destination(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for _ in range(20):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    destination = tmp_path / "clips" / "invalid.mp4"

    with pytest.raises(ValueError, match="Invalid temporal segment"):
        extract_temporal_clip(source, destination, 2.0, 1.0)

    assert not destination.exists()


def test_processed_sequence_validation_catches_nan_inf_and_feature_width(tmp_path):
    root = tmp_path / "external"
    sample_path = root / "processed" / "train" / "broken.npz"
    sequence = np.ones((30, 8), dtype=np.float32)
    sequence[0, 0] = np.nan
    sequence[1, 0] = np.inf
    _write_sample(sample_path, sequence)
    row = {
        "sample_id": "broken",
        "dataset": "MS-ASL",
        "language": "American Sign Language (ASL)",
        "source_record_id": "record",
        "source_split": "train",
        "video_id": "video001",
        "source_url": "",
        "label_id": "0",
        "label": "HELLO",
        "signer_id": "S01",
        "start_time": "0",
        "end_time": "1",
        "sequence_path": str(sample_path),
        "frame_count": "30",
        "feature_count": "8",
        "processing_status": "PROCESSED",
    }
    _write_csv(root / "manifests" / "sequences.csv", SEQUENCE_FIELDS, [row])
    _write_csv(root / "manifests" / "import_manifest.csv", MANIFEST_FIELDS, [_manifest_row(processing_status="PROCESSED")])

    result = validate_processed_dataset(root, expected_feature_count=10)

    assert result["counts"]["FAILED"] == 1
    assert any("NaN or Inf" in reason for reason in result["failed"][0]["reasons"])
    assert any("does not match expected 10" in reason for reason in result["failed"][0]["reasons"])
    assert (root / "review" / "msasl_review_candidates.json").is_file()


def test_msasl_dataset_loads_asl_only_sequence(tmp_path):
    root = tmp_path / "external"
    sample_path = root / "processed" / "train" / "sample.npz"
    _write_sample(sample_path, np.ones((30, 8), dtype=np.float32))
    row = {
        "sample_id": "sample",
        "dataset": "MS-ASL",
        "language": "American Sign Language (ASL)",
        "source_record_id": "record",
        "source_split": "train",
        "video_id": "video001",
        "source_url": "",
        "label_id": "0",
        "label": "HELLO",
        "signer_id": "S01",
        "start_time": "0",
        "end_time": "1",
        "sequence_path": str(sample_path),
        "frame_count": "30",
        "feature_count": "8",
        "processing_status": "PROCESSED",
    }
    manifest = root / "manifests" / "sequences.csv"
    _write_csv(manifest, SEQUENCE_FIELDS, [row])

    dataset = MSASLDataset(manifest)
    sequence, label, metadata = dataset[0]

    assert tuple(sequence.shape) == (30, 8)
    assert label == 0
    assert metadata["dataset"] == "MS-ASL"
    assert metadata["language"] == "American Sign Language (ASL)"
    assert metadata["source_split"] == "train"


def test_official_signer_overlap_is_reported_not_changed(tmp_path):
    manifest = tmp_path / "external" / "manifests" / "import_manifest.csv"
    rows = [
        _manifest_row(source_split="train", source_record_id="tr1", signer_id="SAME"),
        _manifest_row(source_split="val", source_record_id="va1", signer_id="SAME"),
        _manifest_row(source_split="test", source_record_id="te1", signer_id="OTHER"),
    ]
    _write_csv(manifest, MANIFEST_FIELDS, rows)

    result = build_split_manifest(manifest)

    assert result["official_splits_preserved"] is True
    assert result["signer_leakage_detected"] is True
    assert result["signer_overlap"]["train_val"] == ["SAME"]
    assert result["signers_by_split"]["train"] == ["SAME"]
    assert result["signers_by_split"]["val"] == ["SAME"]


def test_duplicate_sample_ids_and_inconsistent_dimensions_are_failed(tmp_path):
    root = tmp_path / "external"
    sample_a = root / "processed" / "train" / "a.npz"
    sample_b = root / "processed" / "train" / "b.npz"
    _write_sample(sample_a, np.ones((30, 8), dtype=np.float32))
    _write_sample(sample_b, np.ones((30, 9), dtype=np.float32))
    rows = []
    for path in (sample_a, sample_b):
        rows.append({
            "sample_id": "duplicate",
            "dataset": "MS-ASL",
            "language": "American Sign Language (ASL)",
            "source_record_id": "record",
            "source_split": "train",
            "video_id": "video001",
            "source_url": "",
            "label_id": "0",
            "label": "HELLO",
            "signer_id": "S01",
            "start_time": "0",
            "end_time": "1",
            "sequence_path": str(path),
            "frame_count": "30",
            "feature_count": str(np.load(path)["sequence"].shape[1]),
            "processing_status": "PROCESSED",
        })
    _write_csv(root / "manifests" / "sequences.csv", SEQUENCE_FIELDS, rows)
    _write_csv(root / "manifests" / "import_manifest.csv", MANIFEST_FIELDS, [_manifest_row(processing_status="PROCESSED")])

    result = validate_processed_dataset(root)

    assert result["duplicate_sample_ids"] == ["duplicate"]
    assert any(issue["type"] == "INCONSISTENT_FEATURE_DIMENSIONS" for issue in result["issues"])
    assert result["counts"]["FAILED"] == 1


def test_corrupt_source_video_and_missing_label_are_reported(tmp_path):
    root = tmp_path / "external"
    corrupt = root / "raw" / "broken.mp4"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not a video")
    rows = [
        _manifest_row(local_video_path=str(corrupt), download_status="DOWNLOADED", processing_status="PENDING"),
        _manifest_row(source_record_id="missing_label", label="", signer_id="", video_id="missing-metadata", processing_status="PENDING"),
    ]
    _write_csv(root / "manifests" / "import_manifest.csv", MANIFEST_FIELDS, rows)
    _write_csv(root / "manifests" / "sequences.csv", SEQUENCE_FIELDS, [])

    result = validate_processed_dataset(root)

    reasons = [entry.get("reason", "") for entry in result["failed"]]
    assert any("Corrupt or unreadable source video" in reason for reason in reasons)
    assert any("Missing label" in reason for reason in reasons)
    assert any(entry.get("video_id") == "missing-metadata" for entry in result["warning"])


def test_duplicate_video_ids_are_warned(tmp_path):
    root = tmp_path / "external"
    rows = [
        _manifest_row(source_record_id=f"record-{i}", video_id="duplicate-video", processing_status="PENDING")
        for i in range(2)
    ]
    _write_csv(root / "manifests" / "import_manifest.csv", MANIFEST_FIELDS, rows)
    _write_csv(root / "manifests" / "sequences.csv", SEQUENCE_FIELDS, [])

    result = validate_processed_dataset(root)

    assert result["duplicate_video_ids"] == ["duplicate-video"]
    assert any(issue["type"] == "DUPLICATE_VIDEO_ID" for issue in result["issues"])


def test_padding_is_excluded_from_missing_landmark_rates(tmp_path):
    root = tmp_path / "external"
    sample_path = root / "processed" / "train" / "padded.npz"
    sample_path.parent.mkdir(parents=True)
    valid = np.concatenate([np.ones(3, dtype=np.float32), np.zeros(27, dtype=np.float32)])
    np.savez_compressed(
        sample_path,
        sequence=np.ones((30, 8), dtype=np.float32),
        raw_sequence=np.zeros((30, 8), dtype=np.float32),
        landmark_presence=np.concatenate([np.ones((3, 2)), np.zeros((27, 2))]).astype(np.float32),
        valid_frame_mask=valid,
        feature_group_names=np.asarray(["left_hand", "pose"], dtype="<U16"),
    )
    row = {
        "sample_id": "padded",
        "source_split": "train",
        "video_id": "video001",
        "label": "HELLO",
        "signer_id": "S01",
        "sequence_path": str(sample_path),
        "frame_count": "30",
        "feature_count": "8",
    }
    _write_csv(root / "manifests" / "sequences.csv", SEQUENCE_FIELDS, [row])
    _write_csv(root / "manifests" / "import_manifest.csv", MANIFEST_FIELDS, [_manifest_row(processing_status="PROCESSED")])

    result = validate_processed_dataset(root, excessive_missing_rate=0.5)

    assert result["counts"]["VALID"] == 1
    assert result["counts"]["WARNING"] == 0


def test_processing_adapter_preserves_asl_metadata_and_uses_shared_preprocessor(tmp_path, monkeypatch):
    root = tmp_path / "external"
    source = root / "raw" / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"mocked source")
    clip = root / "raw" / "clips" / "clip.mp4"
    row = _manifest_row(local_video_path=str(source), clip_path=str(clip), download_status="DOWNLOADED")
    manifest = root / "manifests" / "import_manifest.csv"
    _write_csv(manifest, MANIFEST_FIELDS, [row])

    def fake_extract(source_path, destination, start_time, end_time, force=False):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"mocked clip")
        return {"clip_path": str(destination), "notes": "", "status": "EXTRACTED"}

    def fake_shared_processor(video_path, signer_id, sign, config):
        sequence_path = Path(config["processed_dataset_path"]) / "fixture.npz"
        np.savez_compressed(
            sequence_path,
            sequence=np.ones((30, 8), dtype=np.float32),
            raw_sequence=np.ones((30, 8), dtype=np.float32),
            landmark_presence=np.ones((30, 4), dtype=np.float32),
            valid_frame_mask=np.ones(30, dtype=np.float32),
            feature_group_names=np.asarray(["left_hand", "right_hand", "pose", "face"], dtype="<U16"),
        )
        return {"sequence_windows": [{"sequence_path": str(sequence_path)}]}

    monkeypatch.setattr("tools.msasl_dataset.import_msasl.extract_temporal_clip", fake_extract)
    monkeypatch.setattr("tools.msasl_dataset.import_msasl.process_landmark_video", fake_shared_processor)

    result = process_selected(manifest)
    sequences_manifest = root / "manifests" / "sequences.csv"
    dataset = MSASLDataset(sequences_manifest)
    sequence, _, metadata = dataset[0]

    assert result == {"processed_videos": 1, "failed_videos": 0, "sequences": 1}
    assert tuple(sequence.shape) == (30, 8)
    assert metadata["dataset"] == "MS-ASL"
    assert metadata["language"] == "American Sign Language (ASL)"
    assert metadata["source_split"] == "train"
    updated = next(csv.DictReader(manifest.open(encoding="utf-8")))
    assert updated["processing_status"] == "PROCESSED"


def test_report_generation_states_asl_and_never_msl_training(tmp_path):
    root = tmp_path / "external"
    _write_csv(root / "manifests" / "import_manifest.csv", MANIFEST_FIELDS, [_manifest_row(download_status="UNAVAILABLE", processing_status="FAILED", download_error="deleted")])
    _write_csv(root / "manifests" / "sequences.csv", SEQUENCE_FIELDS, [])
    (root / "metadata" / "metadata_source.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "metadata_source.json").write_text(json.dumps({"metadata_records": 1, "source": "fixture", "license_information": ""}), encoding="utf-8")
    validation = validate_processed_dataset(root)

    reports = generate_reports(root, validation)
    report = Path(reports["import_report"]).read_text(encoding="utf-8")

    assert "LANGUAGE:** American Sign Language (ASL)" in report
    assert "This dataset must not be treated as Malawian Sign Language training data." in report
    assert "Videos unavailable: 1" in report
    assert Path(reports["quality_report"]).is_file()


def test_import_isolated_from_msl_and_protected_static_files(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    protected = [project_root / "data" / "dataset.csv", project_root / "models" / "sign_model_light.pkl"]
    before = {path: _hash(path) for path in protected if path.is_file()}
    msl_root = project_root / "data" / "msl_dataset"
    msl_before = {path.relative_to(msl_root): _hash(path) for path in msl_root.rglob("*") if path.is_file()}
    source = tmp_path / "metadata"
    source.mkdir()
    (source / "MSASL_train.json").write_text(json.dumps([{"url": "https://youtu.be/a", "label": 0, "text": "HELLO"}]), encoding="utf-8")

    result = import_metadata(source, dataset_root=tmp_path / "external")

    msl_after = {path.relative_to(msl_root): _hash(path) for path in msl_root.rglob("*") if path.is_file()}
    after = {path: _hash(path) for path in protected if path.is_file()}
    assert DATASET_ROOT.parts[-3:] == ("data", "external_datasets", "msasl")
    assert result["selected_records"] == 1
    assert before == after
    assert msl_before == msl_after
