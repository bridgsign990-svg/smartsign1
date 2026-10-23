import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.msl_dataset.msl_dataset_pipeline import (
    MSLSignDataset,
    _frame_features,
    discover_videos,
    generate_split_manifest,
)
from tools.msl_dataset.record_video import choose_output_path


def test_discover_videos_extracts_signer_sign_and_video_id(tmp_path):
    video = tmp_path / "S01" / "WATER" / "session_001" / "video_001.mp4"
    video.parent.mkdir(parents=True)
    video.touch()

    discovered = discover_videos(tmp_path)

    assert discovered == [{
        "path": str(video),
        "valid": True,
        "signer_id": "S01",
        "sign": "WATER",
        "video_id": "video_001",
    }]


def test_missing_landmarks_produce_fixed_finite_model_features():
    results = SimpleNamespace(
        left_hand_landmarks=None,
        right_hand_landmarks=[],
        pose_landmarks=None,
        face_landmarks=[],
    )
    config = {
        "left_hand": True,
        "right_hand": True,
        "pose": True,
        "face": True,
        "pose_landmark_subset": list(range(33)),
    }

    model_features, raw_features, presence = _frame_features(results, [0, 1], config, "wrist_relative")

    assert model_features.shape == (268,)
    assert np.isfinite(model_features).all()
    assert np.isnan(raw_features).all()
    assert presence.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_recorder_path_preserves_session_and_recording_number(tmp_path):
    path = choose_output_path(tmp_path, "S01", "WATER", "session_002", 7)

    assert path == tmp_path / "S01" / "WATER" / "session_002" / "video_007.mp4"
    with pytest.raises(ValueError):
        choose_output_path(tmp_path, "../S01", "WATER", "session_002", 1)


def test_signer_split_is_disjoint_and_rejects_overlap(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    rows = [
        {
            "sequence_id": f"S{number:02d}_WATER_video_001",
            "sign": "WATER",
            "signer_id": f"S{number:02d}",
            "video_id": "video_001",
            "session_id": "session_001",
            "sequence_path": "",
            "frame_count": 30,
            "feature_count": 8,
            "processing_status": "PROCESSED",
        }
        for number in range(1, 11)
    ]
    with (manifests / "sequences.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split = generate_split_manifest(
        manifests,
        {"train_signers": [], "validation_signers": [], "test_signers": []},
    )
    train, validation, test = map(set, (
        split["train_signers"],
        split["validation_signers"],
        split["test_signers"],
    ))
    assert len(train) == 7
    assert validation == {"S08"}
    assert test == {"S09", "S10"}
    assert not (train & validation or train & test or validation & test)

    with pytest.raises(ValueError, match="Signer leakage"):
        generate_split_manifest(
            manifests,
            {"train_signers": ["S01"], "validation_signers": ["S01"], "test_signers": []},
        )


def test_split_rejects_shared_source_across_signers(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    rows = [
        {
            "sequence_id": f"sequence_{signer}",
            "sign": "WATER",
            "signer_id": signer,
            "video_id": "video_001",
            "session_id": "session_001",
            "sequence_path": "",
            "source_sha256": "shared-source",
        }
        for signer in ("S01", "S02", "S03")
    ]
    with (manifests / "sequences.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="source_sha256 leakage"):
        generate_split_manifest(
            manifests,
            {"train_signers": ["S01"], "validation_signers": ["S02"], "test_signers": ["S03"]},
        )


def test_default_split_persists_existing_signer_assignments(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    rows = [
        {
            "sequence_id": f"sequence_{signer}",
            "sign": "WATER",
            "signer_id": signer,
            "video_id": "video_001",
            "session_id": "session_001",
            "sequence_path": "",
        }
        for signer in ("S01", "S02", "S03")
    ]
    sequence_manifest = manifests / "sequences.csv"
    with sequence_manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    initial = generate_split_manifest(manifests, {
        "train_signers": [],
        "validation_signers": [],
        "test_signers": [],
    })
    persisted = {
        signer: group
        for group, key in (
            ("train", "train_signers"),
            ("validation", "validation_signers"),
            ("test", "test_signers"),
        )
        for signer in initial[key]
    }
    rows.append({
        "sequence_id": "sequence_S04",
        "sign": "WATER",
        "signer_id": "S04",
        "video_id": "video_001",
        "session_id": "session_001",
        "sequence_path": "",
    })
    with sequence_manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    updated = generate_split_manifest(manifests, {
        "train_signers": [],
        "validation_signers": [],
        "test_signers": [],
    })

    for group, key in (
        ("train", "train_signers"),
        ("validation", "validation_signers"),
        ("test", "test_signers"),
    ):
        assert all(signer in updated[key] for signer, prior_group in persisted.items() if prior_group == group)


def test_pytorch_dataset_loads_sequence_and_metadata(tmp_path):
    sample = tmp_path / "sample.npz"
    np.savez_compressed(
        sample,
        sequence=np.ones((30, 8), dtype=np.float32),
        label=np.asarray("WATER", dtype="<U16"),
        signer_id=np.asarray("S01", dtype="<U32"),
        video_id=np.asarray("video_001", dtype="<U64"),
        session_id=np.asarray("session_001", dtype="<U64"),
    )
    manifest = tmp_path / "sequences.csv"
    row = {
        "sequence_id": "S01_WATER_video_001",
        "sign": "WATER",
        "signer_id": "S01",
        "video_id": "video_001",
        "session_id": "session_001",
        "sequence_path": str(sample),
    }
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    dataset = MSLSignDataset(manifest)
    features, label, metadata = dataset[0]

    assert tuple(features.shape) == (30, 8)
    assert label == 0
    assert metadata == {"signer_id": "S01", "video_id": "video_001", "session_id": "session_001"}
