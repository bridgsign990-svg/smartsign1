from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tools.experimental_dynamic import (
    baseline,
    wlasl_v2,
    wlasl_v2_benchmark,
    wlasl_v2_live_demo,
)


def test_manifest_uses_first_hundred_official_glosses_and_preserves_metadata(tmp_path, monkeypatch):
    metadata = [
        {
            "gloss": f"gloss_{index:03d}",
            "instances": [{
                "video_id": f"video_{index:03d}",
                "signer_id": index,
                "source": "official",
                "split": "train",
                "url": f"https://example.org/{index:03d}.mp4",
                "fps": 25,
                "frame_start": 1,
                "frame_end": -1,
            }],
        }
        for index in range(100)
    ]
    metadata_path = tmp_path / "WLASL_v0.3.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    root = tmp_path / "wlasl_v2"
    monkeypatch.setattr(wlasl_v2, "DATA_ROOT", root)
    monkeypatch.setattr(wlasl_v2, "MANIFEST_PATH", root / "download_manifest.json")
    monkeypatch.setattr(wlasl_v2, "METADATA_PATH", metadata_path)

    manifest_path = wlasl_v2.build_manifest(100)
    result = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["subset"] == "WLASL100"
    assert [item["gloss"] for item in result["glosses"]] == [
        f"gloss_{index:03d}" for index in range(100)
    ]
    assert result["records"][0]["signer_id"] == "0"
    assert result["records"][0]["source_url"] == "https://example.org/000.mp4"
    assert result["records"][0]["download_status"] == "PENDING"
    assert manifest_path.is_relative_to(root)
    with pytest.raises(FileExistsError):
        wlasl_v2.build_manifest(100)


def test_manifest_atomic_replace_retries_transient_windows_file_locks(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(wlasl_v2, "MANIFEST_PATH", manifest_path)
    original_replace = wlasl_v2.os.replace
    replace_calls = 0

    def replace_with_one_transient_lock(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise PermissionError("transient file lock")
        original_replace(source, destination)

    monkeypatch.setattr(wlasl_v2.os, "replace", replace_with_one_transient_lock)
    monkeypatch.setattr(wlasl_v2.time, "sleep", lambda _seconds: None)

    wlasl_v2._persist_manifest({"dataset": "WLASL"})

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"dataset": "WLASL"}
    assert replace_calls == 2
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "reason",
    [
        "HTTPError: HTTP Error 404: Not Found",
        "HTTPError: HTTP Error 403: Forbidden",
        "hostname mismatch",
        "Server returned HTML (text/html), not a video.",
        "yt-dlp exit=1: ERROR: this video is unavailable",
    ],
)
def test_permanent_video_failures_are_not_retried(reason):
    assert not wlasl_v2._retryable_download_failure({"failure_reason": reason})


def test_transient_download_failures_remain_retryable():
    assert wlasl_v2._retryable_download_failure({
        "failure_reason": "yt-dlp timed out after 35 seconds."
    })


def test_duplicate_official_instances_count_as_one_downloaded_source_video():
    records = [
        {"class_index": 2, "video_id": "123", "download_status": "DOWNLOADED"},
        {"class_index": 2, "video_id": "123", "download_status": "DOWNLOADED"},
        {"class_index": 2, "video_id": "124", "download_status": "DOWNLOADED"},
        {"class_index": 3, "video_id": "123", "download_status": "DOWNLOADED"},
    ]

    unique = wlasl_v2._unique_downloaded_records(records)

    assert len(unique) == 3


def test_preprocessing_subset_ranks_verified_video_and_signer_coverage():
    rows = []
    for index, (gloss, video_count, signer_count) in enumerate(
        (("official_a", 2, 2), ("official_b", 3, 2), ("official_c", 3, 3))
    ):
        for video_index in range(video_count):
            rows.append({
                "class_index": index,
                "gloss": gloss,
                "video_id": f"{gloss}_video_{video_index}",
                "signer_id": f"{gloss}_signer_{video_index % signer_count}",
                "download_status": "DOWNLOADED",
            })
    rows.append({
        "class_index": 0,
        "gloss": "official_a",
        "video_id": "not_downloaded",
        "signer_id": "another_signer",
        "download_status": "FAILED",
    })

    selected, counts = wlasl_v2._select_preprocessing_classes(rows, 2)

    assert selected == ["official_c", "official_b"]
    assert counts == {
        "official_c": {"videos": 3, "signers": 3},
        "official_b": {"videos": 3, "signers": 2},
    }


def test_signer_split_keeps_each_signer_and_source_video_in_one_partition(tmp_path):
    samples = []
    for class_index, label in enumerate(("book", "chair", "drink")):
        for signer_index in range(12):
            signer = f"signer_{signer_index:02d}"
            video = f"{label}_{signer}"
            for window in range(2):
                samples.append(baseline.Sample(
                    sample_id=f"{video}_window_{window}",
                    group_id=video,
                    label=label,
                    signer_id=signer,
                    source_split="",
                    sequence_path=tmp_path / f"{video}_{window}.npz",
                    sequence=np.zeros((30, 562), dtype=np.float32),
                    valid_length=30,
                ))

    rows, summary = wlasl_v2._split_samples(samples, seed=17)

    assert summary["signer_separation"] == "AVAILABLE"
    assert summary["signer_overlap"] is False
    split_by_signer = {}
    split_by_video = {}
    for row in rows:
        split_by_signer.setdefault(row["signer_id"], set()).add(row["split"])
        split_by_video.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_signer.values())
    assert all(len(splits) == 1 for splits in split_by_video.values())
    assert summary["split_seed"] >= 17


def test_video_metrics_aggregate_windows_to_source_video():
    result = wlasl_v2._video_predictions(
        [0, 0, 1],
        [[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]],
        [
            {"group_id": "video_a", "label": "book", "signer_id": "s1"},
            {"group_id": "video_a", "label": "book", "signer_id": "s1"},
            {"group_id": "video_b", "label": "chair", "signer_id": "s2"},
        ],
        ["book", "chair"],
    )

    assert result["videos"] == 2
    assert len(result["records"]) == 2
    assert result["records"][0]["predicted_class"] == "book"
    assert result["records"][0]["windows"] == 2


def _feature_config():
    return {
        "face_landmark_indices": list(range(100)),
        "feature_selection": {
            "left_hand": True,
            "right_hand": True,
            "pose": True,
            "face": True,
            "pose_landmark_subset": list(range(33)),
        },
    }


def test_representation_v2_dimensions_and_missing_point_masks():
    sequence = np.zeros((4, 562), dtype=np.float32)
    raw = np.zeros((4, 558), dtype=np.float32)
    presence = np.ones((4, 4), dtype=np.float32)
    valid_frames = np.ones(4, dtype=np.float32)
    sequence[0, 0] = 1.0
    sequence[2, 0] = 3.0
    sequence[3, 0] = 4.0
    raw[0, 0] = 1.0
    raw[1, 0:3] = np.nan
    raw[2, 0] = 3.0
    raw[3, 0] = 4.0
    sequence[1, 0:3] = 0.0

    result = wlasl_v2_benchmark.representation_v2(
        sequence, raw, presence, valid_frames, _feature_config()
    )

    assert result.shape == (4, 912)
    assert np.isfinite(result).all()
    assert result[1, 0] == pytest.approx(2.0)
    # The point mask stays missing after short-gap interpolation.
    assert result[1, 558] == 0.0
    # A temporal difference is present for adjacent observed coordinates.
    assert result[3, 733] == pytest.approx(1.0)


def test_representation_v2_masks_padding_and_rejects_incompatible_dimensions():
    sequence = np.zeros((3, 562), dtype=np.float32)
    raw = np.zeros((3, 558), dtype=np.float32)
    presence = np.ones((3, 4), dtype=np.float32)
    valid_frames = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    output = wlasl_v2_benchmark.representation_v2(
        sequence, raw, presence, valid_frames, _feature_config()
    )
    group_flags_start = 558 + 175 + 175
    assert np.all(output[2] == 0.0)
    assert np.all(output[:, group_flags_start:] <= 1.0)
    with pytest.raises(ValueError, match="arrays must be"):
        wlasl_v2_benchmark.representation_v2(
            np.zeros((3, 15), dtype=np.float32),
            np.zeros((3, 11), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
            np.ones(3, dtype=np.float32),
            _feature_config(),
        )


def test_benchmark_video_metrics_use_saved_label_order_and_video_aggregation():
    result = wlasl_v2_benchmark._video_metrics(
        [
            {"group_id": "0:v1", "video_id": "v1", "class_index": "0",
             "signer_id": "s1", "label": "book"},
            {"group_id": "0:v1", "video_id": "v1", "class_index": "0",
             "signer_id": "s1", "label": "book"},
            {"group_id": "1:v2", "video_id": "v2", "class_index": "1",
             "signer_id": "s2", "label": "chair"},
        ],
        np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]], dtype=np.float32),
        ["book", "chair"],
    )
    assert result["videos"] == 2
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["predictions"][0]["predicted_class"] == "book"
    assert result["predictions"][0]["windows"] == 2


def test_attention_model_accepts_v2_representation():
    import torch

    model = wlasl_v2_benchmark.build_torch_model("bilstm_attention", 912, 10)
    logits = model(torch.zeros((2, 30, 912)), torch.tensor([30, 18]))
    assert logits.shape == (2, 10)
    assert wlasl_v2_benchmark.build_torch_model("bilstm", 912, 10).training


def test_torch_benchmark_checkpoint_loads_its_declared_architecture(tmp_path):
    import torch

    model = wlasl_v2_benchmark.build_torch_model("bilstm_attention", 912, 2)
    checkpoint_path = tmp_path / "attention.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_kind": "bilstm_attention",
        "input_size": 912,
        "labels": ["book", "chair"],
    }, checkpoint_path)
    loaded, metadata = wlasl_v2_benchmark.load_torch_checkpoint(checkpoint_path)
    assert metadata["model_kind"] == "bilstm_attention"
    assert loaded(torch.zeros((1, 30, 912)), torch.tensor([30])).shape == (1, 2)


def test_keras_attention_model_consumes_sequence_mask():
    import tensorflow as tf

    model = wlasl_v2_benchmark._build_keras_model(912, 3, attention=True)
    logits = model(
        [np.zeros((2, 30, 912), dtype=np.float32), np.asarray([30, 18], dtype=np.int32)],
        training=False,
    )
    assert tuple(logits.shape) == (2, 3)


def test_v2_demo_rejects_checkpoints_outside_the_asl_experiment():
    checkpoint = {
        "dataset": "WLASL v0.3 top-100 (WLASL V2)",
        "language": "ASL",
        "experiment_tag": baseline.LSA64_EXPERIMENT_TAG,
        "sequence_length": 30,
        "input_size": 912,
        "representation": "representation_v2",
    }
    with patch.object(
        wlasl_v2_live_demo.wlasl_v2_benchmark,
        "load_torch_checkpoint",
        return_value=(object(), checkpoint),
    ):
        with pytest.raises(ValueError, match="isolated WLASL ASL experiment"):
            wlasl_v2_live_demo.load_v2_model(Path("model.pt"))


def test_v2_prediction_uses_saved_labels_and_returns_low_confidence():
    class Model:
        def eval(self):
            return self

        def __call__(self, values, lengths):
            import torch
            return torch.tensor([[0.0, 0.0, 0.0]])

    checkpoint = {
        "input_size": 912,
        "label_to_index": {
            "official_gloss_a": 0,
            "official_gloss_b": 1,
            "official_gloss_c": 2,
        },
    }
    label, confidence = wlasl_v2_live_demo.predict_sequence(
        Model(), checkpoint, "cpu", np.zeros((30, 912), dtype=np.float32)
    )
    assert label == "official_gloss_a"
    assert 0.0 < confidence < wlasl_v2_live_demo.DISPLAY_CONFIDENCE_THRESHOLD
    with pytest.raises(ValueError, match="Expected a \\[30, 912\\] sequence"):
        wlasl_v2_live_demo.predict_sequence(
            Model(), checkpoint, "cpu", np.zeros((15, 912), dtype=np.float32)
        )


def test_v2_live_transform_matches_benchmark_representation_and_keeps_masks():
    raw = np.zeros((30, 558), dtype=np.float32)
    raw[:, :3] = np.nan
    transformed = wlasl_v2_live_demo.transform_live_buffer(
        np.zeros((30, 562), dtype=np.float32),
        raw,
        np.ones((30, 4), dtype=np.float32),
        _feature_config(),
    )
    assert transformed.shape == (30, 912)
    assert np.all(transformed[:, 558] == 0.0)
