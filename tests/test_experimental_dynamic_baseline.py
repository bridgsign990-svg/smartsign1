import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.experimental_dynamic.baseline import (
    BiLSTMBaseline,
    _validate_msl_ready_rows,
    create_splits,
    evaluate_baseline,
    load_samples,
    predict_sequence,
    train_baseline,
    validate_experimental_source,
)


def _write_fixture_dataset(root: Path, include_signers: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "sequences.csv"
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for signer_index in range(2):
            signer = f"{split}_signer_{signer_index}" if include_signers else ""
            for label_index, label in enumerate(("HELP", "HELLO")):
                sample_id = f"{split}_{signer_index}_{label}"
                sequence_path = root / f"{sample_id}.npz"
                rng = np.random.default_rng(split_index * 100 + signer_index * 10 + label_index)
                sequence = rng.normal(size=(6, 4)).astype(np.float32)
                sequence += label_index * 1.5
                np.savez_compressed(
                    sequence_path,
                    sequence=sequence,
                    valid_frame_mask=np.asarray([1, 1, 1, 1, 0, 0], dtype=np.float32),
                )
                rows.append({
                    "sample_id": sample_id,
                    "source_record_id": sample_id,
                    "source_split": split,
                    "video_id": sample_id,
                    "signer_id": signer,
                    "label": label,
                    "sequence_path": str(sequence_path),
                })
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_manifest_loads_finite_sequence_and_valid_frame_length(tmp_path):
    manifest = _write_fixture_dataset(tmp_path)

    samples = load_samples(manifest)

    assert samples[0].sequence.shape == (6, 4)
    assert samples[0].valid_length == 4
    assert np.isfinite(samples[0].sequence).all()


def test_signer_and_source_sample_never_cross_splits(tmp_path):
    manifest = _write_fixture_dataset(tmp_path)
    samples = load_samples(manifest)

    split_rows, summary = create_splits(samples)

    assert summary["signer_separation"] == "AVAILABLE"
    assert {row["split"] for row in split_rows} == {"train", "validation", "test"}
    for key in ("signer_id", "group_id"):
        membership = {}
        for row in split_rows:
            membership.setdefault(row[key], set()).add(row["split"])
        assert all(len(value) == 1 for value in membership.values())


def test_missing_signer_metadata_is_reported(tmp_path):
    manifest = _write_fixture_dataset(tmp_path, include_signers=False)
    samples = load_samples(manifest)

    _, summary = create_splits(samples)

    assert summary["signer_separation"] == "SIGNER_SEPARATION_UNAVAILABLE"


def test_official_split_signer_leakage_is_rejected(tmp_path):
    manifest = _write_fixture_dataset(tmp_path)
    with manifest.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    rows[0]["signer_id"] = "LEAKED_SIGNER"
    rows[4]["signer_id"] = "LEAKED_SIGNER"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Signer leakage"):
        create_splits(load_samples(manifest))


def test_model_forward_has_one_logit_vector_per_sample():
    model = BiLSTMBaseline(input_size=4, num_classes=3, hidden_size=8, num_layers=1)

    logits = model(torch.ones((2, 6, 4)), torch.tensor([4, 6]))

    assert tuple(logits.shape) == (2, 3)


def test_train_checkpoint_test_evaluation_and_sequence_inference(tmp_path):
    manifest = _write_fixture_dataset(tmp_path / "dataset")
    result = train_baseline(
        manifest,
        tmp_path / "experiment",
        tmp_path / "experiment" / "runs" / "unit",
        epochs=1,
        batch_size=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )

    assert Path(result["checkpoint"]).is_file()
    assert result["metrics"]["test_samples"] == 4
    evaluation = evaluate_baseline(
        manifest,
        result["checkpoint"],
        tmp_path / "experiment" / "runs" / "unit" / "evaluation",
    )
    prediction = predict_sequence(
        next((tmp_path / "dataset").glob("train_0_HELP.npz")),
        result["checkpoint"],
    )

    assert evaluation["test_samples"] == 4
    assert prediction["predicted_class"] in {"HELP", "HELLO"}
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["experiment_tag"] == "EXPERIMENTAL_ASL_ARCHITECTURE_VALIDATION"
    prediction_path = Path(result["metrics"]["test_predictions_path"])
    assert prediction_path.is_file()
    with prediction_path.open(encoding="utf-8") as predictions_file:
        assert len(list(csv.DictReader(predictions_file))) == 4


def test_msl_experiment_requires_gated_production_sequence_manifest(tmp_path):
    manifest = _write_fixture_dataset(tmp_path)

    with pytest.raises(ValueError, match="MSL experiments must use the existing production"):
        validate_experimental_source(manifest, "MSL")
    with pytest.raises(ValueError, match="Only the production processed-sequences manifest"):
        validate_experimental_source(
            Path("data/msl_dataset/manifests/existing_recordings.csv"), "ASL"
        )


def test_unapproved_msl_recording_cannot_enter_experimental_training(tmp_path, monkeypatch):
    import tools.experimental_dynamic.baseline as baseline

    monkeypatch.setattr(baseline, "PROJECT_ROOT", tmp_path)
    msl_root = tmp_path / "data" / "msl_dataset"
    raw_video = msl_root / "raw" / "S01" / "HELLO" / "video.mp4"
    raw_video.parent.mkdir(parents=True)
    raw_video.write_bytes(b"source fixture")
    processed = msl_root / "processed"
    processed.mkdir(parents=True)
    manifests = msl_root / "manifests"
    manifests.mkdir(parents=True)
    recording = {
        "original_path": str(raw_video),
        "validation_status": "METADATA_INCOMPLETE",
        "metadata_confirmation_status": "PENDING",
        "processing_status": "NOT_PROCESSED",
    }
    with (manifests / "existing_recordings.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(recording))
        writer.writeheader()
        writer.writerow(recording)
    sequence = {
        "processing_status": "PROCESSED",
        "file_path": str(raw_video),
        "sequence_path": str(processed / "candidate.npz"),
        "source_sha256": "not-a-real-hash",
        "sign": "HELLO",
        "signer_id": "S01",
        "session_id": "UNKNOWN",
    }

    with pytest.raises(ValueError, match="has not passed all readiness gates"):
        _validate_msl_ready_rows(manifests / "sequences.csv", [sequence])
