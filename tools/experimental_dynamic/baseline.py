from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - optional for preprocessing-only installations
    torch = None
    nn = None
    Dataset = object

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from tools.msl_dataset.msl_dataset_pipeline import (
    PROJECT_ROOT,
    load_config as load_msl_config,
    process_video,
)
from tools.msl_dataset.collection_management import load_sessions, load_signers
from tools.msl_dataset.import_existing_recordings import _processing_blockers


EXPERIMENT_TAG = "EXPERIMENTAL_ASL_ARCHITECTURE_VALIDATION"
LSA64_EXPERIMENT_TAG = "EXPERIMENTAL_LSA_ARCHITECTURE_VALIDATION"
MSL_EXPERIMENT_TAG = "EXPERIMENTAL_MSL_ARCHITECTURE_VALIDATION"
GENERIC_EXPERIMENT_TAG = "EXPERIMENTAL_DYNAMIC_ARCHITECTURE_VALIDATION"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "external_datasets" / "msasl" / "manifests" / "sequences.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "experimental_dynamic"
DEFAULT_RUN = DEFAULT_OUTPUT / "runs" / "dynamic_baseline"
SPLIT_NAMES = ("train", "validation", "test")
UNKNOWN_SIGNERS = {"", "unknown", "none", "null", "na", "n/a"}


def _experiment_tag(language: str) -> str:
    normalized = language.strip().casefold()
    if normalized in {"asl", "american sign language"}:
        return EXPERIMENT_TAG
    if normalized in {"lsa64", "experimental_lsa_architecture_validation"}:
        return LSA64_EXPERIMENT_TAG
    if normalized in {"msl", "malawian sign language"}:
        return MSL_EXPERIMENT_TAG
    return GENERIC_EXPERIMENT_TAG


@dataclass(frozen=True)
class Sample:
    sample_id: str
    group_id: str
    label: str
    signer_id: str
    source_split: str
    sequence_path: Path
    sequence: np.ndarray
    valid_length: int


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for experimental dynamic training.")


def _resolved(path: str | Path) -> Path:
    value = Path(path)
    return (PROJECT_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_experimental_source(manifest_path: str | Path, language: str) -> Path:
    manifest = _resolved(manifest_path)
    production_msl = (PROJECT_ROOT / "data" / "msl_dataset").resolve()
    production_sequence_manifest = production_msl / "manifests" / "sequences.csv"
    is_msl = language.strip().casefold() in {"msl", "malawian sign language"}
    if is_msl and manifest != production_sequence_manifest:
        raise ValueError(
            "MSL experiments must use the existing production sequences manifest; "
            "its READY_FOR_PROCESSING eligibility gates are rechecked."
        )
    if _is_within(manifest, production_msl) and (
        not is_msl or manifest != production_sequence_manifest
    ):
        raise ValueError("Only the production processed-sequences manifest may be used for a gated MSL experiment.")
    if not manifest.is_file():
        raise FileNotFoundError(f"Experimental sequence manifest not found: {manifest}")
    return manifest


def _validate_msl_ready_rows(manifest: Path, rows: list[dict[str, str]]) -> None:
    msl_root = (PROJECT_ROOT / "data" / "msl_dataset").resolve()
    recordings_path = msl_root / "manifests" / "existing_recordings.csv"
    if not recordings_path.is_file():
        raise ValueError("MSL experiment blocked: existing-recordings validation manifest is missing.")
    with recordings_path.open("r", newline="", encoding="utf-8-sig") as source:
        recording_rows = list(csv.DictReader(source))
    recordings_by_path = {
        str(Path(row.get("original_path", "")).resolve()): row
        for row in recording_rows if row.get("original_path")
    }
    signers = load_signers(msl_root / "manifests")
    sessions = load_sessions(msl_root / "manifests")
    vocabulary_path = msl_root / "manifests" / "sign_vocabulary.json"
    vocabulary = (
        json.loads(vocabulary_path.read_text(encoding="utf-8"))
        if vocabulary_path.is_file() else {}
    )
    processed_root = (msl_root / "processed").resolve()
    raw_root = (msl_root / "raw").resolve()

    for index, row in enumerate(rows, start=2):
        if row.get("processing_status") != "PROCESSED":
            raise ValueError(f"MSL sequence manifest row {index} is not marked PROCESSED.")
        source_path_value = _field(row, "file_path", "original_path")
        sequence_path_value = _field(row, "sequence_path")
        if not source_path_value or not sequence_path_value:
            raise ValueError(f"MSL sequence manifest row {index} is missing source or sequence path.")
        source_path = Path(source_path_value).resolve()
        sequence_path = Path(sequence_path_value).resolve()
        if not _is_within(source_path, raw_root) or not _is_within(sequence_path, processed_root):
            raise ValueError(f"MSL sequence manifest row {index} points outside production raw/processed storage.")
        imported = recordings_by_path.get(str(source_path))
        if not imported:
            raise ValueError(f"MSL source video in sequence manifest row {index} is not in the review manifest.")
        if (
            imported.get("validation_status") != "READY_FOR_PROCESSING"
            or imported.get("metadata_confirmation_status") != "CONFIRMED"
            or imported.get("processing_status") != "PROCESSED"
        ):
            raise ValueError(f"MSL source recording {source_path.name} has not passed all readiness gates.")
        blockers = _processing_blockers(imported, signers, sessions, vocabulary)
        if blockers:
            raise ValueError(
                f"MSL source recording {source_path.name} is blocked: {', '.join(blockers)}"
            )
        source_hash = _sha256_file(source_path)
        if (
            source_hash != imported.get("sha256")
            or source_hash != row.get("source_sha256")
        ):
            raise ValueError(f"MSL source hash mismatch for {source_path.name}.")
        if (
            _field(row, "sign") != imported.get("sign")
            or _field(row, "signer_id") != imported.get("signer_id")
            or _field(row, "session_id") != imported.get("session_id")
        ):
            raise ValueError(f"MSL sequence metadata does not match its approved source recording: {source_path.name}")


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def _normalized_split(value: str) -> str:
    value = value.strip().casefold()
    if value in {"train", "training"}:
        return "train"
    if value in {"val", "valid", "validation", "dev"}:
        return "validation"
    if value in {"test", "testing"}:
        return "test"
    return ""


def load_samples(manifest_path: str | Path, language: str = "ASL") -> list[Sample]:
    manifest = validate_experimental_source(manifest_path, language)
    with manifest.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(
            f"No prepared temporal sequences are listed in {manifest}. "
            "Obtain authorized videos and run the existing Holistic preprocessing before training."
        )
    if language.strip().casefold() in {"msl", "malawian sign language"}:
        _validate_msl_ready_rows(manifest, rows)

    samples: list[Sample] = []
    dimensions: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        label = _field(row, "label", "sign")
        sequence_value = _field(row, "sequence_path")
        if not label or not sequence_value:
            raise ValueError(f"Manifest row {index + 2} must include a label and sequence_path.")
        sequence_path = Path(sequence_value)
        if not sequence_path.is_absolute():
            sequence_path = (manifest.parent / sequence_path).resolve()
        if (
            language.strip().casefold() not in {"msl", "malawian sign language"}
            and _is_within(sequence_path.resolve(), (PROJECT_ROOT / "data" / "msl_dataset").resolve())
        ):
            raise ValueError("Production MSL sequence files cannot be loaded by the experimental baseline.")
        if not sequence_path.is_file():
            raise FileNotFoundError(f"Sequence file listed in row {index + 2} is missing: {sequence_path}")
        sample_id = _field(row, "sample_id", "sequence_id", "source_record_id", "recording_id")
        if not sample_id:
            sample_id = f"sample_{index + 1:06d}"
        group_id = _field(row, "source_record_id", "recording_id", "video_id", "source_video_id")
        if not group_id:
            group_id = sample_id
        signer = _field(row, "signer_id")
        if signer.casefold() in UNKNOWN_SIGNERS:
            signer = ""
        split = _normalized_split(_field(row, "source_split", "split"))

        with np.load(sequence_path, allow_pickle=False) as stored:
            if "sequence" not in stored:
                raise ValueError(f"{sequence_path} has no 'sequence' array.")
            sequence = np.asarray(stored["sequence"], dtype=np.float32)
            mask = np.asarray(stored["valid_frame_mask"], dtype=np.float32) if "valid_frame_mask" in stored else None
        if sequence.ndim != 2 or not sequence.shape[0] or not sequence.shape[1]:
            raise ValueError(f"Sequence must be a non-empty [frames, features] array: {sequence_path}")
        if not np.isfinite(sequence).all():
            raise ValueError(f"Model input contains NaN or Inf: {sequence_path}")
        if mask is not None:
            if mask.shape != (sequence.shape[0],) or not np.isfinite(mask).all():
                raise ValueError(f"valid_frame_mask has invalid dimensions or values: {sequence_path}")
            valid_length = int(np.count_nonzero(mask > 0))
        else:
            valid_length = int(sequence.shape[0])
        if valid_length <= 0:
            raise ValueError(f"Sequence has no unpadded frames: {sequence_path}")
        dimensions.add((int(sequence.shape[0]), int(sequence.shape[1])))
        samples.append(Sample(
            sample_id=sample_id,
            group_id=group_id,
            label=label,
            signer_id=signer,
            source_split=split,
            sequence_path=sequence_path,
            sequence=sequence,
            valid_length=valid_length,
        ))

    if len(dimensions) != 1:
        raise ValueError(f"All sequences must have the same [frames, features] shape; got {sorted(dimensions)}.")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("Sequence manifest contains duplicate sample IDs.")
    return samples


def _assign_group_splits(groups: list[str], seed: int) -> dict[str, str]:
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 3:
        raise ValueError(
            "At least three independent signer/video groups are required to form non-empty train, "
            "validation, and test splits without leakage."
        )
    shuffled = list(unique_groups)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * 0.15))
    validation_count = max(1, round(len(shuffled) * 0.15))
    train_count = len(shuffled) - test_count - validation_count
    if train_count < 1:
        raise ValueError("Not enough independent groups for three non-empty data splits.")
    assignment = {}
    for group in shuffled[:train_count]:
        assignment[group] = "train"
    for group in shuffled[train_count:train_count + validation_count]:
        assignment[group] = "validation"
    for group in shuffled[train_count + validation_count:]:
        assignment[group] = "test"
    return assignment


def create_splits(samples: list[Sample], seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not samples:
        raise ValueError("Cannot split an empty sequence dataset.")
    has_any_official_split = any(sample.source_split for sample in samples)
    has_official_splits = all(sample.source_split for sample in samples)
    if has_any_official_split and not has_official_splits:
        raise ValueError("Manifest has incomplete source_split values; refusing to guess official assignments.")
    all_have_signers = all(sample.signer_id for sample in samples)
    if has_official_splits:
        assignment = {sample.sample_id: sample.source_split for sample in samples}
        split_basis = "official_source_split"
        signer_separation_available = all_have_signers
    else:
        signers = sorted({sample.signer_id for sample in samples if sample.signer_id})
        signer_separation_available = bool(signers)
        if signers:
            signer_assignment = _assign_group_splits(signers, seed)
            assignment = {
                sample.sample_id: signer_assignment[sample.signer_id]
                if sample.signer_id else ""
                for sample in samples
            }
            unknown_samples = [sample for sample in samples if not sample.signer_id]
            if unknown_samples:
                unknown_groups = sorted({sample.group_id for sample in unknown_samples})
                random.Random(seed).shuffle(unknown_groups)
                unknown_assignment = {
                    group: SPLIT_NAMES[index % len(SPLIT_NAMES)]
                    for index, group in enumerate(unknown_groups)
                }
                assignment.update({
                    sample.sample_id: unknown_assignment[sample.group_id]
                    for sample in unknown_samples
                })
                signer_separation_available = False
            split_basis = "signer_grouped"
        else:
            video_assignment = _assign_group_splits([sample.group_id for sample in samples], seed)
            assignment = {sample.sample_id: video_assignment[sample.group_id] for sample in samples}
            split_basis = "video_grouped_signer_separation_unavailable"

    unknown_split = set(assignment.values()) - set(SPLIT_NAMES)
    if unknown_split:
        raise ValueError(f"Unsupported source split values: {sorted(unknown_split)}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    signer_splits: dict[str, set[str]] = defaultdict(set)
    split_rows = []
    for sample in samples:
        split = assignment[sample.sample_id]
        group_splits[sample.group_id].add(split)
        if sample.signer_id:
            signer_splits[sample.signer_id].add(split)
        split_rows.append({
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "signer_id": sample.signer_id,
            "label": sample.label,
            "split": split,
            "sequence_path": str(sample.sequence_path),
        })
    group_leaks = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    signer_leaks = {key: sorted(value) for key, value in signer_splits.items() if len(value) > 1}
    if group_leaks:
        raise ValueError(f"Source video/sample leakage across splits: {group_leaks}")
    if signer_leaks:
        raise ValueError(f"Signer leakage across splits: {signer_leaks}")

    counts = Counter(row["split"] for row in split_rows)
    missing = [name for name in SPLIT_NAMES if counts[name] == 0]
    if missing:
        raise ValueError(f"Dataset split is missing non-empty partitions: {', '.join(missing)}.")
    summary = {
        "split_basis": split_basis,
        "signer_separation": (
            "AVAILABLE" if signer_separation_available
            else "PARTIAL_SIGNER_METADATA" if any(sample.signer_id for sample in samples)
            else "SIGNER_SEPARATION_UNAVAILABLE"
        ),
        "sample_counts": {name: counts[name] for name in SPLIT_NAMES},
        "group_counts": {
            name: len({row["group_id"] for row in split_rows if row["split"] == name})
            for name in SPLIT_NAMES
        },
        "signer_counts": {
            name: len({row["signer_id"] for row in split_rows if row["split"] == name and row["signer_id"]})
            for name in SPLIT_NAMES
        },
        "signer_overlap": False,
        "group_overlap": False,
    }
    return split_rows, summary


def write_splits(output_root: str | Path, split_rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    split_dir = Path(output_root) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / "split_manifest.csv"
    fields = ["sample_id", "group_id", "signer_id", "label", "split", "sequence_path"]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(split_rows)
    (split_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


if torch is not None:
    class BiLSTMBaseline(nn.Module):
        def __init__(
            self,
            input_size: int,
            num_classes: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, values: Any, lengths: Any | None = None) -> Any:
            if lengths is None:
                _, (hidden, _) = self.lstm(values)
            else:
                packed = pack_padded_sequence(
                    values, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
                )
                _, (hidden, _) = self.lstm(packed)
            representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.classifier(self.dropout(representation))
else:
    class BiLSTMBaseline:  # pragma: no cover - used only to provide a clear optional-dependency failure
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


class SequenceDataset(Dataset):
    def __init__(self, samples: list[Sample], label_to_index: dict[str, int]) -> None:
        self.samples = samples
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int, int, dict[str, str]]:
        sample = self.samples[index]
        return (
            torch.from_numpy(sample.sequence),
            self.label_to_index[sample.label],
            sample.valid_length,
            {
                "sample_id": sample.sample_id,
                "group_id": sample.group_id,
                "signer_id": sample.signer_id,
                "label": sample.label,
            },
        )


def _dataset_for_split(samples: list[Sample], split_rows: list[dict[str, Any]], split: str) -> list[Sample]:
    selected = {row["sample_id"] for row in split_rows if row["split"] == split}
    return [sample for sample in samples if sample.sample_id in selected]


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _check_label_coverage(
    label_to_index: dict[str, int], validation_samples: list[Sample], test_samples: list[Sample]
) -> None:
    unseen = sorted({sample.label for sample in validation_samples + test_samples} - set(label_to_index))
    if unseen:
        raise ValueError(
            "Validation/test contain labels not present in training, so they cannot be classified by this "
            f"closed-set baseline: {unseen}. Select a sample with shared class coverage."
        )


def _predict(model: Any, loader: Any, device: Any) -> tuple[list[int], list[int], list[list[float]], list[str]]:
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    probabilities: list[list[float]] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for values, targets, lengths, metadata in loader:
            logits = model(values.to(device), lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predicted.extend(np.argmax(probs, axis=1).tolist())
            truth.extend(targets.numpy().tolist())
            probabilities.extend(probs.tolist())
            sample_ids.extend(metadata["sample_id"])
    return truth, predicted, probabilities, sample_ids


def _classification_metrics(
    truth: list[int], predicted: list[int], labels: list[str]
) -> dict[str, Any]:
    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=indices, zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=indices)
    return {
        "accuracy": float(accuracy_score(truth, predicted)) if truth else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": labels,
        "test_samples": len(truth),
    }


def _write_experiment_report(
    output_root: Path,
    dataset: str,
    language: str,
    samples: list[Sample],
    split_summary: dict[str, Any],
    model_info: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    labels = sorted({sample.label for sample in samples})
    disclaimer = (
        "> EXPERIMENTAL. LSA64 represents Argentine Sign Language; these results are not MSL recognition performance."
        if language.strip().casefold() in {"lsa64", "experimental_lsa_architecture_validation"}
        else "> EXPERIMENTAL. NOT VALIDATED MSL. Results from ASL or synthetic test fixtures are not MSL accuracy."
    )
    lines = [
        "# Experimental Dynamic Baseline",
        "",
        f"**Status:** {'TRAINED (EXPERIMENTAL)' if metrics else 'BLOCKED: NO RUNNABLE EXPERIMENT'}",
        f"**Experiment tag:** `{_experiment_tag(language)}`",
        f"**Dataset:** {dataset}",
        f"**Language:** {language}",
        "",
        disclaimer,
        "> Production MSL recordings and metadata are never changed. MSL inputs must come through the existing "
        "processed-sequences manifest and pass its signer, consent, session, vocabulary, review, technical QC, "
        "and source-hash gates again.",
        "",
        "## Dataset",
        "",
        f"- Classes: {len(labels)}" + (f" ({', '.join(labels)})" if labels else ""),
        f"- Sequence samples/windows: {len(samples)}",
        f"- Source video/sample groups: {len({sample.group_id for sample in samples})}",
        f"- Signers: {len({sample.signer_id for sample in samples if sample.signer_id})}",
        f"- Signer separation: {split_summary.get('signer_separation', 'NOT_RUN')}",
        f"- Split counts: {json.dumps(split_summary.get('sample_counts', {}), sort_keys=True)}",
        f"- Split signer counts: {json.dumps(split_summary.get('signer_counts', {}), sort_keys=True)}",
        "",
        "## Training and evaluation",
        "",
    ]
    if model_info:
        lines.extend([
            f"- Architecture: {model_info['architecture']}",
            f"- Input: [batch, {model_info['sequence_length']}, {model_info['feature_dimension']}]",
            f"- Sequence stride: {model_info['stride']} frames",
            f"- FPS handling: {model_info['fps_handling']}",
            f"- Normalization: {model_info['normalization']}",
            f"- Missing landmarks: {model_info['missing_landmarks']}",
            f"- Parameters: {model_info['parameter_count']}",
            f"- Epochs requested: {model_info['epochs']}",
            f"- Batch size: {model_info['batch_size']}",
            f"- Learning rate: {model_info['learning_rate']}",
        ])
    if metrics:
        lines.extend([
            f"- Test accuracy: {metrics['accuracy']:.4f}",
            f"- Test macro F1: {metrics['macro_f1']:.4f}",
            f"- Test samples: {metrics['test_samples']}",
            f"- Test signers: {split_summary.get('signer_counts', {}).get('test', 0)}",
            f"- Confusion matrix: `{metrics.get('confusion_matrix_path', '')}`",
        ])
    else:
        lines.append("No training or evaluation was performed because no eligible experimental sequences are available.")
    lines.extend([
        "",
        "A signer-separated split is required for signer generalization claims. If signer metadata is missing,",
        "the report must retain `SIGNER_SEPARATION_UNAVAILABLE`; video-level splits are not signer-generalization tests.",
        "",
    ])
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "reports" / "baseline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def train_baseline(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT,
    run_dir: str | Path = DEFAULT_RUN,
    *,
    dataset: str = "MS-ASL",
    language: str = "ASL",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> dict[str, Any]:
    _require_torch()
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    manifest = validate_experimental_source(manifest_path, language)
    try:
        samples = load_samples(manifest, language)
    except ValueError as exc:
        if "No prepared temporal sequences" in str(exc):
            (Path(output_root) / "reports").mkdir(parents=True, exist_ok=True)
            (Path(output_root) / "reports" / "baseline_report.md").write_text(
                "# Experimental Dynamic Baseline\n\n"
                f"**Status:** BLOCKED - {_experiment_tag(language)}\n\n"
                f"Dataset: {dataset}; language: {language}.\n\n"
                f"No training was performed. The manifest contains no prepared temporal sequences. "
                f"This experiment is tagged {_experiment_tag(language)} and is NOT a production model.\n",
                encoding="utf-8",
            )
        raise
    split_rows, split_summary = create_splits(samples, seed)
    write_splits(output_root, split_rows, split_summary)
    train_samples = _dataset_for_split(samples, split_rows, "train")
    validation_samples = _dataset_for_split(samples, split_rows, "validation")
    test_samples = _dataset_for_split(samples, split_rows, "test")
    labels = sorted({sample.label for sample in train_samples})
    if len(labels) < 2:
        raise ValueError("At least two distinct training labels are required for classification.")
    label_to_index = {label: index for index, label in enumerate(labels)}
    _check_label_coverage(label_to_index, validation_samples, test_samples)

    generator = _seed_everything(seed)
    experiment_tag = _experiment_tag(language)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = SequenceDataset(train_samples, label_to_index)
    validation_data = SequenceDataset(validation_samples, label_to_index)
    test_data = SequenceDataset(test_samples, label_to_index)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    feature_dimension = int(samples[0].sequence.shape[1])
    sequence_length = int(samples[0].sequence.shape[0])
    model = BiLSTMBaseline(
        feature_dimension, len(labels), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run / "best_model.pt"
    best_validation_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for values, targets, lengths, _metadata in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.to(device), lengths)
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        validation_loss_values = []
        validation_truth: list[int] = []
        validation_predictions: list[int] = []
        with torch.no_grad():
            for values, targets, lengths, _metadata in validation_loader:
                logits = model(values.to(device), lengths)
                validation_loss_values.append(float(criterion(logits, targets.to(device)).item()))
                validation_truth.extend(targets.numpy().tolist())
                validation_predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        validation_loss = float(np.mean(validation_loss_values))
        epoch_record = {
            "epoch": float(epoch),
            "training_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
            "validation_accuracy": float(accuracy_score(validation_truth, validation_predictions)),
        }
        history.append(epoch_record)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_index": label_to_index,
                "input_size": feature_dimension,
                "sequence_length": sequence_length,
                "architecture": "BiLSTM",
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "dataset": dataset,
                "language": language,
                "experiment_tag": experiment_tag,
                "seed": seed,
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_truth, test_predictions, test_probabilities, test_sample_ids = _predict(model, test_loader, device)
    ordered_labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    metrics = _classification_metrics(test_truth, test_predictions, ordered_labels)
    matrix_path = run / "confusion_matrix.json"
    matrix_path.write_text(json.dumps({
        "labels": metrics.pop("confusion_matrix_labels"),
        "matrix": metrics["confusion_matrix"],
    }, indent=2) + "\n", encoding="utf-8")
    metrics["confusion_matrix_path"] = str(matrix_path)
    samples_by_id = {sample.sample_id: sample for sample in test_samples}
    prediction_path = run / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as output:
        fields = [
            "sample_id", "video_id", "signer_id", "actual_class", "predicted_class",
            "confidence", "class_probabilities",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for sample_id, truth_index, predicted_index, probabilities in zip(
            test_sample_ids, test_truth, test_predictions, test_probabilities
        ):
            sample = samples_by_id[sample_id]
            writer.writerow({
                "sample_id": sample_id,
                "video_id": sample.group_id,
                "signer_id": sample.signer_id,
                "actual_class": ordered_labels[truth_index],
                "predicted_class": ordered_labels[predicted_index],
                "confidence": float(probabilities[predicted_index]),
                "class_probabilities": json.dumps(dict(zip(ordered_labels, probabilities))),
            })
    metrics["test_predictions_path"] = str(prediction_path)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_info = {
        "architecture": f"BiLSTM(hidden_size={hidden_size}, num_layers={num_layers}, bidirectional=True)",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "sequence_length": sequence_length,
        "feature_dimension": feature_dimension,
        "stride": int(load_msl_config().get("stride", 15)),
        "fps_handling": "Original video FPS is used for MediaPipe timestamps; frames are not resampled.",
        "normalization": str(load_msl_config().get("normalization_method", "wrist_relative")),
        "missing_landmarks": "Model feature groups are zero-filled; landmark-presence masks remain in the feature vector.",
        "parameter_count": parameter_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    configuration = {
        **model_info,
        "dataset": dataset,
        "language": language,
        "experiment_tag": experiment_tag,
        "label_to_index": label_to_index,
        "split_summary": split_summary,
        "training_history": history,
        "best_checkpoint": str(checkpoint_path),
        "metrics": metrics,
    }
    (run / "config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_path = _write_experiment_report(
        Path(output_root), dataset, language, samples, split_summary, model_info, metrics
    )
    return {"checkpoint": str(checkpoint_path), "report": str(report_path), "metrics": metrics}


def load_checkpoint(checkpoint_path: str | Path, device: Any | None = None) -> tuple[Any, dict[str, Any], Any]:
    _require_torch()
    target_device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    if checkpoint.get("experiment_tag") not in {
        EXPERIMENT_TAG, LSA64_EXPERIMENT_TAG, MSL_EXPERIMENT_TAG, GENERIC_EXPERIMENT_TAG
    }:
        raise ValueError("Checkpoint is not tagged as an isolated experimental dynamic baseline.")
    model = BiLSTMBaseline(
        checkpoint["input_size"],
        len(checkpoint["label_to_index"]),
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, target_device


def evaluate_baseline(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    language: str = "ASL",
    seed: int = 42,
) -> dict[str, Any]:
    samples = load_samples(manifest_path, language)
    split_rows, split_summary = create_splits(samples, seed)
    model, checkpoint, device = load_checkpoint(checkpoint_path)
    train_labels = checkpoint["label_to_index"]
    test_samples = _dataset_for_split(samples, split_rows, "test")
    _check_label_coverage(train_labels, [], test_samples)
    test_loader = DataLoader(SequenceDataset(test_samples, train_labels), batch_size=8, shuffle=False)
    truth, predicted, _probabilities, _sample_ids = _predict(model, test_loader, device)
    labels = [label for label, _ in sorted(train_labels.items(), key=lambda item: item[1])]
    metrics = _classification_metrics(truth, predicted, labels)
    metrics.update({
        "dataset": checkpoint["dataset"],
        "language": checkpoint["language"],
        "experiment_tag": checkpoint["experiment_tag"],
        "test_signers": split_summary["signer_counts"]["test"],
        "signer_separation": split_summary["signer_separation"],
        "split_counts": split_summary["sample_counts"],
    })
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        (target / "confusion_matrix.json").write_text(json.dumps({
            "labels": labels,
            "matrix": metrics["confusion_matrix"],
        }, indent=2) + "\n", encoding="utf-8")
    return metrics


def predict_sequence(sequence_path: str | Path, checkpoint_path: str | Path) -> dict[str, Any]:
    model, checkpoint, device = load_checkpoint(checkpoint_path)
    with np.load(sequence_path, allow_pickle=False) as stored:
        sequence = np.asarray(stored["sequence"], dtype=np.float32)
        mask = np.asarray(stored["valid_frame_mask"], dtype=np.float32) if "valid_frame_mask" in stored else None
    if sequence.ndim != 2 or not np.isfinite(sequence).all():
        raise ValueError("Inference sequence must be a finite [frames, features] array.")
    if sequence.shape[1] != checkpoint["input_size"]:
        raise ValueError(
            f"Inference feature dimension {sequence.shape[1]} does not match model input {checkpoint['input_size']}."
        )
    if sequence.shape[0] != checkpoint["sequence_length"]:
        raise ValueError(
            f"Inference sequence length {sequence.shape[0]} does not match model length {checkpoint['sequence_length']}."
        )
    length = int(np.count_nonzero(mask > 0)) if mask is not None else sequence.shape[0]
    if length <= 0 or (mask is not None and mask.shape != (sequence.shape[0],)):
        raise ValueError("Inference valid_frame_mask is invalid.")
    with torch.no_grad():
        scores = torch.softmax(
            model(torch.from_numpy(sequence).unsqueeze(0).to(device), torch.tensor([length])), dim=1
        )[0].cpu().numpy()
    index_to_label = {value: key for key, value in checkpoint["label_to_index"].items()}
    class_index = int(np.argmax(scores))
    return {
        "predicted_class": index_to_label[class_index],
        "confidence": float(scores[class_index]),
        "class_probabilities": {
            index_to_label[index]: float(score) for index, score in enumerate(scores)
        },
        "experiment_tag": checkpoint["experiment_tag"],
        "dataset": checkpoint["dataset"],
        "language": checkpoint["language"],
    }


def predict_video(video_path: str | Path, checkpoint_path: str | Path) -> dict[str, Any]:
    source = Path(video_path).resolve()
    protected_raw = (PROJECT_ROOT / "data" / "msl_dataset" / "raw").resolve()
    if _is_within(source, protected_raw):
        raise ValueError("Use only experimental-source videos for trained-model inference; MSL raw videos are protected.")
    if not source.is_file():
        raise FileNotFoundError(source)
    model, checkpoint, device = load_checkpoint(checkpoint_path)
    config = load_msl_config()
    config["processed_dataset_path"] = tempfile.mkdtemp(prefix="dynamic-inference-")
    try:
        extracted = process_video(source, "UNKNOWN", "UNLABELLED", config=config, session_id="unknown")
        predictions = [
            predict_sequence(record["sequence_path"], checkpoint_path)
            for record in extracted["sequence_windows"]
        ]
        aggregate = np.mean(
            [[result["class_probabilities"][label] for label in checkpoint["label_to_index"]]
             for result in predictions],
            axis=0,
        )
        labels = list(checkpoint["label_to_index"])
        best = int(np.argmax(aggregate))
        return {
            "actual_class": None,
            "predicted_class": labels[best],
            "confidence": float(aggregate[best]),
            "sequence_count": len(predictions),
            "video_id": source.stem,
            "experiment_tag": checkpoint["experiment_tag"],
            "dataset": checkpoint["dataset"],
            "language": checkpoint["language"],
        }
    finally:
        shutil.rmtree(config["processed_dataset_path"], ignore_errors=False)


def _write_status_report(output_root: str | Path, manifest_path: str | Path) -> Path:
    manifest = _resolved(manifest_path)
    records = []
    if manifest.is_file():
        with manifest.open("r", newline="", encoding="utf-8-sig") as source:
            records = list(csv.DictReader(source))
    dataset_summary_path = PROJECT_ROOT / "data" / "external_datasets" / "msasl" / "reports" / "dataset_summary.json"
    selected_metadata_records = 0
    if dataset_summary_path.is_file():
        dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        selected_metadata_records = int(dataset_summary.get("selected_records", 0))

    def local_file_exists(value: str) -> bool:
        path = Path(value)
        if not path.is_absolute():
            path = manifest.parent / path
        return path.is_file()

    prepared = sum(
        local_file_exists(_field(row, "sequence_path"))
        for row in records if _field(row, "sequence_path")
    )
    videos = sum(
        local_file_exists(_field(row, "local_video_path", "clip_path"))
        for row in records if _field(row, "local_video_path", "clip_path")
    )
    report = [
        "# Experimental Dynamic Baseline - Availability",
        "",
        f"**Experiment tag:** `{EXPERIMENT_TAG}`",
        "**Status:** BLOCKED - no prepared experimental sequences are available.",
        "**Language:** ASL (American Sign Language), not MSL.",
        "",
        "This report is an architecture-readiness snapshot, not a trained model result.",
        "No model training, accuracy evaluation, or trained inference was performed.",
        "",
        "## Local MS-ASL availability",
        "",
        f"- Sequence-manifest records: {len(records)}",
        f"- Selected metadata candidates (MS-ASL import report): {selected_metadata_records}",
        f"- Records with local video/clip paths: {videos}",
        f"- Prepared sequence records: {prepared}",
        "- Imported videos (per current MS-ASL report): 0",
        "- Extracted sequences (per current MS-ASL report): 0",
        "- Signer-separated splits: unavailable until videos are processed into sequences.",
        "",
        f"The selected MS-ASL metadata currently has {selected_metadata_records} candidates, "
        "but the first video source returned HTML;",
        "the remaining 99 downloads were not attempted. No alternate source was used. Do not infer",
        "training readiness from metadata-only records.",
        "",
        "The 13 existing MSL videos are pending human validation and are not used as training data.",
        "No MSL review gates, manifests, videos, static assets, or production vocabulary are changed.",
        "",
        "## Exact next step",
        "",
        "Obtain an authorized set of actual video files for the selected external dataset, preprocess them",
        "with the existing MediaPipe Holistic extractor into finite temporal sequences, and create a",
        "sequence manifest with labels, video/sample IDs, signer IDs when available, and official source",
        "splits. Then run the training command in the collection guide.",
        "",
    ]
    report_path = Path(output_root) / "reports" / "baseline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an isolated experimental dynamic-sign BiLSTM; never an MSL production model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status", help="Report whether local prepared sequences are available.")
    train = subparsers.add_parser("train", help="Train and evaluate using a prepared temporal-sequence manifest.")
    infer_sequence = subparsers.add_parser("predict-sequence", help="Predict a prepared sequence using an experiment checkpoint.")
    infer_video = subparsers.add_parser("predict-video", help="Run Holistic plus trained inference on an experimental video.")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate an experiment checkpoint on the held-out test split.")
    for command in (status, train):
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--dataset", default="MS-ASL")
        command.add_argument("--language", default="ASL")
    train.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--hidden-size", type=int, default=128)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.3)
    infer_sequence.add_argument("--sequence", type=Path, required=True)
    infer_sequence.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "best_model.pt")
    infer_video.add_argument("--video", type=Path, required=True)
    infer_video.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "best_model.pt")
    infer_video.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "reports" / "inference.json")
    evaluate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "best_model.pt")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    evaluate.add_argument("--language", default="ASL")
    evaluate.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.command == "status":
        report = _write_status_report(args.output_root, args.manifest)
        print(f"Availability report written to {report}")
    elif args.command == "train":
        result = train_baseline(
            args.manifest,
            args.output_root,
            args.run_dir,
            dataset=args.dataset,
            language=args.language,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "predict-sequence":
        print(json.dumps(predict_sequence(args.sequence, args.checkpoint), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_baseline(
            args.manifest, args.checkpoint, args.output_dir, language=args.language, seed=args.seed
        ), indent=2))
    else:
        result = predict_video(args.video, args.checkpoint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
