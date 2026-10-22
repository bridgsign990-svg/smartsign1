from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.experimental_dynamic.baseline import load_checkpoint, predict_video, train_baseline
from tools.msl_dataset.msl_dataset_pipeline import PROJECT_ROOT, load_config, process_video


DATA_ROOT = PROJECT_ROOT / "data" / "experimental_dynamic" / "lsa64"
DEFAULT_SOURCE = DATA_ROOT / "all_cut"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "experimental" / "lsa64"
CLASS_COUNT = 64
SIGNER_COUNT = 10
REPETITION_COUNT = 5
DEFAULT_CLASS_COUNT = 5
DEFAULT_SEED = 42
FILENAME_PATTERN = re.compile(r"^(?P<class>\d{3})_(?P<signer>\d{3})_(?P<repetition>\d{3})$")
MANIFEST_FIELDS = [
    "source_record_id",
    "video_path",
    "source_relative_path",
    "video_id",
    "class_id",
    "sign_label",
    "signer_id",
    "repetition_id",
    "source_split",
    "file_size_bytes",
    "fps",
    "frame_count",
    "duration_seconds",
    "width",
    "height",
    "decoded_frames",
    "validation_status",
    "validation_reasons",
    "processing_status",
    "processing_error",
    "usable_frames",
    "detected_frames",
    "left_hand_detection_rate",
    "right_hand_detection_rate",
    "pose_detection_rate",
    "face_detection_rate",
    "sequence_window_count",
    "sequence_paths_json",
    "source_sha256",
]


def parse_lsa64_filename(path: str | Path) -> dict[str, str]:
    video_path = Path(path)
    match = FILENAME_PATTERN.fullmatch(video_path.stem)
    if not match:
        raise ValueError(f"Video filename does not use the LSA64 class_signer_repetition format: {video_path.name}")
    class_number = int(match.group("class"))
    signer_number = int(match.group("signer"))
    repetition_number = int(match.group("repetition"))
    if not 1 <= class_number <= CLASS_COUNT:
        raise ValueError(f"LSA64 class ID out of range in {video_path.name}: {class_number}")
    if not 1 <= signer_number <= SIGNER_COUNT:
        raise ValueError(f"LSA64 signer ID out of range in {video_path.name}: {signer_number}")
    if not 1 <= repetition_number <= REPETITION_COUNT:
        raise ValueError(f"LSA64 repetition ID out of range in {video_path.name}: {repetition_number}")
    class_id = f"{class_number:03d}"
    return {
        "source_record_id": video_path.stem,
        "video_id": video_path.stem,
        "class_id": class_id,
        "sign_label": f"LSA64_CLASS_{class_id}",
        "signer_id": f"LSA64_SIGNER_{signer_number:03d}",
        "repetition_id": f"REP_{repetition_number:03d}",
    }


def scan_dataset(source_dir: str | Path) -> list[dict[str, str]]:
    root = Path(source_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Extracted LSA64 video directory not found: {root}")
    videos = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".mp4"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not videos:
        raise ValueError(f"No MP4 videos found under {root}; archive extraction may be required.")

    records = []
    identities: set[str] = set()
    combinations: set[tuple[str, str, str]] = set()
    for path in videos:
        identity = parse_lsa64_filename(path)
        if identity["source_record_id"] in identities:
            raise ValueError(f"Duplicate source identity: {identity['source_record_id']}")
        combination = (identity["class_id"], identity["signer_id"], identity["repetition_id"])
        if combination in combinations:
            raise ValueError(f"Duplicate class/signer/repetition combination: {combination}")
        identities.add(identity["source_record_id"])
        combinations.add(combination)
        records.append({
            **identity,
            "video_path": str(path),
            "source_relative_path": path.relative_to(root).as_posix(),
            "source_split": "",
            "file_size_bytes": str(path.stat().st_size),
            "fps": "",
            "frame_count": "",
            "duration_seconds": "",
            "width": "",
            "height": "",
            "decoded_frames": "",
            "validation_status": "NOT_CHECKED",
            "validation_reasons": "",
            "processing_status": "NOT_SELECTED",
            "processing_error": "",
            "usable_frames": "",
            "detected_frames": "",
            "left_hand_detection_rate": "",
            "right_hand_detection_rate": "",
            "pose_detection_rate": "",
            "face_detection_rate": "",
            "sequence_window_count": "0",
            "sequence_paths_json": "[]",
            "source_sha256": "",
        })

    expected = {
        (f"{class_number:03d}", f"LSA64_SIGNER_{signer_number:03d}", f"REP_{repetition_number:03d}")
        for class_number in range(1, CLASS_COUNT + 1)
        for signer_number in range(1, SIGNER_COUNT + 1)
        for repetition_number in range(1, REPETITION_COUNT + 1)
    }
    missing = sorted(expected - combinations)
    if len(records) != CLASS_COUNT * SIGNER_COUNT * REPETITION_COUNT or missing:
        raise ValueError(
            "LSA64 structure validation failed: expected 3,200 unique class/signer/repetition videos "
            f"and found {len(records)}; missing combinations include {missing[:10]}."
        )
    return records


def select_subset(records: list[dict[str, str]], class_count: int, seed: int) -> tuple[list[str], list[dict[str, str]]]:
    if not 5 <= class_count <= 10:
        raise ValueError("The experiment subset must contain between 5 and 10 classes.")
    by_class: dict[str, list[dict[str, str]]] = {}
    for record in records:
        by_class.setdefault(record["class_id"], []).append(record)
    eligible = [
        class_id for class_id, class_records in by_class.items()
        if len(class_records) == SIGNER_COUNT * REPETITION_COUNT
        and len({row["signer_id"] for row in class_records}) == SIGNER_COUNT
        and len({row["repetition_id"] for row in class_records}) == REPETITION_COUNT
    ]
    if len(eligible) < class_count:
        raise ValueError(
            f"Only {len(eligible)} classes have complete signer/repetition coverage; {class_count} are required."
        )
    ordered = sorted(
        eligible,
        key=lambda class_id: hashlib.sha256(f"LSA64:{seed}:{class_id}".encode("ascii")).hexdigest(),
    )
    selected_classes = sorted(ordered[:class_count])
    selected = sorted(
        (record for record in records if record["class_id"] in selected_classes),
        key=lambda record: (
            record["class_id"],
            record["signer_id"],
            record["repetition_id"],
        ),
    )
    return selected_classes, selected


def _write_csv(
    path: Path,
    records: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        columns = fieldnames or (list(records[0]) if records else MANIFEST_FIELDS)
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_check(record: dict[str, str], thresholds: dict[str, Any]) -> dict[str, str]:
    path = Path(record["video_path"])
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {**record, "validation_status": "REJECTED", "validation_reasons": "OPEN_FAILED"}

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    decoded_frames = 0
    has_image_content = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames += 1
            if frame.size and float(np.std(frame)) > 0.25:
                has_image_content = True
    finally:
        capture.release()

    reasons = []
    if decoded_frames == 0:
        reasons.append("NO_DECODED_FRAMES")
    if decoded_frames != reported_frames:
        reasons.append("FRAME_COUNT_MISMATCH")
    if decoded_frames < int(thresholds["min_frames"]):
        reasons.append("TOO_FEW_FRAMES")
    if decoded_frames > int(thresholds.get("max_frames", decoded_frames)):
        reasons.append("TOO_MANY_FRAMES")
    if fps < float(thresholds["min_fps"]) or fps > float(thresholds["max_fps"]) + 0.1:
        reasons.append("FPS_OUT_OF_RANGE")
    if width < int(thresholds["min_width"]) or height < int(thresholds["min_height"]):
        reasons.append("RESOLUTION_TOO_LOW")
    if not has_image_content:
        reasons.append("EMPTY_IMAGE_CONTENT")
    return {
        **record,
        "fps": f"{fps:.6f}",
        "frame_count": str(reported_frames),
        "duration_seconds": f"{reported_frames / fps:.6f}" if fps > 0 else "0",
        "width": str(width),
        "height": str(height),
        "decoded_frames": str(decoded_frames),
        "validation_status": "REJECTED" if reasons else "VALID",
        "validation_reasons": ";".join(reasons),
        "source_sha256": _sha256(path),
    }


def _restore_selection(
    discovered: list[dict[str, str]], selected_path: Path, classes: list[str], seed: int
) -> list[dict[str, str]]:
    existing = {row["source_record_id"]: row for row in _read_csv(selected_path)}
    restored = []
    for record in discovered:
        previous = existing.get(record["source_record_id"])
        if previous:
            restored.append({**record, **{key: value for key, value in previous.items() if value != ""}})
        else:
            restored.append(record)
    metadata = {
        "experiment_tag": "EXPERIMENTAL_LSA_ARCHITECTURE_VALIDATION",
        "seed": seed,
        "selected_class_ids": classes,
        "selected_class_count": len(classes),
        "selected_video_count": len(restored),
        "selection_method": (
            "Classes with complete signer/repetition coverage ranked by SHA-256 of "
            f"'LSA64:{seed}:<class_id>'; all recordings for the first {len(classes)} class IDs selected."
        ),
    }
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.with_name("subset_selection.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return restored


def prepare_dataset(
    source_dir: str | Path,
    output_root: str | Path = DATA_ROOT,
    *,
    class_count: int = DEFAULT_CLASS_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    metadata_dir = root / "metadata"
    processed_dir = root / "processed"
    reports_dir = root / "reports"
    for directory in (metadata_dir, processed_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = scan_dataset(source_dir)
    classes, selected = select_subset(records, class_count, seed)
    selected = [{**record, "processing_status": "SELECTED"} for record in selected]
    selected_ids = {record["source_record_id"] for record in selected}
    all_records = [
        {**record, "processing_status": "SELECTED" if record["source_record_id"] in selected_ids else "NOT_SELECTED"}
        for record in records
    ]
    source_manifest = metadata_dir / "lsa64_manifest.csv"
    _write_csv(source_manifest, all_records)
    selected_manifest = metadata_dir / "selected_videos.csv"
    selected = _restore_selection(selected, selected_manifest, classes, seed)

    pipeline_config = load_config(PROJECT_ROOT / "config" / "msl_dataset.json")
    thresholds = pipeline_config["quality_thresholds"]
    qc_records = []
    for index, record in enumerate(selected, start=1):
        if record.get("validation_status") in {"VALID", "REJECTED"}:
            checked = record
        else:
            checked = _quality_check(record, thresholds)
        qc_records.append(checked)
        _write_csv(selected_manifest, qc_records + selected[index:])
        print(
            f"QC {index}/{len(selected)} {record['source_record_id']}: "
            f"{checked['validation_status']} {checked['validation_reasons']}",
            flush=True,
        )
    selected = qc_records

    pipeline_config = deepcopy(pipeline_config)
    pipeline_config["raw_dataset_path"] = str(Path(source_dir).resolve())
    pipeline_config["processed_dataset_path"] = str(processed_dir)
    pipeline_config["manifests_path"] = str(metadata_dir)
    sequence_rows = []
    successful = 0
    failed = 0

    def sync_master_manifest() -> None:
        state = {row["source_record_id"]: row for row in selected}
        _write_csv(
            source_manifest,
            [{**row, **state.get(row["source_record_id"], {})} for row in all_records],
        )

    sync_master_manifest()
    for index, record in enumerate(selected, start=1):
        sequence_paths = json.loads(record.get("sequence_paths_json") or "[]")
        current_sha256 = _sha256(Path(record["video_path"]))
        if record.get("source_sha256") and current_sha256 != record["source_sha256"]:
            record["processing_status"] = "FAILED"
            record["processing_error"] = "Source video changed after quality validation."
            failed += 1
            _write_csv(selected_manifest, selected)
            sync_master_manifest()
            continue
        if (
            record["validation_status"] == "VALID"
            and record["processing_status"] == "PROCESSED"
            and sequence_paths
            and all(Path(value).is_file() for value in sequence_paths)
        ):
            successful += 1
            sequence_rows.extend(_sequence_rows(record, sequence_paths))
            print(f"SKIP {index}/{len(selected)} {record['source_record_id']}: already processed", flush=True)
            continue
        if record["validation_status"] != "VALID":
            record["processing_status"] = "REJECTED"
            failed += 1
            continue

        record["processing_status"] = "PROCESSING"
        _write_csv(selected_manifest, selected)
        try:
            result = process_video(
                record["video_path"],
                record["signer_id"],
                record["sign_label"],
                config=pipeline_config,
                session_id="lsa64",
            )
            if result["source_sha256"] != record["source_sha256"]:
                raise RuntimeError(f"Source hash changed since QC for {record['source_record_id']}.")
            sequence_paths = [window["sequence_path"] for window in result["sequence_windows"]]
            record["sequence_window_count"] = str(len(sequence_paths))
            record["sequence_paths_json"] = json.dumps(sequence_paths)
            record["processing_status"] = "PROCESSED"
            record["processing_error"] = ""
            record["usable_frames"] = str(result["usable_frames"])
            record["detected_frames"] = str(result["detected_frames"])
            record["left_hand_detection_rate"] = f"{1.0 - result['missing_landmark_rates']['left_hand']:.6f}"
            record["right_hand_detection_rate"] = f"{1.0 - result['missing_landmark_rates']['right_hand']:.6f}"
            record["pose_detection_rate"] = f"{1.0 - result['missing_landmark_rates']['pose']:.6f}"
            record["face_detection_rate"] = f"{1.0 - result['missing_landmark_rates']['face']:.6f}"
            sequence_rows.extend(_sequence_rows(record, sequence_paths))
            successful += 1
        except (ValueError, RuntimeError, OSError, cv2.error) as exc:
            record["processing_status"] = "FAILED"
            record["processing_error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        _write_csv(selected_manifest, selected)
        _write_csv(metadata_dir / "sequences.csv", sequence_rows)
        sync_master_manifest()
        print(
            f"PROCESS {index}/{len(selected)} {record['source_record_id']}: "
            f"{record['processing_status']} windows={record['sequence_window_count']} "
            f"{record['processing_error']}",
            flush=True,
        )

    sequence_manifest = metadata_dir / "sequences.csv"
    _write_csv(selected_manifest, selected)
    sync_master_manifest()
    _write_csv(sequence_manifest, sequence_rows)
    return {
        "source_manifest": str(source_manifest),
        "selected_manifest": str(selected_manifest),
        "sequence_manifest": str(sequence_manifest),
        "selected_classes": classes,
        "selected_videos": len(selected),
        "valid_videos": sum(row["validation_status"] == "VALID" for row in selected),
        "rejected_videos": sum(row["validation_status"] == "REJECTED" for row in selected),
        "processed_videos": successful,
        "failed_videos": failed,
        "sequence_count": len(sequence_rows),
    }


def _sequence_rows(record: dict[str, str], sequence_paths: list[str]) -> list[dict[str, str]]:
    rows = []
    for index, sequence_path in enumerate(sequence_paths, start=1):
        rows.append({
            "sample_id": f"{record['source_record_id']}_window_{index:03d}",
            "source_record_id": record["source_record_id"],
            "source_split": "",
            "experiment_split": "",
            "video_id": record["video_id"],
            "original_video_path": record["video_path"],
            "class_id": record["class_id"],
            "label": record["sign_label"],
            "signer_id": record["signer_id"],
            "repetition_id": record["repetition_id"],
            "sequence_window_index": str(index),
            "sequence_path": sequence_path,
            "source_sha256": record["source_sha256"],
            "validation_status": record["validation_status"],
            "processing_status": "PROCESSED",
        })
    return rows


SEQUENCE_FIELDS = [
    "sample_id",
    "source_record_id",
    "source_split",
    "experiment_split",
    "video_id",
    "original_video_path",
    "class_id",
    "label",
    "signer_id",
    "repetition_id",
    "sequence_window_index",
    "sequence_path",
    "source_sha256",
    "validation_status",
    "processing_status",
]


def train_and_report(
    output_root: str | Path = DATA_ROOT,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    *,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = DEFAULT_SEED,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    model_dir = Path(model_root).resolve()
    manifest = root / "metadata" / "sequences.csv"
    rows = _read_csv(manifest)
    if not rows:
        raise ValueError(f"No processed LSA64 sequences are available in {manifest}.")
    if any(row.get("processing_status") != "PROCESSED" for row in rows):
        raise ValueError("Sequence manifest contains a record not marked PROCESSED.")
    selected = _read_csv(root / "metadata" / "selected_videos.csv")
    for video in selected:
        if video["processing_status"] == "PROCESSED":
            if _sha256(Path(video["video_path"])) != video["source_sha256"]:
                raise RuntimeError(f"LSA64 source video changed since preprocessing: {video['video_path']}")
    for row in rows:
        with np.load(row["sequence_path"], allow_pickle=False) as saved:
            sequence = np.asarray(saved["sequence"], dtype=np.float32)
        if sequence.shape != (30, 562) or not np.isfinite(sequence).all():
            raise ValueError(f"Expected finite (30, 562) LSA64 sequence: {row['sequence_path']}")
    if reuse_existing:
        required_artifacts = (
            model_dir / "best_model.pt",
            model_dir / "config.json",
            model_dir / "metrics.json",
            model_dir / "test_predictions.csv",
            root / "splits" / "split_manifest.csv",
            root / "splits" / "split_summary.json",
        )
        missing = [str(path) for path in required_artifacts if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot reuse incomplete LSA64 training artifacts: {missing}")
        result = {
            "checkpoint": str(model_dir / "best_model.pt"),
            "report": str(root / "reports" / "baseline_report.md"),
        }
    else:
        result = train_baseline(
            manifest,
            root,
            model_dir,
            dataset="LSA64",
            language="LSA64",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
        )

    split_rows = _read_csv(root / "splits" / "split_manifest.csv")
    split_by_id = {row["sample_id"]: row["split"] for row in split_rows}
    for row in rows:
        row["experiment_split"] = split_by_id.get(row["sample_id"], "")
        if not row["experiment_split"]:
            raise ValueError(f"No train/validation/test assignment for sequence {row['sample_id']}.")
    _write_csv(manifest, rows, SEQUENCE_FIELDS)

    signer_splits: dict[str, set[str]] = {}
    for row in rows:
        signer_splits.setdefault(row["signer_id"], set()).add(row["experiment_split"])
    leakage = {signer: sorted(splits) for signer, splits in signer_splits.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"Signer leakage across LSA64 train/validation/test splits: {leakage}")
    video_split = {
        row["video_id"]: row["experiment_split"]
        for row in rows
    }
    predictions = _read_csv(model_dir / "test_predictions.csv")
    test_videos = sorted(video_id for video_id, split in video_split.items() if split == "test")
    if not test_videos:
        raise ValueError("The signer-aware split produced no held-out test videos.")
    held_video_id = test_videos[0]
    held_row = next(row for row in rows if row["video_id"] == held_video_id and row["experiment_split"] == "test")
    inference = predict_video(held_row["original_video_path"], result["checkpoint"])
    inference["actual_class"] = held_row["label"]
    inference["signer_id"] = held_row["signer_id"]
    inference["repetition_id"] = held_row["repetition_id"]
    inference["experiment_tag"] = "EXPERIMENTAL_LSA_ARCHITECTURE_VALIDATION"
    inference_path = root / "reports" / "held_out_video_inference.json"
    inference_path.parent.mkdir(parents=True, exist_ok=True)
    inference_path.write_text(json.dumps(inference, indent=2) + "\n", encoding="utf-8")

    split_summary = json.loads((root / "splits" / "split_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    _model, checkpoint, _device = load_checkpoint(result["checkpoint"])
    config.update({
        "hidden_size": checkpoint["hidden_size"],
        "num_layers": checkpoint["num_layers"],
        "dropout": checkpoint["dropout"],
    })
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    selected = _read_csv(root / "metadata" / "selected_videos.csv")
    source_manifest = root / "metadata" / "lsa64_manifest.csv"
    all_records = _read_csv(source_manifest)
    subset_selection = json.loads((root / "metadata" / "subset_selection.json").read_text(encoding="utf-8"))
    class_ids = sorted({row["class_id"] for row in selected})
    video_counts = {
        name: len({row["video_id"] for row in rows if row["experiment_split"] == name})
        for name in ("train", "validation", "test")
    }
    signer_counts = {
        name: len({row["signer_id"] for row in rows if row["experiment_split"] == name})
        for name in ("train", "validation", "test")
    }
    class_counts = {
        name: sorted({row["label"] for row in rows if row["experiment_split"] == name})
        for name in ("train", "validation", "test")
    }
    report_path = root / "reports" / "lsa64_experiment_report.md"
    report_path.write_text(
        _render_report(
            selected,
            class_ids,
            rows,
            video_counts,
            signer_counts,
            class_counts,
            split_summary,
            metrics,
            config,
            predictions,
            inference,
            source_manifest,
            all_records,
            subset_selection["seed"],
        ),
        encoding="utf-8",
    )
    return {
        **result,
        "report": str(report_path),
        "held_out_inference": str(inference_path),
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "video_counts": video_counts,
        "signer_counts": signer_counts,
    }


def _render_report(
    selected: list[dict[str, str]],
    class_ids: list[str],
    sequences: list[dict[str, str]],
    video_counts: dict[str, int],
    signer_counts: dict[str, int],
    classes_per_split: dict[str, list[str]],
    split_summary: dict[str, Any],
    metrics: dict[str, Any],
    training: dict[str, Any],
    predictions: list[dict[str, str]],
    inference: dict[str, Any],
    source_manifest: Path,
    all_records: list[dict[str, str]],
    selection_seed: int,
) -> str:
    history = training.get("training_history", [])
    history_lines = [
        "| {epoch} | {training_loss:.6f} | {validation_loss:.6f} | {validation_accuracy:.4f} |".format(**row)
        for row in history
    ]
    per_class = metrics["per_class"]
    result_lines = [
        f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
        f"{values['f1']:.4f} | {values['support']} |"
        for label, values in per_class.items()
    ]
    matrix = metrics["confusion_matrix"]
    matrix_labels = sorted(per_class)
    matrix_lines = [
        "| " + " | ".join([label, *(str(value) for value in row)]) + " |"
        for label, row in zip(matrix_labels, matrix)
    ]
    rejected = [row for row in selected if row["validation_status"] == "REJECTED"]
    failed = [row for row in selected if row["processing_status"] == "FAILED"]
    best_epoch = min(history, key=lambda row: row["validation_loss"]) if history else None
    root_dir = Path(selected[0]["video_path"]).parent if selected else source_manifest.parent
    archive_paths = sorted(root_dir.parent.parent.glob("lsa64_cut*.zip"))
    archive_note = (
        "- Source archive(s): " + "; ".join(
            f"`{path}` ({path.stat().st_size} bytes)" for path in archive_paths
        ) + "."
        if archive_paths else "- No source archive was found alongside the extracted dataset."
    )
    rejected_lines = [
        f"- `{row['source_record_id']}`: "
        f"{row['validation_reasons'] if row['validation_status'] == 'REJECTED' else row['processing_error']}"
        for row in rejected + failed
    ] or ["- None."]
    prediction_lines = [
        f"- `{row['sample_id']}` ({row['signer_id']}): actual `{row['actual_class']}`, "
        f"predicted `{row['predicted_class']}`, confidence {float(row['confidence']):.4f}."
        for row in predictions[:20]
    ]
    if len(predictions) > 20:
        prediction_lines.append(
            f"- {len(predictions) - 20} additional per-window predictions are in "
            "`../../../models/experimental/lsa64/test_predictions.csv`."
        )
    split_sequence_counts = {
        name: sum(row["experiment_split"] == name for row in sequences)
        for name in ("train", "validation", "test")
    }
    return "\n".join([
        "# LSA64 Experimental Dynamic Recognition Report",
        "",
        "**Experiment tag:** `EXPERIMENTAL_LSA_ARCHITECTURE_VALIDATION`",
        "**Dataset:** LSA64, Argentine Sign Language. This experiment is not MSL.",
        "",
        "> LSA64 was used solely to validate the engineering pipeline for dynamic sign recognition. "
        "Because LSA64 represents Argentine Sign Language, its results are not interpreted as Malawian "
        "Sign Language recognition performance. Verified MSL data will be processed through the same "
        "pipeline once available.",
        "",
        "## Source and dataset",
        "",
        f"- Source directory: `{root_dir}`.",
        archive_note,
        f"- Full dataset manifest: `{source_manifest}`",
        f"- Extracted video payload: {sum(int(row['file_size_bytes']) for row in all_records)} bytes "
        f"({sum(int(row['file_size_bytes']) for row in all_records) / (1024 ** 3):.3f} GiB).",
        f"- Downloaded dataset videos: {len(all_records)}",
        "- Archive/folder contents included only the `all_cut/` video directory and MP4 files; no license, "
        "citation, glossary, or class-to-gloss mapping metadata was included. License terms therefore remain unverified.",
        "- No official train/validation/test split metadata was included; the experiment uses a signer-grouped split.",
        "- Source filename identity format: `class_id_signer_id_repetition_id.mp4`; source class IDs and "
        "signer/repetition IDs are retained. No glosses were inferred.",
        f"- Classes found: {len({row['class_id'] for row in all_records})} (numeric source class IDs); "
        f"videos found: {len(all_records)}; signers found: {len({row['signer_id'] for row in all_records})}.",
        f"- Experiment subset: {len(class_ids)} classes ({', '.join(class_ids)}), {len(selected)} selected videos.",
        "- Selection: all recordings for the complete-coverage classes ranked by seeded SHA-256 "
        f"(`LSA64:{selection_seed}:<class_id>`); no individual clips were cherry-picked.",
        "",
        "## Video quality and preprocessing",
        "",
        f"- QC-valid selected videos: {sum(row['validation_status'] == 'VALID' for row in selected)}",
        f"- QC-rejected selected videos: {len(rejected)}",
        f"- Processed videos: {sum(row['processing_status'] == 'PROCESSED' for row in selected)}",
        f"- Processing failures: {len(failed)}",
        *rejected_lines,
        f"- Generated sequences: {len(sequences)}",
        f"- Total decoded frames in processed videos: {sum(int(row['usable_frames'] or 0) for row in selected)}",
        "- Detection rates (mean across processed videos): "
        + ", ".join(
            f"{name} {np.mean([float(row[key]) for row in selected if row['processing_status'] == 'PROCESSED']):.3f}"
            for name, key in (
                ("left hand", "left_hand_detection_rate"),
                ("right hand", "right_hand_detection_rate"),
                ("pose", "pose_detection_rate"),
                ("face", "face_detection_rate"),
            )
        ),
        "- Sequence shape: 30 x 562; window length 30, stride 15.",
        "- Features: existing MediaPipe Tasks Holistic, project wrist-relative normalization, presence features, "
        "and 562-dimensional frame representation. Original FPS timestamps were retained.",
        "- Sequence manifest: `../metadata/sequences.csv`; QC manifest: `../metadata/selected_videos.csv`.",
        "",
        "## Signer-aware splits",
        "",
        f"- Split basis: {split_summary['split_basis']}; signer separation: {split_summary['signer_separation']}.",
        f"- Train: {video_counts['train']} videos, {split_sequence_counts['train']} sequences, {signer_counts['train']} signers.",
        f"- Validation: {video_counts['validation']} videos, {split_sequence_counts['validation']} sequences, {signer_counts['validation']} signers.",
        f"- Test: {video_counts['test']} videos, {split_sequence_counts['test']} sequences, {signer_counts['test']} signers.",
        f"- Classes per split: {json.dumps(classes_per_split, sort_keys=True)}.",
        f"- Signer leakage check: {'PASS' if split_summary.get('signer_overlap') is False else 'FAIL'}; "
        f"signers by split: {json.dumps({key: sorted({r['signer_id'] for r in sequences if r['experiment_split'] == key}) for key in ('train', 'validation', 'test')}, sort_keys=True)}.",
        "",
        "## Model and training",
        "",
        f"- Architecture: bidirectional LSTM; input 562, sequence length 30, hidden size {training['hidden_size']}, "
        f"{training['num_layers']} layers, dropout {training['dropout']}.",
        f"- Training: {training['epochs']} epochs maximum, batch size {training['batch_size']}, "
        f"learning rate {training['learning_rate']}, seed {training['seed']}.",
        f"- Best validation checkpoint (lowest validation loss): `{training['best_checkpoint']}`.",
        (
            f"- Best validation epoch: {int(best_epoch['epoch'])}; loss {best_epoch['validation_loss']:.6f}; "
            f"accuracy {best_epoch['validation_accuracy']:.4f}."
            if best_epoch else "- Best validation result: not available."
        ),
        "- Checkpoint and artifacts: `../../../models/experimental/lsa64/`.",
        "",
        "### Training history",
        "",
        "| Epoch | Training loss | Validation loss | Validation accuracy |",
        "|---:|---:|---:|---:|",
        *(history_lines or ["| - | - | - | - |"]),
        "",
        "## Held-out test evaluation",
        "",
        f"- Test accuracy: {metrics['accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Confusion matrix artifact: `../../../models/experimental/lsa64/confusion_matrix.json`.",
        f"- Per-window held-out predictions: `../../../models/experimental/lsa64/test_predictions.csv` "
        f"({len(predictions)} rows).",
        "",
        "### Held-out per-window predictions (first 20)",
        "",
        *(prediction_lines or ["- No per-window predictions were emitted."]),
        "",
        "| LSA64 class ID | Precision | Recall | F1 | Support (test windows) |",
        "|---|---:|---:|---:|---:|",
        *result_lines,
        "",
        "Confusion matrix (rows: actual, columns: predicted; label order follows the per-class table):",
        "",
        "| Actual / Predicted | " + " | ".join(matrix_labels) + " |",
        "|---|" + "---:|" * len(matrix_labels),
        *matrix_lines,
        "",
        "### Inference on a held-out test video",
        "",
        f"- Video: `{inference['video_id']}` (signer `{inference['signer_id']}`, "
        f"repetition `{inference['repetition_id']}`).",
        f"- Predicted: `{inference['predicted_class']}`",
        f"- Confidence: {inference['confidence']:.6f}",
        f"- Actual: `{inference['actual_class']}`",
        f"- Result JSON: `held_out_video_inference.json`.",
        "",
        "## Limitations",
        "",
        "- Class glosses and license information were not present in the downloaded files; numeric class IDs "
        "are used without translating them.",
        "- This is a five-class (configurable to 5-10), local subset from one dataset source, with only ten "
        "encoded signers and signer-held-out evaluation; it is not a production model.",
        "- MediaPipe detection failures and dataset-specific capture conditions may limit generalization.",
        "- No live webcam mode was enabled; production inference code was intentionally left unchanged.",
        "- LSA64 is Argentine Sign Language and is NOT Malawian Sign Language (MSL).",
        "",
        "## Reproduction",
        "",
        "From the project root, run `python -m tools.experimental_dynamic.lsa64 prepare --source-dir "
        "<extracted-LSA64-all_cut> --classes 5 --seed 42`, then run `python -m "
        "tools.experimental_dynamic.lsa64 train --epochs 10 --batch-size 8 --learning-rate 0.001 --seed 42`.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and train the isolated LSA64 dynamic-sign experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Inspect, select, QC, and preprocess a seeded LSA64 subset.")
    prepare.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--output-root", type=Path, default=DATA_ROOT)
    prepare.add_argument("--classes", type=int, default=DEFAULT_CLASS_COUNT)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train = subparsers.add_parser("train", help="Train and evaluate on the prepared LSA64 sequences.")
    train.add_argument("--output-root", type=Path, default=DATA_ROOT)
    train.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Generate evaluation/report artifacts from an existing completed experimental checkpoint.",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_dataset(args.source_dir, args.output_root, class_count=args.classes, seed=args.seed)
    else:
        result = train_and_report(
            args.output_root,
            args.model_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            reuse_existing=args.reuse_existing,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
