from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental_dynamic import baseline
from tools.msl_dataset.msl_dataset_pipeline import load_config as load_msl_config, process_video


DATA_ROOT = PROJECT_ROOT / "data" / "experimental_dynamic" / "wlasl"
METADATA_PATH = DATA_ROOT / "metadata" / "WLASL_v0.3.json"
MANIFEST_PATH = DATA_ROOT / "manifests" / "video_manifest.csv"
SEQUENCE_MANIFEST_PATH = DATA_ROOT / "manifests" / "sequences.csv"
QC_PATH = DATA_ROOT / "reports" / "dataset_qc.json"
MODEL_ROOT = PROJECT_ROOT / "models" / "experimental" / "wlasl"
METRICS_ROOT = PROJECT_ROOT / "metrics" / "experimental" / "wlasl"
REPORT_ROOT = PROJECT_ROOT / "reports" / "experimental" / "wlasl"
PROCESSED_ROOT = DATA_ROOT / "processed"
RAW_ROOT = DATA_ROOT / "raw"
CLIPS_ROOT = DATA_ROOT / "clips"

DEFAULT_GLOSSES = ("book", "drink", "computer", "chair", "who")
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}
VIDEO_FIELDS = [
    "gloss", "video_id", "signer_id", "source", "source_split", "url",
    "fps", "source_fps", "target_fps", "frame_start", "frame_end", "variation_id", "bbox",
    "download_status", "download_error", "downloaded_path", "local_video_path",
    "download_sha256", "download_bytes", "decoded_frames", "process_status",
    "process_error", "sequences_generated",
]
SEQUENCE_FIELDS = [
    "sample_id", "group_id", "label", "signer_id", "source_split",
    "sequence_path", "local_video_path", "video_id",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def select_candidates(
    metadata: list[dict[str, Any]],
    glosses: tuple[str, ...] = DEFAULT_GLOSSES,
    candidates_per_class: int = 16,
) -> list[dict[str, Any]]:
    """Select unique-signer, direct video instances without altering WLASL glosses."""
    by_gloss = {str(row.get("gloss", "")): row for row in metadata}
    missing = [gloss for gloss in glosses if gloss not in by_gloss]
    if missing:
        raise ValueError(f"Requested glosses are absent from WLASL metadata: {missing}")
    rows: list[dict[str, Any]] = []
    for gloss in glosses:
        seen_signers: set[str] = set()
        entry = by_gloss[gloss]
        for instance in entry.get("instances", []):
            url = str(instance.get("url", "")).strip()
            parsed = urllib.parse.urlparse(url)
            signer = str(instance.get("signer_id", "")).strip()
            if (
                parsed.scheme not in {"http", "https"}
                or Path(parsed.path).suffix.lower() not in VIDEO_EXTENSIONS
                or not signer
                or signer.casefold() in baseline.UNKNOWN_SIGNERS
                or signer in seen_signers
            ):
                continue
            seen_signers.add(signer)
            rows.append({
                "gloss": gloss,
                "video_id": str(instance.get("video_id", "")),
                "signer_id": signer,
                "source": str(instance.get("source", "")),
                "source_split": str(instance.get("split", "")),
                "url": url,
                "fps": str(instance.get("fps", 25)),
                "source_fps": "",
                "target_fps": "25",
                "frame_start": str(instance.get("frame_start", 1)),
                "frame_end": str(instance.get("frame_end", -1)),
                "variation_id": str(instance.get("variation_id", "")),
                "bbox": json.dumps(instance.get("bbox", []), separators=(",", ":")),
                "download_status": "PENDING",
                "download_error": "",
                "downloaded_path": "",
                "local_video_path": "",
                "download_sha256": "",
                "download_bytes": "0",
                "decoded_frames": "0",
                "process_status": "PENDING",
                "process_error": "",
                "sequences_generated": "0",
            })
            if len(seen_signers) >= candidates_per_class:
                break
        if not any(row["gloss"] == gloss for row in rows):
            raise ValueError(f"No usable direct-video instances with signer IDs for WLASL gloss {gloss!r}.")
    return rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required WLASL manifest is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def create_manifest(
    glosses: tuple[str, ...] = DEFAULT_GLOSSES,
    candidates_per_class: int = 16,
    *,
    force: bool = False,
) -> Path:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Official WLASL metadata is missing: {METADATA_PATH}")
    if MANIFEST_PATH.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {MANIFEST_PATH}. "
            "Use --force only if you intend to regenerate it."
        )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    rows = select_candidates(metadata, glosses, candidates_per_class)
    _write_csv(MANIFEST_PATH, VIDEO_FIELDS, rows)
    return MANIFEST_PATH


def _decode_video(path: Path) -> tuple[int, str | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0, "OpenCV could not open the downloaded file."
    decoded = 0
    first_frame_ok = False
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded += 1
            first_frame_ok = True
    finally:
        capture.release()
    if not first_frame_ok or decoded < 12:
        return decoded, f"Video decoded only {decoded} frames; at least 12 are required."
    return decoded, None


def _download_one(row: dict[str, str], destination: Path, max_bytes: int) -> tuple[int, str]:
    if not _is_within(destination, DATA_ROOT):
        raise ValueError(f"Refusing to write outside the WLASL experiment directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        row["url"],
        headers={"User-Agent": "BridgeSign WLASL experiment (metadata-listed dataset video)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=35) as response, temporary.open("wb") as output:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError(f"Video URL returned HTML ({content_type}).")
            size = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f"Download exceeded the {max_bytes}-byte safety limit.")
                output.write(block)
        if temporary.stat().st_size < 10_000:
            raise ValueError(f"Downloaded payload is only {temporary.stat().st_size} bytes, not a usable video.")
        decoded_frames, error = _decode_video(temporary)
        if error:
            raise ValueError(error)
        os.replace(temporary, destination)
        return decoded_frames, ""
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def download_subset(
    videos_per_class: int = 8,
    max_bytes: int = 120 * 1024 * 1024,
) -> dict[str, int]:
    rows = _read_csv(MANIFEST_PATH)
    succeeded = Counter()
    failed = Counter()
    for row in rows:
        if row["download_status"] in {"DOWNLOADED", "DOWNLOADED_EXISTING"}:
            existing = Path(row["downloaded_path"])
            if existing.is_file():
                succeeded[row["gloss"]] += 1
                continue
            row["download_status"] = "MISSING"
            row["download_error"] = "Manifest artifact path no longer exists."
        if succeeded[row["gloss"]] >= videos_per_class:
            if row["download_status"] == "PENDING":
                row["download_status"] = "NOT_ATTEMPTED_TARGET_MET"
            continue
        extension = Path(urllib.parse.urlparse(row["url"]).path).suffix.lower()
        destination = RAW_ROOT / row["gloss"] / f"{row['video_id']}{extension}"
        if destination.is_file():
            frames, error = _decode_video(destination)
            if error:
                row["download_status"] = "REJECTED"
                row["download_error"] = error
                failed[row["gloss"]] += 1
                continue
            row["download_status"] = "DOWNLOADED_EXISTING"
            row["decoded_frames"] = str(frames)
            row["downloaded_path"] = str(destination.resolve())
            row["download_sha256"] = _sha256(destination)
            row["download_bytes"] = str(destination.stat().st_size)
            succeeded[row["gloss"]] += 1
            continue
        try:
            frames, error = _download_one(row, destination, max_bytes)
            row["download_status"] = "DOWNLOADED"
            row["download_error"] = error
            row["downloaded_path"] = str(destination.resolve())
            row["download_sha256"] = _sha256(destination)
            row["download_bytes"] = str(destination.stat().st_size)
            row["decoded_frames"] = str(frames)
            succeeded[row["gloss"]] += 1
            print(f"DOWNLOADED {row['gloss']} video_id={row['video_id']} signer={row['signer_id']} ({frames} frames)")
        except Exception as exc:
            row["download_status"] = "FAILED"
            row["download_error"] = f"{type(exc).__name__}: {exc}"
            failed[row["gloss"]] += 1
            print(f"FAILED {row['gloss']} video_id={row['video_id']}: {row['download_error']}")
        _write_csv(MANIFEST_PATH, VIDEO_FIELDS, rows)
    _write_csv(MANIFEST_PATH, VIDEO_FIELDS, rows)
    return {"downloaded": sum(succeeded.values()), "failed": sum(failed.values())}


def _materialize_segment(row: dict[str, str]) -> Path:
    source = Path(row["downloaded_path"]).resolve()
    if not _is_within(source, DATA_ROOT) or not source.is_file():
        raise ValueError(f"Downloaded video is missing or outside the WLASL data root: {source}")
    start = int(row.get("frame_start") or 1)
    end = int(row.get("frame_end") or -1)
    if start <= 1 and end == -1:
        row["local_video_path"] = str(source)
        return source
    fps_meta = float(row.get("fps") or 25)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not reopen downloaded video: {source}")
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps_meta)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_index = max(0, round((start - 1) * actual_fps / fps_meta))
    end_index = total if end == -1 else min(total, round(end * actual_fps / fps_meta))
    if start_index >= end_index or end_index - start_index < 12:
        capture.release()
        raise ValueError(
            f"Metadata frame interval [{start}, {end}] yields fewer than 12 frames "
            f"from {total}-frame video at {actual_fps:g}fps."
        )
    segment = CLIPS_ROOT / row["gloss"] / f"{row['video_id']}_segment.mp4"
    if not _is_within(segment, DATA_ROOT):
        capture.release()
        raise ValueError(f"Refusing to write video segment outside WLASL data root: {segment}")
    segment.parent.mkdir(parents=True, exist_ok=True)
    temporary = segment.with_name(segment.stem + ".part.mp4")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create metadata-trimmed video: {temporary}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    written = 0
    try:
        for _ in range(end_index - start_index):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written < 12 or temporary.stat().st_size < 10_000:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Metadata-trimmed segment is unusable ({written} decoded frames).")
    os.replace(temporary, segment)
    row["local_video_path"] = str(segment.resolve())
    return segment.resolve()


def _resample_video(source: Path, row: dict[str, str], target_fps: float = 25.0) -> Path:
    """Resample metadata-trimmed WLASL footage to its documented 25fps cadence."""
    if target_fps <= 0:
        raise ValueError("Target video FPS must be positive.")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video for FPS normalization: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or target_fps)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    row["source_fps"] = f"{source_fps:.3f}"
    row["target_fps"] = f"{target_fps:.3f}"
    target = DATA_ROOT / "resampled" / row["gloss"] / f"{row['video_id']}_{target_fps:g}fps.mp4"
    if not _is_within(target, DATA_ROOT):
        capture.release()
        raise ValueError(f"Refusing to write resampled video outside the WLASL data root: {target}")
    if target.is_file():
        existing = cv2.VideoCapture(str(target))
        existing_fps = float(existing.get(cv2.CAP_PROP_FPS) or 0.0)
        existing.release()
        _decoded, error = _decode_video(target)
        if not error and abs(existing_fps - target_fps) < 0.05:
            capture.release()
            row["local_video_path"] = str(target.resolve())
            return target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".part.mp4")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create 25fps WLASL video: {temporary}")
    frame_index = 0
    output_count = 0
    next_output_time = 0.0
    duration = frame_count / source_fps if source_fps > 0 else 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_time = frame_index / source_fps
            while next_output_time < duration and next_output_time <= source_time + 1e-8:
                writer.write(frame)
                output_count += 1
                next_output_time += 1.0 / target_fps
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if output_count < 12 or not temporary.is_file() or temporary.stat().st_size < 10_000:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"FPS-normalized video is unusable ({output_count} output frames).")
    os.replace(temporary, target)
    row["local_video_path"] = str(target.resolve())
    return target.resolve()


def _feature_config() -> dict[str, Any]:
    config = load_msl_config()
    config["raw_dataset_path"] = str(RAW_ROOT)
    config["processed_dataset_path"] = str(PROCESSED_ROOT)
    config["manifests_path"] = str((DATA_ROOT / "manifests").resolve())
    config["review_path"] = str((DATA_ROOT / "review").resolve())
    config["sequence_length"] = 30
    config["stride"] = 30
    config["target_fps"] = 25.0
    config["min_frames"] = 12
    config["max_frames"] = 300
    config["random_seed"] = 42
    config["dataset_version"] = "WLASL_v0.3"
    config["normalization_method"] = "wrist_relative"
    config["allow_padding"] = True
    return config


def preprocess_subset() -> dict[str, int]:
    rows = _read_csv(MANIFEST_PATH)
    config = _feature_config()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    sequence_rows: list[dict[str, str]] = []
    processed_videos = 0
    rejected_videos = 0
    generated_sequences = 0
    qc_by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"videos": 0, "decoded_frames": 0, "sequences_generated": 0, "sequences_selected": 0}
    )
    for row in rows:
        if row["download_status"] not in {"DOWNLOADED", "DOWNLOADED_EXISTING"}:
            continue
        try:
            video_path = _materialize_segment(row)
            video_path = _resample_video(video_path, row, float(config["target_fps"]))
            decoded_frames, decode_error = _decode_video(video_path)
            if decode_error:
                raise ValueError(decode_error)
            result = process_video(
                video_path,
                row["signer_id"],
                row["gloss"],
                config=config,
                session_id=f"wlasl_{row['video_id']}",
            )
            windows = result["sequence_windows"]
            if not windows:
                raise ValueError("Holistic preprocessing produced no temporal sequences.")
            generated_sequences += len(windows)
            qc_by_class[row["gloss"]]["videos"] += 1
            qc_by_class[row["gloss"]]["decoded_frames"] += decoded_frames
            qc_by_class[row["gloss"]]["sequences_generated"] += len(windows)
            # One center window per source video avoids treating highly correlated windows as independent samples.
            selected = windows[len(windows) // 2]
            with np.load(selected["sequence_path"], allow_pickle=False) as stored:
                presence = np.asarray(stored["landmark_presence"], dtype=np.float32)
                mask = np.asarray(stored["valid_frame_mask"], dtype=np.float32)
            if presence.ndim != 2 or not np.isfinite(presence).all() or not np.any(presence * mask[:, None]):
                raise ValueError("Selected sequence contains no detected Holistic landmarks.")
            sample_id = f"wlasl_{row['video_id']}"
            sequence_rows.append({
                "sample_id": sample_id,
                "group_id": row["video_id"],
                "label": row["gloss"],
                "signer_id": row["signer_id"],
                "source_split": "",
                "sequence_path": str(Path(selected["sequence_path"]).resolve()),
                "local_video_path": str(video_path),
                "video_id": row["video_id"],
            })
            qc_by_class[row["gloss"]]["sequences_selected"] += 1
            row["process_status"] = "PROCESSED"
            row["process_error"] = ""
            row["sequences_generated"] = str(len(windows))
            processed_videos += 1
            print(
                f"PROCESSED {row['gloss']} video_id={row['video_id']} "
                f"frames={decoded_frames} sequences={len(windows)}"
            )
        except Exception as exc:
            row["process_status"] = "REJECTED"
            row["process_error"] = f"{type(exc).__name__}: {exc}"
            rejected_videos += 1
            print(f"REJECTED {row['gloss']} video_id={row['video_id']}: {row['process_error']}")
        _write_csv(MANIFEST_PATH, VIDEO_FIELDS, rows)

    _write_csv(MANIFEST_PATH, VIDEO_FIELDS, rows)
    _write_csv(SEQUENCE_MANIFEST_PATH, SEQUENCE_FIELDS, sequence_rows)
    summary = {
        "dataset": "WLASL v0.3",
        "language": "ASL",
        "status": "QC_COMPLETE" if sequence_rows else "NO_USABLE_SEQUENCES",
        "metadata_path": str(METADATA_PATH.resolve()),
        "metadata_sha256": _sha256(METADATA_PATH),
        "videos_downloaded": sum(row["download_status"] in {"DOWNLOADED", "DOWNLOADED_EXISTING"} for row in rows),
        "videos_processed": processed_videos,
        "videos_rejected_during_processing": rejected_videos,
        "temporal_sequences_generated": generated_sequences,
        "sequence_samples_selected_for_training": len(sequence_rows),
        "target_video_fps": config["target_fps"],
        "classes": sorted({row["label"] for row in sequence_rows}),
        "signer_count": len({row["signer_id"] for row in sequence_rows}),
        "signer_ids": sorted({row["signer_id"] for row in sequence_rows}),
        "class_statistics": dict(qc_by_class),
        "feature_config": config,
        "source_manifest": str(MANIFEST_PATH.resolve()),
        "sequence_manifest": str(SEQUENCE_MANIFEST_PATH.resolve()),
        "selection_policy": "One center sequence per WLASL source video; all windows from any source stay together.",
        "disclaimer": "WLASL is an ASL dataset and is being used here to validate the dynamic video-recognition engineering pipeline. It is not Malawi Sign Language.",
    }
    QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    QC_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (DATA_ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "videos_processed": processed_videos,
        "videos_rejected": rejected_videos,
        "sequences_generated": generated_sequences,
        "training_samples": len(sequence_rows),
    }


def _select_signer_split_seed(manifest_path: Path, first_seed: int = 42, attempts: int = 20_000) -> int:
    samples = baseline.load_samples(manifest_path, language="ASL")
    labels = {sample.label for sample in samples}
    if len(labels) < 2:
        raise ValueError("At least two WLASL glosses need usable processed videos before training.")
    for seed in range(first_seed, first_seed + attempts):
        split_rows, summary = baseline.create_splits(samples, seed)
        if summary["signer_separation"] != "AVAILABLE" or summary["signer_overlap"]:
            continue
        split_labels: dict[str, set[str]] = {name: set() for name in baseline.SPLIT_NAMES}
        for row in split_rows:
            split_labels[row["split"]].add(row["label"])
        if all(labels.issubset(split_labels[split]) for split in baseline.SPLIT_NAMES):
            return seed
    raise ValueError(
        "Could not find a deterministic signer-disjoint split containing every gloss in all three partitions. "
        "Acquire more distinct WLASL signers per class before training; no fallback video-random split was used."
    )


def train_subset(epochs: int = 20, batch_size: int = 8) -> dict[str, Any]:
    if not SEQUENCE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("Run the WLASL Holistic preprocessing stage before training.")
    seed = _select_signer_split_seed(SEQUENCE_MANIFEST_PATH)
    result = baseline.train_baseline(
        SEQUENCE_MANIFEST_PATH,
        DATA_ROOT,
        MODEL_ROOT,
        dataset="WLASL v0.3",
        language="ASL",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        seed=seed,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    )
    metrics = result["metrics"]
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    metrics["manifest_sha256"] = _sha256(SEQUENCE_MANIFEST_PATH)
    metrics["metadata_sha256"] = _sha256(METADATA_PATH)
    metrics["signer_disjoint_split_seed"] = seed
    (MODEL_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(MODEL_ROOT / "metrics.json", METRICS_ROOT / "metrics.json")
    config = json.loads((MODEL_ROOT / "config.json").read_text(encoding="utf-8"))
    split_rows = _read_csv(DATA_ROOT / "splits" / "split_manifest.csv")
    signer_split = {
        split: {
            "video_count": sum(row["split"] == split for row in split_rows),
            "signer_ids": sorted({row["signer_id"] for row in split_rows if row["split"] == split and row["signer_id"]}),
            "classes": sorted({row["label"] for row in split_rows if row["split"] == split}),
        }
        for split in baseline.SPLIT_NAMES
    }
    signer_split["signer_overlap"] = not (
        set(signer_split["train"]["signer_ids"]).isdisjoint(signer_split["validation"]["signer_ids"])
        and set(signer_split["train"]["signer_ids"]).isdisjoint(signer_split["test"]["signer_ids"])
        and set(signer_split["validation"]["signer_ids"]).isdisjoint(signer_split["test"]["signer_ids"])
    )
    (DATA_ROOT / "reports" / "signer_split.json").write_text(
        json.dumps(signer_split, indent=2) + "\n", encoding="utf-8"
    )
    config.update({
        "dataset_version": "WLASL_v0.3",
        "dataset_manifest": str(MANIFEST_PATH.resolve()),
        "sequence_manifest": str(SEQUENCE_MANIFEST_PATH.resolve()),
        "metadata_sha256": _sha256(METADATA_PATH),
        "sequence_manifest_sha256": _sha256(SEQUENCE_MANIFEST_PATH),
        "feature_config": _feature_config(),
        "stride": int(_feature_config()["stride"]),
        "fps_handling": "Source clips resampled to 25fps from timestamps; webcam sequence sampling uses the same target FPS.",
        "signer_split_seed": seed,
        "signer_split": signer_split,
        "selected_video_ids": sorted({row["group_id"] for row in split_rows}),
        "checkpoint_path": str((MODEL_ROOT / "best_model.pt").resolve()),
    })
    (MODEL_ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (METRICS_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_report(metrics)
    return result


def write_report(metrics: dict[str, Any] | None = None) -> Path:
    qc = json.loads(QC_PATH.read_text(encoding="utf-8")) if QC_PATH.is_file() else {}
    split_path = DATA_ROOT / "splits" / "split_summary.json"
    split_summary = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}
    signer_split_path = DATA_ROOT / "reports" / "signer_split.json"
    signer_split = json.loads(signer_split_path.read_text(encoding="utf-8")) if signer_split_path.is_file() else {}
    rows = _read_csv(MANIFEST_PATH) if MANIFEST_PATH.is_file() else []
    sequence_rows = _read_csv(SEQUENCE_MANIFEST_PATH) if SEQUENCE_MANIFEST_PATH.is_file() else []
    model_config_path = MODEL_ROOT / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8")) if model_config_path.is_file() else {}
    qc["signer_count"] = len({row["signer_id"] for row in sequence_rows if row.get("signer_id")})
    qc["signer_ids"] = sorted({row["signer_id"] for row in sequence_rows if row.get("signer_id")})
    if QC_PATH.is_file():
        QC_PATH.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    classes = sorted({row["gloss"] for row in rows if row.get("download_status") in {"DOWNLOADED", "DOWNLOADED_EXISTING"}})
    download_failures = Counter()
    for row in rows:
        if row.get("download_status") != "FAILED":
            continue
        error = row.get("download_error", "")
        if "404" in error:
            category = "HTTP 404"
        elif "403" in error:
            category = "HTTP 403"
        elif "CERTIFICATE_VERIFY_FAILED" in error:
            category = "TLS certificate mismatch"
        elif "HTML" in error:
            category = "HTML response, not video"
        else:
            category = "Other recorded download error"
        download_failures[category] += 1
    lines = [
        "# WLASL Dynamic Sign Experiment",
        "",
        "**Status:** " + ("TRAINED (EXPERIMENTAL)" if metrics else "ACQUISITION / PROCESSING STATUS"),
        "",
        "> WLASL is an ASL dataset and is being used here to validate the dynamic video-recognition engineering pipeline. It is not Malawi Sign Language.",
        "",
        "The experiment is isolated from production BridgeSign Live, static A-Z, Call, Learn, MSL production data, and LSA64 assets.",
        "Videos are obtained only through URLs listed in official WLASL v0.3 metadata. WLASL requires compliance with its C-UDA and is for academic/computational use only.",
        "",
        "## Acquisition and quality control",
        "",
        f"- Selected WLASL glosses: {', '.join(classes) if classes else 'not yet downloaded'}",
        f"- Metadata candidates in manifest: {len(rows)}",
        f"- Actual videos downloaded and decoded: {qc.get('videos_downloaded', 0)}",
        f"- Failed/unavailable metadata URLs: {sum(download_failures.values())}; reasons: `{json.dumps(dict(download_failures), sort_keys=True)}`",
        f"- Videos successfully processed by Holistic: {qc.get('videos_processed', 0)}",
        f"- Videos rejected during processing: {qc.get('videos_rejected_during_processing', 0)}",
        f"- Temporal windows generated: {qc.get('temporal_sequences_generated', 0)}",
        f"- Center sequence samples selected: {qc.get('sequence_samples_selected_for_training', 0)}",
        f"- Unique signers represented: {qc.get('signer_count', 0)}",
        f"- Downloaded WLASL video IDs: `{json.dumps({label: [row['video_id'] for row in rows if row.get('gloss') == label and row.get('download_status') in {'DOWNLOADED', 'DOWNLOADED_EXISTING'}] for label in classes}, sort_keys=True)}`",
        f"- Per-class QC: `{json.dumps(qc.get('class_statistics', {}), sort_keys=True)}`",
        "",
        "## Representation and model",
        "",
        "- Training and webcam inference use the MSL pipeline's MediaPipe Holistic per-frame features, wrist-relative normalization, and fixed 30-frame sequence.",
        f"- Model input is `[batch, {model_config.get('sequence_length', 30)}, {model_config.get('feature_dimension', 0)}]`.",
        "- WLASL video clips are resampled to 25fps (the dataset's documented decoding cadence); the webcam demo samples at the same cadence.",
        "- This representation is not the production motion model's hand-only 15 x 42 input and is not routed through the production GestureClassifier.",
        f"- Model: `{MODEL_ROOT / 'best_model.pt'}`",
        "- Architecture: isolated experimental bidirectional LSTM with 2 recurrent layers; class labels are the official WLASL glosses preserved from metadata.",
        "- Live demo status: isolated webcam entry point is ready; the checkpoint loaded and inferred on a real extracted sequence. Physical webcam interaction was not exercised in this run.",
        "",
        "## Split and held-out evaluation",
        "",
        f"- Split summary: `{json.dumps(split_summary, sort_keys=True)}`",
        f"- Signer IDs by partition: `{json.dumps(signer_split, sort_keys=True)}`",
    ]
    if metrics:
        lines.extend([
            f"- Test accuracy: {metrics.get('accuracy', 0):.4f}",
            f"- Test macro F1: {metrics.get('macro_f1', 0):.4f}",
            f"- Held-out sequences: {metrics.get('test_samples', 0)}",
            f"- Per-class metrics: `{json.dumps(metrics.get('per_class', {}), sort_keys=True)}`",
            f"- Confusion matrix: `{json.dumps(metrics.get('confusion_matrix', []))}`",
            f"- Predictions: `{MODEL_ROOT / 'test_predictions.csv'}`",
        ])
    else:
        lines.append("No model was trained yet; held-out metrics are not available.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is a small ASL engineering experiment, not MSL recognition and not a claim of general ASL accuracy.",
        "- A signer-disjoint test on a small subset has high variance; inspect the per-class support and confusion matrix.",
        "- The webcam demo is a closed-set classifier and its confidence is not calibrated; predictions below the configured display threshold are shown as uncertain.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python tools\\experimental_dynamic\\wlasl_experiment.py manifest --candidates-per-class 16",
        "python tools\\experimental_dynamic\\wlasl_experiment.py download --videos-per-class 8",
        "python tools\\experimental_dynamic\\wlasl_experiment.py process",
        "python tools\\experimental_dynamic\\wlasl_experiment.py train --epochs 20 --batch-size 8",
        "python tools\\experimental_dynamic\\wlasl_live_demo.py --camera 0",
        "```",
        "",
    ])
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / "wlasl_experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_all(epochs: int, batch_size: int, videos_per_class: int) -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        print(f"Using existing acquisition manifest: {MANIFEST_PATH}")
    else:
        print(f"Created acquisition manifest: {create_manifest()}")
    download_result = download_subset(videos_per_class=videos_per_class)
    if download_result["downloaded"] <= 0:
        write_report()
        raise RuntimeError("No actual WLASL videos were downloaded; preprocessing and training were not attempted.")
    qc_result = preprocess_subset()
    if qc_result["training_samples"] <= 0:
        write_report()
        raise RuntimeError("No usable Holistic sequences were generated; model training was not attempted.")
    trained = train_subset(epochs=epochs, batch_size=batch_size)
    report_path = write_report(trained["metrics"])
    return {
        "downloads": download_result,
        "preprocessing": qc_result,
        "training": trained,
        "report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated WLASL acquisition, Holistic preprocessing, and BiLSTM experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("manifest", help="Create the WLASL metadata-derived subset manifest before downloading.")
    create.add_argument("--force", action="store_true")
    create.add_argument("--candidates-per-class", type=int, default=16)
    download = subparsers.add_parser("download", help="Download only manifest-listed WLASL direct-video URLs.")
    download.add_argument("--videos-per-class", type=int, default=8)
    preprocess = subparsers.add_parser("process", help="Run the isolated MediaPipe Holistic preprocessing.")
    train = subparsers.add_parser("train", help="Train/evaluate the isolated signer-disjoint BiLSTM.")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=8)
    all_command = subparsers.add_parser("all", help="Run manifest, acquisition, preprocessing, training, and reporting.")
    all_command.add_argument("--videos-per-class", type=int, default=8)
    all_command.add_argument("--epochs", type=int, default=20)
    all_command.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.command == "manifest":
        print(create_manifest(candidates_per_class=args.candidates_per_class, force=args.force))
    elif args.command == "download":
        print(json.dumps(download_subset(videos_per_class=args.videos_per_class), indent=2))
    elif args.command == "process":
        print(json.dumps(preprocess_subset(), indent=2))
    elif args.command == "train":
        print(json.dumps(train_subset(args.epochs, args.batch_size), indent=2))
    else:
        print(json.dumps(run_all(args.epochs, args.batch_size, args.videos_per_class), indent=2))


if __name__ == "__main__":
    main()
