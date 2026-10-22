from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional for preprocessing-only environments
    torch = None

from tools.msl_dataset.msl_dataset_pipeline import (
    DEFAULT_FACE_INDICES,
    PROJECT_ROOT,
    load_config as load_msl_config,
    process_video as process_landmark_video,
)


DATASET_ROOT = PROJECT_ROOT / "data" / "external_datasets" / "msasl"
METADATA_DIR = DATASET_ROOT / "metadata"
RAW_DIR = DATASET_ROOT / "raw"
CLIPS_DIR = RAW_DIR / "clips"
PROCESSED_DIR = DATASET_ROOT / "processed"
MANIFESTS_DIR = DATASET_ROOT / "manifests"
REPORTS_DIR = DATASET_ROOT / "reports"
REVIEW_DIR = DATASET_ROOT / "review"
SPLIT_NAMES = ("train", "val", "test")
METADATA_FILES = {
    "train": "MSASL_train.json",
    "val": "MSASL_val.json",
    "test": "MSASL_test.json",
    "classes": "MSASL_classes.json",
    "synonym": "MSASL_synonym.json",
}
MANIFEST_FIELDS = [
    "dataset", "source_split", "source_record_id", "video_id", "source_url",
    "label_id", "label", "signer_id", "start_time", "end_time", "width",
    "height", "fps", "local_video_path", "clip_path", "download_status",
    "download_error", "processing_status", "processing_error", "notes",
]
SEQUENCE_FIELDS = [
    "sample_id", "dataset", "language", "source_record_id", "source_split", "video_id", "source_url",
    "label_id", "label", "signer_id", "start_time", "end_time", "sequence_path",
    "frame_count", "feature_count", "processing_status",
]
SUPPORTED_RAW_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_FEATURES = {
    "left_hand": True,
    "right_hand": True,
    "pose": True,
    "face": True,
    "pose_landmark_subset": list(range(33)),
}
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


def ensure_dataset_directories(root: str | Path = DATASET_ROOT) -> dict[str, Path]:
    root = Path(root)
    dirs = {
        "root": root,
        "metadata": root / "metadata",
        "raw": root / "raw",
        "clips": root / "raw" / "clips",
        "processed": root / "processed",
        "manifests": root / "manifests",
        "reports": root / "reports",
        "review": root / "review",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_metadata_sources(source: str | Path) -> tuple[dict[str, Any], str]:
    source_path = Path(source)
    if source_path.is_dir():
        files = {path.name.casefold(): path.read_bytes() for path in source_path.iterdir() if path.is_file()}
        origin = str(source_path.resolve())
    elif source_path.is_file() and zipfile.is_zipfile(source_path):
        files = {}
        with zipfile.ZipFile(source_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                basename = Path(member.filename).name
                if basename.casefold() in {name.casefold() for name in METADATA_FILES.values()} or (
                    "license" in basename.casefold() or "readme" in basename.casefold()
                ):
                    files[basename.casefold()] = archive.read(member)
        origin = str(source_path.resolve())
    elif source_path.is_file() and source_path.suffix.lower() == ".json":
        files = {source_path.name.casefold(): source_path.read_bytes()}
        origin = str(source_path.resolve())
    else:
        raise FileNotFoundError(f"Metadata input must be a directory, JSON file, or ZIP file: {source_path}")

    parsed: dict[str, Any] = {}
    for key, filename in METADATA_FILES.items():
        raw = files.get(filename.casefold())
        if raw is not None:
            try:
                parsed[key] = json.loads(raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not parse {filename}: {exc}") from exc
    parsed["_license_text"] = "\n".join(
        payload.decode("utf-8", errors="replace")[:12000]
        for name, payload in files.items()
        if "license" in name or "readme" in name
    )
    if not any(name in parsed for name in SPLIT_NAMES):
        raise ValueError("No MSASL_train.json, MSASL_val.json, or MSASL_test.json files were found.")
    return parsed, origin


def _as_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "videos", "annotations", "records"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            return [dict(value, **({"video_id": key} if not any(k in value for k in ("video_id", "id")) else {})) for key, value in payload.items()]
    return []


def _class_lookup(payload: Any) -> dict[str, str]:
    if isinstance(payload, list):
        result: dict[str, str] = {}
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                label_id = item.get("label_id", item.get("id", item.get("index", index)))
                label = item.get("label", item.get("text", item.get("name", "")))
                result[str(label_id)] = _safe_text(label)
            else:
                result[str(index)] = _safe_text(item)
        return result
    if isinstance(payload, dict):
        for key in ("classes", "labels", "data"):
            if isinstance(payload.get(key), (list, dict)):
                return _class_lookup(payload[key])
        result: dict[str, str] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                label = value.get("label", value.get("text", value.get("name", "")))
                label_id = value.get("label_id", value.get("id", key))
                result[str(label_id)] = _safe_text(label)
            else:
                result[str(key)] = _safe_text(value)
        return result
    return {}


def _video_id_from_record(record: dict[str, Any], url: str) -> str:
    for key in ("video_id", "videoId", "youtube_id", "youtubeId", "id"):
        value = _safe_text(record.get(key))
        if value:
            return value
    parsed = urllib.parse.urlparse(url)
    query_id = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
    if query_id:
        return query_id
    if parsed.netloc.casefold().endswith("youtu.be"):
        return parsed.path.strip("/").split("/")[0]
    return Path(parsed.path).stem


def _first_value(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def _normalized_record(
    record: dict[str, Any],
    source_split: str,
    index: int,
    classes: dict[str, str],
) -> dict[str, str]:
    url = _safe_text(_first_value(record, ("url", "source_url", "video_url", "youtube_url")))
    raw_label_id = _first_value(record, ("label_id", "label", "class_id", "class"))
    label_id = _safe_text(raw_label_id)
    explicit_label = _safe_text(_first_value(record, ("clean_text", "text", "gloss", "label_name", "sign")))
    label = explicit_label or classes.get(label_id, "")
    if isinstance(raw_label_id, str) and not raw_label_id.strip().isdigit() and not explicit_label:
        label = raw_label_id.strip()
    video_id = _video_id_from_record(record, url)
    signer_id = _safe_text(_first_value(record, ("signer_id", "signer", "subject_id")))
    start = _finite_number(_first_value(record, ("start_time", "start", "start_seconds")))
    end = _finite_number(_first_value(record, ("end_time", "end", "end_seconds")))
    width = _finite_number(record.get("width"))
    height = _finite_number(record.get("height"))
    fps = _finite_number(record.get("fps"))
    source_record_id = f"{source_split}_{index:07d}_{hashlib.sha1(f'{source_split}|{video_id}|{label_id}|{index}'.encode()).hexdigest()[:8]}"
    return {
        "dataset": "MS-ASL",
        "source_split": source_split,
        "source_record_id": source_record_id,
        "video_id": video_id,
        "source_url": url,
        "label_id": label_id,
        "label": label,
        "signer_id": signer_id,
        "start_time": "" if start is None else str(start),
        "end_time": "" if end is None else str(end),
        "width": "" if width is None else str(int(width)),
        "height": "" if height is None else str(int(height)),
        "fps": "" if fps is None else str(fps),
        "local_video_path": "",
        "clip_path": "",
        "download_status": "PENDING",
        "download_error": "",
        "processing_status": "PENDING",
        "processing_error": "",
        "notes": "",
    }


def _write_csv(path: str | Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def select_records(records: list[dict[str, str]], limit: int = 100) -> list[dict[str, str]]:
    if limit < 1:
        raise ValueError("--limit must be a positive integer.")
    grouped: dict[str, list[dict[str, str]]] = {split: [] for split in SPLIT_NAMES}
    for row in records:
        grouped.setdefault(row.get("source_split", ""), []).append(row)
    total = len(records)
    if total <= limit:
        return list(records)
    nonempty = {split: rows for split, rows in grouped.items() if rows}
    quotas = {split: (limit * len(rows)) // total for split, rows in nonempty.items()}
    remaining = limit - sum(quotas.values())
    order = sorted(
        nonempty,
        key=lambda split: ((limit * len(nonempty[split])) % total, -SPLIT_NAMES.index(split)),
        reverse=True,
    )
    for split in order[:remaining]:
        quotas[split] += 1
    selected: list[dict[str, str]] = []
    for split in SPLIT_NAMES:
        rows = nonempty.get(split, [])
        count = quotas.get(split, 0)
        if count == 0:
            continue
        indexes = sorted({min(len(rows) - 1, int((position + 0.5) * len(rows) / count)) for position in range(count)})
        selected.extend(rows[index] for index in indexes)
    selected_ids = {row["source_record_id"] for row in selected}
    if len(selected) < limit:
        selected.extend(row for row in records if row["source_record_id"] not in selected_ids and len(selected) < limit)
    return selected[:limit]


def import_metadata(
    source: str | Path,
    limit: int = 100,
    dataset_root: str | Path = DATASET_ROOT,
) -> dict[str, Any]:
    dirs = ensure_dataset_directories(dataset_root)
    payloads, origin = _read_metadata_sources(source)
    classes = _class_lookup(payloads.get("classes"))
    for key, filename in METADATA_FILES.items():
        if key in payloads:
            (dirs["metadata"] / filename).write_text(
                json.dumps(payloads[key], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    if payloads.get("_license_text"):
        (dirs["metadata"] / "source_license_readme.txt").write_text(
            payloads["_license_text"],
            encoding="utf-8",
        )
    all_rows: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        source_records = _as_records(payloads.get(split))
        source_counts[split] = len(source_records)
        for index, record in enumerate(source_records):
            all_rows.append(_normalized_record(record, split, index, classes))

    selected = select_records(all_rows, limit)
    _write_csv(dirs["metadata"] / "msasl_metadata.csv", MANIFEST_FIELDS, all_rows)
    _write_csv(dirs["manifests"] / "import_manifest.csv", MANIFEST_FIELDS, selected)
    _write_csv(dirs["manifests"] / "sequences.csv", SEQUENCE_FIELDS, [])
    (dirs["metadata"] / "metadata_source.json").write_text(
        json.dumps({
            "dataset": "MS-ASL",
            "language": "American Sign Language (ASL)",
            "source": origin,
            "metadata_records": len(all_rows),
            "selected_records": len(selected),
            "source_split_counts": source_counts,
            "class_definitions_found": len(classes),
            "license_information": payloads.get("_license_text", ""),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "metadata_records": len(all_rows),
        "selected_records": len(selected),
        "source_split_counts": source_counts,
        "manifest_path": str(dirs["manifests"] / "import_manifest.csv"),
        "source": origin,
        "license_information": payloads.get("_license_text", ""),
    }


def _sample_file_stem(row: dict[str, str]) -> str:
    source = "|".join((
        row.get("source_record_id", ""),
        row.get("video_id", ""),
        row.get("label_id", ""),
        row.get("signer_id", ""),
    ))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", row.get("label", "")).strip("_")[:32] or "unlabelled"
    signer = re.sub(r"[^A-Za-z0-9_-]+", "_", row.get("signer_id", "")).strip("_")[:24] or "unknown"
    video_id = re.sub(r"[^A-Za-z0-9_-]+", "_", row.get("video_id", "")).strip("_")[:40] or "unknown"
    return f"{video_id}_{label}_{signer}_{digest}"


def _write_manifest_rows(path: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(path, MANIFEST_FIELDS, rows)


def download_selected(
    manifest_path: str | Path = MANIFESTS_DIR / "import_manifest.csv",
    force: bool = False,
    timeout: float = 30.0,
    retries: int = 2,
) -> dict[str, int]:
    manifest_path = Path(manifest_path)
    dataset_root = manifest_path.parents[1]
    raw_dir = dataset_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(manifest_path)
    counts: Counter[str] = Counter()
    for row in rows:
        url = row.get("source_url", "").strip()
        if row.get("local_video_path") and Path(row["local_video_path"]).is_file() and not force:
            row["download_status"] = "ALREADY_EXISTS"
            row["download_error"] = ""
            counts["ALREADY_EXISTS"] += 1
            _write_manifest_rows(manifest_path, rows)
            continue
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            row["download_status"] = "FAILED"
            row["download_error"] = "Source URL is missing or is not HTTP(S); no alternate source was searched."
            counts["FAILED"] += 1
            _write_manifest_rows(manifest_path, rows)
            continue
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in SUPPORTED_RAW_FORMATS:
            suffix = ".mp4"
        target = raw_dir / f"{_sample_file_stem(row)}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and not force:
            row["local_video_path"] = str(target)
            row["download_status"] = "ALREADY_EXISTS"
            row["download_error"] = ""
            counts["ALREADY_EXISTS"] += 1
            _write_manifest_rows(manifest_path, rows)
            continue
        temporary_path: Path | None = None
        last_error = ""
        unavailable = False
        row["download_status"] = "PENDING"
        for _ in range(max(0, retries) + 1):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "BridgeSign-MSASL-Importer/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content_type = response.headers.get_content_type().casefold()
                    if content_type in {"text/html", "application/xhtml+xml"}:
                        raise ValueError(f"Source returned {content_type}, not a direct video file.")
                    fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".part", dir=target.parent)
                    os.close(fd)
                    temporary_path = Path(temp_name)
                    bytes_written = 0
                    with temporary_path.open("wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            bytes_written += len(chunk)
                            if bytes_written > MAX_DOWNLOAD_BYTES:
                                raise ValueError(f"Download exceeded the {MAX_DOWNLOAD_BYTES}-byte safety limit.")
                            output.write(chunk)
                    if bytes_written == 0:
                        raise ValueError("Source returned an empty response.")
                    os.replace(temporary_path, target)
                    temporary_path = None
                    row["local_video_path"] = str(target)
                    row["download_status"] = "DOWNLOADED"
                    row["download_error"] = ""
                    counts["DOWNLOADED"] += 1
                    break
            except urllib.error.HTTPError as exc:
                unavailable = exc.code in {401, 403, 404, 410}
                last_error = f"HTTP {exc.code}: {exc.reason}"
                if unavailable:
                    break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                    temporary_path = None
        else:
            unavailable = False
        if row.get("download_status") != "DOWNLOADED":
            row["download_status"] = "UNAVAILABLE" if unavailable else "FAILED"
            row["download_error"] = last_error
            counts[row["download_status"]] += 1
        _write_manifest_rows(manifest_path, rows)
    return dict(counts)


def _video_metadata(path: Path) -> tuple[float, int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frames / fps if fps > 0 else 0.0
        if frames <= 0:
            raise ValueError(f"Video has zero frames: {path}")
        return fps, frames, width, height, duration
    finally:
        capture.release()


def extract_temporal_clip(
    source_path: str | Path,
    destination: str | Path,
    start_time: str | float | None,
    end_time: str | float | None,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(source_path)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not force:
        fps, frames, width, height, duration = _video_metadata(target)
        return {
            "clip_path": str(target), "frames": frames, "fps": fps,
            "width": width, "height": height, "duration": duration,
            "status": "ALREADY_EXISTS", "notes": "",
        }
    fps, _, width, height, source_duration = _video_metadata(source)
    start = _finite_number(start_time)
    end = _finite_number(end_time)
    if start is None and end is None:
        shutil.copy2(source, target)
        return {
            "clip_path": str(target), "frames": int(source_duration * fps),
            "fps": fps, "width": width, "height": height,
            "duration": source_duration, "status": "COPIED_FULL_VIDEO",
            "notes": "No valid temporal segment was supplied; copied full source video.",
        }
    if start is None or end is None or start < 0 or end <= start or start >= source_duration:
        raise ValueError(f"Invalid temporal segment start={start_time!r}, end={end_time!r}, duration={source_duration:.3f}s.")
    notes = ""
    if end > source_duration:
        end = source_duration
        notes = "Temporal segment end exceeded source duration and was clamped."
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"OpenCV cannot open source video: {source}")
    temporary = target.with_name(f".{target.stem}.partial{target.suffix}")
    writer = None
    frames_written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise ValueError(f"Could not seek to segment start at {start:.3f}s.")
        frame_height, frame_width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise ValueError(f"Could not create extracted clip: {temporary}")
        current = first_frame
        while ok and capture.get(cv2.CAP_PROP_POS_MSEC) <= end * 1000:
            writer.write(current)
            frames_written += 1
            ok, current = capture.read()
        writer.release()
        writer = None
        if frames_written <= 0:
            raise ValueError("Temporal segment extraction produced zero frames.")
        os.replace(temporary, target)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
    return {
        "clip_path": str(target),
        "frames": frames_written,
        "fps": fps,
        "width": frame_width,
        "height": frame_height,
        "duration": frames_written / fps if fps > 0 else 0.0,
        "status": "EXTRACTED",
        "notes": notes,
    }


def process_selected(
    manifest_path: str | Path = MANIFESTS_DIR / "import_manifest.csv",
    force: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    rows = _read_csv(manifest_path)
    dirs = ensure_dataset_directories(manifest_path.parents[1])
    sequence_rows: list[dict[str, str]] = []
    existing_sequences = _read_csv(dirs["manifests"] / "sequences.csv")
    existing_by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for existing in existing_sequences:
        if existing.get("source_record_id"):
            existing_by_record[existing["source_record_id"]].append(existing)
    summary_counts: Counter[str] = Counter()
    config = load_msl_config()
    config["holistic_model_path"] = str(config.get("holistic_model_path", PROJECT_ROOT / "models" / "holistic_landmarker.task"))
    config["sequence_length"] = int(config.get("sequence_length", 30))
    config["stride"] = int(config.get("stride", 15))
    config["min_frames"] = int(config.get("min_frames", 12))
    config["allow_padding"] = True
    config["face_landmark_indices"] = config.get("face_landmark_indices", DEFAULT_FACE_INDICES)
    config["feature_selection"] = config.get("feature_selection", DEFAULT_FEATURES)

    for row in rows:
        if row.get("processing_status") == "PROCESSED" and not force:
            existing = existing_by_record.get(row.get("source_record_id", ""), [])
            if existing and all(Path(sample.get("sequence_path", "")).is_file() for sample in existing):
                sequence_rows.extend(existing)
                summary_counts["processed"] += 1
                continue
        row["processing_status"] = "PENDING"
        row["processing_error"] = ""
        video_path = Path(row.get("local_video_path", ""))
        if row.get("download_status") not in {"DOWNLOADED", "ALREADY_EXISTS"} or not video_path.is_file():
            row["processing_status"] = "FAILED"
            row["processing_error"] = "No downloaded local video is available."
            summary_counts["failed"] += 1
            continue
        if not row.get("label"):
            row["processing_status"] = "FAILED"
            row["processing_error"] = "Metadata record has no label; it was not relabelled."
            summary_counts["failed"] += 1
            continue
        if not row.get("signer_id"):
            row["notes"] = "; ".join(filter(None, [row.get("notes", ""), "Missing signer ID in source metadata."]))
        if not row.get("clip_path"):
            row["clip_path"] = str(dirs["clips"] / f"{_sample_file_stem(row)}.mp4")
        try:
            clip = extract_temporal_clip(
                video_path,
                row["clip_path"],
                row.get("start_time"),
                row.get("end_time"),
                force=force,
            )
            row["clip_path"] = clip["clip_path"]
            row["notes"] = "; ".join(filter(None, [row.get("notes", ""), clip.get("notes", "")]))
            processor_signer = row.get("signer_id") or "UNKNOWN"
            processor_label = re.sub(r"[^A-Za-z0-9_]+", "_", row["label"]).strip("_") or "UNLABELLED"
            split_dir = dirs["processed"] / row.get("source_split", "unknown")
            split_dir.mkdir(parents=True, exist_ok=True)
            sequence_number = 0
            with tempfile.TemporaryDirectory(prefix=".msasl-staging-", dir=dirs["processed"]) as staging_dir:
                config["raw_dataset_path"] = str(dirs["clips"])
                config["processed_dataset_path"] = staging_dir
                result = process_landmark_video(
                    row["clip_path"],
                    processor_signer,
                    processor_label,
                    config=config,
                )
                for sequence in result["sequence_windows"]:
                    sequence_number += 1
                    source_npz = Path(sequence["sequence_path"])
                    with np.load(source_npz, allow_pickle=False) as sample:
                        payload = {key: np.asarray(sample[key]) for key in sample.files}
                    sample_id = f"{row['source_record_id']}_seq{sequence_number:04d}"
                    final_path = split_dir / f"{sample_id}.npz"
                    if final_path.exists() and not force:
                        with np.load(final_path, allow_pickle=False) as existing:
                            if str(existing["source_record_id"]) == row["source_record_id"]:
                                final_sequence = np.asarray(existing["sequence"], dtype=np.float32)
                            else:
                                raise FileExistsError(f"Processed sample ID collision: {final_path}")
                    else:
                        payload.update({
                            "label": np.asarray(row["label"], dtype="<U256"),
                            "label_id": np.asarray(row.get("label_id", ""), dtype="<U64"),
                            "signer_id": np.asarray(row.get("signer_id", ""), dtype="<U128"),
                            "video_id": np.asarray(row.get("video_id", ""), dtype="<U256"),
                            "session_id": np.asarray("", dtype="<U64"),
                            "dataset": np.asarray("MS-ASL", dtype="<U16"),
                            "language": np.asarray("American Sign Language (ASL)", dtype="<U64"),
                            "source_split": np.asarray(row.get("source_split", ""), dtype="<U16"),
                            "source_record_id": np.asarray(row["source_record_id"], dtype="<U64"),
                            "source_url": np.asarray(row.get("source_url", ""), dtype="<U2048"),
                            "start_time": np.asarray(_finite_number(row.get("start_time")) or 0.0, dtype=np.float32),
                            "end_time": np.asarray(_finite_number(row.get("end_time")) or 0.0, dtype=np.float32),
                        })
                        np.savez_compressed(final_path, **payload)
                        final_sequence = np.asarray(payload["sequence"], dtype=np.float32)
                    sequence_rows.append({
                        "sample_id": sample_id,
                        "dataset": "MS-ASL",
                        "language": "American Sign Language (ASL)",
                        "source_record_id": row["source_record_id"],
                        "source_split": row.get("source_split", ""),
                        "video_id": row.get("video_id", ""),
                        "source_url": row.get("source_url", ""),
                        "label_id": row.get("label_id", ""),
                        "label": row["label"],
                        "signer_id": row.get("signer_id", ""),
                        "start_time": row.get("start_time", ""),
                        "end_time": row.get("end_time", ""),
                        "sequence_path": str(final_path),
                        "frame_count": str(final_sequence.shape[0]),
                        "feature_count": str(final_sequence.shape[-1]),
                        "processing_status": "PROCESSED",
                    })
            row["processing_status"] = "PROCESSED"
            summary_counts["processed"] += 1
        except Exception as exc:
            row["processing_status"] = "FAILED"
            row["processing_error"] = f"{type(exc).__name__}: {exc}"
            summary_counts["failed"] += 1
    _write_manifest_rows(manifest_path, rows)
    _write_csv(dirs["manifests"] / "sequences.csv", SEQUENCE_FIELDS, sequence_rows)
    return {"processed_videos": summary_counts["processed"], "failed_videos": summary_counts["failed"], "sequences": len(sequence_rows)}


def build_split_manifest(manifest_path: str | Path = MANIFESTS_DIR / "import_manifest.csv") -> dict[str, Any]:
    rows = _read_csv(manifest_path)
    signers: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    classes: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    counts: Counter[str] = Counter()
    for row in rows:
        split = row.get("source_split", "")
        if split not in signers:
            continue
        counts[split] += 1
        if row.get("signer_id"):
            signers[split].add(row["signer_id"])
        if row.get("label"):
            classes[split].add(row["label"])
    overlap: dict[str, list[str]] = {}
    for index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[index + 1:]:
            overlap[f"{first}_{second}"] = sorted(signers[first] & signers[second])
    result = {
        "dataset": "MS-ASL",
        "official_splits_preserved": True,
        "sample_counts": dict(counts),
        "signers_by_split": {split: sorted(values) for split, values in signers.items()},
        "classes_by_split": {split: sorted(values) for split, values in classes.items()},
        "signer_overlap": overlap,
        "signer_leakage_detected": any(overlap.values()),
    }
    output = Path(manifest_path).parent / "split_manifest.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def validate_processed_dataset(
    dataset_root: str | Path = DATASET_ROOT,
    excessive_missing_rate: float = 0.5,
    expected_feature_count: int | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    manifest_rows = _read_csv(root / "manifests" / "import_manifest.csv")
    sequence_rows = _read_csv(root / "manifests" / "sequences.csv")
    categories: dict[str, list[dict[str, Any]]] = {"VALID": [], "WARNING": [], "FAILED": []}
    issues: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    duplicate_video_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    duplicate_sample_ids: set[str] = set()

    for row in manifest_rows:
        video_id = row.get("video_id", "")
        if video_id and video_id in seen_video_ids:
            duplicate_video_ids.add(video_id)
        seen_video_ids.add(video_id)
        if row.get("download_status") in {"FAILED", "UNAVAILABLE"}:
            categories["FAILED"].append({
                "source_record_id": row.get("source_record_id", ""),
                "video_id": video_id,
                "reason": row.get("download_error") or f"Download status: {row['download_status']}.",
            })
        if row.get("processing_status") == "FAILED":
            categories["FAILED"].append({
                "source_record_id": row.get("source_record_id", ""),
                "video_id": video_id,
                "reason": row.get("processing_error") or row.get("download_error") or "Processing failed.",
            })
        if row.get("label") == "":
            categories["FAILED"].append({
                "source_record_id": row.get("source_record_id", ""),
                "video_id": video_id,
                "reason": "Missing label in source metadata.",
            })
        if not row.get("signer_id"):
            categories["WARNING"].append({
                "source_record_id": row.get("source_record_id", ""),
                "video_id": video_id,
                "reasons": ["Missing signer ID in source metadata."],
            })
        local_path = Path(row.get("local_video_path", ""))
        if row.get("download_status") in {"DOWNLOADED", "ALREADY_EXISTS"} and local_path.is_file():
            try:
                source_fps, _, _, _, _ = _video_metadata(local_path)
                if source_fps > 0 and (source_fps < 5 or source_fps > 120):
                    categories["WARNING"].append({
                        "source_record_id": row.get("source_record_id", ""),
                        "video_id": video_id,
                        "reasons": [f"Unusual source frame rate: {source_fps:.2f} FPS."],
                    })
            except Exception as exc:
                categories["FAILED"].append({
                    "source_record_id": row.get("source_record_id", ""),
                    "video_id": video_id,
                    "reason": f"Corrupt or unreadable source video: {type(exc).__name__}: {exc}",
                })

    feature_dimensions: set[int] = set()
    for row in sequence_rows:
        sample_id = row.get("sample_id", "")
        reasons: list[str] = []
        warnings: list[str] = []
        if not sample_id:
            reasons.append("Missing processed sample ID.")
        elif sample_id in seen_sample_ids:
            duplicate_sample_ids.add(sample_id)
            reasons.append("Duplicate processed sample ID.")
        seen_sample_ids.add(sample_id)
        if not row.get("label"):
            reasons.append("Missing label.")
        if not row.get("signer_id"):
            warnings.append("Missing signer ID in source metadata.")
        path = Path(row.get("sequence_path", ""))
        if not path.is_file():
            reasons.append(f"Processed sequence file is missing: {path}")
        else:
            try:
                with np.load(path, allow_pickle=False) as sample:
                    sequence = np.asarray(sample["sequence"], dtype=np.float32)
                    raw = np.asarray(sample["raw_sequence"], dtype=np.float32) if "raw_sequence" in sample else None
                    mask = np.asarray(sample["landmark_presence"], dtype=np.float32) if "landmark_presence" in sample else None
                    valid_frames = np.asarray(sample["valid_frame_mask"], dtype=np.float32) if "valid_frame_mask" in sample else None
                    groups = [str(item) for item in sample["feature_group_names"].tolist()] if "feature_group_names" in sample else []
                if sequence.ndim != 2 or sequence.shape[0] == 0 or sequence.shape[1] == 0:
                    reasons.append("Sequence has invalid or empty dimensions.")
                if sequence.size and not np.isfinite(sequence).all():
                    reasons.append("Model sequence contains NaN or Inf.")
                if sequence.size:
                    feature_dimensions.add(int(sequence.shape[1]))
                if raw is not None and np.isinf(raw).any():
                    reasons.append("Raw landmark sequence contains Inf.")
                if mask is None or mask.ndim != 2 or mask.shape[0] != sequence.shape[0]:
                    warnings.append("Landmark presence mask is missing or has inconsistent dimensions.")
                elif mask.size:
                    valid_rows = (
                        valid_frames > 0
                        if valid_frames is not None and valid_frames.shape == (sequence.shape[0],)
                        else np.ones(sequence.shape[0], dtype=bool)
                    )
                    for index, group in enumerate(groups):
                        if index >= mask.shape[1]:
                            break
                        if not valid_rows.any():
                            warnings.append("Sequence contains no unpadded frames.")
                            break
                        missing_rate = 1.0 - float(np.mean(mask[valid_rows, index] > 0))
                        if missing_rate > excessive_missing_rate:
                            warnings.append(f"Excessive missing {group} landmarks ({missing_rate:.1%}).")
                if row.get("frame_count") and sequence.ndim == 2 and int(row["frame_count"]) != sequence.shape[0]:
                    reasons.append("Manifest frame count does not match stored sequence.")
                if row.get("feature_count") and sequence.ndim == 2 and int(row["feature_count"]) != sequence.shape[1]:
                    reasons.append("Manifest feature count does not match stored sequence.")
                if expected_feature_count is not None and sequence.ndim == 2 and sequence.shape[1] != expected_feature_count:
                    reasons.append(f"Feature dimension {sequence.shape[1]} does not match expected {expected_feature_count}.")
            except Exception as exc:
                reasons.append(f"Could not load processed sequence: {type(exc).__name__}: {exc}")
        sample = {
            "sample_id": sample_id,
            "video_id": row.get("video_id", ""),
            "source_split": row.get("source_split", ""),
            "sequence_path": str(path),
            "reasons": reasons,
            "warnings": warnings,
        }
        if reasons:
            categories["FAILED"].append(sample)
        elif warnings:
            categories["WARNING"].append(sample)
        else:
            categories["VALID"].append(sample)

    for video_id in sorted(duplicate_video_ids):
        issues.append({"severity": "WARNING", "type": "DUPLICATE_VIDEO_ID", "video_id": video_id})
    for sample_id in sorted(duplicate_sample_ids):
        issues.append({"severity": "FAILED", "type": "DUPLICATE_SAMPLE_ID", "sample_id": sample_id})
    if len(feature_dimensions) > 1:
        issues.append({
            "severity": "FAILED",
            "type": "INCONSISTENT_FEATURE_DIMENSIONS",
            "dimensions": sorted(feature_dimensions),
        })
    pending_downloads = sum(row.get("download_status") == "PENDING" for row in manifest_rows)
    if pending_downloads:
        issues.append({
            "severity": "WARNING",
            "type": "DOWNLOADS_NOT_ATTEMPTED",
            "count": pending_downloads,
            "reason": "The batch was stopped after a source URL returned a webpage instead of a direct video file.",
        })
    result = {
        "dataset": "MS-ASL",
        "valid": categories["VALID"],
        "warning": categories["WARNING"],
        "failed": categories["FAILED"],
        "issues": issues,
        "duplicate_video_ids": sorted(duplicate_video_ids),
        "duplicate_sample_ids": sorted(duplicate_sample_ids),
        "feature_dimensions": sorted(feature_dimensions),
        "counts": {key: len(value) for key, value in categories.items()},
    }
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "quality_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    review_candidates = []
    for status in ("WARNING", "FAILED"):
        for sample in categories[status]:
            review_candidates.append({
                **sample,
                "status": status,
                "review_required": True,
            })
    for row in manifest_rows:
        if "clamped" in row.get("notes", "").casefold():
            review_candidates.append({
                "source_record_id": row.get("source_record_id", ""),
                "video_id": row.get("video_id", ""),
                "source_split": row.get("source_split", ""),
                "status": "WARNING",
                "review_required": True,
                "reasons": [row["notes"]],
            })
    for issue in issues:
        if issue.get("severity") in {"WARNING", "FAILED"} and issue.get("type") != "DOWNLOADS_NOT_ATTEMPTED":
            review_candidates.append({
                **issue,
                "status": issue["severity"],
                "review_required": True,
                "reasons": [f"Dataset quality check: {issue.get('type', 'unknown issue')}."],
            })
    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "msasl_review_candidates.json").write_text(
        json.dumps(review_candidates, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def generate_reports(
    dataset_root: str | Path = DATASET_ROOT,
    validation: dict[str, Any] | None = None,
) -> dict[str, str]:
    root = Path(dataset_root)
    dirs = ensure_dataset_directories(root)
    manifest_rows = _read_csv(dirs["manifests"] / "import_manifest.csv")
    sequence_rows = _read_csv(dirs["manifests"] / "sequences.csv")
    split_result = build_split_manifest(dirs["manifests"] / "import_manifest.csv")
    validation = validation or validate_processed_dataset(root)
    source_metadata_path = dirs["metadata"] / "metadata_source.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8")) if source_metadata_path.exists() else {}
    download_counts = Counter(row.get("download_status", "PENDING") for row in manifest_rows)
    processing_counts = Counter(row.get("processing_status", "PENDING") for row in manifest_rows)
    sequence_signers: dict[str, set[str]] = defaultdict(set)
    sequence_classes: dict[str, set[str]] = defaultdict(set)
    for row in sequence_rows:
        split = row.get("source_split", "")
        if row.get("signer_id"):
            sequence_signers[split].add(row["signer_id"])
        if row.get("label"):
            sequence_classes[split].add(row["label"])
    feature_count = validation.get("feature_dimensions", [])
    report_lines = [
        "# MS-ASL Import Report",
        "",
        "**DATASET:** MS-ASL",
        "",
        "**LANGUAGE:** American Sign Language (ASL)",
        "",
        "**PURPOSE:** External dataset for dynamic sign-recognition pipeline validation/research.",
        "",
        "> This dataset must not be treated as Malawian Sign Language training data.",
        "",
        "## Import statistics",
        "",
        f"- Metadata records found: {source_metadata.get('metadata_records', len(manifest_rows))}",
        f"- Selected records: {len(manifest_rows)}",
        f"- Videos downloaded: {download_counts.get('DOWNLOADED', 0)}",
        f"- Existing videos reused: {download_counts.get('ALREADY_EXISTS', 0)}",
        f"- Videos unavailable: {download_counts.get('UNAVAILABLE', 0)}",
        f"- Download failures: {download_counts.get('FAILED', 0)}",
        f"- Downloads not attempted: {download_counts.get('PENDING', 0)}",
        f"- Clips extracted/copied: {sum(bool(row.get('clip_path')) and Path(row['clip_path']).is_file() for row in manifest_rows)}",
        f"- Videos processed: {processing_counts.get('PROCESSED', 0)}",
        f"- Processing failures: {processing_counts.get('FAILED', 0)}",
        f"- Processing not attempted: {processing_counts.get('PENDING', 0)}",
        f"- First download failure detail: {next((row.get('download_error') for row in manifest_rows if row.get('download_error')), 'none')}",
        f"- Classes represented: {len({row.get('label', '') for row in manifest_rows if row.get('label')})}",
        f"- Signers represented: {len({row.get('signer_id', '') for row in manifest_rows if row.get('signer_id')})}",
        f"- Videos represented: {len({row.get('video_id', '') for row in manifest_rows if row.get('video_id')})}",
        f"- Processed sequences: {len(sequence_rows)}",
        f"- Feature dimension(s): {feature_count or 'not available'}",
        f"- Temporal window length: {load_msl_config().get('sequence_length', 30)}",
        "",
        "## Original split distribution",
        "",
    ]
    for split in SPLIT_NAMES:
        split_rows = [row for row in manifest_rows if row.get("source_split") == split]
        report_lines.append(
            f"- {split}: {len(split_rows)} selected metadata records; "
            f"{sum(row.get('processing_status') == 'PROCESSED' for row in split_rows)} processed; "
            f"{sum(sample.get('source_split') == split for sample in sequence_rows)} sequences; "
            f"{len(sequence_classes.get(split, set()))} processed classes "
            f"({', '.join(sorted(sequence_classes.get(split, set()))) or 'none'})"
        )
    report_lines.extend(["", "## Signer overlap in official splits", ""])
    for pair, signers in split_result["signer_overlap"].items():
        report_lines.append(f"- {pair}: {len(signers)} shared signer(s)" + (f" ({', '.join(signers)})" if signers else ""))
    report_lines.extend([
        "",
        "## Source and license",
        "",
        f"- Metadata source: {source_metadata.get('source', 'not recorded')}",
        f"- License information: {source_metadata.get('license_information') or 'not provided in the imported metadata; consult the official MS-ASL terms and individual source-video restrictions before use.'}",
        "",
        "## Limitations",
        "",
        "- MS-ASL is ASL and is isolated from all MSL data and labels.",
        "- Video availability depends on the original source URL; deleted, private, restricted, or non-direct links are reported and are not bypassed.",
        "- No model was trained and no MSL recognition claim is made.",
        "- Dataset statistics describe only the imported/selected records and successfully processed samples.",
    ])
    import_report = dirs["reports"] / "MSASL_IMPORT_REPORT.md"
    import_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    quality_lines = ["# MS-ASL Quality Report", ""]
    for category in ("VALID", "WARNING", "FAILED"):
        entries = validation.get(category.casefold(), [])
        quality_lines.extend([f"## {category} ({len(entries)})", ""])
        if not entries:
            quality_lines.append("- None.")
        for entry in entries[:500]:
            details = entry.get("reasons") or entry.get("warnings") or []
            if not details and entry.get("reason"):
                details = [entry["reason"]]
            quality_lines.append(f"- {entry.get('sample_id') or entry.get('video_id') or 'unknown'}: {'; '.join(details) or 'passed checks'}")
        quality_lines.append("")
    quality_lines.extend(["## Dataset-level issues", ""])
    if validation.get("issues"):
        quality_lines.extend(f"- {issue}" for issue in validation["issues"])
    else:
        quality_lines.append("- None.")
    quality_report = dirs["reports"] / "MSASL_QUALITY_REPORT.md"
    quality_report.write_text("\n".join(quality_lines) + "\n", encoding="utf-8")
    (dirs["reports"] / "dataset_summary.json").write_text(
        json.dumps({
            "dataset": "MS-ASL",
            "language": "American Sign Language (ASL)",
            "metadata_records": source_metadata.get("metadata_records", len(manifest_rows)),
            "selected_records": len(manifest_rows),
            "download_status_counts": dict(download_counts),
            "processing_status_counts": dict(processing_counts),
            "processed_sequences": len(sequence_rows),
            "videos": sorted({row.get("video_id", "") for row in manifest_rows if row.get("video_id")}),
            "classes": sorted({row.get("label", "") for row in manifest_rows if row.get("label")}),
            "signers": sorted({row.get("signer_id", "") for row in manifest_rows if row.get("signer_id")}),
            "classes_by_processed_split": {split: sorted(values) for split, values in sequence_classes.items()},
            "signer_overlap": split_result["signer_overlap"],
            "quality_counts": validation.get("counts", {}),
            "feature_dimensions": feature_count,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"import_report": str(import_report), "quality_report": str(quality_report)}


if torch is not None:
    class MSASLDataset(torch.utils.data.Dataset):
        """PyTorch dataset for external MS-ASL sequences only."""

        def __init__(self, manifest_path: str | Path, label_to_index: dict[str, int] | None = None):
            self.manifest_path = Path(manifest_path)
            self.rows = _read_csv(self.manifest_path)
            self.samples = [row for row in self.rows if row.get("sequence_path")]
            labels = sorted({row.get("label", "") for row in self.samples if row.get("label")})
            self.label_to_index = dict(label_to_index) if label_to_index is not None else {
                label: index for index, label in enumerate(labels)
            }

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Any, int, dict[str, str]]:
            if index < 0 or index >= len(self.samples):
                raise IndexError(index)
            row = self.samples[index]
            with np.load(row["sequence_path"], allow_pickle=False) as sample:
                if str(sample["dataset"]) != "MS-ASL" or str(sample["language"]) != "American Sign Language (ASL)":
                    raise ValueError(f"Sample is not explicitly marked as MS-ASL: {row['sequence_path']}")
                sequence = np.asarray(sample["sequence"], dtype=np.float32)
                label = str(sample["label"])
                metadata = {
                    "dataset": "MS-ASL",
                    "language": "American Sign Language (ASL)",
                    "source_split": str(sample["source_split"]),
                    "source_record_id": str(sample["source_record_id"]),
                    "video_id": str(sample["video_id"]),
                    "signer_id": str(sample["signer_id"]),
                    "start_time": str(float(sample["start_time"])),
                    "end_time": str(float(sample["end_time"])),
                }
            if label not in self.label_to_index:
                raise ValueError(f"MS-ASL label '{label}' is missing from the label mapping.")
            return torch.from_numpy(sequence), self.label_to_index[label], metadata
else:
    class MSASLDataset:
        """Fallback reader for environments without the optional PyTorch dependency."""

        def __init__(self, manifest_path: str | Path, label_to_index: dict[str, int] | None = None):
            self.manifest_path = Path(manifest_path)
            self.rows = _read_csv(self.manifest_path)
            self.samples = [row for row in self.rows if row.get("sequence_path")]
            labels = sorted({row.get("label", "") for row in self.samples if row.get("label")})
            self.label_to_index = dict(label_to_index) if label_to_index is not None else {label: i for i, label in enumerate(labels)}

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[np.ndarray, int, dict[str, str]]:
            row = self.samples[index]
            with np.load(row["sequence_path"], allow_pickle=False) as sample:
                if str(sample["dataset"]) != "MS-ASL":
                    raise ValueError(f"Sample is not MS-ASL: {row['sequence_path']}")
                sequence = np.asarray(sample["sequence"], dtype=np.float32)
                label = str(sample["label"])
                metadata = {"dataset": "MS-ASL", "language": "American Sign Language (ASL)", "source_split": str(sample["source_split"]), "video_id": str(sample["video_id"]), "signer_id": str(sample["signer_id"])}
            if label not in self.label_to_index:
                raise ValueError(f"MS-ASL label '{label}' is missing from the label mapping.")
            return sequence, self.label_to_index[label], metadata


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import a controlled sample of external MS-ASL (American Sign Language) "
            "metadata/videos into an isolated BridgeSign dataset area. No model training is performed."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Official MS-ASL metadata ZIP, extracted metadata directory, or a JSON metadata file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of records selected (default: 100; no automatic full-dataset download).",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--download", action="store_true", help="Download only the selected source URLs; failures are recorded and processing continues.")
    parser.add_argument("--process", action="store_true", help="Extract temporal clips and process locally downloaded videos with MediaPipe Holistic.")
    parser.add_argument("--split", action="store_true", help="Write a manifest of the official split distribution and signer overlap.")
    parser.add_argument("--report", action="store_true", help="Generate the import and quality Markdown reports.")
    parser.add_argument("--validate", action="store_true", help="Validate processed samples and update human-review candidates.")
    parser.add_argument("--force", action="store_true", help="Replace downloaded files and/or reprocess existing samples.")
    parser.add_argument("--dry-run", action="store_true", help="Select records and print a preview without downloading or processing.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request download timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed URL; default 2 retries after the first attempt.")
    parser.add_argument("--expected-feature-count", type=int, default=None, help="Optional required feature-vector width for validation.")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be a positive integer.")
    if args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be positive and --retries cannot be negative.")

    dirs = ensure_dataset_directories(args.dataset_root)
    manifest_path = dirs["manifests"] / "import_manifest.csv"
    if args.metadata:
        imported = import_metadata(args.metadata, args.limit, args.dataset_root)
        print(json.dumps(imported, indent=2, ensure_ascii=False))
    elif not manifest_path.exists():
        parser.error("Provide --metadata on the first run or point to an existing dataset-root.")

    if args.dry_run:
        selected = _read_csv(manifest_path)
        print(json.dumps({
            "dry_run": True,
            "selected_records": len(selected),
            "sample_preview": [
                {key: row.get(key, "") for key in ("source_split", "video_id", "label_id", "label", "signer_id", "source_url")}
                for row in selected[:10]
            ],
            "downloads_started": False,
            "dataset": "MS-ASL",
            "language": "American Sign Language (ASL)",
        }, indent=2, ensure_ascii=False))
        return 0

    if args.download:
        print(json.dumps({"download_status_counts": download_selected(manifest_path, force=args.force, timeout=args.timeout, retries=args.retries)}, indent=2))
    if args.process:
        print(json.dumps(process_selected(manifest_path, force=args.force), indent=2))
    split_result = build_split_manifest(manifest_path) if args.split else None
    if split_result is not None:
        print(json.dumps(split_result, indent=2, ensure_ascii=False))
    validation = None
    if args.validate or args.report:
        validation = validate_processed_dataset(args.dataset_root, expected_feature_count=args.expected_feature_count)
        if args.validate:
            print(json.dumps({
                "quality_counts": validation["counts"],
                "feature_dimensions": validation["feature_dimensions"],
                "duplicate_video_ids": validation["duplicate_video_ids"],
                "issues": validation["issues"],
            }, indent=2))
    if args.report:
        print(json.dumps(generate_reports(args.dataset_root, validation), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
