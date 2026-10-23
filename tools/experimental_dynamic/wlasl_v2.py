from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental_dynamic import baseline
from tools.msl_dataset.msl_dataset_pipeline import load_config as load_msl_config, process_video


DATA_ROOT = PROJECT_ROOT / "data" / "experimental" / "wlasl_v2"
METADATA_PATH = PROJECT_ROOT / "data" / "experimental_dynamic" / "wlasl" / "metadata" / "WLASL_v0.3.json"
MANIFEST_PATH = DATA_ROOT / "download_manifest.json"
SEQUENCE_MANIFEST_PATH = DATA_ROOT / "manifests" / "sequences.csv"
QC_PATH = DATA_ROOT / "reports" / "dataset_qc.json"
SPLIT_DIR = DATA_ROOT / "splits"
MODEL_ROOT = PROJECT_ROOT / "models" / "experimental" / "wlasl_v2"
METRICS_ROOT = PROJECT_ROOT / "metrics" / "experimental" / "wlasl_v2"
REPORT_PATH = PROJECT_ROOT / "reports" / "experimental" / "wlasl" / "wlasl_v2_report.md"
RAW_ROOT = DATA_ROOT / "raw"
CLIP_ROOT = DATA_ROOT / "clips"
PROCESSED_ROOT = DATA_ROOT / "processed"
SUPPORTED_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi"}
MAX_WINDOWS_PER_VIDEO = 3
DEFAULT_CLASSES = 100
DEFAULT_CAP_PER_CLASS = 20
DEFAULT_ATTEMPTS_PER_CLASS = 30
SEQUENCE_FIELDS = [
    "sample_id", "group_id", "label", "signer_id", "source_split",
    "sequence_path", "local_video_path", "video_id", "class_index",
    "source", "source_url", "instance_id", "variation_id",
]
_manifest_lock = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_downloaded_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {
        (int(row["class_index"]), str(row["video_id"])): row
        for row in records
        if row.get("download_status") == "DOWNLOADED"
    }
    return list(unique.values())


def _within(path: Path, root: Path = DATA_ROOT) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_metadata(path: Path = METADATA_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Official WLASL metadata not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"Official WLASL metadata is empty or malformed: {path}")
    return metadata


def _ranked_instances(metadata: list[dict[str, Any]], class_count: int) -> list[dict[str, Any]]:
    if class_count not in {100, 300, 1000, 2000}:
        raise ValueError("Use an official WLASL subset size: 100, 300, 1000, or 2000.")
    if class_count > len(metadata):
        raise ValueError(f"Requested WLASL{class_count}, but metadata has {len(metadata)} glosses.")
    return metadata[:class_count]


def build_manifest(class_count: int = DEFAULT_CLASSES, *, force: bool = False) -> Path:
    if MANIFEST_PATH.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite the WLASL V2 manifest: {MANIFEST_PATH}. "
            "Pass --force only when deliberately resetting acquisition state."
        )
    metadata = _load_metadata(METADATA_PATH)
    selected_classes = _ranked_instances(metadata, class_count)
    records: list[dict[str, Any]] = []
    for class_index, entry in enumerate(selected_classes):
        gloss = str(entry["gloss"])
        for instance in entry.get("instances", []):
            original_url = str(instance.get("url", "")).strip()
            suffix = Path(urllib.parse.urlparse(original_url).path).suffix.lower()
            is_youtube = urllib.parse.urlparse(original_url).netloc.lower() in {"youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be"}
            extension_supported = suffix in SUPPORTED_EXTENSIONS
            if not (is_youtube or extension_supported):
                status = "UNSUPPORTED_SOURCE_FORMAT"
                failure = f"Official URL extension {suffix or '(none)'} is not a supported video format."
            else:
                status = "PENDING"
                failure = ""
            records.append({
                "class_index": class_index,
                "gloss": gloss,
                "video_id": str(instance.get("video_id", "")),
                "instance_id": instance.get("instance_id"),
                "signer_id": str(instance.get("signer_id", "")),
                "source": str(instance.get("source", "")),
                "source_url": original_url,
                "official_split": str(instance.get("split", "")),
                "fps": instance.get("fps", 25),
                "frame_start": instance.get("frame_start", 1),
                "frame_end": instance.get("frame_end", -1),
                "variation_id": instance.get("variation_id"),
                "bbox": instance.get("bbox"),
                "download_status": status,
                "http_status": None,
                "local_filename": "",
                "download_bytes": 0,
                "download_sha256": "",
                "decode_status": "NOT_CHECKED",
                "decoded_frames": 0,
                "failure_reason": failure,
                "section_pretrimmed": False,
                "preprocessing_status": "PENDING",
                "preprocessing_error": "",
                "generated_windows": 0,
            })
    manifest = {
        "dataset": "WLASL",
        "metadata_version": "WLASL_v0.3",
        "subset": f"WLASL{class_count}",
        "class_selection": "First K gloss entries in official WLASL_v0.3.json ordering; this ordering defines WLASL100/300/1000/2000.",
        "class_count": class_count,
        "glosses": [
            {"class_index": index, "gloss": str(entry["gloss"]), "metadata_video_count": len(entry.get("instances", []))}
            for index, entry in enumerate(selected_classes)
        ],
        "metadata_sha256": _sha256(METADATA_PATH),
        "academic_use_notice": "WLASL is ASL, governed by the Computational Use of Data Agreement (C-UDA), and is not MSL.",
        "records": records,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return MANIFEST_PATH


def _record_filename(record: dict[str, Any]) -> Path:
    extension = Path(urllib.parse.urlparse(record["source_url"]).path).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        extension = ".mp4"
    output = RAW_ROOT / f"class_{record['class_index']:04d}" / str(record["video_id"])
    return output.with_suffix(extension)


def _decode_check(path: Path) -> tuple[int, str]:
    if not path.is_file() or path.stat().st_size < 5_000:
        return 0, "Downloaded payload is missing or smaller than 5 KB."
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0, "OpenCV could not open the downloaded file (not a decodable video)."
    count = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    if count < 12:
        return count, f"Only {count} frames decoded; at least 12 required."
    return count, ""


def _direct_download(record: dict[str, Any], destination: Path) -> tuple[int | None, int, str]:
    temporary = destination.with_name(destination.stem + ".partial" + destination.suffix)
    request = urllib.request.Request(
        record["source_url"],
        headers={"User-Agent": "BridgeSign WLASL V2 academic research acquisition"},
    )
    http_status: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=12) as response, temporary.open("wb") as output:
            http_status = int(response.status)
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError(f"Server returned HTML ({content_type}), not a video.")
            size = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > 300 * 1024 * 1024:
                    raise ValueError("Video exceeded 300 MB safety limit.")
                output.write(block)
        frames, failure = _decode_check(temporary)
        if failure:
            raise ValueError(failure)
        os.replace(temporary, destination)
        return http_status, frames, ""
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, urllib.error.HTTPError):
            http_status = exc.code
        return http_status, 0, f"{type(exc).__name__}: {exc}"


def _youtube_download(record: dict[str, Any], destination: Path) -> tuple[int | None, int, str]:
    fps = float(record.get("fps") or 25)
    start = max(0.0, (float(record.get("frame_start") or 1) - 1.0) / fps - 0.5)
    raw_end = int(record.get("frame_end") or -1)
    end = None if raw_end < 0 else raw_end / fps + 0.5
    directory = destination.parent / f".yt_{record['video_id']}"
    directory.mkdir(parents=True, exist_ok=True)
    output_template = str(directory / f"{record['video_id']}.%(ext)s")
    command = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist", "--no-warnings", "--quiet",
        "--socket-timeout", "15", "--retries", "1", "--fragment-retries", "1",
        "-f", "best[height<=480][ext=mp4]/best[height<=480]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
    ]
    if end is not None and end > start:
        command.extend(["--download-sections", f"*{start:.3f}-{end:.3f}"])
        record["section_pretrimmed"] = True
    command.extend(["-o", output_template, record["source_url"]])
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=35, check=False
        )
        artifacts = [
            path for path in directory.glob(f"{record['video_id']}.*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if completed.returncode != 0 or not artifacts:
            details = (completed.stderr or completed.stdout or "yt-dlp produced no video file").strip()
            return None, 0, f"yt-dlp exit={completed.returncode}: {details[-1200:]}"
        source = max(artifacts, key=lambda path: path.stat().st_size)
        frames, failure = _decode_check(source)
        if failure:
            return None, frames, failure
        if destination.suffix.lower() != source.suffix.lower():
            destination = destination.with_suffix(source.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        return None, frames, ""
    except subprocess.TimeoutExpired:
        return None, 0, "yt-dlp timed out after 35 seconds."
    except Exception as exc:
        return None, 0, f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _retryable_download_failure(record: dict[str, Any]) -> bool:
    reason = str(record.get("failure_reason", "")).casefold()
    permanent_markers = (
        "http error 403",
        "http error 404",
        "certificate_verify_failed",
        "hostname mismatch",
        "server returned html",
        "not a video",
        "requested format is not available",
        "this video is unavailable",
        "private video",
        "could not open the downloaded file",
    )
    return not any(marker in reason for marker in permanent_markers)


def _persist_manifest(manifest: dict[str, Any]) -> None:
    with _manifest_lock:
        temporary = MANIFEST_PATH.with_name(
            f"{MANIFEST_PATH.name}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        try:
            for attempt in range(6):
                try:
                    os.replace(temporary, MANIFEST_PATH)
                    return
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.1 * (2 ** attempt))
        finally:
            temporary.unlink(missing_ok=True)


def _acquire_gloss(
    gloss_record: dict[str, Any],
    manifest: dict[str, Any],
    max_per_class: int,
    max_attempts_per_class: int,
) -> tuple[str, int, int]:
    gloss = gloss_record["gloss"]
    records = [
        row for row in manifest["records"]
        if row["gloss"] == gloss and (
            row["download_status"] in {
                "PENDING", "IN_PROGRESS", "NOT_ATTEMPTED_CLASS_CAP"
            }
            or (
                row["download_status"] == "FAILED"
                and _retryable_download_failure(row)
            )
        )
    ]
    success_signers = {
        row["signer_id"] for row in manifest["records"]
        if row["gloss"] == gloss and row["download_status"] == "DOWNLOADED"
    }
    signer_order: dict[str, int] = {}
    signer_occurrences: dict[str, int] = defaultdict(int)
    acquisition_order: dict[str, tuple[int, int]] = {}
    for row in records:
        signer = row["signer_id"]
        if signer:
            signer_order.setdefault(signer, len(signer_order))
            acquisition_order[row["video_id"]] = (
                signer_occurrences[signer], signer_order[signer]
            )
            signer_occurrences[signer] += 1
        else:
            acquisition_order[row["video_id"]] = (10**6, 10**6)
    records.sort(key=lambda row: (
        "youtube.com" in row["source_url"] or "youtu.be" in row["source_url"],
        row["signer_id"] in success_signers if row["signer_id"] else True,
        acquisition_order[row["video_id"]],
        row["official_split"] != "train",
        str(row["video_id"]),
    ))
    success_ids = {
        row["video_id"] for row in manifest["records"]
        if row["gloss"] == gloss and row["download_status"] == "DOWNLOADED"
    }
    success = len(success_ids)
    failures = 0
    attempts = 0
    for record in records:
        if success >= max_per_class:
            record["download_status"] = "NOT_ATTEMPTED_CLASS_CAP"
            continue
        if record["video_id"] in success_ids:
            continue
        if attempts >= max_attempts_per_class:
            break
        destination = _record_filename(record)
        if not _within(destination):
            record["download_status"] = "FAILED"
            record["failure_reason"] = "Resolved destination is outside the WLASL V2 root."
            failures += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            frames, error = _decode_check(destination)
            if not error:
                record.update({
                    "download_status": "DOWNLOADED",
                    "local_filename": str(destination.resolve()),
                    "download_bytes": destination.stat().st_size,
                    "download_sha256": _sha256(destination),
                    "decode_status": "DECODED",
                    "decoded_frames": frames,
                    "failure_reason": "",
                })
                success += 1
                success_ids.add(record["video_id"])
                success_signers.add(record["signer_id"])
                _persist_manifest(manifest)
                continue
            destination.unlink()
        record["download_status"] = "IN_PROGRESS"
        attempts += 1
        _persist_manifest(manifest)
        is_youtube = "youtube.com" in record["source_url"] or "youtu.be" in record["source_url"]
        if is_youtube:
            http_status, frames, failure = _youtube_download(record, destination)
        else:
            http_status, frames, failure = _direct_download(record, destination)
        if failure:
            record.update({
                "download_status": "FAILED",
                "http_status": http_status,
                "decode_status": "REJECTED" if frames else "FAILED",
                "failure_reason": failure,
                "decoded_frames": frames,
            })
            failures += 1
        else:
            final_path = Path(destination)
            if not final_path.is_file():
                alternatives = [
                    candidate for candidate in destination.parent.glob(destination.stem + ".*")
                    if candidate.suffix.lower() in SUPPORTED_EXTENSIONS and candidate.is_file()
                ]
                if alternatives:
                    final_path = alternatives[0]
            record.update({
                "download_status": "DOWNLOADED",
                "http_status": http_status,
                "local_filename": str(final_path.resolve()),
                "download_bytes": final_path.stat().st_size,
                "download_sha256": _sha256(final_path),
                "decode_status": "DECODED",
                "decoded_frames": frames,
                "failure_reason": "",
            })
            success += 1
            success_ids.add(record["video_id"])
            success_signers.add(record["signer_id"])
        _persist_manifest(manifest)
        print(
            f"{record['download_status']} gloss={gloss} id={record['video_id']} "
            f"success={success}/{max_per_class} {record['failure_reason'][:140]}",
            flush=True,
        )
    return gloss, success, failures


def acquire(
    class_count: int = DEFAULT_CLASSES,
    max_per_class: int = DEFAULT_CAP_PER_CLASS,
    workers: int = 4,
    max_attempts_per_class: int = DEFAULT_ATTEMPTS_PER_CLASS,
) -> dict[str, int]:
    if min(max_per_class, workers, max_attempts_per_class) < 1:
        raise ValueError("max_per_class, workers, and max_attempts_per_class must be positive.")
    if not MANIFEST_PATH.is_file():
        build_manifest(class_count)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["class_count"] != class_count:
        raise ValueError(
            f"Manifest is for {manifest['subset']}, but requested WLASL{class_count}; create a separate manifest."
        )
    succeeded = Counter(
        row["gloss"] for row in manifest["records"] if row["download_status"] == "DOWNLOADED"
    )
    tasks = [
        gloss for gloss in manifest["glosses"]
        if succeeded[gloss["gloss"]] < max_per_class
    ]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _acquire_gloss, gloss, manifest, max_per_class, max_attempts_per_class
            )
            for gloss in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return {
        "glosses": len(manifest["glosses"]),
        "downloaded": len(_unique_downloaded_records(manifest["records"])),
        "downloaded_manifest_records": sum(
            row["download_status"] == "DOWNLOADED" for row in manifest["records"]
        ),
        "failed": sum(row["download_status"] == "FAILED" for row in manifest["records"]),
        "not_attempted_or_unsupported": sum(
            row["download_status"] not in {"DOWNLOADED", "FAILED", "PENDING", "IN_PROGRESS"}
            for row in manifest["records"]
        ),
    }


def _materialize_clip(record: dict[str, Any]) -> Path:
    source = Path(record["local_filename"]).resolve()
    if not _within(source) or not source.is_file():
        raise ValueError(f"Downloaded video missing or outside WLASL V2: {source}")
    if record.get("section_pretrimmed"):
        return source
    fps_meta = float(record.get("fps") or 25)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot reopen video: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps_meta)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(record.get("frame_start") or 1)
    end = int(record.get("frame_end") or -1)
    first = max(0, round((start - 1) * source_fps / fps_meta))
    last = count if end < 0 else min(count, round(end * source_fps / fps_meta))
    if last <= first:
        capture.release()
        raise ValueError(f"Invalid WLASL frame range [{start}, {end}] for {count} frames.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = CLIP_ROOT / f"class_{int(record['class_index']):04d}" / f"{record['video_id']}.mp4"
    if not _within(output):
        capture.release()
        raise ValueError(f"Refusing to save clip outside WLASL V2 root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".part.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create clip file: {temporary}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    written = 0
    try:
        for _ in range(last - first):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written < 12:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Metadata clip decoded only {written} frames.")
    os.replace(temporary, output)
    return output.resolve()


def _resample_clip(source: Path, record: dict[str, Any], target_fps: float = 25.0) -> Path:
    destination = DATA_ROOT / "resampled" / f"class_{int(record['class_index']):04d}" / f"{record['video_id']}_25fps.mp4"
    if not _within(destination):
        raise ValueError(f"Refusing to save resampled clip outside WLASL V2 root: {destination}")
    if destination.is_file():
        capture = cv2.VideoCapture(str(destination))
        cached_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()
        _frames, failure = _decode_check(destination)
        if not failure and abs(cached_fps - target_fps) < 0.05:
            return destination.resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open clip for temporal resampling: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or target_fps)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = count / source_fps if source_fps else 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create 25fps clip: {temporary}")
    frame_index = output_count = 0
    next_time = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_time = frame_index / source_fps
            while next_time < duration and next_time <= source_time + 1e-8:
                writer.write(frame)
                output_count += 1
                next_time += 1.0 / target_fps
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if output_count < 12:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Resampling yielded only {output_count} frames.")
    os.replace(temporary, destination)
    return destination.resolve()


def _feature_config() -> dict[str, Any]:
    config = load_msl_config()
    config.update({
        "raw_dataset_path": str(RAW_ROOT.resolve()),
        "processed_dataset_path": str(PROCESSED_ROOT.resolve()),
        "manifests_path": str((DATA_ROOT / "manifests").resolve()),
        "review_path": str((DATA_ROOT / "review").resolve()),
        "sequence_length": 30,
        "stride": 15,
        "target_fps": 25.0,
        "min_frames": 12,
        "max_frames": 300,
        "random_seed": 42,
        "dataset_version": "WLASL_V2",
        "normalization_method": "wrist_relative",
        "allow_padding": True,
    })
    return config


def _select_preprocessing_classes(
    records: list[dict[str, Any]],
    max_classes: int,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    if max_classes < 1:
        raise ValueError("max_classes must be positive.")
    downloaded_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["download_status"] == "DOWNLOADED":
            downloaded_by_class[row["gloss"]].append(row)
    ranked = sorted(
        downloaded_by_class,
        key=lambda label: (
            -len({row["video_id"] for row in downloaded_by_class[label]}),
            -len({row["signer_id"] for row in downloaded_by_class[label] if row["signer_id"]}),
            min(int(row["class_index"]) for row in downloaded_by_class[label]),
        ),
    )
    selected = ranked[:max_classes]
    counts = {
        label: {
            "videos": len({row["video_id"] for row in downloaded_by_class[label]}),
            "signers": len({row["signer_id"] for row in downloaded_by_class[label] if row["signer_id"]}),
        }
        for label in selected
    }
    return selected, counts


def preprocess(max_classes: int = 10) -> dict[str, Any]:
    if max_classes < 1:
        raise ValueError("max_classes must be positive.")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Create/acquire the WLASL V2 manifest before preprocessing.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config = _feature_config()
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    sequence_rows: list[dict[str, str]] = []
    selected_glosses, selection_counts = _select_preprocessing_classes(
        manifest["records"], max_classes
    )
    selected_gloss_set = set(selected_glosses)
    for row in manifest["records"]:
        if row["download_status"] != "DOWNLOADED":
            continue
        if row["gloss"] not in selected_gloss_set:
            row["preprocessing_status"] = "NOT_SELECTED_FOR_PREPROCESSING"
            row["preprocessing_error"] = (
                f"Outside the top {max_classes} verified-video/signer-coverage classes selected for this run."
            )
        elif row["preprocessing_status"] == "NOT_SELECTED_FOR_PREPROCESSING":
            row["preprocessing_status"] = "PENDING"
            row["preprocessing_error"] = ""
    manifest["preprocessing_selection"] = {
        "max_classes": max_classes,
        "ranking": "Descending verified downloaded videos, then distinct WLASL signers, then official class index.",
        "selected_glosses": selected_glosses,
        "class_counts": selection_counts,
    }
    _persist_manifest(manifest)
    cache_records = {
        (int(row["class_index"]), row["video_id"]): row
        for row in manifest["records"]
        if row["download_status"] == "DOWNLOADED" and row["gloss"] in selected_gloss_set
    }
    videos_processed = windows_total = rejected = 0
    class_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"downloaded_videos": 0, "processed_videos": 0, "windows_generated": 0, "windows_selected": 0}
    )
    for (class_index, video_id), record in cache_records.items():
        try:
            clip = _materialize_clip(record)
            fps_clip = _resample_clip(clip, record, config["target_fps"])
            current_hash = _sha256(fps_clip)
            cache_dir = PROCESSED_ROOT / f"class_{int(record['class_index']):04d}" / str(video_id)
            cached_manifest = cache_dir / "windows.json"
            windows: list[dict[str, Any]]
            if cached_manifest.is_file():
                cached = json.loads(cached_manifest.read_text(encoding="utf-8"))
                if cached.get("clip_sha256") == current_hash and all(Path(x["sequence_path"]).is_file() for x in cached.get("windows", [])):
                    windows = cached["windows"]
                    quality = cached.get("quality")
                else:
                    windows = []
                    quality = None
            else:
                windows = []
                quality = None
            if not windows:
                result = process_video(
                    fps_clip,
                    str(record["signer_id"]),
                    str(record["gloss"]),
                    config=config,
                    session_id=f"wlasl_v2_{video_id}",
                )
                windows = result["sequence_windows"]
                quality = {
                    "fps": result["fps"],
                    "frame_count": result["frame_count"],
                    "usable_frames": result["usable_frames"],
                    "detected_frames": result["detected_frames"],
                    "missing_landmark_rates": result["missing_landmark_rates"],
                }
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached_manifest.write_text(json.dumps({
                    "video_id": video_id,
                    "gloss": record["gloss"],
                    "signer_id": record["signer_id"],
                    "clip_sha256": current_hash,
                    "windows": windows,
                    "quality": quality,
                }, indent=2) + "\n", encoding="utf-8")
            if not windows:
                raise ValueError("Holistic preprocessing yielded no temporal windows.")
            if not isinstance(quality, dict):
                raise ValueError("Cached preprocessing is missing its landmark-quality record.")
            select_count = min(MAX_WINDOWS_PER_VIDEO, len(windows))
            indices = sorted(set(np.linspace(0, len(windows) - 1, select_count, dtype=int).tolist()))
            for index in indices:
                item = windows[index]
                with np.load(item["sequence_path"], allow_pickle=False) as packed:
                    sequence = np.asarray(packed["sequence"], dtype=np.float32)
                    presence = np.asarray(packed["landmark_presence"], dtype=np.float32)
                    mask = np.asarray(packed["valid_frame_mask"], dtype=np.float32)
                if sequence.shape != (30, 562) or not np.isfinite(sequence).all():
                    raise ValueError(f"Sequence shape/content invalid: {sequence.shape}.")
                if not np.isfinite(presence).all() or not np.any(presence * mask[:, None]):
                    raise ValueError("Window contains no detected Holistic landmarks.")
                sample_id = f"wlasl_v2_{class_index}_{video_id}_window_{index:04d}"
                sequence_rows.append({
                    "sample_id": sample_id,
                    "group_id": f"{class_index}:{video_id}",
                    "label": str(record["gloss"]),
                    "signer_id": str(record["signer_id"]),
                    "source_split": "",
                    "sequence_path": str(Path(item["sequence_path"]).resolve()),
                    "local_video_path": str(fps_clip),
                    "video_id": str(video_id),
                    "class_index": str(record["class_index"]),
                    "source": str(record["source"]),
                    "source_url": str(record["source_url"]),
                    "instance_id": str(record["instance_id"]),
                    "variation_id": str(record["variation_id"]),
                })
            record["preprocessing_status"] = "PROCESSED"
            record["preprocessing_error"] = ""
            record["generated_windows"] = len(windows)
            record["selected_windows"] = len(indices)
            record["local_clip_25fps"] = str(fps_clip)
            record["processing_quality"] = quality
            class_stats[record["gloss"]]["downloaded_videos"] += 1
            class_stats[record["gloss"]]["processed_videos"] += 1
            class_stats[record["gloss"]]["windows_generated"] += len(windows)
            class_stats[record["gloss"]]["windows_selected"] += len(indices)
            windows_total += len(windows)
            videos_processed += 1
        except Exception as exc:
            record["preprocessing_status"] = "REJECTED"
            record["preprocessing_error"] = f"{type(exc).__name__}: {exc}"
            rejected += 1
        if videos_processed % 10 == 0 or record["preprocessing_status"] == "REJECTED":
            manifest["records"] = list(cache_records.values()) + [
                row for row in manifest["records"]
                if (int(row["class_index"]), row["video_id"]) not in cache_records
            ]
            _persist_manifest(manifest)
        if record["preprocessing_status"] == "PROCESSED":
            print(
                f"PROCESSED {record['gloss']} id={video_id} "
                f"windows={record['generated_windows']} selected={record['selected_windows']}",
                flush=True,
            )
    manifest["records"] = list(cache_records.values()) + [
        row for row in manifest["records"] if row["video_id"] not in cache_records
    ]
    _persist_manifest(manifest)
    SEQUENCE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEQUENCE_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=SEQUENCE_FIELDS)
        writer.writeheader()
        writer.writerows(sequence_rows)
    qc = {
        "dataset": "WLASL v0.3",
        "experiment": "WLASL V2",
        "subset": manifest["subset"],
        "status": "QC_COMPLETE",
        "metadata_sha256": manifest["metadata_sha256"],
        "videos_downloaded": len(_unique_downloaded_records(manifest["records"])),
        "videos_processed": videos_processed,
        "videos_rejected": rejected,
        "temporal_windows_generated": windows_total,
        "training_windows_selected": len(sequence_rows),
        "classes_downloaded": sorted({row["label"] for row in sequence_rows}),
        "selected_glosses": selected_glosses,
        "class_selection": manifest["preprocessing_selection"],
        "rejection_reasons": {
            f"{row['class_index']}:{row['video_id']}": row["preprocessing_error"]
            for row in manifest["records"]
            if row["preprocessing_status"] == "REJECTED"
        },
        "landmark_quality_by_video": {
            f"{row['class_index']}:{row['video_id']}": row["processing_quality"]
            for row in manifest["records"]
            if row["preprocessing_status"] == "PROCESSED" and row.get("processing_quality")
        },
        "class_statistics": dict(class_stats),
        "feature_dimension": 562,
        "sequence_length": 30,
        "normalization": "wrist-relative using project Holistic MSL normalization helpers",
        "temporal_sampling": "Videos resampled from timestamps to 25fps; 30-frame windows with 15-frame stride; at most 3 evenly spaced windows selected per source clip.",
        "missing_landmarks": "Model pipeline zero-fills absent groups and stores landmark presence; windows with no detected Holistic landmarks are rejected.",
        "augmentation": "Only applied to training sequences during model fitting; temporal shift +/-2 frames and low-amplitude Gaussian landmark jitter. Presence channels remain unchanged.",
        "config": config,
    }
    QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    QC_PATH.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    (DATA_ROOT / "preprocessing_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "videos_processed": videos_processed,
        "videos_rejected": rejected,
        "classes_selected": len(selected_glosses),
        "windows_generated": windows_total,
        "windows_selected": len(sequence_rows),
    }


def _split_samples(samples: list[baseline.Sample], seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = {sample.label for sample in samples}
    if len(labels) < 2:
        raise ValueError("At least two WLASL gloss classes need usable sequences.")
    best: tuple[int, list[dict[str, Any]], dict[str, Any], int] | None = None
    for candidate_seed in range(seed, seed + 5_000):
        rows, summary = baseline.create_splits(samples, candidate_seed)
        if summary["signer_separation"] != "AVAILABLE" or summary["signer_overlap"]:
            continue
        classes_by_split = {
            name: {row["label"] for row in rows if row["split"] == name}
            for name in baseline.SPLIT_NAMES
        }
        if not labels.issubset(classes_by_split["train"]):
            continue
        score = 100 * len(classes_by_split["test"]) + 10 * len(classes_by_split["validation"])
        if best is None or score > best[0]:
            best = (score, rows, summary, candidate_seed)
        if labels.issubset(classes_by_split["validation"]) and labels.issubset(classes_by_split["test"]):
            best = (score, rows, summary, candidate_seed)
            break
    if best is None:
        raise ValueError("Could not construct nonempty train/validation/test partitions with signer separation.")
    _, rows, summary, chosen_seed = best
    classes_by_split = {
        name: sorted({row["label"] for row in rows if row["split"] == name})
        for name in baseline.SPLIT_NAMES
    }
    summary["class_counts"] = {name: len(classes_by_split[name]) for name in baseline.SPLIT_NAMES}
    summary["classes_missing_validation"] = sorted(labels - set(classes_by_split["validation"]))
    summary["classes_missing_test"] = sorted(labels - set(classes_by_split["test"]))
    summary["class_coverage_by_split"] = classes_by_split
    summary["split_seed"] = chosen_seed
    return rows, summary


def _write_split_manifest(rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / "split_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        fields = ["sample_id", "group_id", "signer_id", "label", "split", "sequence_path"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (SPLIT_DIR / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _video_predictions(
    truth: list[int],
    probabilities: list[list[float]],
    rows: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    by_video: dict[str, dict[str, Any]] = {}
    for target, scores, row in zip(truth, probabilities, rows):
        video = by_video.setdefault(row["group_id"], {
            "label": row["label"],
            "signer_id": row["signer_id"],
            "class_index": target,
            "scores": [],
        })
        video["scores"].append(scores)
    y_true: list[int] = []
    y_pred: list[int] = []
    top5: list[bool] = []
    records = []
    for video_id, value in by_video.items():
        mean_scores = np.mean(value["scores"], axis=0)
        order = np.argsort(mean_scores)[::-1]
        prediction = int(order[0])
        y_true.append(int(value["class_index"]))
        y_pred.append(prediction)
        top5.append(int(value["class_index"]) in order[:min(5, len(labels))])
        records.append({
            "video_id": video_id,
            "signer_id": value["signer_id"],
            "actual_class": value["label"],
            "predicted_class": labels[prediction],
            "confidence": float(mean_scores[prediction]),
            "windows": len(value["scores"]),
            "top5_correct": int(value["class_index"]) in order[:min(5, len(labels))],
        })
    precision, recall, f1, support = baseline.precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    matrix = baseline.confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return {
        "accuracy": float(baseline.accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "videos": len(y_true),
        "top5_accuracy": float(np.mean(top5)) if top5 else 0.0,
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
        "records": records,
    }


def _predict_loader(model: Any, loader: Any, device: Any) -> tuple[list[int], list[list[float]], list[dict[str, Any]]]:
    import torch

    truth: list[int] = []
    probabilities: list[list[float]] = []
    metadata: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for values, targets, lengths, batch_metadata in loader:
            logits = model(values.to(device), lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.extend(probs.tolist())
            truth.extend(targets.numpy().tolist())
            count = len(batch_metadata["sample_id"])
            metadata.extend({
                "sample_id": batch_metadata["sample_id"][i],
                "group_id": batch_metadata["group_id"][i],
                "signer_id": batch_metadata["signer_id"][i],
                "label": batch_metadata["label"][i],
            } for i in range(count))
    return truth, probabilities, metadata


def _train_trial(
    trial_name: str,
    train_samples: list[baseline.Sample],
    validation_samples: list[baseline.Sample],
    labels: list[str],
    split_rows: list[dict[str, Any]],
    *,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 8e-4,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    label_to_index = {label: index for index, label in enumerate(labels)}
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = baseline.BiLSTMBaseline(
        input_size=train_samples[0].sequence.shape[1],
        num_classes=len(labels),
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.35,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model.to(device)
    train_dataset = baseline.SequenceDataset(train_samples, label_to_index)
    validation_dataset = baseline.SequenceDataset(validation_samples, label_to_index)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    video_counts = Counter(sample.label for sample in train_samples)
    class_weights = torch.tensor(
        [len(train_samples) / max(1, len(labels) * video_counts[label]) for label in labels],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    output_dir = MODEL_ROOT / "trials" / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_key: tuple[float, float, float] | None = None
    best_checkpoint = output_dir / "best_validation.pt"
    presence_channels = 4
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for values, targets, lengths, batch_metadata in train_loader:
            _ = batch_metadata
            values = values.to(device)
            targets = targets.to(device)
            # Mild temporal jitter and coordinate noise only; presence flags are never perturbed.
            if values.shape[1] > 3:
                shift = torch.randint(-2, 3, (values.shape[0], 1, 1), device=device)
                frame_positions = torch.arange(values.shape[1], device=device).view(1, -1, 1)
                shifted_positions = (frame_positions + shift).clamp(0, values.shape[1] - 1)
                values = torch.gather(
                    values, 1,
                    shifted_positions.expand(-1, -1, values.shape[2]),
                )
            coordinate_dims = values.shape[-1] - presence_channels
            if coordinate_dims > 0:
                noise = torch.randn_like(values[:, :, :coordinate_dims]) * 0.003
                values = torch.cat((values[:, :, :coordinate_dims] + noise, values[:, :, coordinate_dims:]), dim=-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation_truth, validation_probabilities, validation_rows = _predict_loader(
            model, validation_loader, device
        )
        window_predictions = [int(np.argmax(row)) for row in validation_probabilities]
        window_loss = float(np.mean([
            -np.log(max(1e-12, probability[target]))
            for target, probability in zip(validation_truth, validation_probabilities)
        ])) if validation_truth else float("inf")
        video = _video_predictions(
            validation_truth, validation_probabilities, validation_rows, labels
        )
        key = (video["accuracy"], video["macro_f1"], -window_loss)
        history.append({
            "epoch": epoch,
            "training_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_window_loss": window_loss,
            "validation_window_accuracy": float(np.mean(
                np.asarray(validation_truth) == np.asarray(window_predictions)
            )) if validation_truth else 0.0,
            "validation_video_accuracy": video["accuracy"],
            "validation_video_macro_f1": video["macro_f1"],
        })
        scheduler.step(window_loss)
        if best_key is None or key > best_key:
            best_key = key
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_index": label_to_index,
                "input_size": int(train_samples[0].sequence.shape[1]),
                "sequence_length": int(train_samples[0].sequence.shape[0]),
                "architecture": "BiLSTM",
                "hidden_size": hidden_size,
                "num_layers": 2,
                "dropout": 0.35,
                "dataset": "WLASL v0.3 top-100 (WLASL V2)",
                "language": "ASL",
                "experiment_tag": baseline.EXPERIMENT_TAG,
                "seed": seed,
            }, best_checkpoint)
        print(
            f"{trial_name} epoch {epoch}/{epochs} "
            f"val_video_acc={video['accuracy']:.4f} val_macro_f1={video['macro_f1']:.4f}",
            flush=True,
        )
    return {
        "trial": trial_name,
        "checkpoint": str(best_checkpoint),
        "hidden_size": hidden_size,
        "epochs_run": epochs,
        "best_validation_video_accuracy": best_key[0] if best_key else 0.0,
        "best_validation_video_macro_f1": best_key[1] if best_key else 0.0,
        "history": history,
    }


def train(epochs: int = 20, batch_size: int = 16, seed: int = 42) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    if not SEQUENCE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("Run WLASL V2 preprocessing first.")
    samples = baseline.load_samples(SEQUENCE_MANIFEST_PATH, language="ASL")
    if not samples:
        raise ValueError("No usable WLASL V2 temporal sequences.")
    split_rows, split_summary = _split_samples(samples, seed)
    _write_split_manifest(split_rows, split_summary)
    train_ids = {row["sample_id"] for row in split_rows if row["split"] == "train"}
    validation_ids = {row["sample_id"] for row in split_rows if row["split"] == "validation"}
    test_ids = {row["sample_id"] for row in split_rows if row["split"] == "test"}
    train_samples = [sample for sample in samples if sample.sample_id in train_ids]
    validation_samples = [sample for sample in samples if sample.sample_id in validation_ids]
    test_samples = [sample for sample in samples if sample.sample_id in test_ids]
    labels = sorted({sample.label for sample in train_samples})
    if len(labels) < 2 or not validation_samples or not test_samples:
        raise ValueError("Training, validation, and test must all contain samples and at least two train labels.")
    if not validation_ids or not test_ids:
        raise ValueError("Signer-disjoint validation/test split unexpectedly empty.")

    trials = []
    for hidden in (64, 128):
        trial = _train_trial(
            f"bilstm_h{hidden}_seed{seed}",
            train_samples,
            validation_samples,
            labels,
            split_rows,
            hidden_size=hidden,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        trials.append(trial)
    # Only validation metrics choose the model. The held-out test is evaluated once below.
    selected = max(
        trials,
        key=lambda item: (
            item["best_validation_video_accuracy"],
            item["best_validation_video_macro_f1"],
        ),
    )
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    final_checkpoint = MODEL_ROOT / "best_model.pt"
    shutil.copy2(selected["checkpoint"], final_checkpoint)
    model, checkpoint, device = baseline.load_checkpoint(final_checkpoint)
    test_dataset = baseline.SequenceDataset(test_samples, checkpoint["label_to_index"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    truth, probabilities, metadata = _predict_loader(model, test_loader, device)
    labels = [label for label, _ in sorted(
        checkpoint["label_to_index"].items(), key=lambda item: item[1]
    )]
    prediction_indices = [int(np.argmax(row)) for row in probabilities]
    top5_flags = [
        target in np.argsort(row)[::-1][:min(5, len(labels))]
        for target, row in zip(truth, probabilities)
    ]
    window_precision, window_recall, window_f1, window_support = baseline.precision_recall_fscore_support(
        truth, prediction_indices, labels=list(range(len(labels))), zero_division=0
    )
    window_metrics = {
        "accuracy": float(baseline.accuracy_score(truth, prediction_indices)),
        "top5_accuracy": float(np.mean(top5_flags)) if top5_flags else 0.0,
        "macro_f1": float(np.mean(window_f1)),
        "per_class": {
            label: {
                "precision": float(window_precision[index]),
                "recall": float(window_recall[index]),
                "f1": float(window_f1[index]),
                "support": int(window_support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": baseline.confusion_matrix(truth, prediction_indices, labels=list(range(len(labels)))).tolist(),
        "test_windows": len(truth),
    }
    video_metrics = _video_predictions(truth, probabilities, metadata, labels)
    train_dataset = baseline.SequenceDataset(train_samples, checkpoint["label_to_index"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_truth, train_probabilities, train_metadata = _predict_loader(model, train_loader, device)
    train_video_metrics = _video_predictions(
        train_truth, train_probabilities, train_metadata, labels
    )
    signer_scores: dict[str, list[bool]] = defaultdict(list)
    for record in video_metrics["records"]:
        signer_scores[record["signer_id"]].append(record["actual_class"] == record["predicted_class"])
    confidence = [max(row) for row in probabilities]
    metrics = {
        "dataset": checkpoint["dataset"],
        "language": "ASL",
        "experiment_tag": checkpoint["experiment_tag"],
        "model_selected_by": "Validation video-level top-1, then validation video macro F1; test set not used for selection.",
        "best_model": selected["trial"],
        "selection_trials": [
            {key: value for key, value in trial.items() if key != "history"}
            for trial in trials
        ],
        "split_summary": split_summary,
        "training_video_level": {
            key: value for key, value in train_video_metrics.items() if key != "records"
        },
        "sequence_level": window_metrics,
        "video_level": {key: value for key, value in video_metrics.items() if key != "records"},
        "video_predictions": video_metrics["records"],
        "signer_video_accuracy": {
            signer: float(np.mean(values)) for signer, values in sorted(signer_scores.items())
        },
        "confidence_distribution": {
            "mean": float(np.mean(confidence)) if confidence else 0.0,
            "median": float(np.median(confidence)) if confidence else 0.0,
            "p90": float(np.percentile(confidence, 90)) if confidence else 0.0,
            "below_0_45_fraction": float(np.mean(np.asarray(confidence) < 0.45)) if confidence else 0.0,
        },
        "top5_video_accuracy": video_metrics["top5_accuracy"],
        "class_count": len(labels),
        "classes": labels,
        "preprocessing_selection": json.loads(QC_PATH.read_text(encoding="utf-8"))["class_selection"],
        "seed": seed,
        "checkpoint": str(final_checkpoint.resolve()),
        "metadata_sha256": json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["metadata_sha256"],
        "sequence_manifest_sha256": _sha256(SEQUENCE_MANIFEST_PATH),
    }
    # The split has one held-out prediction per source video; window confusion is also retained.
    support = video_metrics["per_class"]
    metrics["best_classes"] = sorted(
        (label for label, score in support.items() if score["support"]),
        key=lambda label: support[label]["f1"],
        reverse=True,
    )[:10]
    metrics["worst_classes"] = sorted(
        (label for label, score in support.items() if score["support"]),
        key=lambda label: support[label]["f1"],
    )[:10]
    metrics["most_common_confusions"] = _top_confusions(
        video_metrics["confusion_matrix"], labels
    )
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    METRICS_ROOT.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    selected_config = {
        "dataset": checkpoint["dataset"],
        "dataset_version": "WLASL_v0.3",
        "subset": json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["subset"],
        "class_count": len(labels),
        "preprocessing_selection": metrics["preprocessing_selection"],
        "label_to_index": checkpoint["label_to_index"],
        "architecture": f"BiLSTM(hidden_size={checkpoint['hidden_size']}, num_layers=2, bidirectional=True)",
        "sequence_length": checkpoint["sequence_length"],
        "feature_dimension": checkpoint["input_size"],
        "fps": 25,
        "normalization": "wrist_relative",
        "missing_landmarks": "Zero-filled feature groups with explicit presence channels; windows lacking all Holistic landmarks rejected.",
        "windows_per_video_max": MAX_WINDOWS_PER_VIDEO,
        "augmentation": ["temporal shift +/-2 frames", "coordinate Gaussian jitter std=0.003; presence features unchanged"],
        "split_method": "Deterministic signer-grouped 70/15/15; all windows from a WLASL video share one partition.",
        "split_seed": seed,
        "metadata_sha256": metrics["metadata_sha256"],
        "sequence_manifest_sha256": metrics["sequence_manifest_sha256"],
        "model_selection": metrics["model_selected_by"],
        "selected_trial": selected["trial"],
        "trial_history": {trial["trial"]: trial["history"] for trial in trials},
    }
    (MODEL_ROOT / "config.json").write_text(json.dumps(selected_config, indent=2) + "\n", encoding="utf-8")
    (MODEL_ROOT / "test_video_predictions.csv").write_text(
        _csv_text(video_metrics["records"]), encoding="utf-8"
    )
    write_report(metrics)
    return metrics


def _csv_text(rows: list[dict[str, Any]]) -> str:
    import io
    output = io.StringIO()
    fields = ["video_id", "signer_id", "actual_class", "predicted_class", "confidence", "windows", "top5_correct"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _top_confusions(matrix: list[list[int]], labels: list[str], limit: int = 20) -> list[dict[str, Any]]:
    result = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            if i != j and count:
                result.append({"actual": labels[i], "predicted": labels[j], "count": int(count)})
    return sorted(result, key=lambda item: item["count"], reverse=True)[:limit]


def write_report(metrics: dict[str, Any] | None = None) -> Path:
    if metrics is None and (METRICS_ROOT / "metrics.json").is_file():
        metrics = json.loads((METRICS_ROOT / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else {}
    records = manifest.get("records", [])
    qc = json.loads(QC_PATH.read_text(encoding="utf-8")) if QC_PATH.is_file() else {}
    split_path = SPLIT_DIR / "split_summary.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}
    free_space = shutil.disk_usage(DATA_ROOT if DATA_ROOT.exists() else PROJECT_ROOT)
    fail_categories = Counter()
    for row in records:
        if row.get("download_status") != "FAILED":
            continue
        reason = row.get("failure_reason", "")
        if "404" in reason:
            category = "HTTP 404"
        elif "403" in reason:
            category = "HTTP 403"
        elif "timed out" in reason.lower() or "timeout" in reason.lower():
            category = "timeouts"
        elif "not a video" in reason.lower() or "could not open" in reason.lower():
            category = "invalid/non-video payload"
        else:
            category = "other errors"
        fail_categories[category] += 1
    downloaded_rows = [
        row for row in records if row.get("download_status") == "DOWNLOADED"
    ]
    downloaded = _unique_downloaded_records(records)
    per_class_download = Counter(row["gloss"] for row in downloaded)
    quality_rows = list(qc.get("landmark_quality_by_video", {}).values())
    def mean_missing_rate(name: str) -> float | None:
        rates = [
            row["missing_landmark_rates"][name]
            for row in quality_rows
            if name in row.get("missing_landmark_rates", {})
        ]
        return float(np.mean(rates)) if rates else None

    mean_missing_landmark_rates = {
        name: mean_missing_rate(name)
        for name in ("left_hand", "right_hand", "pose", "face")
    }
    lines = [
        "# WLASL V2 Dataset-Scale Experiment",
        "",
        "**Status:** " + ("TRAINED (EXPERIMENTAL)" if metrics else "ACQUISITION / PREPROCESSING"),
        "",
        "> WLASL is an ASL dataset. It is used here to validate a dynamic video-recognition pipeline and is not Malawi Sign Language.",
        "",
        "The experiment and its artifacts are isolated from BridgeSign production models/data and from the earlier WLASL V1 run.",
        "WLASL is subject to the C-UDA; the upstream description restricts it to academic/computational use and prohibits commercial use.",
        "",
        "## Objective and official dataset information",
        "",
        "- Official source: `dxli94/WLASL`, metadata `WLASL_v0.3.json`.",
        "- Official metadata: 2,000 gloss classes, 21,083 video instances and 119 distinct signer IDs (as counted locally).",
        "- Official top-K gloss subsets are the first 100/300/1000/2000 gloss entries in the metadata.",
        "- Upstream acquisition: metadata URLs, direct sources and YouTube (`yt-dlp`); inaccessible videos are expected and recorded as missing/failed.",
        "- Upstream says preprocessed videos may be requested by agreeing to C-UDA; they are not bundled in the official GitHub repository.",
        "",
        "## Machine capacity and processed-mirror check",
        "",
        "- Machine: 16 GB RAM, Intel i7-1185G7 CPU, Intel Iris Xe graphics, no CUDA; only about 2 GB RAM was free at inspection.",
        f"- Free disk at report generation: {free_space.free / (1024**3):.1f} GiB.",
        "- Kaggle mirror `risangbaskoro/wlasl-processed`: its public metadata describes about 12k MP4s, 2k gloss labels and a 5.40 GB archive; its description repeats the WLASL C-UDA academic/computational-use restriction and links back to the official source.",
        "- Kaggle's anonymous dataset-download endpoint returned HTTP 404 in this environment, and no Kaggle CLI/API credential is configured. The mirror was not downloaded or treated as verified local content.",
        "- `UnmannedArchive/signspeak` describes a separate 225-clip/18-gloss project and states 44.1% test accuracy; its downloader uses metadata-listed direct sources and YouTube. It is a code/project reference, not evidence that the Kaggle archive or a full local dataset is accessible.",
        f"- The V2 experiment is using official metadata links; no third-party mirror files were used. V2 raw acquisition currently uses {len(downloaded)} decoded video files ({sum(int(row.get('download_bytes') or 0) for row in downloaded) / (1024**3):.2f} GiB).",
        "",
        "## V2 acquisition",
        "",
        f"- Selected subset: {manifest.get('subset', 'not created')} ({manifest.get('class_count', 0)} official metadata-ranked gloss classes).",
        f"- Candidate metadata instances recorded: {len(records)}.",
        f"- Actual downloaded and decoded videos: {len(downloaded)}.",
        f"- Duplicate downloaded metadata records: {len(downloaded_rows) - len(downloaded)}; these repeated class/video IDs are preserved in the manifest but treated as a single source video.",
        f"- Failed downloads and reasons: {sum(fail_categories.values())}; `{json.dumps(dict(fail_categories), sort_keys=True)}`.",
        f"- Unsupported/non-video source formats recorded: {sum(row.get('download_status') == 'UNSUPPORTED_SOURCE_FORMAT' for row in records)}.",
        f"- Per-gloss unique video counts: `{json.dumps(dict(per_class_download), sort_keys=True)}`.",
        f"- All exact glosses and source records are preserved in `{MANIFEST_PATH}`.",
        "",
        "## Preprocessing and split",
        "",
        f"- Training vocabulary selection: {len(qc.get('selected_glosses', []))} glosses ranked by verified video count, then signer diversity, then official class index: `{json.dumps(qc.get('selected_glosses', []))}`.",
        "- Other downloaded videos remain preserved in the acquisition manifest and are explicitly marked as not selected for compute-limited preprocessing.",
        f"- Holistic videos processed: {qc.get('videos_processed', 0)}; rejected: {qc.get('videos_rejected', 0)}.",
        f"- Temporal windows generated: {qc.get('temporal_windows_generated', 0)}; windows retained: {qc.get('training_windows_selected', 0)}.",
        "- Feature representation: MediaPipe Holistic, normalized 562-feature frames, 30 frames/window, 15-frame stride, source clips resampled to 25fps.",
        "- At most three evenly spaced windows are retained per clip. All windows remain grouped by source video for splitting and video-level evaluation.",
        "- Per-video frame counts, detected-frame counts, landmark missing rates, and rejection reasons are stored in the QC JSON and acquisition manifest.",
        f"- Mean missing-landmark rates across selected videos: `{json.dumps(mean_missing_landmark_rates, sort_keys=True)}`.",
        f"- Split summary: `{json.dumps(split, sort_keys=True)}`.",
        "- Split is signer-grouped and deterministic. Any class missing from a held-out partition is identified in the split summary and not silently counted as evaluated.",
        "",
        "## Model selection and evaluation",
        "",
        "- Compared BiLSTM hidden sizes 64 and 128, two layers, dropout 0.35. Training-only augmentation: temporal shift +/-2 frames and low-amplitude coordinate jitter; presence channels unchanged.",
        "- Selection criterion: validation video-level accuracy, then validation video macro F1. Test data is not used for architecture/epoch selection.",
    ]
    if metrics:
        seq = metrics["sequence_level"]
        video = metrics["video_level"]
        lines.extend([
            f"- Selected model: {metrics['best_model']}.",
            f"- Sequence/window test accuracy: {seq['accuracy']:.4f}; top-5: {seq['top5_accuracy']:.4f}; macro F1: {seq['macro_f1']:.4f}; windows: {seq['test_windows']}.",
            f"- Training video-level accuracy (resubstitution): {metrics['training_video_level']['accuracy']:.4f}; macro F1: {metrics['training_video_level']['macro_f1']:.4f}; videos: {metrics['training_video_level']['videos']}.",
            f"- Video-level signer-disjoint test accuracy (primary): {video['accuracy']:.4f}; top-5: {video['top5_accuracy']:.4f}; macro F1: {video['macro_f1']:.4f}; videos: {video['videos']}.",
            f"- Best/worst evaluated classes by F1: {json.dumps({'best': metrics['best_classes'], 'worst': metrics['worst_classes']})}.",
            f"- Most common confusions: `{json.dumps(metrics['most_common_confusions'])}`.",
            f"- Signer video accuracy: `{json.dumps(metrics['signer_video_accuracy'], sort_keys=True)}`.",
            f"- Confidence distribution: `{json.dumps(metrics['confidence_distribution'], sort_keys=True)}`.",
            f"- Test collapse checks: evaluated class support={sum(x['support'] > 0 for x in video['per_class'].values())}; predicted class counts are in the video prediction CSV.",
        ])
    else:
        lines.append("No training/evaluation completed yet.")
    lines.extend([
        "",
        "## Webcam / quality gate",
        "",
        "- Isolated V2 webcam demo is provided separately and loads only the WLASL V2 checkpoint.",
        "- Physical webcam recognition accuracy has not been measured unless manually exercised and recorded.",
        f"- 90% gate achieved: {'YES' if metrics and metrics['video_level']['accuracy'] >= 0.90 else 'NO / NOT ESTABLISHED'}. This gate requires >=90% signer-disjoint video-level top-1 for the intended vocabulary, macro F1 and a successful live webcam test.",
        "- WLASL results are ASL-only engineering results; they do not establish MSL recognition.",
        "",
        "## Reproduction commands",
        "",
        "```powershell",
        "python tools\\experimental_dynamic\\wlasl_v2.py manifest --classes 100",
        "python tools\\experimental_dynamic\\wlasl_v2.py acquire --classes 100 --max-per-class 20 --workers 4 --max-attempts-per-class 30",
        "python tools\\experimental_dynamic\\wlasl_v2.py preprocess --max-classes 10",
        "python tools\\experimental_dynamic\\wlasl_v2.py train --epochs 20 --batch-size 16 --seed 42",
        "python tools\\experimental_dynamic\\wlasl_v2_live_demo.py --camera 0",
        "```",
        "",
    ])
    benchmark_json = (
        PROJECT_ROOT / "reports" / "experimental" / "wlasl" / "wlasl_v2_model_comparison.json"
    )
    if benchmark_json.is_file():
        comparison = json.loads(benchmark_json.read_text(encoding="utf-8"))
        best = comparison["best_model"]
        selected = next(
            item for item in comparison["experiments"] if item["name"] == best["name"]
        )
        test = selected["test_video_level"]
        lines.extend([
            "",
            "## Missingness-aware model benchmark and webcam selection",
            "",
            "> This follow-up benchmark supersedes the preceding baseline checkpoint selection for the experimental webcam demo only. It does not change production models.",
            "",
            f"- Landmark diagnostic: `reports/experimental/wlasl/wlasl_v2_landmark_diagnostic.md`; per-video frame coverage and cautious attribution limits are recorded there.",
            f"- Five controlled variants used the same {comparison['sample_windows']} windows, labels, saved signer-disjoint split and augmentation policy: `{json.dumps([item['name'] for item in comparison['experiments']])}`.",
            f"- Validation-selected checkpoint: `{best['checkpoint']}` ({best['name']}, {best['framework']}, {best['representation']}, {best['input_dimension']} input features/frame).",
            f"- Held-out evaluation: {test['videos']} videos, top-1 accuracy {test['accuracy']:.4f}, macro F1 {test['macro_f1']:.4f}, top-5 {test['top5_accuracy']:.4f}. The >=90% top-1 and macro-F1 target was {'reached' if comparison['target_reached'] else 'not reached'}.",
            f"- Detailed comparison: `reports/experimental/wlasl/wlasl_v2_model_comparison.md`; held-out error analysis: `reports/experimental/wlasl/wlasl_v2_error_analysis.md`.",
            "- The webcam demo explicitly loads the validation-selected experimental checkpoint and applies representation_v2 to live Holistic frames. Physical webcam accuracy has not been measured.",
            "- Reproduce the diagnostics and five-model benchmark with `python -m tools.experimental_dynamic.wlasl_v2_benchmark diagnose` and `python -m tools.experimental_dynamic.wlasl_v2_benchmark benchmark --epochs 24`.",
            "",
        ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated WLASL V2 acquisition, Holistic, and BiLSTM experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--classes", type=int, choices=(100, 300, 1000, 2000), default=DEFAULT_CLASSES)
    manifest.add_argument("--force", action="store_true")
    download = commands.add_parser("acquire")
    download.add_argument("--classes", type=int, choices=(100, 300, 1000, 2000), default=DEFAULT_CLASSES)
    download.add_argument("--max-per-class", type=int, default=DEFAULT_CAP_PER_CLASS)
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--max-attempts-per-class", type=int, default=DEFAULT_ATTEMPTS_PER_CLASS)
    preprocess_command = commands.add_parser("preprocess")
    preprocess_command.add_argument("--max-classes", type=int, default=10)
    train_command = commands.add_parser("train")
    train_command.add_argument("--epochs", type=int, default=20)
    train_command.add_argument("--batch-size", type=int, default=16)
    train_command.add_argument("--seed", type=int, default=42)
    commands.add_parser("report")
    args = parser.parse_args()
    if args.command == "manifest":
        print(build_manifest(args.classes, force=args.force))
    elif args.command == "acquire":
        print(json.dumps(acquire(
            args.classes, args.max_per_class, args.workers, args.max_attempts_per_class
        ), indent=2))
    elif args.command == "preprocess":
        print(json.dumps(preprocess(args.max_classes), indent=2))
    elif args.command == "train":
        print(json.dumps(train(args.epochs, args.batch_size, args.seed), indent=2))
    else:
        print(write_report())


if __name__ == "__main__":
    main()
