from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np

import config
from feature_extractor import FEATURE_VECTOR_SIZE, FeatureExtractor

OUTPUT_FILENAME = "static_video_samples.csv"
SAMPLE_INTERVAL_MS = 50.0
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_SIGN_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,39}$")
_SIGNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
FEATURE_COLUMNS = [f"feature_{index:02d}" for index in range(FEATURE_VECTOR_SIZE)]
OUTPUT_COLUMNS = [
    "sample_id",
    "sign_label",
    "signer_id",
    "source_video_sha256",
    "source_frame_index",
    "timestamp_ms",
    *FEATURE_COLUMNS,
]


class VideoImportError(ValueError):
    pass


def normalize_sign_label(value: str) -> str:
    label = str(value or "").strip().upper().replace(" ", "_")
    if not _SIGN_RE.fullmatch(label):
        raise VideoImportError("Choose a valid sign label from the existing sign list.")
    return label


def validate_signer_id(value: str) -> str:
    signer_id = str(value or "").strip()
    if not _SIGNER_RE.fullmatch(signer_id):
        raise VideoImportError("Signer ID must use 1-32 letters, numbers, underscores, or hyphens.")
    return signer_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_signer_ids(output_path: str | Path | None = None) -> list[str]:
    path = Path(output_path or (Path(config.DATA_DIR) / OUTPUT_FILENAME))
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "signer_id" not in reader.fieldnames:
            return []
        return sorted({row["signer_id"].strip() for row in reader if row.get("signer_id", "").strip()})


def extract_static_video(
    video_path: str | Path,
    sign_label: str,
    signer_id: str,
    output_path: str | Path | None = None,
    sample_interval_ms: float = SAMPLE_INTERVAL_MS,
    detector=None,
    extractor=None,
) -> dict:
    """Extract mirrored static-sign XY features into a separate staging CSV."""
    path = Path(video_path)
    label = normalize_sign_label(sign_label)
    signer = validate_signer_id(signer_id)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise VideoImportError("Unsupported video type. Use MP4, AVI, MOV, MKV, WEBM, or M4V.")
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoImportError("The selected video is empty or unavailable.")
    if sample_interval_ms <= 0:
        raise VideoImportError("Sample interval must be greater than zero.")

    destination = Path(output_path or (Path(config.DATA_DIR) / OUTPUT_FILENAME))
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoImportError("MediaPipe could not open the selected video.")

    owns_detector = detector is None
    if detector is None:
        from hand_detector import HandDetector
        detector = HandDetector(mode=False, max_hands=1)
    hand_detector = detector
    feature_extractor = extractor or FeatureExtractor()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    records = []
    frame_index = 0
    next_sample_ms = 0.0
    frames_processed = 0
    frames_with_hands = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = frame_index * 1000.0 / fps
            if timestamp_ms + 1e-6 >= next_sample_ms:
                mirrored_frame = cv2.flip(frame, 1)
                mirrored_frame, results = hand_detector.find_hands(mirrored_frame, draw=False)
                frames_processed += 1
                landmarks = hand_detector.get_landmarks(mirrored_frame, hand_no=0)
                if landmarks:
                    frames_with_hands += 1
                    features = feature_extractor.extract_features(landmarks)
                    if features is not None and len(features) == FEATURE_VECTOR_SIZE and np.isfinite(features).all():
                        records.append({
                            "sample_id": f"{source_hash[:16]}-{frame_index:08d}",
                            "sign_label": label,
                            "signer_id": signer,
                            "source_video_sha256": source_hash,
                            "source_frame_index": frame_index,
                            "timestamp_ms": round(timestamp_ms, 3),
                            **{column: format(float(value), ".9g") for column, value in zip(FEATURE_COLUMNS, features)},
                        })
                while next_sample_ms <= timestamp_ms + 1e-6:
                    next_sample_ms += sample_interval_ms
            frame_index += 1
    finally:
        capture.release()
        if owns_detector:
            close = getattr(getattr(hand_detector, "_detector", None), "close", None)
            if callable(close):
                close()

    if not records:
        raise VideoImportError("No usable hand landmarks were found; no samples were saved.")

    if destination.exists() and destination.stat().st_size:
        with destination.open("r", newline="", encoding="utf-8") as existing:
            header = next(csv.reader(existing), [])
        if header != OUTPUT_COLUMNS:
            raise VideoImportError("The staging file has an unexpected format; no samples were appended.")

    write_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(records)

    return {
        "sign_label": label,
        "signer_id": signer,
        "source_video_sha256": source_hash,
        "frames_in_video": frame_index,
        "frames_sampled": frames_processed,
        "sampled_frames_with_hand": frames_with_hands,
        "samples_saved": len(records),
        "sample_interval_ms": sample_interval_ms,
        "output_filename": destination.name,
        "output_path": str(destination),
    }
