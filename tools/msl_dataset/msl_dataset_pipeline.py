from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

try:
    from .normalization import normalize_face_landmarks, normalize_hand_landmarks, normalize_pose_landmarks
    from .collection_management import (
        append_failure,
        collection_status,
        load_recordings,
        load_sessions,
        load_signers,
        update_dataset_version,
    )
except ImportError:  # Supports direct CLI imports from tools/msl_dataset.
    from normalization import normalize_face_landmarks, normalize_hand_landmarks, normalize_pose_landmarks
    from collection_management import (
        append_failure,
        collection_status,
        load_recordings,
        load_sessions,
        load_signers,
        update_dataset_version,
    )

try:
    import torch
except Exception:  # pragma: no cover - optional in minimal test environments
    torch = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - environment is expected to provide MediaPipe
    mp = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "msl_dataset.json"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_FACE_INDICES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
]


def default_config() -> dict[str, Any]:
    return {
        "raw_dataset_path": str(PROJECT_ROOT / "data" / "msl_dataset" / "raw"),
        "processed_dataset_path": str(PROJECT_ROOT / "data" / "msl_dataset" / "processed"),
        "manifests_path": str(PROJECT_ROOT / "data" / "msl_dataset" / "manifests"),
        "review_path": str(PROJECT_ROOT / "data" / "msl_dataset" / "review"),
        "holistic_model_path": str(PROJECT_ROOT / "models" / "holistic_landmarker.task"),
        "sequence_length": 30,
        "stride": 15,
        "min_frames": 12,
        "max_frames": 300,
        "random_seed": 42,
        "dataset_version": "0.1",
        "quality_thresholds": {
            "min_frames": 12,
            "max_frames": 300,
            "min_fps": 10,
            "max_fps": 60,
            "min_width": 320,
            "min_height": 240,
        },
        "collection_targets": {
            "signers_per_sign": 5,
            "recordings_per_signer_sign": 3,
        },
        "allow_padding": True,
        "allowed_video_extensions": sorted(SUPPORTED_VIDEO_EXTENSIONS),
        "face_landmark_indices": DEFAULT_FACE_INDICES,
        "normalization_method": "wrist_relative",
        "train_signers": [],
        "validation_signers": [],
        "test_signers": [],
        "feature_selection": {
            "left_hand": True,
            "right_hand": True,
            "pose": True,
            "face": True,
            "pose_landmark_subset": list(range(0, 33)),
        },
    }


def ensure_default_config(path: str | Path | None = None) -> Path:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(json.dumps(default_config(), indent=2) + "\n", encoding="utf-8")
    return config_path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        config_path = ensure_default_config(config_path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Could not parse MSL config at {config_path}: {exc}") from exc
    merged = default_config()
    merged.update(data)
    return merged


if torch is not None:
    class MSLSignDataset(torch.utils.data.Dataset):
        """Simple PyTorch-compatible dataset wrapper for processed MSL sequences."""

        def __init__(self, manifest_path: str | Path, label_to_index: dict[str, int] | None = None):
            self.manifest_path = Path(manifest_path)
            self.rows = []
            if self.manifest_path.exists():
                with self.manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
                    self.rows = list(csv.DictReader(csv_file))
            self.samples = [row for row in self.rows if row.get("sequence_path")]
            if label_to_index is None:
                labels = sorted({row.get("sign", "") for row in self.samples if row.get("sign")})
                self.label_to_index = {label: index for index, label in enumerate(labels)}
            else:
                self.label_to_index = dict(label_to_index)
            self.sequence_paths = [row["sequence_path"] for row in self.samples]

        def __len__(self) -> int:
            return len(self.sequence_paths)

        def __getitem__(self, index: int) -> tuple[Any, int, dict[str, str]]:
            if index < 0 or index >= len(self.sequence_paths):
                raise IndexError(f"Sequence index {index} out of range for dataset of length {len(self.sequence_paths)}.")
            path = Path(self.sequence_paths[index])
            with np.load(path, allow_pickle=False) as data:
                sequence = np.asarray(data["sequence"], dtype=np.float32)
                label = str(data.get("label", "UNKNOWN"))
                metadata = {
                    "signer_id": str(data.get("signer_id", "UNKNOWN")),
                    "video_id": str(data.get("video_id", "UNKNOWN")),
                    "session_id": str(data.get("session_id", "UNKNOWN")),
                }
            if label not in self.label_to_index:
                raise ValueError(f"Label '{label}' is missing from the dataset label mapping.")
            return torch.from_numpy(sequence), self.label_to_index[label], metadata
else:
    class MSLSignDataset:
        """Simple fallback dataset implementation when PyTorch is unavailable."""

        def __init__(self, manifest_path: str | Path, label_to_index: dict[str, int] | None = None):
            self.manifest_path = Path(manifest_path)
            self.rows = []
            if self.manifest_path.exists():
                with self.manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
                    self.rows = list(csv.DictReader(csv_file))
            self.samples = [row for row in self.rows if row.get("sequence_path")]
            self.label_to_index = dict(label_to_index or {})
            self.sequence_paths = [row["sequence_path"] for row in self.samples]

        def __len__(self) -> int:
            return len(self.sequence_paths)

        def __getitem__(self, index: int) -> tuple[np.ndarray, int, dict[str, str]]:
            if index < 0 or index >= len(self.sequence_paths):
                raise IndexError(f"Sequence index {index} out of range for dataset of length {len(self.sequence_paths)}.")
            path = Path(self.sequence_paths[index])
            with np.load(path, allow_pickle=False) as data:
                sequence = np.asarray(data["sequence"], dtype=np.float32)
                label = str(data.get("label", "UNKNOWN"))
                metadata = {
                    "signer_id": str(data.get("signer_id", "UNKNOWN")),
                    "video_id": str(data.get("video_id", "UNKNOWN")),
                    "session_id": str(data.get("session_id", "UNKNOWN")),
                }
            target = self.label_to_index.get(label, 0)
            return sequence, int(target), metadata


def _validate_signer_id(value: str) -> bool:
    return bool(re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$", str(value).strip()))


def _validate_sign_label(value: str) -> bool:
    return bool(re.fullmatch(r"^[A-Z][A-Z0-9_]{0,39}$", str(value).strip().upper()))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_landmarks(markmarks: Any) -> np.ndarray:
    if markmarks is None:
        return np.full((0,), np.nan, dtype=np.float32)
    points = []
    for landmark in markmarks:
        if hasattr(landmark, "x"):
            points.append([landmark.x, landmark.y, getattr(landmark, "z", 0.0)])
        elif isinstance(landmark, (list, tuple)) and len(landmark) >= 3:
            points.append([float(landmark[0]), float(landmark[1]), float(landmark[2])])
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.full((0,), np.nan, dtype=np.float32)
    return arr.reshape(-1)


@dataclass
class VideoRecord:
    signer_id: str
    sign: str
    video_path: Path
    video_id: str
    session_id: str
    recording_date: str
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    file_size: int
    status: str = "PREPARED"
    quality_status: str = "UNKNOWN"
    notes: str = ""
    missing_landmark_rates: dict[str, float] | None = None

    def as_manifest_row(self) -> dict[str, str | int | float]:
        return {
            "signer_id": self.signer_id,
            "sign": self.sign,
            "video_id": self.video_id,
            "session_id": self.session_id,
            "recording_date": self.recording_date,
            "file_path": str(self.video_path),
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "processing_status": self.status,
            "quality_status": self.quality_status,
            "notes": self.notes,
            **{
                f"{name}_missing_rate": rate
                for name, rate in (self.missing_landmark_rates or {}).items()
            },
        }


def discover_videos(raw_dir: str | Path, allowed_extensions: Iterable[str] | None = None) -> list[dict[str, Any]]:
    root = Path(raw_dir)
    if not root.exists():
        return []
    extensions = {f".{ext.lower().lstrip('.')}" for ext in (allowed_extensions or SUPPORTED_VIDEO_EXTENSIONS)}
    discovered: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 2:
            discovered.append({
                "path": str(path),
                "valid": False,
                "error": "Invalid folder layout. Expected <signer>/<sign>/<video>.",
            })
            continue
        signer_id = str(parts[0]).strip()
        sign = str(parts[1]).strip()
        if not _validate_signer_id(signer_id):
            discovered.append({
                "path": str(path),
                "valid": False,
                "error": f"Invalid signer_id '{signer_id}'.",
            })
            continue
        sign_label = sign.upper().replace(" ", "_")
        if not _validate_sign_label(sign_label):
            discovered.append({
                "path": str(path),
                "valid": False,
                "error": f"Invalid sign label '{sign}'.",
            })
            continue
        discovered.append({
            "path": str(path),
            "valid": True,
            "signer_id": signer_id,
            "sign": sign_label,
            "video_id": path.stem,
        })
    return discovered


def _read_video_metadata(path: Path) -> tuple[float, int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0
        return fps, frame_count, duration, width, height
    finally:
        capture.release()


def _safe_landmarks_from_result(landmarks: Any) -> np.ndarray:
    if landmarks is None:
        return np.full((0,), np.nan, dtype=np.float32)
    return _flatten_landmarks(landmarks.landmark if hasattr(landmarks, "landmark") else landmarks)


def _landmark_points(landmarks: Any, include_visibility: bool = False) -> np.ndarray:
    if landmarks is None:
        return np.empty((0, 4 if include_visibility else 3), dtype=np.float32)
    points = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    if len(points) == 0:
        return np.empty((0, 4 if include_visibility else 3), dtype=np.float32)
    result = []
    for landmark in points:
        values = [landmark.x, landmark.y, getattr(landmark, "z", 0.0)]
        if include_visibility:
            values.append(getattr(landmark, "visibility", np.nan))
        result.append(values)
    return np.asarray(result, dtype=np.float32)


def _create_holistic_detector(config: dict[str, Any]) -> tuple[Any, str]:
    if mp is None:
        raise RuntimeError("MediaPipe is not available in the current Python environment.")
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "holistic"):
        detector = solutions.holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        return detector, "legacy"

    model_path = Path(config.get("holistic_model_path", PROJECT_ROOT / "models" / "holistic_landmarker.task"))
    if not model_path.is_file():
        raise FileNotFoundError(
            "This MediaPipe installation provides the Tasks Holistic API but no model asset was found. "
            f"Set holistic_model_path to a downloaded holistic_landmarker.task file (expected: {model_path})."
        )
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )
    return vision.HolisticLandmarker.create_from_options(options), "tasks"


def _detect_frame(detector: Any, backend: str, frame: np.ndarray, timestamp_ms: int) -> Any:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if backend == "legacy":
        return detector.process(rgb)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return detector.detect_for_video(image, timestamp_ms)


def _frame_features(
    results: Any,
    face_indices: Sequence[int],
    feature_selection: dict[str, Any],
    normalization_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    landmarks = {
        "left_hand": _landmark_points(getattr(results, "left_hand_landmarks", None)),
        "right_hand": _landmark_points(getattr(results, "right_hand_landmarks", None)),
        "pose": _landmark_points(getattr(results, "pose_landmarks", None), include_visibility=True),
        "face": _landmark_points(getattr(results, "face_landmarks", None)),
    }
    raw_parts: list[np.ndarray] = []
    model_parts: list[np.ndarray] = []
    presence: list[float] = []
    enabled_names: list[str] = []
    for name in ("left_hand", "right_hand", "pose", "face"):
        if not feature_selection.get(name, False):
            continue
        enabled_names.append(name)
        points = landmarks[name]
        indices: Sequence[int]
        if name == "pose":
            indices = feature_selection.get("pose_landmark_subset", list(range(33)))
        elif name == "face":
            indices = face_indices
        else:
            indices = range(21)
        valid_indices = [int(i) for i in indices if 0 <= int(i) < len(points)]
        count = len(indices)
        has_visibility = name == "pose"
        dims = 4 if has_visibility else 3
        expected = np.full((count, dims), np.nan, dtype=np.float32)
        if points.size:
            for output_index, landmark_index in enumerate(valid_indices):
                expected[output_index, :min(dims, points.shape[1])] = points[landmark_index, :min(dims, points.shape[1])]
        is_present = bool(np.isfinite(points[:, :3]).all(axis=1).any()) if points.size else False
        presence.append(float(is_present))
        raw_parts.append(expected.reshape(-1))
        full_coords = points[:, :3].copy()
        if normalization_method == "wrist_relative":
            if name in {"left_hand", "right_hand"} and is_present:
                full_coords = normalize_hand_landmarks(full_coords).reshape(-1, 3)
            elif name == "pose" and is_present:
                full_coords = normalize_pose_landmarks(full_coords).reshape(-1, 3)
            elif name == "face" and is_present:
                full_coords = normalize_face_landmarks(full_coords).reshape(-1, 3)
        coords = np.full((count, 3), np.nan, dtype=np.float32)
        for output_index, landmark_index in enumerate(valid_indices):
            coords[output_index] = full_coords[landmark_index]
        if has_visibility:
            model_points = np.concatenate([coords, expected[:, 3:4]], axis=1)
        else:
            model_points = coords
        model_parts.append(np.nan_to_num(model_points.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0))
    if not model_parts:
        raise ValueError("At least one feature group must be enabled in feature_selection.")
    presence_array = np.asarray(presence, dtype=np.float32)
    model_vector = np.concatenate([*model_parts, presence_array]).astype(np.float32)
    raw_vector = np.concatenate(raw_parts).astype(np.float32)
    return model_vector, raw_vector, presence_array


def normalize_feature_array(values: np.ndarray, reference: np.ndarray | None = None, scale_value: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.full_like(values, np.nan, dtype=np.float32)
    if values.shape[-1] >= 3:
        values = values.reshape(-1, 3)
        if reference is None:
            reference = np.nanmean(values, axis=0)
        centered = values - reference
        if scale_value is None:
            distances = np.linalg.norm(centered, axis=1)
            magnitude = float(np.nanmax(distances)) if np.nanmax(distances) > 0 else 1.0
        else:
            magnitude = float(scale_value)
        if magnitude <= 0:
            magnitude = 1.0
        normalized = centered / magnitude
        return normalized.reshape(-1)
    return values


def _normalize_frame(frame_data: dict[str, np.ndarray], normalization_method: str) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for name, values in frame_data.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            normalized[name] = arr.reshape(-1)
            continue
        arr = arr.reshape(-1, 3) if arr.size % 3 == 0 else arr.reshape(-1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if normalization_method == "wrist_relative":
            if name in {"left_hand", "right_hand"}:
                if arr.shape[0] > 0:
                    wrist = arr[0]
                    centered = arr - wrist
                    scale = float(np.linalg.norm(centered[:10])) if centered.size else 1.0
                    if scale <= 0:
                        scale = 1.0
                    normalized[name] = (centered / scale).reshape(-1)
                else:
                    normalized[name] = arr.reshape(-1)
            elif name == "pose":
                if arr.shape[0] >= 2:
                    shoulders = np.nanmean(arr[[11, 12]] if arr.shape[0] > 12 else arr[:2], axis=0)
                    centered = arr - shoulders
                    scale = float(np.linalg.norm(centered[:15])) if centered.size else 1.0
                    if scale <= 0:
                        scale = 1.0
                    normalized[name] = (centered / scale).reshape(-1)
                else:
                    normalized[name] = arr.reshape(-1)
            elif name == "face":
                if arr.shape[0] > 0:
                    nose = arr[0] if arr.shape[0] >= 1 else np.zeros(3)
                    centered = arr - nose
                    scale = float(np.linalg.norm(centered)) if centered.size else 1.0
                    if scale <= 0:
                        scale = 1.0
                    normalized[name] = (centered / scale).reshape(-1)
                else:
                    normalized[name] = arr.reshape(-1)
            else:
                normalized[name] = arr.reshape(-1)
        else:
            normalized[name] = arr.reshape(-1)
    return normalized


def _flatten_frame(frame_data: dict[str, np.ndarray], feature_selection: dict[str, Any]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for name in ("left_hand", "right_hand", "pose", "face"):
        if not feature_selection.get(name, False):
            continue
        values = np.asarray(frame_data.get(name, np.full((0,), np.nan, dtype=np.float32)), dtype=np.float32)
        vectors.append(values.reshape(-1))
    if not vectors:
        return np.full((0,), np.nan, dtype=np.float32)
    return np.concatenate(vectors).astype(np.float32)


def _extract_landmarks_for_video(
    video_path: str | Path,
    face_indices: Sequence[int],
    config: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    path = Path(video_path)
    holistic, backend = _create_holistic_detector(config)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        holistic.close()
        raise ValueError(f"Failed to open video: {path}")
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)

    sequences: list[np.ndarray] = []
    raw_sequences: list[np.ndarray] = []
    presence_sequences: list[np.ndarray] = []
    stats: dict[str, Any] = {
        "frame_count": 0,
        "detected_frames": 0,
        "left_hand_detected": 0,
        "right_hand_detected": 0,
        "pose_detected": 0,
        "face_detected": 0,
    }
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stats["frame_count"] += 1
            timestamp_ms = int((stats["frame_count"] - 1) * 1000 / max(video_fps, 1.0))
            results = _detect_frame(holistic, backend, frame, timestamp_ms)
            model_features, raw_features, presence = _frame_features(
                results,
                face_indices,
                config.get("feature_selection", {}),
                config.get("normalization_method", "wrist_relative"),
            )
            sequences.append(model_features)
            raw_sequences.append(raw_features)
            presence_sequences.append(presence)
            stats["detected_frames"] += int(bool(np.any(presence)))
            active_names = [name for name in ("left_hand", "right_hand", "pose", "face") if config.get("feature_selection", {}).get(name, False)]
            for name, detected in zip(active_names, presence):
                stats[f"{name}_detected"] += int(detected)
    finally:
        capture.release()
        holistic.close()
    return sequences, raw_sequences, presence_sequences, stats


def _sequence_windows(sequence: np.ndarray, seq_len: int, stride: int, min_frames: int, allow_padding: bool) -> list[np.ndarray]:
    if sequence.shape[0] < min_frames:
        return []
    if sequence.shape[0] <= seq_len:
        candidate = sequence.copy()
        if allow_padding and candidate.shape[0] < seq_len:
            pad = np.zeros((seq_len - candidate.shape[0], candidate.shape[1]), dtype=np.float32)
            candidate = np.concatenate([candidate, pad], axis=0)
        return [candidate[:seq_len]]
    windows: list[np.ndarray] = []
    for start in range(0, max(sequence.shape[0] - seq_len + 1, 1), max(stride, 1)):
        clip = sequence[start:start + seq_len]
        if clip.shape[0] < seq_len:
            if allow_padding:
                pad = np.zeros((seq_len - clip.shape[0], clip.shape[1]), dtype=np.float32)
                clip = np.concatenate([clip, pad], axis=0)
            else:
                continue
        windows.append(clip[:seq_len])
    if not windows:
        windows = [sequence[:seq_len]]
    return windows


def _window_starts(frame_count: int, sequence_length: int, stride: int, allow_padding: bool) -> list[int]:
    if frame_count <= sequence_length:
        return [0] if allow_padding else ([0] if frame_count == sequence_length else [])
    return list(range(0, frame_count - sequence_length + 1, stride))


def process_video(
    video_path: str | Path,
    signer_id: str,
    sign: str,
    config: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    path = Path(video_path)
    if not path.exists():
        raise ValueError(f"Video file does not exist: {path}")
    fps, frame_count, duration, width, height = _read_video_metadata(path)
    if frame_count <= 0:
        raise ValueError(f"Video has no usable frames: {path}")
    if fps <= 0:
        fps = 30.0
    recording_date = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    if session_id is None:
        session_id = "unknown"
        try:
            rel = path.relative_to(Path(config.get("raw_dataset_path", PROJECT_ROOT / "data" / "msl_dataset" / "raw")))
            if len(rel.parts) >= 4:
                session_id = str(rel.parts[2])
        except ValueError:
            pass
    video_id = path.stem
    source_sha256 = _sha256(path)
    processed_dir = Path(config.get("processed_dataset_path", PROJECT_ROOT / "data" / "msl_dataset" / "processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    sequence_length = int(config.get("sequence_length", 30))
    stride = int(config.get("stride", 15))
    min_frames = int(config.get("min_frames", 12))
    allow_padding = bool(config.get("allow_padding", True))
    face_indices = config.get("face_landmark_indices", DEFAULT_FACE_INDICES)
    if sequence_length < 1 or stride < 1 or min_frames < 1:
        raise ValueError("sequence_length, stride, and min_frames must all be positive.")
    try:
        features, raw_features, presence, stats = _extract_landmarks_for_video(path, face_indices, config)
    except Exception as exc:  # pragma: no cover - heavy operation exercised by integration tests
        raise ValueError(f"Holistic extraction failed for {path}: {exc}") from exc
    if _sha256(path) != source_sha256:
        raise RuntimeError(f"Raw video changed during preprocessing; refusing to publish sequences: {path}")
    frame_count_actual = len(features)
    if frame_count_actual < min_frames:
        raise ValueError(f"Video has only {frame_count_actual} decoded frames (< {min_frames}) and is rejected.")
    sequence_matrix = np.asarray(features, dtype=np.float32)
    raw_matrix = np.asarray(raw_features, dtype=np.float32)
    presence_matrix = np.asarray(presence, dtype=np.float32)
    starts = _window_starts(frame_count_actual, sequence_length, stride, allow_padding)
    if not starts:
        raise ValueError(f"No valid sequence windows could be generated for {path}.")
    output_records: list[dict[str, Any]] = []
    for index, start in enumerate(starts, start=1):
        end = min(start + sequence_length, frame_count_actual)
        window = sequence_matrix[start:end]
        raw_window = raw_matrix[start:end]
        presence_window = presence_matrix[start:end]
        if window.shape[0] < sequence_length:
            pad_count = sequence_length - window.shape[0]
            valid_frame_mask = np.concatenate(
                [np.ones(window.shape[0], dtype=np.float32), np.zeros(pad_count, dtype=np.float32)]
            )
            window = np.pad(window, ((0, pad_count), (0, 0)), mode="constant")
            raw_window = np.pad(raw_window, ((0, pad_count), (0, 0)), mode="constant", constant_values=np.nan)
            presence_window = np.pad(presence_window, ((0, pad_count), (0, 0)), mode="constant")
        else:
            valid_frame_mask = np.ones(window.shape[0], dtype=np.float32)
        safe_session = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id)
        safe_video_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id)
        seq_id = f"{signer_id}_{sign}_{safe_session}_{safe_video_id}_{source_sha256[:8]}_seq{index:03d}"
        sequence_path = processed_dir / f"{seq_id}.npz"
        np.savez_compressed(
            sequence_path,
            sequence=window.astype(np.float32),
            raw_sequence=raw_window.astype(np.float32),
            landmark_presence=presence_window.astype(np.float32),
            valid_frame_mask=valid_frame_mask,
            feature_group_names=np.asarray(
                [name for name in ("left_hand", "right_hand", "pose", "face") if config.get("feature_selection", {}).get(name, False)],
                dtype="<U16",
            ),
            label=np.asarray(sign, dtype='<U16'),
            signer_id=np.asarray(signer_id, dtype='<U32'),
            video_id=np.asarray(video_id, dtype='<U64'),
            session_id=np.asarray(session_id, dtype='<U64'),
        )
        output_records.append({
            "sequence_id": seq_id,
            "sign": sign,
            "signer_id": signer_id,
            "video_id": video_id,
            "session_id": session_id,
            "source_sha256": source_sha256,
            "file_path": str(path.resolve()),
            "sequence_path": str(sequence_path),
            "frame_count": int(window.shape[0]),
            "feature_count": int(window.shape[1]),
            "processing_status": "PROCESSED",
        })
    return {
        "video_id": video_id,
        "sign": sign,
        "signer_id": signer_id,
        "session_id": session_id,
        "source_sha256": source_sha256,
        "recording_date": recording_date,
        "path": str(path),
        "sequence_windows": output_records,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
        "usable_frames": frame_count_actual,
        "detected_frames": stats["detected_frames"],
        "missing_landmark_rates": {
            name: 1.0 - (stats[f"{name}_detected"] / max(frame_count_actual, 1))
            for name in ("left_hand", "right_hand", "pose", "face")
        },
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def prepare_msl_dataset(
    raw_dataset_path: str | Path | None = None,
    config_path: str | Path | None = None,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(raw_dataset_path) if raw_dataset_path is not None else Path(config.get("raw_dataset_path", PROJECT_ROOT / "data" / "msl_dataset" / "raw"))
    if not root.is_dir():
        raise FileNotFoundError(f"MSL raw dataset directory does not exist: {root}")
    config["raw_dataset_path"] = str(root)
    manifests_dir = Path(config.get("manifests_path", PROJECT_ROOT / "data" / "msl_dataset" / "manifests"))
    processed_dir = Path(config.get("processed_dataset_path", PROJECT_ROOT / "data" / "msl_dataset" / "processed"))
    review_dir = Path(config.get("review_path", PROJECT_ROOT / "data" / "msl_dataset" / "review"))
    discovered = discover_videos(root, config.get("allowed_video_extensions"))
    signers = load_signers(manifests_dir)
    sessions = load_sessions(manifests_dir)
    vocabulary_path = manifests_dir / "sign_vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8")) if vocabulary_path.is_file() else {}
    registrations = load_recordings(manifests_dir)
    registration_by_path = {
        str(Path(str(row.get("file_path", ""))).resolve()): row
        for row in registrations if row.get("file_path")
    }
    previous_videos: list[dict[str, Any]] = []
    previous_sequences: list[dict[str, Any]] = []
    if (manifests_dir / "videos.csv").exists():
        with (manifests_dir / "videos.csv").open("r", newline="", encoding="utf-8") as source:
            previous_videos = list(csv.DictReader(source))
    if (manifests_dir / "sequences.csv").exists():
        with (manifests_dir / "sequences.csv").open("r", newline="", encoding="utf-8") as source:
            previous_sequences = list(csv.DictReader(source))
    previous_processed = {
        (str(Path(row.get("file_path", "")).resolve()), str(row.get("source_sha256", ""))): row
        for row in previous_videos if row.get("processing_status") == "PROCESSED"
    }
    valid_videos: list[dict[str, Any]] = []
    invalid_videos: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    process_entries: list[tuple[dict[str, Any], Path, str, dict[str, Any], str]] = []
    reused_paths: set[str] = set()
    plan: list[dict[str, str]] = []
    for entry in discovered:
        if not entry.get("valid"):
            invalid_videos.append({
                "file_path": entry["path"],
                "signer_id": "",
                "sign": "",
                "processing_status": "REJECTED",
                "quality_status": "INVALID_LAYOUT",
                "notes": entry["error"],
            })
            plan.append({"file_path": entry["path"], "action": "skip", "reason": "INVALID_LAYOUT"})
            continue
        path = Path(entry["path"])
        relative = path.relative_to(root)
        if len(relative.parts) != 4:
            invalid_videos.append({
                "file_path": str(path), "signer_id": entry["signer_id"], "sign": entry["sign"],
                "video_id": entry["video_id"], "processing_status": "SKIPPED",
                "quality_status": "LEGACY_LAYOUT",
                "notes": "Legacy layout preserved; expected <signer>/<sign>/<session>/<video>.",
            })
            plan.append({"file_path": str(path), "action": "skip", "reason": "LEGACY_LAYOUT"})
            continue
        signer_id, sign, session_id = entry["signer_id"], entry["sign"], relative.parts[2]
        issue = ""
        if signer_id not in signers or signers[signer_id].get("status") != "ACTIVE" or signers[signer_id].get("consent_status") != "GRANTED":
            issue = "CONSENT_BLOCKED"
        elif f"{signer_id}/{session_id}" not in sessions or sessions[f"{signer_id}/{session_id}"].get("status") != "ACTIVE":
            issue = "SESSION_UNREGISTERED"
        resolved = str(path.resolve())
        current_hash = _sha256(path)
        registration = registration_by_path.get(resolved)
        if not issue and not registration:
            issue = "UNREGISTERED_RECORDING"
        elif not issue and registration.get("sha256") != current_hash:
            issue = "RAW_HASH_CHANGED"
        elif not issue and registration.get("validation_status") != "VALID":
            issue = "VALIDATION_FAILED"
        thresholds = config.get("quality_thresholds", {})
        if not issue and int(registration.get("frame_count", 0) or 0) < int(thresholds.get("min_frames", 12)):
            issue = "TOO_FEW_FRAMES"
        elif not issue and int(registration.get("frame_count", 0) or 0) > int(thresholds.get("max_frames", 300)):
            issue = "FRAME_LIMIT_EXCEEDED"
        elif not issue and not (
            float(thresholds.get("min_fps", 10)) <= _safe_float(registration.get("fps"))
            <= float(thresholds.get("max_fps", 60))
        ):
            issue = "FPS_OUT_OF_RANGE"
        elif not issue and (
            int(registration.get("width", 0) or 0) < int(thresholds.get("min_width", 320))
            or int(registration.get("height", 0) or 0) < int(thresholds.get("min_height", 240))
        ):
            issue = "LOW_RESOLUTION"
        if issue:
            invalid_videos.append({
                "file_path": str(path), "signer_id": signer_id, "sign": sign, "video_id": entry["video_id"],
                "session_id": session_id, "source_sha256": current_hash, "processing_status": "SKIPPED",
                "quality_status": issue,
                "notes": {
                    "CONSENT_BLOCKED": f"Signer {signer_id} is not registered with active, granted consent.",
                    "SESSION_UNREGISTERED": f"Session {session_id} is not registered and active for {signer_id}.",
                    "UNREGISTERED_RECORDING": "Recording is not present in the registration manifest.",
                    "RAW_HASH_CHANGED": "Recording bytes changed after registration.",
                    "VALIDATION_FAILED": f"Recording validation status is {(registration or {}).get('validation_status', 'UNKNOWN')}.",
                    "TOO_FEW_FRAMES": "Recording is below the configured minimum frame count.",
                    "FRAME_LIMIT_EXCEEDED": "Recording exceeds the configured maximum frame count.",
                    "FPS_OUT_OF_RANGE": "Recording FPS is outside the configured quality range.",
                    "LOW_RESOLUTION": "Recording dimensions are below the configured minimum.",
                }[issue],
            })
            plan.append({"file_path": str(path), "action": "skip", "reason": issue})
            continue
        cached = previous_processed.get((resolved, current_hash))
        if cached and not force:
            cached_sequences = [
                row for row in previous_sequences
                if str(Path(row.get("file_path", "")).resolve()) == resolved
                and row.get("source_sha256") == current_hash
            ]
            if cached_sequences and all(Path(row.get("sequence_path", "")).is_file() for row in cached_sequences):
                valid_videos.append(cached)
                sequence_rows.extend(cached_sequences)
                reused_paths.add(resolved)
                plan.append({"file_path": str(path), "action": "reuse", "reason": "ALREADY_PROCESSED"})
                continue
        process_entries.append((entry, path, session_id, registration, current_hash))
        plan.append({"file_path": str(path), "action": "process", "reason": "ELIGIBLE"})

    imported_manifest_path = manifests_dir / "existing_recordings.csv"
    imported_rows: list[dict[str, str]] = []
    if imported_manifest_path.is_file():
        with imported_manifest_path.open("r", newline="", encoding="utf-8") as source:
            imported_rows = list(csv.DictReader(source))
    queued_paths = {
        str(path.resolve()) for _, path, _, _, _ in process_entries
    }
    for imported in imported_rows:
        path = Path(imported.get("original_path", ""))
        if path.is_file():
            resolved = str(path.resolve())
            current_hash = _sha256(path)
            cached = previous_processed.get((resolved, current_hash))
            if (
                cached
                and imported.get("processing_status") == "PROCESSED"
                and resolved not in queued_paths
                and resolved not in registration_by_path
            ):
                cached_sequences = [
                    row for row in previous_sequences
                    if str(Path(row.get("file_path", "")).resolve()) == resolved
                    and row.get("source_sha256") == current_hash
                ]
                if cached_sequences and all(
                    Path(row.get("sequence_path", "")).is_file() for row in cached_sequences
                ):
                    valid_videos.append(cached)
                    sequence_rows.extend(cached_sequences)
                    reused_paths.add(resolved)
                    queued_paths.add(resolved)
                    plan.append({
                        "file_path": str(path),
                        "action": "reuse",
                        "reason": "IMPORTED_ALREADY_PROCESSED",
                    })
                    continue
        if (
            imported.get("validation_status") != "READY_FOR_PROCESSING"
            or imported.get("metadata_confirmation_status") != "CONFIRMED"
            or imported.get("review_status") != "APPROVED"
            or imported.get("signer_identity_status") != "CONFIRMED"
            or imported.get("language") != "MSL"
            or imported.get("language_validation_state") != "VALIDATED"
            or imported.get("label_validation_state") != "VALIDATED"
            or imported.get("source_hash_changed") == "true"
            or imported.get("signer_review_status") != "CONFIRMED"
            or imported.get("session_review_status") != "CONFIRMED"
            or imported.get("label_review_status") != "CONFIRMED"
            or imported.get("language_review_status") != "MSL"
            or imported.get("recording_quality_status") != "APPROVED"
            or imported.get("human_review_status") != "APPROVED"
            or imported.get("technical_qc_status") not in {"READY_FOR_REVIEW", "READY_FOR_PROCESSING"}
            or imported.get("finite_model_input") != "true"
        ):
            continue
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved in queued_paths or resolved in registration_by_path:
            continue
        current_hash = _sha256(path)
        if current_hash != imported.get("sha256"):
            continue
        signer_id = imported.get("signer_id", "")
        sign = imported.get("sign", "")
        session_id = imported.get("session_id", "")
        signer = signers.get(signer_id, {})
        session = sessions.get(f"{signer_id}/{session_id}", {})
        vocabulary_entry = vocabulary.get(sign, {})
        if (
            not signer
            or signer.get("status") != "ACTIVE"
            or signer.get("consent_status") != "GRANTED"
            or not session
            or session.get("status") != "ACTIVE"
            or vocabulary_entry.get("language") != "MSL"
            or vocabulary_entry.get("status") != "ACTIVE"
            or vocabulary_entry.get("human_validation") != "VALIDATED"
        ):
            continue
        entry = {
            "path": str(path),
            "valid": True,
            "signer_id": signer_id,
            "sign": sign,
            "video_id": path.stem,
        }
        registration = {
            "recording_id": imported.get("recording_id", ""),
            "sha256": current_hash,
            "validation_status": "VALID",
        }
        process_entries.append((entry, path, session_id, registration, current_hash))
        queued_paths.add(resolved)
        plan.append({
            "file_path": str(path),
            "action": "process",
            "reason": "HUMAN_CONFIRMED_IMPORT",
        })
    if dry_run:
        return {
            "dry_run": True,
            "plan": plan,
            "would_process": len(process_entries),
            "would_reuse": len(reused_paths),
            "would_skip": sum(item["action"] == "skip" for item in plan),
            "video_rows": [],
            "sequence_rows": [],
        }
    manifests_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    for entry, path, session_id, registration, current_hash in process_entries:
        try:
            record = VideoRecord(
                signer_id=entry["signer_id"],
                sign=entry["sign"],
                video_path=path,
                video_id=entry["video_id"],
                session_id=session_id,
                recording_date=datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"),
                fps=0.0,
                frame_count=0,
                duration=0.0,
                width=0,
                height=0,
                file_size=path.stat().st_size,
                status="PREPARED",
                quality_status="UNKNOWN",
            )
            fps, frame_count, duration, width, height = _read_video_metadata(path)
            record.fps = fps
            record.frame_count = frame_count
            record.duration = duration
            record.width = width
            record.height = height
            record.status = "PROCESSING"
            result = process_video(
                path, record.signer_id, record.sign, config=config, session_id=session_id
            )
            record.status = "PROCESSED"
            record.quality_status = "VALID"
            record.missing_landmark_rates = result["missing_landmark_rates"]
            video_row = record.as_manifest_row()
            video_row.update({
                "session_id": session_id,
                "source_sha256": current_hash,
                "recording_id": registration.get("recording_id", ""),
            })
            valid_videos.append(video_row)
            sequence_rows.extend(result["sequence_windows"])
        except Exception as exc:  # invalid or rejected video
            invalid_videos.append({
                "file_path": str(path),
                "signer_id": entry.get("signer_id", ""),
                "sign": entry.get("sign", ""),
                "video_id": entry.get("video_id", ""),
                "session_id": session_id,
                "processing_status": "REJECTED",
                "quality_status": "REVIEW_REQUIRED",
                "notes": str(exc),
            })
            append_failure(registration.get("recording_id", ""), "preprocessing", exc, manifests_dir)
    if imported_rows:
        processed_paths = {
            str(Path(row.get("file_path", "")).resolve())
            for row in valid_videos
        }
        failed_by_path = {
            str(Path(row.get("file_path", "")).resolve()): row.get("notes", "")
            for row in invalid_videos
        }
        changed_import_manifest = False
        for imported in imported_rows:
            resolved = str(Path(imported.get("original_path", "")).resolve())
            if resolved in processed_paths:
                imported["processing_status"] = "PROCESSED"
                imported["already_processed"] = "true"
                imported["processing_notes"] = ""
                changed_import_manifest = True
            elif resolved in failed_by_path:
                imported["processing_status"] = "FAILED_PREPROCESSING"
                imported["processing_notes"] = failed_by_path[resolved]
                changed_import_manifest = True
        if changed_import_manifest:
            _write_csv(
                imported_manifest_path,
                list(imported_rows[0].keys()),
                imported_rows,
            )
    video_rows = [
        {
            "signer_id": row.get("signer_id", ""),
            "sign": row.get("sign", ""),
            "video_id": row.get("video_id", ""),
            "session_id": row.get("session_id", "unknown"),
            "recording_date": row.get("recording_date", "unknown"),
            "file_path": row.get("file_path", ""),
            "source_sha256": row.get("source_sha256", ""),
            "recording_id": row.get("recording_id", ""),
            "fps": row.get("fps", 0),
            "frame_count": row.get("frame_count", 0),
            "duration": row.get("duration", 0),
            "width": row.get("width", 0),
            "height": row.get("height", 0),
            "processing_status": row.get("processing_status", "REJECTED"),
            "quality_status": row.get("quality_status", "UNKNOWN"),
            "notes": row.get("notes", ""),
            "left_hand_missing_rate": row.get("left_hand_missing_rate", ""),
            "right_hand_missing_rate": row.get("right_hand_missing_rate", ""),
            "pose_missing_rate": row.get("pose_missing_rate", ""),
            "face_missing_rate": row.get("face_missing_rate", ""),
        }
        for row in valid_videos + invalid_videos
    ]
    video_fieldnames = [
        "signer_id", "sign", "video_id", "session_id", "recording_date", "file_path", "source_sha256", "recording_id", "fps",
        "frame_count", "duration", "width",         "height", "processing_status", "quality_status", "notes", "left_hand_missing_rate",
        "right_hand_missing_rate", "pose_missing_rate", "face_missing_rate"
    ]
    _write_csv(manifests_dir / "videos.csv", video_fieldnames, video_rows)
    sequence_fieldnames = [
        "sequence_id", "sign", "signer_id", "video_id", "session_id", "sequence_path",
        "source_sha256", "file_path", "frame_count", "feature_count", "processing_status"
    ]
    current_sources = {
        (str(Path(row.get("file_path", "")).resolve()), str(row.get("source_sha256", "")))
        for row in valid_videos
    }
    preserved_sequences = [
        row for row in previous_sequences
        if (str(Path(row.get("file_path", "")).resolve()), str(row.get("source_sha256", ""))) in current_sources
        and Path(row.get("sequence_path", "")).is_file()
    ]
    known_sequence_ids = {row.get("sequence_id") for row in sequence_rows}
    sequence_rows.extend(row for row in preserved_sequences if row.get("sequence_id") not in known_sequence_ids)
    _write_csv(manifests_dir / "sequences.csv", sequence_fieldnames, sequence_rows)
    _write_pending_reviews(review_dir, sequence_rows)
    summary = build_dataset_summary(manifests_dir)
    (manifests_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status = collection_status(manifests_dir, review_dir)
    update_dataset_version(
        manifests_dir,
        status["number_of_signers"],
        status["total_signs"],
        status["number_of_recordings"],
        len(sequence_rows),
    )
    return {
        "video_rows": video_rows,
        "sequence_rows": sequence_rows,
        "summary": summary,
        "processed_dir": str(processed_dir),
        "manifests_dir": str(manifests_dir),
        "dry_run": False,
    }


def _write_pending_reviews(review_dir: Path, sequence_rows: list[dict[str, Any]]) -> None:
    path = review_dir / "review_manifest.csv"
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    known = {row.get("sequence_id") for row in rows}
    for sequence in sequence_rows:
        if sequence.get("sequence_id") not in known:
            rows.append({
                "sequence_id": sequence.get("sequence_id", ""),
                "status": "PENDING",
                "reviewer": "",
                "note": "",
                "updated_at": "",
            })
    _write_csv(review_dir / "review_manifest.csv", ["sequence_id", "status", "reviewer", "note", "updated_at"], rows)


def build_dataset_summary(manifests_dir: str | Path) -> dict[str, Any]:
    manifests_root = Path(manifests_dir)
    videos_path = manifests_root / "videos.csv"
    sequences_path = manifests_root / "sequences.csv"
    video_rows = []
    if videos_path.exists():
        with videos_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            video_rows = list(reader)
    sequence_rows = []
    if sequences_path.exists():
        with sequences_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            sequence_rows = list(reader)
    valid_videos = [row for row in video_rows if row.get("processing_status") == "PROCESSED"]
    rejected = [row for row in video_rows if row.get("processing_status") in {"REJECTED", "REVIEW_REQUIRED"}]
    class_counts: dict[str, dict[str, int]] = {}
    signer_counts: dict[str, int] = {}
    sign_to_signer_counts: dict[str, int] = {}
    sequence_counts_by_signer: dict[str, int] = {}
    sequence_counts_by_sign: dict[str, int] = {}
    avg_duration = 0.0
    avg_frame_count = 0.0
    if valid_videos:
        durations = [float(row.get("duration", 0.0) or 0.0) for row in valid_videos]
        frames = [int(row.get("frame_count", 0) or 0) for row in valid_videos]
        avg_duration = round(sum(durations) / len(durations), 3) if durations else 0.0
        avg_frame_count = round(sum(frames) / len(frames), 3) if frames else 0.0
    missing_rates = {
        name: (
            round(
                sum(float(row.get(f"{name}_missing_rate") or 0.0) for row in valid_videos)
                / len(valid_videos),
                6,
            )
            if valid_videos
            else 0.0
        )
        for name in ("left_hand", "right_hand", "pose", "face")
    }
    summary_rows = valid_videos if valid_videos else sequence_rows
    for row in summary_rows:
        sign = str(row.get("sign", "UNKNOWN")).strip() or "UNKNOWN"
        signer = str(row.get("signer_id", "UNKNOWN")).strip() or "UNKNOWN"
        class_counts.setdefault(sign, {})
        class_counts[sign][signer] = class_counts[sign].get(signer, 0) + 1
        signer_counts[signer] = signer_counts.get(signer, 0) + 1
        sign_to_signer_counts[sign] = sign_to_signer_counts.get(sign, 0) + 1
    for row in sequence_rows:
        signer = str(row.get("signer_id", "")).strip()
        sign = str(row.get("sign", "")).strip()
        if signer:
            sequence_counts_by_signer[signer] = sequence_counts_by_signer.get(signer, 0) + 1
        if sign:
            sequence_counts_by_sign[sign] = sequence_counts_by_sign.get(sign, 0) + 1
    feature_dim = 0
    for row in sequence_rows:
        sequence_path = Path(row.get("sequence_path", ""))
        if not sequence_path.exists():
            continue
        try:
            with np.load(sequence_path, allow_pickle=False) as payload:
                sequence = np.asarray(payload["sequence"], dtype=np.float32)
            if sequence.size:
                feature_dim = int(sequence.shape[-1])
                break
        except Exception:
            continue
    warnings: list[str] = []
    if not valid_videos:
        warnings.append("No valid videos were processed.")
    if not sequence_rows:
        warnings.append("No processed sequences were generated.")
    if len(rejected) > 0:
        warnings.append(f"{len(rejected)} videos were rejected or require review.")
    train_count = validation_count = test_count = 0
    split_manifest_path = manifests_root / "split_manifest.json"
    if split_manifest_path.exists():
        try:
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
            train_count = int(split_manifest.get("train_count", 0) or 0)
            validation_count = int(split_manifest.get("validation_count", 0) or 0)
            test_count = int(split_manifest.get("test_count", 0) or 0)
        except Exception:
            pass
    return {
        "number_of_signers": len(sorted({str(row.get("signer_id", "")).strip() for row in summary_rows if row.get("signer_id")})),
        "number_of_signs": len(sorted({str(row.get("sign", "")).strip() for row in summary_rows if row.get("sign")})),
        "videos_per_signer": signer_counts,
        "videos_per_sign": {label: count for label, count in sorted(sign_to_signer_counts.items())},
        "sequences_per_signer": {signer: count for signer, count in sorted(sequence_counts_by_signer.items())},
        "sequences_per_sign": {label: count for label, count in sorted(sequence_counts_by_sign.items())},
        "average_duration_seconds": avg_duration,
        "average_frame_count": avg_frame_count,
        "missing_landmark_rates": missing_rates,
        "feature_dimensions": feature_dim,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "total_videos": len(video_rows),
        "total_valid_videos": len(valid_videos),
        "total_rejected_videos": len(rejected),
        "total_sequences": len(sequence_rows),
        "classes": sorted({str(row["sign"]) for row in valid_videos if row.get("sign")}),
        "samples_per_sign": {label: sum(1 for row in sequence_rows if row.get("sign") == label) for label in sorted({str(row.get("sign", "")) for row in sequence_rows if row.get("sign")})},
        "samples_per_signer": {
            signer: sum(1 for row in sequence_rows if row.get("signer_id") == signer)
            for signer in sorted({str(row.get("signer_id", "")) for row in sequence_rows if row.get("signer_id")})
        },
        "signer_distribution": class_counts,
        "warnings": warnings,
    }


def generate_split_manifest(manifests_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    manifests_root = Path(manifests_dir)
    sequence_path = manifests_root / "sequences.csv"
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence manifest not found: {sequence_path}")
    with sequence_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("No sequences available to split.")
    all_signers = sorted({str(row.get("signer_id", "")).strip() for row in rows if row.get("signer_id")})
    train_signers = [str(value).strip() for value in config.get("train_signers", []) if str(value).strip()]
    validation_signers = [str(value).strip() for value in config.get("validation_signers", []) if str(value).strip()]
    test_signers = [str(value).strip() for value in config.get("test_signers", []) if str(value).strip()]
    if not train_signers and not validation_signers and not test_signers:
        prior_path = manifests_root / "split_manifest.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
        existing_groups = {
            "train": [signer for signer in prior.get("train_signers", []) if signer in all_signers],
            "validation": [signer for signer in prior.get("validation_signers", []) if signer in all_signers],
            "test": [signer for signer in prior.get("test_signers", []) if signer in all_signers],
        }
        if any(existing_groups.values()):
            assigned = {signer for group in existing_groups.values() for signer in group}
            if len(assigned) != sum(len(group) for group in existing_groups.values()):
                raise ValueError("Persisted signer split contains overlapping signer assignments.")
            weights = {"train": 0.7, "validation": 0.1, "test": 0.2}
            for signer in sorted(set(all_signers) - assigned):
                destination = min(
                    existing_groups,
                    key=lambda name: len(existing_groups[name]) / weights[name],
                )
                existing_groups[destination].append(signer)
            train_signers = existing_groups["train"]
            validation_signers = existing_groups["validation"]
            test_signers = existing_groups["test"]
        elif len(all_signers) >= 10:
            train_signers = all_signers[:7]
            validation_signers = all_signers[7:8]
            test_signers = all_signers[8:]
        elif len(all_signers) >= 3:
            train_signers = all_signers[:-2]
            validation_signers = all_signers[-2:-1]
            test_signers = all_signers[-1:]
        elif len(all_signers) == 2:
            train_signers = [all_signers[0]]
            validation_signers = []
            test_signers = [all_signers[1]]
        elif len(all_signers) == 1:
            train_signers = [all_signers[0]]
            validation_signers = []
            test_signers = []
    split_lookup = {signer: 0 for signer in all_signers}
    for split_name, signer_group in (("train", train_signers), ("validation", validation_signers), ("test", test_signers)):
        for signer in signer_group:
            if signer not in split_lookup:
                raise ValueError(f"Signer '{signer}' in {split_name} split is not present in the sequence manifest.")
            split_lookup[signer] += 1
    if any(value > 1 for value in split_lookup.values()):
        offenders = [signer for signer, count in split_lookup.items() if count > 1]
        raise ValueError(f"Signer leakage detected across splits: {offenders}")
    assigned_signers = {signer for signer, count in split_lookup.items() if count == 1}
    unassigned_signers = sorted(set(all_signers) - assigned_signers)
    if unassigned_signers:
        raise ValueError(f"Signers are missing from the configured splits: {unassigned_signers}")
    train_rows = [row for row in rows if row.get("signer_id") in train_signers]
    validation_rows = [row for row in rows if row.get("signer_id") in validation_signers]
    test_rows = [row for row in rows if row.get("signer_id") in test_signers]
    _assert_split_identity_separation(train_rows, validation_rows, test_rows)
    split_manifest = {
        "train_signers": train_signers,
        "validation_signers": validation_signers,
        "test_signers": test_signers,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
        "deterministic": True,
    }
    (manifests_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    for name, subset in {"train.csv": train_rows, "validation.csv": validation_rows, "test.csv": test_rows}.items():
        _write_csv(manifests_root / name, list(rows[0].keys()), subset)
    return split_manifest


def _assert_split_identity_separation(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    identity_fields = ("sequence_id", "recording_id", "file_path", "source_sha256")
    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}
    for field in identity_fields:
        seen: dict[str, str] = {}
        for split_name, rows in splits.items():
            for row in rows:
                identity = str(row.get(field, "")).strip()
                if not identity:
                    continue
                previous_split = seen.get(identity)
                if previous_split is not None and previous_split != split_name:
                    raise ValueError(
                        f"{field} leakage detected between {previous_split} and {split_name}: {identity}"
                    )
                seen[identity] = split_name


def generate_dataset_report(manifests_dir: str | Path, config: dict[str, Any] | None = None) -> str:
    manifests_root = Path(manifests_dir)
    config = config or load_config()
    summary = build_dataset_summary(manifests_root)
    status = collection_status(
        manifests_root,
        config.get("review_path", manifests_root.parent / "review"),
        targets=config.get("collection_targets", {}),
    )
    split_manifest_path = manifests_root / "split_manifest.json"
    if split_manifest_path.exists():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        summary["train_validation_test"] = {
            "train": split_manifest.get("train_count", 0),
            "validation": split_manifest.get("validation_count", 0),
            "test": split_manifest.get("test_count", 0),
        }
    text = [
        "# MSL Dataset Report",
        "",
        f"- signers: {summary.get('number_of_signers', 0)}",
        f"- signs: {summary.get('number_of_signs', 0)}",
        f"- total videos: {summary.get('total_videos', 0)}",
        f"- total valid videos: {summary.get('total_valid_videos', 0)}",
        f"- total rejected videos: {summary.get('total_rejected_videos', 0)}",
        f"- total sequences: {summary.get('total_sequences', 0)}",
        f"- average duration (s): {summary.get('average_duration_seconds', 0.0)}",
        f"- average frame count: {summary.get('average_frame_count', 0.0)}",
        f"- feature dimensions: {summary.get('feature_dimensions', 0)}",
        f"- consented signers: {status.get('consent_granted_signers', 0)}",
        f"- registered sessions: {status.get('number_of_sessions', 0)}",
        f"- failed processing attempts: {status.get('failure_count', 0)}",
        f"- pending or retake reviews: {status.get('pending_review_sequences', 0)}",
        "",
        "## Class summary",
    ]
    for label, count in summary.get("samples_per_sign", {}).items():
        text.append(f"- {label}: {count} samples")
    text.append("")
    text.append("## Signer summary")
    for signer, count in summary.get("videos_per_signer", {}).items():
        text.append(f"- {signer}: {count} videos")
    text.append("")
    text.append("## Collection priorities")
    priorities = status.get("collection_priorities", [])
    if priorities:
        for priority in priorities:
            if priority["action"] == "ADD_NEW_SIGNER":
                text.append(
                    f"- {priority['sign']}: add {priority['gap']} new signer(s) "
                    f"({priority['current_signers']}/{priority['target_signers']} present)."
                )
            else:
                text.append(
                    f"- {priority['sign']} / {priority['signer_id']}: add {priority['gap']} "
                    f"recording(s) ({priority['recordings']}/{priority['target_recordings']} present)."
                )
    else:
        text.append("- No vocabulary entries or outstanding collection targets.")
    text.append("")
    text.append("## Split counts")
    text.append(f"- train: {summary.get('train_count', 0)}")
    text.append(f"- validation: {summary.get('validation_count', 0)}")
    text.append(f"- test: {summary.get('test_count', 0)}")
    if summary.get("warnings"):
        text.append("")
        text.append("## Warnings")
        for warning in summary["warnings"]:
            text.append(f"- {warning}")
    report_path = manifests_root.parent / "MSL_DATASET_REPORT.md"
    report_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return str(report_path)


def validate_dataset(manifests_dir: str | Path) -> dict[str, Any]:
    manifests_root = Path(manifests_dir)
    video_rows = []
    if (manifests_root / "videos.csv").exists():
        with (manifests_root / "videos.csv").open("r", newline="", encoding="utf-8") as csv_file:
            video_rows = list(csv.DictReader(csv_file))
    sequence_rows = []
    if (manifests_root / "sequences.csv").exists():
        with (manifests_root / "sequences.csv").open("r", newline="", encoding="utf-8") as csv_file:
            sequence_rows = list(csv.DictReader(csv_file))
    issues: list[str] = []
    duplicate_paths: set[str] = set()
    duplicate_sequences: set[str] = set()
    seen_videos: set[str] = set()
    seen_sequences: set[str] = set()
    for row in video_rows:
        video_path = row.get("file_path", "")
        if video_path in seen_videos:
            duplicate_paths.add(video_path)
            issues.append(f"Duplicate video path detected: {video_path}")
        seen_videos.add(video_path)
        if not row.get("signer_id") or not row.get("sign"):
            issues.append(f"Video metadata missing signer or sign label: {video_path or row}")
    for row in sequence_rows:
        sequence_id = str(row.get("sequence_id", "")).strip()
        sequence_path = str(row.get("sequence_path", "")).strip()
        if not sequence_id:
            issues.append("Encountered a sequence row missing a sequence_id.")
        if sequence_id in seen_sequences:
            duplicate_sequences.add(sequence_id)
            issues.append(f"Duplicate sequence detected: {sequence_id}")
        seen_sequences.add(sequence_id)
        if sequence_path and not Path(sequence_path).exists():
            issues.append(f"Sequence file missing for manifest row: {sequence_path}")
    if not video_rows:
        issues.append("No video metadata rows were found.")
    split_files = {
        "train": manifests_root / "train.csv",
        "validation": manifests_root / "validation.csv",
        "test": manifests_root / "test.csv",
    }
    signer_sets: dict[str, set[str]] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_path in split_files.items():
        if not split_path.exists():
            continue
        with split_path.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        split_rows[split_name] = rows
        signer_sets[split_name] = {str(row.get("signer_id", "")).strip() for row in rows if row.get("signer_id")}
    overlaps: list[str] = []
    split_names = list(signer_sets)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            shared = sorted(signer_sets[left_name].intersection(signer_sets[right_name]))
            if shared:
                overlaps.extend([f"Signer leakage between {left_name} and {right_name}: {', '.join(shared)}"])
                issues.extend(overlaps[-1:])
    try:
        _assert_split_identity_separation(
            split_rows.get("train", []),
            split_rows.get("validation", []),
            split_rows.get("test", []),
        )
    except ValueError as exc:
        issues.append(str(exc))
        overlaps.append(str(exc))
    if not video_rows and not sequence_rows:
        issues.append("Dataset has no rows to validate.")
    return {
        "issues": issues,
        "duplicate_paths": sorted(duplicate_paths),
        "duplicate_sequences": sorted(duplicate_sequences),
        "signer_leakage": overlaps,
    }


def build_review_html(sequence_path: str | Path) -> str:
    seq_path = Path(sequence_path)
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>MSL Sequence Review</title></head><body>",
        f"<h1>MSL Sequence Review</h1>",
        f"<p>File: {seq_path}</p>",
        "<ul>",
        "<li>Sequence review utility is present for manual QC.</li>",
        "<li>Reviewers can inspect sequence shape, signer, sign, and data quality.</li>",
        "</ul>",
        "</body></html>",
    ]
    return "\n".join(html)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the BridgeSign MSL dataset.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the MSL dataset config JSON file.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Optional override for the raw dataset directory.")
    args = parser.parse_args()
    result = prepare_msl_dataset(raw_dataset_path=args.raw_dir, config_path=args.config)
    print(json.dumps({"videos": len(result["video_rows"]), "sequences": len(result["sequence_rows"])}, indent=2))
