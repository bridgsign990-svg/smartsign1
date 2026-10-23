from __future__ import annotations

from typing import Any

import numpy as np


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


def normalize_hand_landmarks(points: np.ndarray, wrist_index: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    if points.shape[0] == 0:
        return points.reshape(-1)
    ref = points[wrist_index] if wrist_index < points.shape[0] else points[0]
    centered = points - ref
    scale = float(np.linalg.norm(centered[:10])) if centered.size else 1.0
    if scale <= 0:
        scale = 1.0
    return (centered / scale).reshape(-1)


def normalize_pose_landmarks(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    if points.shape[0] == 0:
        return points.reshape(-1)
    shoulders = np.nanmean(points[[11, 12]] if points.shape[0] > 12 else points[:2], axis=0)
    centered = points - shoulders
    scale = float(np.linalg.norm(centered[:15])) if centered.size else 1.0
    if scale <= 0:
        scale = 1.0
    return (centered / scale).reshape(-1)


def normalize_face_landmarks(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    if points.shape[0] == 0:
        return points.reshape(-1)
    nose = points[0]
    centered = points - nose
    scale = float(np.linalg.norm(centered)) if centered.size else 1.0
    if scale <= 0:
        scale = 1.0
    return (centered / scale).reshape(-1)


def normalize_frame(frame_data: dict[str, Any], normalization_method: str = "wrist_relative") -> dict[str, np.ndarray]:
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
                normalized[name] = normalize_hand_landmarks(arr)
            elif name == "pose":
                normalized[name] = normalize_pose_landmarks(arr)
            elif name == "face":
                normalized[name] = normalize_face_landmarks(arr)
            else:
                normalized[name] = arr.reshape(-1)
        else:
            normalized[name] = arr.reshape(-1)
    return normalized


def normalize_temporal_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.size == 0:
        return sequence
    if sequence.ndim == 1:
        return sequence.reshape(1, -1)
    return sequence
