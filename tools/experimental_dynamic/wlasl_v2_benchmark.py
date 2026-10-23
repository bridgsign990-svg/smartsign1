from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from tools.experimental_dynamic import baseline, wlasl_v2


PROJECT_ROOT = wlasl_v2.PROJECT_ROOT
DATA_ROOT = wlasl_v2.DATA_ROOT
MODEL_ROOT = wlasl_v2.MODEL_ROOT
REPORT_ROOT = PROJECT_ROOT / "reports" / "experimental" / "wlasl"
BENCHMARK_ROOT = MODEL_ROOT / "benchmark"
DIAGNOSTIC_JSON = REPORT_ROOT / "wlasl_v2_landmark_diagnostic.json"
DIAGNOSTIC_REPORT = REPORT_ROOT / "wlasl_v2_landmark_diagnostic.md"
COMPARISON_JSON = REPORT_ROOT / "wlasl_v2_model_comparison.json"
COMPARISON_REPORT = REPORT_ROOT / "wlasl_v2_model_comparison.md"
ERROR_REPORT = REPORT_ROOT / "wlasl_v2_error_analysis.md"
REPRESENTATION_VERSION = "representation_v2"
SEED = 42
MAX_EPOCHS = 24
PATIENCE = 6
BATCH_SIZE = 16
HIDDEN_SIZE = 64
LAYER_COUNT = 2
DROPOUT = 0.35
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _video_key(class_index: int | str, video_id: str) -> str:
    return f"{int(class_index)}:{video_id}"


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(max(values)),
    }


def _reconstruct_frame_presence(
    cache_manifest: Path,
    usable_frames: int,
    sequence_length: int = 30,
    stride: int = 15,
) -> tuple[np.ndarray, int]:
    cached = json.loads(cache_manifest.read_text(encoding="utf-8"))
    frames = np.zeros((usable_frames, 4), dtype=np.float32)
    assigned = np.zeros(usable_frames, dtype=bool)
    for window_index, window in enumerate(cached["windows"]):
        sequence_path = Path(window["sequence_path"])
        with np.load(sequence_path, allow_pickle=False) as stored:
            presence = np.asarray(stored["landmark_presence"], dtype=np.float32)
            valid = np.asarray(stored["valid_frame_mask"], dtype=np.float32) > 0
        start = window_index * stride
        for offset in np.flatnonzero(valid):
            frame_index = start + int(offset)
            if frame_index >= usable_frames:
                break
            value = presence[offset]
            if assigned[frame_index] and not np.array_equal(frames[frame_index], value):
                raise ValueError(
                    f"Overlapping WLASL windows disagree on landmark presence: {sequence_path}"
                )
            frames[frame_index] = value
            assigned[frame_index] = True
    if not assigned.any():
        raise ValueError(f"Cached windows contain no valid frames: {cache_manifest}")
    return frames[assigned], int(assigned.sum())


def run_landmark_diagnostics() -> dict[str, Any]:
    manifest = json.loads(wlasl_v2.MANIFEST_PATH.read_text(encoding="utf-8"))
    qc = json.loads(wlasl_v2.QC_PATH.read_text(encoding="utf-8"))
    downloads = wlasl_v2._unique_downloaded_records(manifest["records"])
    sequence_rows = _read_rows(wlasl_v2.SEQUENCE_MANIFEST_PATH)
    selected_class_ids = {
        int(row["class_index"])
        for row in sequence_rows
    }
    selected_video_keys = {
        _video_key(row["class_index"], row["video_id"])
        for row in sequence_rows
    }
    quality_records = qc.get("landmark_quality_by_video", {})
    video_rows: dict[str, dict[str, Any]] = {}
    class_downloads: dict[str, set[str]] = defaultdict(set)
    signer_downloads: dict[str, set[str]] = defaultdict(set)
    for record in downloads:
        key = _video_key(record["class_index"], record["video_id"])
        class_downloads[record["gloss"]].add(key)
        signer = str(record.get("signer_id") or "")
        if signer:
            signer_downloads[signer].add(key)
        video_path = Path(record["local_filename"])
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Previously verified WLASL video can no longer be opened: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        video_rows[key] = {
            "video_id": str(record["video_id"]),
            "class_index": int(record["class_index"]),
            "label": str(record["gloss"]),
            "signer_id": signer,
            "source_split": str(record.get("official_split") or ""),
            "path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps if fps > 0 else None,
            "width": width,
            "height": height,
            "preprocessed": key in selected_video_keys,
        }

    per_class_frames: dict[str, list[float]] = defaultdict(list)
    per_signer_frames: dict[str, list[float]] = defaultdict(list)
    per_class_rates: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_signer_rates: dict[str, list[dict[str, float]]] = defaultdict(list)
    presence_counts: Counter[str] = Counter()
    total_presence_frames = 0
    for key in sorted(selected_video_keys):
        record = video_rows[key]
        qc_key = key
        quality = quality_records.get(qc_key)
        if quality is None:
            raise ValueError(f"Landmark QC is missing the selected video {key}.")
        cache_path = (
            wlasl_v2.PROCESSED_ROOT
            / f"class_{record['class_index']:04d}"
            / record["video_id"]
            / "windows.json"
        )
        presence, diagnostic_frames = _reconstruct_frame_presence(
            cache_path, int(quality["usable_frames"])
        )
        counts = {
            "left_hand": float(presence[:, 0].mean()),
            "right_hand": float(presence[:, 1].mean()),
            "pose": float(presence[:, 2].mean()),
            "face": float(presence[:, 3].mean()),
        }
        joint = {
            "both_hands": float(np.logical_and(presence[:, 0] > 0, presence[:, 1] > 0).mean()),
            "right_hand_only": float(np.logical_and(presence[:, 1] > 0, presence[:, 0] == 0).mean()),
            "left_hand_only": float(np.logical_and(presence[:, 0] > 0, presence[:, 1] == 0).mean()),
            "neither_hand": float(np.logical_and(presence[:, 0] == 0, presence[:, 1] == 0).mean()),
        }
        record.update({
            "fps": float(quality["fps"]),
            "frame_count": int(quality["frame_count"]),
            "usable_frames": int(quality["usable_frames"]),
            "landmark_diagnostic_frames": diagnostic_frames,
            "landmark_diagnostic_frame_coverage": (
                diagnostic_frames / int(quality["usable_frames"])
            ),
            "landmark_uncovered_frames": (
                int(quality["usable_frames"]) - diagnostic_frames
            ),
            "detected_frames": int(quality["detected_frames"]),
            "missing_landmark_rates": quality["missing_landmark_rates"],
            "hand_frame_rates": counts,
            "joint_hand_frame_rates": joint,
        })
        per_class_frames[record["label"]].append(float(quality["usable_frames"]))
        per_signer_frames[record["signer_id"]].append(float(quality["usable_frames"]))
        per_class_rates[record["label"]].append({**counts, **joint})
        per_signer_rates[record["signer_id"]].append({**counts, **joint})
        for name, count in (
            ("both_hands", np.count_nonzero((presence[:, 0] > 0) & (presence[:, 1] > 0))),
            ("right_hand_only", np.count_nonzero((presence[:, 1] > 0) & (presence[:, 0] == 0))),
            ("left_hand_only", np.count_nonzero((presence[:, 0] > 0) & (presence[:, 1] == 0))),
            ("neither_hand", np.count_nonzero((presence[:, 0] == 0) & (presence[:, 1] == 0))),
        ):
            presence_counts[name] += int(count)
        total_presence_frames += len(presence)

    def mean_by_group(values: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
        return {
            name: {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in (
                    "left_hand", "right_hand", "pose", "face", "both_hands",
                    "right_hand_only", "left_hand_only", "neither_hand",
                )
            }
            for name, rows in sorted(values.items())
        }

    all_videos = list(video_rows.values())
    selected_videos = [video_rows[key] for key in selected_video_keys]
    config = wlasl_v2._feature_config()
    diagnostic = {
        "dataset": "Official WLASL v0.3, ASL",
        "scope": {
            "unique_downloaded_videos": len(downloads),
            "official_classes_in_manifest": len(manifest["glosses"]),
            "selected_preprocessed_classes": sorted({row["label"] for row in selected_videos}),
            "selected_preprocessed_videos": len(selected_videos),
            "unprocessed_downloaded_videos": len(downloads) - len(selected_videos),
            "unprocessed_note": "Raw-video/FPS metadata is reported for every successful download; per-frame Holistic landmark rates are available only for the 84 videos selected and already preprocessed.",
        },
        "video_counts_by_class": {
            name: len(values) for name, values in sorted(class_downloads.items())
        },
        "video_counts_by_signer": {
            name: len(values) for name, values in sorted(signer_downloads.items())
        },
        "fps_distribution_all_downloaded": _stats([
            row["fps"] for row in all_videos if row["fps"] > 0
        ]),
        "frames_per_video_all_downloaded": _stats([
            row["frame_count"] for row in all_videos if row["frame_count"] > 0
        ]),
        "frames_per_video_preprocessed": _stats([
            row["usable_frames"] for row in selected_videos
        ]),
        "landmark_diagnostic_frame_coverage": _stats([
            row["landmark_diagnostic_frame_coverage"] for row in selected_videos
        ]),
        "sequence_lengths": _stats([
            float(np.load(row["sequence_path"], allow_pickle=False)["valid_frame_mask"].sum())
            for row in sequence_rows
        ]),
        "hand_presence_frame_counts": dict(presence_counts),
        "joint_hand_frame_rates": {
            name: count / total_presence_frames if total_presence_frames else 0.0
            for name, count in presence_counts.items()
        },
        "mean_hand_pose_face_detection_rates": {
            name: float(np.mean([
                row["missing_landmark_rates"][name]
                for row in selected_videos
            ]))
            for name in ("left_hand", "right_hand", "pose", "face")
        },
        "per_class_video_frames_and_hand_rates": {
            label: {
                "downloaded_videos": len(class_downloads[label]),
                "preprocessed_videos": sum(row["label"] == label for row in selected_videos),
                "frames": _stats(per_class_frames[label]),
                "mean_presence_rates": mean_by_group({label: per_class_rates[label]})[label]
                if label in per_class_rates else {},
            }
            for label in sorted(class_downloads)
        },
        "per_signer_video_frames_and_hand_rates": {
            signer: {
                "downloaded_videos": len(video_keys),
                "preprocessed_videos": sum(row["signer_id"] == signer for row in selected_videos),
                "frames": _stats(per_signer_frames[signer]),
                "mean_presence_rates": mean_by_group({signer: per_signer_rates[signer]})[signer]
                if signer in per_signer_rates else {},
            }
            for signer, video_keys in sorted(signer_downloads.items())
        },
        "videos": all_videos,
        "preprocessing_observations": {
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "missing_hand_flags_are_set_before_normalization": True,
            "model_coordinates_zero_fill_missing_landmarks": True,
            "original_representation_has_one_presence_flag_per_feature_group": 4,
            "interpretation": [
                "Pose and face detections remain present for nearly all frames, which argues against whole-frame decode or total Holistic failure as the dominant cause.",
                "The much higher left-hand missingness than right-hand missingness points to side-specific visibility, framing, signer behavior, or handedness/tracking differences, but metadata alone cannot distinguish them.",
                "Normalization does not cause the recorded hand-missing rate: presence is determined from raw MediaPipe landmark coordinates before normalization.",
                "A 0.5 MediaPipe detection/tracking confidence threshold is configured. No evidence in the cached records isolates threshold rejection from genuine absence or occlusion.",
                "Visual review is required to separate out-of-frame hands and one-handed signing from MediaPipe tracking failures; no causal conclusion is inferred from missingness statistics alone.",
            ],
        },
        "feature_configuration": {
            "version": "representation_v1",
            "sequence_length": config["sequence_length"],
            "fps": config["target_fps"],
            "feature_dimension": 562,
            "normalization": config["normalization_method"],
            "groups": ["left_hand", "right_hand", "pose", "face"],
        },
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_JSON.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    _write_diagnostic_report(diagnostic)
    return diagnostic


def _write_diagnostic_report(diagnostic: dict[str, Any]) -> None:
    rates = diagnostic["mean_hand_pose_face_detection_rates"]
    lines = [
        "# WLASL V2 Landmark and Video Diagnostic",
        "",
        "> WLASL is ASL; this engineering diagnostic does not establish MSL performance.",
        "",
        "## Scope",
        "",
        f"- Unique decodable downloaded videos: {diagnostic['scope']['unique_downloaded_videos']} across {diagnostic['scope']['official_classes_in_manifest']} official WLASL100 glosses.",
        f"- Detailed per-frame landmark diagnosis: {diagnostic['scope']['selected_preprocessed_videos']} already-preprocessed videos across ten glosses. Other downloaded videos were not discarded; their count, signer, FPS, dimensions and decoded frame-count metadata are included, but they were not run through Holistic.",
        f"- Unprocessed downloaded videos: {diagnostic['scope']['unprocessed_downloaded_videos']}. {diagnostic['scope']['unprocessed_note']}",
        "",
        "## Dataset distribution",
        "",
        f"- FPS distribution for all acquired files: `{json.dumps(diagnostic['fps_distribution_all_downloaded'], sort_keys=True)}`.",
        f"- Decoded frame-count distribution for all acquired files: `{json.dumps(diagnostic['frames_per_video_all_downloaded'], sort_keys=True)}`.",
        f"- Usable frame-count distribution for Holistic-processed videos: `{json.dumps(diagnostic['frames_per_video_preprocessed'], sort_keys=True)}`.",
        f"- Cached-window coverage of Holistic-processed videos for frame-level landmark rates: `{json.dumps(diagnostic['landmark_diagnostic_frame_coverage'], sort_keys=True)}`. Landmark rates use only frames represented by cached windows; uncovered trailing frames are explicitly counted per video in the JSON and are not treated as missing-landmark frames.",
        f"- Temporal sequence valid-length distribution: `{json.dumps(diagnostic['sequence_lengths'], sort_keys=True)}`.",
        "",
        "## Video counts per class",
        "",
        "| Gloss | Downloaded videos | Holistic processed | Mean frames | Both hands | Right only | Left only | Neither |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in diagnostic["per_class_video_frames_and_hand_rates"].items():
        presence = item.get("mean_presence_rates", {})
        lines.append(
            f"| {label} | {item['downloaded_videos']} | {item['preprocessed_videos']} | "
            f"{item['frames']['median'] if item['frames']['median'] is not None else 'n/a'} | "
            f"{presence.get('both_hands', 0):.3f} | {presence.get('right_hand_only', 0):.3f} | "
            f"{presence.get('left_hand_only', 0):.3f} | {presence.get('neither_hand', 0):.3f} |"
        )
    lines.extend([
        "",
        "## Signer counts and landmark rates",
        "",
        f"- Downloaded video counts per signer: `{json.dumps(diagnostic['video_counts_by_signer'], sort_keys=True)}`.",
        f"- Mean per-frame missing rates over the Holistic-processed videos: `{json.dumps({name: 1.0 - value for name, value in rates.items()}, sort_keys=True)}`.",
        f"- Across {sum(diagnostic['hand_presence_frame_counts'].values())} reconstructed per-video frames, joint hand categories are: `{json.dumps(diagnostic['joint_hand_frame_rates'], sort_keys=True)}`.",
        "- Per-signer and per-class presence percentages are available in the companion JSON because the full tables are large.",
        "",
        "## Interpretation and limits of attribution",
        "",
        "- The existing collector calls MediaPipe Holistic with `min_detection_confidence=0.5` and `min_tracking_confidence=0.5`.",
        "- Landmark presence is measured from raw coordinates before normalization; wrist-relative normalization therefore cannot explain missing-landmark counts.",
        "- Existing representation v1 converts missing coordinates to zero and appends only four group-level presence flags (left hand, right hand, pose, face). It does not carry per-landmark missing masks.",
        "- Pose/face persistence alongside asymmetric hand absence argues against complete decoder/Holistic failure, but does not distinguish occlusion/framing, one-handed signing, hand tracking failure, or signer handedness.",
        "- Visual review is required to attribute the missing-hand cause. These data alone do not prove why a given hand is absent.",
        "",
        "## Existing feature representation",
        "",
        "- representation_v1: 558 normalized landmark coordinate/visibility values + 4 group-presence flags = 562 features/frame; 30 frames at 25 fps.",
        "- This diagnostic does not remove, rewrite, or discard any downloaded video.",
        "",
    ])
    DIAGNOSTIC_REPORT.write_text("\n".join(lines), encoding="utf-8")


def representation_v2(
    sequence: np.ndarray,
    raw_sequence: np.ndarray,
    group_presence: np.ndarray,
    valid_frame_mask: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    raw_sequence = np.asarray(raw_sequence, dtype=np.float32)
    group_presence = np.asarray(group_presence, dtype=np.float32)
    valid_frame_mask = np.asarray(valid_frame_mask, dtype=np.float32)
    if (
        sequence.ndim != 2
        or sequence.shape[1] != 562
        or raw_sequence.shape != (sequence.shape[0], 558)
    ):
        raise ValueError("representation_v1 arrays must be [T,562] and [T,558].")
    if group_presence.shape != (sequence.shape[0], 4) or valid_frame_mask.shape != (sequence.shape[0],):
        raise ValueError("Presence/mask arrays are incompatible with the [T,562] representation.")
    selected = config["feature_selection"]
    groups = [
        ("left_hand", 21, 3),
        ("right_hand", 21, 3),
        ("pose", len(selected.get("pose_landmark_subset", list(range(33)))), 4),
        ("face", len(config["face_landmark_indices"]), 3),
    ]
    frame_count = sequence.shape[0]
    raw_offset = 0
    model_offset = 0
    coordinates: list[np.ndarray] = []
    point_masks: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    group_masks: list[np.ndarray] = []
    frame_valid = valid_frame_mask > 0
    for group_index, (name, point_count, point_dims) in enumerate(groups):
        if not selected.get(name, False):
            raise ValueError(f"representation_v2 requires the configured {name} feature group.")
        width = point_count * point_dims
        raw_group = raw_sequence[:, raw_offset:raw_offset + width].reshape(
            frame_count, point_count, point_dims
        )
        model_group = sequence[:, model_offset:model_offset + width].reshape(
            frame_count, point_count, point_dims
        ).copy()
        point_valid = np.isfinite(raw_group[:, :, :3]).all(axis=2)
        point_valid &= frame_valid[:, None]
        model_group[~frame_valid, :, :] = 0.0

        for point_index in range(point_count):
            observed = np.flatnonzero(point_valid[:, point_index])
            for left, right in zip(observed[:-1], observed[1:]):
                gap = int(right - left - 1)
                if 0 < gap <= 2:
                    for frame_index in range(int(left) + 1, int(right)):
                        fraction = (frame_index - int(left)) / (int(right) - int(left))
                        model_group[frame_index, point_index, :3] = (
                            model_group[left, point_index, :3] * (1.0 - fraction)
                            + model_group[right, point_index, :3] * fraction
                        )

        speed = np.zeros((frame_count, point_count), dtype=np.float32)
        if frame_count > 1:
            consecutive = point_valid[1:] & point_valid[:-1]
            delta = model_group[1:, :, :3] - model_group[:-1, :, :3]
            speed[1:] = np.linalg.norm(delta, axis=2) * consecutive
        coordinates.append(model_group.reshape(frame_count, width))
        point_masks.append(point_valid.astype(np.float32))
        velocities.append(speed)
        group_masks.append((group_presence[:, group_index] * frame_valid).astype(np.float32)[:, None])
        raw_offset += width
        model_offset += width
    if raw_offset != raw_sequence.shape[1] or model_offset != sequence.shape[1] - 4:
        raise ValueError("Configured feature groups do not account for all 558 landmark features.")
    result = np.concatenate(
        [*coordinates, *point_masks, *velocities, *group_masks], axis=1
    ).astype(np.float32)
    if result.shape != (frame_count, 912) or not np.isfinite(result).all():
        raise ValueError(f"representation_v2 must be finite [T,912], got {result.shape}.")
    return result


def _load_benchmark_samples() -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    samples = baseline.load_samples(wlasl_v2.SEQUENCE_MANIFEST_PATH, language="ASL")
    split_rows = _read_rows(wlasl_v2.SPLIT_DIR / "split_manifest.csv")
    split_by_id = {row["sample_id"]: row["split"] for row in split_rows}
    if len(split_by_id) != len(split_rows):
        raise ValueError("The saved signer split contains duplicate sample IDs.")
    if set(split_by_id) != {sample.sample_id for sample in samples}:
        raise ValueError("Saved signer split does not exactly cover the current sequence manifest.")
    source_rows = {
        row["sample_id"]: row
        for row in _read_rows(wlasl_v2.SEQUENCE_MANIFEST_PATH)
    }
    checkpoint = torch.load(
        MODEL_ROOT / "best_model.pt", map_location="cpu", weights_only=True
    )
    labels = [
        label for label, _ in sorted(
            checkpoint["label_to_index"].items(), key=lambda item: item[1]
        )
    ]
    if set(labels) != {sample.label for sample in samples}:
        raise ValueError("Saved label order does not match the current WLASL sequence dataset.")
    diagnostic = (
        json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
        if DIAGNOSTIC_JSON.is_file() else {}
    )
    result = []
    for sample in samples:
        if sample.label not in labels:
            raise ValueError(f"Unknown label {sample.label!r} in the WLASL V2 sequence manifest.")
        row = source_rows[sample.sample_id]
        with np.load(sample.sequence_path, allow_pickle=False) as stored:
            sequence = np.asarray(stored["sequence"], dtype=np.float32)
            raw = np.asarray(stored["raw_sequence"], dtype=np.float32)
            presence = np.asarray(stored["landmark_presence"], dtype=np.float32)
            mask = np.asarray(stored["valid_frame_mask"], dtype=np.float32)
        result.append({
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "video_id": sample.video_id if hasattr(sample, "video_id") else row["video_id"],
            "class_index": row["class_index"],
            "signer_id": sample.signer_id,
            "label": sample.label,
            "split": split_by_id[sample.sample_id],
            "sequence": sequence,
            "raw_sequence": raw,
            "presence": presence,
            "valid_frame_mask": mask,
            "valid_length": int(np.count_nonzero(mask > 0)),
            "sequence_path": str(sample.sequence_path),
        })
    # Use the manifest's class-index/video identity for grouping, retaining the exact saved split.
    for row in result:
        row["group_id"] = _video_key(row["class_index"], row["video_id"])
    signer_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    for row in result:
        signer_splits[row["signer_id"]].add(row["split"])
        video_splits[row["group_id"]].add(row["split"])
    signer_overlap = {key: sorted(value) for key, value in signer_splits.items() if len(value) > 1}
    video_overlap = {key: sorted(value) for key, value in video_splits.items() if len(value) > 1}
    if signer_overlap or video_overlap:
        raise ValueError(f"Leakage found in saved split: signers={signer_overlap}, videos={video_overlap}")
    split_summary = {
        "split_counts": dict(Counter(row["split"] for row in result)),
        "signer_counts": {
            name: len({row["signer_id"] for row in result if row["split"] == name})
            for name in ("train", "validation", "test")
        },
        "video_counts": {
            name: len({row["group_id"] for row in result if row["split"] == name})
            for name in ("train", "validation", "test")
        },
        "signer_overlap": False,
        "video_overlap": False,
        "classes_by_split": {
            name: sorted({row["label"] for row in result if row["split"] == name})
            for name in ("train", "validation", "test")
        },
    }
    if any(set(rows) != set(labels) for rows in split_summary["classes_by_split"].values()):
        raise ValueError("The saved signer split does not contain every label in all partitions.")
    return result, labels, split_summary


def _prepare_arrays(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for row in rows:
        features[row["sample_id"]] = representation_v2(
            row["sequence"],
            row["raw_sequence"],
            row["presence"],
            row["valid_frame_mask"],
            config,
        )
    return features


def _video_metrics(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = defaultdict(list)
    for row, scores in zip(rows, probabilities):
        grouped[row["group_id"]].append((row, scores))
    label_to_index = {label: index for index, label in enumerate(labels)}
    true: list[int] = []
    predicted: list[int] = []
    top5: list[bool] = []
    records = []
    for group_id, windows in sorted(grouped.items()):
        mean_scores = np.mean([scores for _, scores in windows], axis=0)
        order = np.argsort(mean_scores)[::-1]
        first = windows[0][0]
        actual = label_to_index[first["label"]]
        predicted_index = int(order[0])
        true.append(actual)
        predicted.append(predicted_index)
        top5.append(actual in order[:min(5, len(labels))])
        records.append({
            "group_id": group_id,
            "video_id": first["video_id"],
            "class_index": first["class_index"],
            "signer_id": first["signer_id"],
            "actual_class": first["label"],
            "predicted_class": labels[predicted_index],
            "confidence": float(mean_scores[predicted_index]),
            "windows": len(windows),
            "top5_correct": bool(actual in order[:min(5, len(labels))]),
            "probabilities": mean_scores.astype(float).tolist(),
        })
    precision, recall, f1, support = baseline.precision_recall_fscore_support(
        true, predicted, labels=list(range(len(labels))), zero_division=0
    )
    matrix = confusion_matrix(true, predicted, labels=list(range(len(labels))))
    return {
        "videos": len(true),
        "accuracy": float(baseline.accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)) if true else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "top5_accuracy": float(np.mean(top5)) if top5 else 0.0,
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
        "predictions": records,
    }


def _window_metrics(
    rows: list[dict[str, Any]], probabilities: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    truth = np.asarray([label_to_index[row["label"]] for row in rows])
    predicted = probabilities.argmax(axis=1) if len(probabilities) else np.asarray([], dtype=int)
    precision, recall, f1, support = baseline.precision_recall_fscore_support(
        truth, predicted, labels=list(range(len(labels))), zero_division=0
    )
    top5 = [
        target in np.argsort(score)[::-1][:min(5, len(labels))]
        for target, score in zip(truth, probabilities)
    ]
    return {
        "windows": len(rows),
        "accuracy": float(baseline.accuracy_score(truth, predicted)) if len(truth) else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "top5_accuracy": float(np.mean(top5)) if top5 else 0.0,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=list(range(len(labels)))
        ).tolist(),
    }


def _padded_batch(
    rows: list[dict[str, Any]],
    feature_arrays: dict[str, np.ndarray],
    label_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.stack([feature_arrays[row["sample_id"]] for row in rows])
    lengths = np.asarray([row["valid_length"] for row in rows], dtype=np.int32)
    targets = np.asarray([label_to_index[row["label"]] for row in rows], dtype=np.int64)
    return values, lengths, targets


class AttentionBiLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = LAYER_COUNT,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = torch.nn.Linear(hidden_size * 2, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        outputs, _ = self.lstm(values)
        scores = self.attention(torch.tanh(outputs)).squeeze(-1)
        if lengths is not None:
            valid = torch.arange(values.shape[1], device=values.device)[None, :] < lengths.to(values.device)[:, None]
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(context))


def build_torch_model(
    model_kind: str, input_size: int, num_classes: int
) -> torch.nn.Module:
    if model_kind == "bilstm":
        return baseline.BiLSTMBaseline(
            input_size,
            num_classes,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYER_COUNT,
            dropout=DROPOUT,
        )
    if model_kind == "bilstm_attention":
        return AttentionBiLSTM(input_size, num_classes)
    raise ValueError(f"Unsupported PyTorch benchmark model kind: {model_kind}")


def load_torch_checkpoint(path: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_torch_model(
        checkpoint["model_kind"], checkpoint["input_size"], len(checkpoint["labels"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _torch_probabilities(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    feature_arrays: dict[str, np.ndarray],
    label_to_index: dict[str, int],
) -> np.ndarray:
    values, lengths, _targets = _padded_batch(rows, feature_arrays, label_to_index)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            logits = model(
                torch.from_numpy(values[start:start + BATCH_SIZE]),
                torch.from_numpy(lengths[start:start + BATCH_SIZE]),
            )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def _train_torch(
    name: str,
    representation_name: str,
    model_kind: str,
    rows: list[dict[str, Any]],
    labels: list[str],
    feature_arrays: dict[str, np.ndarray],
    input_size: int,
    feature_spec: dict[str, Any],
) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    train_x, train_lengths, train_y = _padded_batch(train_rows, feature_arrays, label_to_index)
    validation_x, validation_lengths, _ = _padded_batch(validation_rows, feature_arrays, label_to_index)
    class_counts = Counter(row["label"] for row in train_rows)
    weights = torch.tensor(
        [len(train_rows) / (len(labels) * class_counts[label]) for label in labels],
        dtype=torch.float32,
    )
    model = build_torch_model(model_kind, input_size, len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    root = BENCHMARK_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "best_validation.pt"
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = np.random.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch_x = torch.from_numpy(train_x[indices])
            batch_lengths = torch.from_numpy(train_lengths[indices])
            batch_y = torch.from_numpy(train_y[indices])
            shift = torch.randint(-2, 3, (len(indices), 1, 1))
            positions = torch.arange(batch_x.shape[1]).view(1, -1, 1)
            batch_x = torch.gather(
                batch_x, 1, (positions + shift).clamp(0, batch_x.shape[1] - 1).expand_as(batch_x)
            )
            batch_x = torch.cat((
                batch_x[:, :, :558]
                + torch.randn_like(batch_x[:, :, :558]) * 0.003,
                batch_x[:, :, 558:],
            ), dim=2)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_lengths)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation_probs = _torch_probabilities(
            model, validation_rows, feature_arrays, label_to_index
        )
        validation_metrics = _video_metrics(validation_rows, validation_probs, labels)
        validation_truth = np.asarray([label_to_index[row["label"]] for row in validation_rows])
        window_loss = float(np.mean([
            -np.log(max(1e-12, float(prob[target])))
            for target, prob in zip(validation_truth, validation_probs)
        ]))
        key = (
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
            -window_loss,
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_video_accuracy": validation_metrics["accuracy"],
            "validation_video_macro_f1": validation_metrics["macro_f1"],
            "validation_loss": window_loss,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_kind": model_kind,
                "framework": "pytorch",
                "language": "ASL",
                "labels": labels,
                "label_to_index": label_to_index,
                "input_size": input_size,
                "sequence_length": 30,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": LAYER_COUNT,
                "dropout": DROPOUT,
                "representation": representation_name,
                "feature_spec": feature_spec,
                "seed": SEED,
                "experiment_tag": baseline.EXPERIMENT_TAG,
                "dataset": "WLASL v0.3 top-100, WLASL V2",
            }, checkpoint_path)
        else:
            stale += 1
        print(
            f"{name} epoch={epoch} val-video={validation_metrics['accuracy']:.3f} "
            f"val-macro-f1={validation_metrics['macro_f1']:.3f}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    training_seconds = time.perf_counter() - started
    best_model, _ = load_torch_checkpoint(checkpoint_path)
    test_probs = _torch_probabilities(best_model, test_rows, feature_arrays, label_to_index)
    window_probs = _torch_probabilities(best_model, rows, feature_arrays, label_to_index)
    result = _assemble_result(
        name, "pytorch", model_kind, representation_name, rows, labels, test_probs,
        test_rows, window_probs, feature_arrays, label_to_index, input_size, feature_spec,
        best_epoch, training_seconds, history, checkpoint_path,
    )
    return result


def _keras_attention_layer() -> type:
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="BridgeSignExperimental")
    class TemporalAttention(tf.keras.layers.Layer):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.score_layer = tf.keras.layers.Dense(1, use_bias=False)
            self.supports_masking = True

        def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
            values, mask = inputs
            scores = tf.squeeze(self.score_layer(tf.tanh(values)), axis=-1)
            mask = tf.cast(mask, tf.bool)
            scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
            weights = tf.nn.softmax(scores, axis=1)
            return tf.reduce_sum(values * weights[:, :, None], axis=1)

        def get_config(self) -> dict[str, Any]:
            return super().get_config()

        def compute_mask(self, inputs: Any, mask: Any = None) -> None:
            return None

    return TemporalAttention


def load_keras_model(path: str | Path) -> Any:
    import tensorflow as tf

    attention_layer = _keras_attention_layer()
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={"TemporalAttention": attention_layer},
    )


def _build_keras_model(
    input_size: int, num_classes: int, *, attention: bool
) -> Any:
    import tensorflow as tf

    sequence_input = tf.keras.Input(shape=(30, input_size), dtype=tf.float32, name="sequence")
    length_input = tf.keras.Input(shape=(), dtype=tf.int32, name="valid_length")
    mask = tf.keras.ops.less(
        tf.keras.ops.expand_dims(tf.keras.ops.arange(30), axis=0),
        tf.keras.ops.expand_dims(length_input, axis=1),
    )
    values = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, dropout=DROPOUT)
    )(sequence_input, mask=mask)
    if attention:
        values = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, dropout=DROPOUT)
        )(values, mask=mask)
        context = _keras_attention_layer()(name="temporal_attention")((values, mask))
    else:
        context = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(HIDDEN_SIZE, return_sequences=False, dropout=DROPOUT)
        )(values, mask=mask)
    context = tf.keras.layers.Dropout(DROPOUT)(context)
    output = tf.keras.layers.Dense(num_classes, name="class_logits")(context)
    return tf.keras.Model([sequence_input, length_input], output)


def _keras_predict(model: Any, rows: list[dict[str, Any]], feature_arrays: dict[str, np.ndarray]) -> np.ndarray:
    values = np.stack([feature_arrays[row["sample_id"]] for row in rows]).astype(np.float32)
    lengths = np.asarray([row["valid_length"] for row in rows], dtype=np.int32)
    logits = model.predict([values, lengths], batch_size=BATCH_SIZE, verbose=0)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _train_keras(
    name: str,
    *,
    attention: bool,
    rows: list[dict[str, Any]],
    labels: list[str],
    feature_arrays: dict[str, np.ndarray],
    input_size: int,
    feature_spec: dict[str, Any],
) -> dict[str, Any]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    train_x = np.stack([feature_arrays[row["sample_id"]] for row in train_rows]).astype(np.float32)
    train_lengths = np.asarray([row["valid_length"] for row in train_rows], dtype=np.int32)
    train_y = np.asarray([label_to_index[row["label"]] for row in train_rows], dtype=np.int64)
    weights_count = Counter(row["label"] for row in train_rows)
    class_weight = {
        index: len(train_rows) / (len(labels) * weights_count[label])
        for label, index in label_to_index.items()
    }
    model = _build_keras_model(input_size, len(labels), attention=attention)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    root = BENCHMARK_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "best_validation.keras"
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            batch_values = train_x[index].copy()
            shifts = rng.integers(-2, 3, size=len(index))
            positions = np.arange(batch_values.shape[1])[None, :]
            shifted = np.clip(positions + shifts[:, None], 0, batch_values.shape[1] - 1)
            batch_values = np.take_along_axis(
                batch_values, shifted[:, :, None], axis=1
            )
            batch_values[:, :, :558] += rng.normal(
                0.0, 0.003, size=batch_values[:, :, :558].shape
            ).astype(np.float32)
            sample_weights = np.asarray([
                class_weight[int(train_y[row_index])] for row_index in index
            ], dtype=np.float32)
            train_metrics = model.train_on_batch(
                [batch_values, train_lengths[index]],
                train_y[index],
                sample_weight=sample_weights,
                return_dict=True,
            )
            losses.append(float(train_metrics["loss"]))
        validation_probs = _keras_predict(model, validation_rows, feature_arrays)
        validation_metrics = _video_metrics(validation_rows, validation_probs, labels)
        validation_truth = np.asarray([label_to_index[row["label"]] for row in validation_rows])
        loss = float(np.mean([
            -np.log(max(1e-12, float(prob[target])))
            for target, prob in zip(validation_truth, validation_probs)
        ]))
        key = (
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
            -loss,
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_video_accuracy": validation_metrics["accuracy"],
            "validation_video_macro_f1": validation_metrics["macro_f1"],
            "validation_loss": loss,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            model.save(checkpoint_path)
        else:
            stale += 1
        print(
            f"{name} epoch={epoch} val-video={validation_metrics['accuracy']:.3f} "
            f"val-macro-f1={validation_metrics['macro_f1']:.3f}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    training_seconds = time.perf_counter() - started
    best_model = load_keras_model(checkpoint_path)
    test_probs = _keras_predict(best_model, test_rows, feature_arrays)
    all_probs = _keras_predict(best_model, rows, feature_arrays)
    result = _assemble_result(
        name, "tensorflow_keras", "bilstm_attention" if attention else "bilstm",
        REPRESENTATION_VERSION, rows, labels, test_probs, test_rows, all_probs,
        feature_arrays, label_to_index, input_size, feature_spec, best_epoch,
        training_seconds, history, checkpoint_path,
    )
    return result


def _assemble_result(
    name: str,
    framework: str,
    model_kind: str,
    representation_name: str,
    rows: list[dict[str, Any]],
    labels: list[str],
    test_probabilities: np.ndarray,
    test_rows: list[dict[str, Any]],
    all_probabilities: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    label_to_index: dict[str, int],
    input_size: int,
    feature_spec: dict[str, Any],
    best_epoch: int,
    training_seconds: float,
    history: list[dict[str, Any]],
    checkpoint_path: Path,
) -> dict[str, Any]:
    window_rows = [row for row in rows if row["split"] == "test"]
    video = _video_metrics(test_rows, test_probabilities, labels)
    windows = _window_metrics(window_rows, test_probabilities, labels)
    by_sample = {
        row["sample_id"]: probabilities
        for row, probabilities in zip(rows, all_probabilities)
    }
    train_video = _video_metrics(
        [row for row in rows if row["split"] == "train"],
        np.stack([by_sample[row["sample_id"]] for row in rows if row["split"] == "train"]),
        labels,
    )
    best_validation = max(
        (item for item in history),
        key=lambda item: (
            item["validation_video_accuracy"],
            item["validation_video_macro_f1"],
            -item["validation_loss"],
        ),
    )
    return {
        "name": name,
        "framework": framework,
        "model_kind": model_kind,
        "representation": representation_name,
        "input_dimension": input_size,
        "test_video_level": video,
        "test_window_level": windows,
        "training_video_resubstitution": {
            key: train_video[key]
            for key in ("videos", "accuracy", "balanced_accuracy", "macro_f1", "top5_accuracy")
        },
        "best_epoch": best_epoch,
        "best_validation_video_accuracy": best_validation["validation_video_accuracy"],
        "best_validation_video_macro_f1": best_validation["validation_video_macro_f1"],
        "training_seconds": training_seconds,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "feature_spec": feature_spec,
    }


def _write_benchmark_checkpoint_metadata(
    result: dict[str, Any], labels: list[str]
) -> None:
    path = Path(result["checkpoint"])
    metadata = {
        "dataset": "WLASL v0.3 top-100, WLASL V2",
        "language": "ASL",
        "experiment_tag": baseline.EXPERIMENT_TAG,
        "framework": result["framework"],
        "model_kind": result["model_kind"],
        "representation": result["representation"],
        "input_size": result["input_dimension"],
        "sequence_length": 30,
        "labels": labels,
        "label_to_index": {label: index for index, label in enumerate(labels)},
        "feature_spec": result["feature_spec"],
        "best_epoch": result["best_epoch"],
        "seed": SEED,
        "split_manifest": str((wlasl_v2.SPLIT_DIR / "split_manifest.csv").resolve()),
        "training_seconds": result["training_seconds"],
        "checkpoint": str(path.resolve()),
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def run_benchmark(max_epochs: int = MAX_EPOCHS) -> dict[str, Any]:
    global MAX_EPOCHS
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive.")
    MAX_EPOCHS = max_epochs
    rows, labels, split_summary = _load_benchmark_samples()
    config = json.loads((DATA_ROOT / "preprocessing_config.json").read_text(encoding="utf-8"))
    feature_arrays_v1 = {row["sample_id"]: row["sequence"] for row in rows}
    feature_arrays_v2 = _prepare_arrays(rows, config)
    dimension_v2 = next(iter(feature_arrays_v2.values())).shape[1]
    feature_spec_v1 = {
        "version": "representation_v1",
        "groups": ["left_hand", "right_hand", "pose", "face"],
        "coordinates": "Existing wrist-relative model features; missing coordinates zero-filled.",
        "presence": "Four group-level flags appended after 558 coordinate/visibility values.",
        "dimensions": 562,
        "normalization": "wrist_relative",
        "temporal": "30 x 562; 25fps; no explicit velocity features.",
    }
    feature_spec_v2 = {
        "version": REPRESENTATION_VERSION,
        "groups": ["left_hand", "right_hand", "pose", "face"],
        "coordinates": "Existing normalized coordinates; gaps of up to 2 frames are linearly interpolated only when bracketed by observed frames.",
        "point_masks": "One observed/missing bit per landmark point; masks remain 0 for imputed coordinates.",
        "velocity": "Euclidean frame-to-frame normalized xyz displacement per landmark, zero unless that point is observed in adjacent valid frames.",
        "group_presence": "Original four group-level flags retained.",
        "dimensions": dimension_v2,
        "normalization": "Existing wrist_relative feature normalization retained.",
        "temporal": "30 x feature_dimension at 25fps; temporal differences supplement recurrent sequence modeling.",
        "missing_values": "Longer gaps stay zero and are distinguished by point/group presence masks.",
    }
    results = []
    results.append(_train_torch(
        "exp1_v1_current_bilstm", "representation_v1", "bilstm",
        rows, labels, feature_arrays_v1, 562, feature_spec_v1,
    ))
    results.append(_train_torch(
        "exp2_v2_current_bilstm", REPRESENTATION_VERSION, "bilstm",
        rows, labels, feature_arrays_v2, dimension_v2, feature_spec_v2,
    ))
    results.append(_train_torch(
        "exp3_v2_attention_bilstm", REPRESENTATION_VERSION, "bilstm_attention",
        rows, labels, feature_arrays_v2, dimension_v2, feature_spec_v2,
    ))
    results.append(_train_keras(
        "exp4_v2_tensorflow_bilstm",
        attention=False,
        rows=rows,
        labels=labels,
        feature_arrays=feature_arrays_v2,
        input_size=dimension_v2,
        feature_spec=feature_spec_v2,
    ))
    results.append(_train_keras(
        "exp5_v2_tensorflow_attention",
        attention=True,
        rows=rows,
        labels=labels,
        feature_arrays=feature_arrays_v2,
        input_size=dimension_v2,
        feature_spec=feature_spec_v2,
    ))
    for result in results:
        _write_benchmark_checkpoint_metadata(result, labels)
    # The final choice uses validation video accuracy then macro F1, never test performance.
    best = max(
        results,
        key=lambda item: (
            item["best_validation_video_accuracy"],
            item["best_validation_video_macro_f1"],
            -item["training_seconds"],
        ),
    )
    comparison = {
        "status": "EXPERIMENTAL",
        "target": {"video_accuracy": 0.90, "macro_f1": 0.90},
        "target_reached": (
            best["test_video_level"]["accuracy"] >= 0.90
            and best["test_video_level"]["macro_f1"] >= 0.90
        ),
        "dataset": "Official WLASL v0.3 ASL; selected WLASL100 subset",
        "labels": labels,
        "sample_windows": len(rows),
        "split": split_summary,
        "split_manifest_sha256": wlasl_v2._sha256(wlasl_v2.SPLIT_DIR / "split_manifest.csv"),
        "sequence_manifest_sha256": wlasl_v2._sha256(wlasl_v2.SEQUENCE_MANIFEST_PATH),
        "diagnostic": str(DIAGNOSTIC_JSON.resolve()),
        "feature_representations": {
            "representation_v1": feature_spec_v1,
            REPRESENTATION_VERSION: feature_spec_v2,
        },
        "experiments": results,
        "best_model": {
            "name": best["name"],
            "framework": best["framework"],
            "model_kind": best["model_kind"],
            "representation": best["representation"],
            "input_dimension": best["input_dimension"],
            "checkpoint": best["checkpoint"],
            "best_epoch": best["best_epoch"],
            "test_video_accuracy": best["test_video_level"]["accuracy"],
            "test_macro_f1": best["test_video_level"]["macro_f1"],
        },
        "model_selection": "Highest signer-disjoint validation video accuracy, then validation macro F1; test metrics are not used to select the model.",
        "training_config": {
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "layers": LAYER_COUNT,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "class_weighting": "inverse training-window class frequency",
            "window_augmentation": "Shared across experiments: temporal shift +/-2 frames plus Gaussian jitter std=0.003 on coordinate values only; presence and velocity channels unchanged.",
            "runtime_versions": {
                "python": ".".join(map(str, sys.version_info[:3])),
                "pytorch": torch.__version__,
            },
        },
    }
    comparison["target_reached"] = (
        best["test_video_level"]["accuracy"] >= 0.90
        and best["test_video_level"]["macro_f1"] >= 0.90
    )
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    COMPARISON_JSON.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    selected_meta = {
        **comparison["best_model"],
        "dataset": comparison["dataset"],
        "language": "ASL",
        "experiment_tag": baseline.EXPERIMENT_TAG,
        "labels": labels,
        "label_to_index": {label: index for index, label in enumerate(labels)},
        "sequence_length": 30,
        "preprocessing_config": str((DATA_ROOT / "preprocessing_config.json").resolve()),
        "split_manifest": str((wlasl_v2.SPLIT_DIR / "split_manifest.csv").resolve()),
        "split_manifest_sha256": comparison["split_manifest_sha256"],
        "feature_spec": best["feature_spec"],
    }
    (BENCHMARK_ROOT / "best_model.json").write_text(
        json.dumps(selected_meta, indent=2) + "\n", encoding="utf-8"
    )
    _write_comparison_report(comparison)
    write_error_analysis(comparison, diagnostic=json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8")))
    return comparison


def _write_comparison_report(comparison: dict[str, Any]) -> None:
    lines = [
        "# WLASL V2 Model and Representation Comparison",
        "",
        "> WLASL is ASL, not MSL. These results apply only to this small selected 10-class experiment.",
        "",
        f"**Target:** {'EXPERIMENTAL TARGET REACHED' if comparison['target_reached'] else 'EXPERIMENTAL TARGET NOT REACHED'} (>=90% signer-disjoint video accuracy and macro F1).",
        "",
        f"- Primary evaluation: one prediction per held-out source video, by averaging probabilities over that video's windows.",
        f"- Labels: `{json.dumps(comparison['labels'])}`.",
        f"- Input: {comparison['sample_windows']} windows, signer-disjoint split `{json.dumps(comparison['split'], sort_keys=True)}`.",
        f"- Saved split SHA-256: `{comparison['split_manifest_sha256']}`.",
        "- All experiments consume identical sequence rows, labels, saved partitions, and held-out videos. Only the representation/model framework varies as identified below.",
        "- Model selection uses validation video accuracy then validation macro F1; held-out test results do not choose model or epoch.",
        "",
        "## Model comparison",
        "",
        "| Experiment | Framework / architecture | Representation | Input dim | Best epoch | Train time (s) | Test videos | Video accuracy | Balanced accuracy | Macro F1 | Top-5 | Window accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in comparison["experiments"]:
        video = result["test_video_level"]
        window = result["test_window_level"]
        lines.append(
            f"| {result['name']} | {result['framework']} / {result['model_kind']} | "
            f"{result['representation']} | {result['input_dimension']} | {result['best_epoch']} | "
            f"{result['training_seconds']:.1f} | {video['videos']} | {video['accuracy']:.4f} | "
            f"{video['balanced_accuracy']:.4f} | {video['macro_f1']:.4f} | "
            f"{video['top5_accuracy']:.4f} | {window['accuracy']:.4f} |"
        )
    best = comparison["best_model"]
    selected = next(row for row in comparison["experiments"] if row["name"] == best["name"])
    lines.extend([
        "",
        "## Selected checkpoint",
        "",
        f"- Best model: `{best['name']}` ({best['framework']}, `{best['representation']}`, {best['input_dimension']} features/frame).",
        f"- Checkpoint: `{best['checkpoint']}`.",
        f"- Selection was based on validation only. Held-out video accuracy: {selected['test_video_level']['accuracy']:.4f}; macro F1: {selected['test_video_level']['macro_f1']:.4f}; top-5: {selected['test_video_level']['top5_accuracy']:.4f}.",
        "",
        "## Representation definitions",
        "",
        "- representation_v1 retains the original 558 wrist-relative coordinates/visibility values plus four group-presence bits (=562). Missing coordinates were zero-filled.",
        "- representation_v2 retains those 558 normalized values, adds 175 point-presence masks, 175 per-landmark xyz displacement magnitudes, and four group-presence flags (=912). Gaps of <=2 frames are interpolated only when bracketed; masks continue to denote observed landmarks. Longer gaps stay zero and masked.",
        "- representation_v2 does not concatenate new raw landmarks or alter frame rate/sequence boundaries. It improves missingness observability and supplies a compact temporal-difference feature per landmark.",
        "",
        "## Error patterns",
        "",
        f"- Detailed per-held-out-video predictions and error groupings: `{ERROR_REPORT}`.",
        f"- Confusion matrix for the selected model: `{json.dumps(selected['test_video_level']['confusion_matrix'])}`.",
        "- See the landmark diagnostic for measured per-class and per-signer hand rates and the limits on causal attribution.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python tools\\experimental_dynamic\\wlasl_v2_benchmark.py diagnose",
        "python tools\\experimental_dynamic\\wlasl_v2_benchmark.py benchmark --epochs 24",
        "python -m pytest -q",
        "```",
        "",
        "Runtime versions are recorded in the companion JSON and the optional benchmark requirements file.",
        "",
    ])
    COMPARISON_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_error_analysis(
    comparison: dict[str, Any], diagnostic: dict[str, Any] | None = None
) -> None:
    diagnostic = diagnostic or json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
    selected = next(
        row for row in comparison["experiments"]
        if row["name"] == comparison["best_model"]["name"]
    )
    predictions = selected["test_video_level"]["predictions"]
    video_quality = {row["video_id"] + ":" + str(row["class_index"]): row for row in diagnostic["videos"]}
    lines = [
        "# WLASL V2 Held-Out Error Analysis",
        "",
        "> WLASL is ASL, not MSL. Predictions are from the signer-disjoint test partition only.",
        "",
        f"- Best validation-selected model: `{selected['name']}`.",
        f"- Held-out videos: {len(predictions)}; correct: {sum(row['actual_class'] == row['predicted_class'] for row in predictions)}; accuracy: {selected['test_video_level']['accuracy']:.4f}; macro F1: {selected['test_video_level']['macro_f1']:.4f}.",
        "",
        "| Video | Signer | True | Predicted | Correct | Confidence | Windows | Both hands | Right only | Left only | Neither | Frames |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_signer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        key = str(prediction["video_id"]) + ":" + str(prediction["class_index"])
        quality = video_quality.get(key, {})
        joint = quality.get("joint_hand_frame_rates", {})
        correct = prediction["actual_class"] == prediction["predicted_class"]
        prediction["correct"] = correct
        prediction["joint_hand_frame_rates"] = joint
        prediction["frame_count"] = quality.get("usable_frames", quality.get("frame_count"))
        by_signer[prediction["signer_id"]].append(prediction)
        by_class[prediction["actual_class"]].append(prediction)
        lines.append(
            f"| {prediction['video_id']} | {prediction['signer_id']} | {prediction['actual_class']} | "
            f"{prediction['predicted_class']} | {'yes' if correct else 'NO'} | "
            f"{prediction['confidence']:.3f} | {prediction['windows']} | "
            f"{joint.get('both_hands', 0):.3f} | {joint.get('right_hand_only', 0):.3f} | "
            f"{joint.get('left_hand_only', 0):.3f} | {joint.get('neither_hand', 0):.3f} | "
            f"{prediction['frame_count'] or 'n/a'} |"
        )
    errors = [row for row in predictions if not row["correct"]]
    lines.extend([
        "",
        "## Error concentration",
        "",
        f"- Error count: {len(errors)} / {len(predictions)} held-out videos.",
        f"- Errors by true class: `{json.dumps(dict(Counter(row['actual_class'] for row in errors)), sort_keys=True)}`.",
        f"- Errors by predicted class: `{json.dumps(dict(Counter(row['predicted_class'] for row in errors)), sort_keys=True)}`.",
        f"- Errors by signer: `{json.dumps(dict(Counter(row['signer_id'] for row in errors)), sort_keys=True)}`.",
        "- Per-signer accuracy and per-class support:",
    ])
    for signer, items in sorted(by_signer.items()):
        lines.append(
            f"  - Signer {signer}: {sum(row['correct'] for row in items)}/{len(items)} correct."
        )
    for label, items in sorted(by_class.items()):
        lines.append(
            f"  - {label}: {sum(row['correct'] for row in items)}/{len(items)} correct; "
            f"signers={sorted({row['signer_id'] for row in items})}."
        )
    if errors:
        mean_both_error = float(np.mean([
            row["joint_hand_frame_rates"].get("both_hands", 0.0) for row in errors
        ]))
        mean_both_correct = float(np.mean([
            row["joint_hand_frame_rates"].get("both_hands", 0.0)
            for row in predictions if row["correct"]
        ])) if any(row["correct"] for row in predictions) else 0.0
        lines.extend([
            f"- Mean both-hand detection rate in errors: {mean_both_error:.3f}; in correct predictions: {mean_both_correct:.3f}. This is descriptive, not a causal test.",
            "- Video duration, frame count, and per-video hand-detection patterns are reported above. No claim about framing or sign similarity is made without a video review.",
        ])
    else:
        lines.append("- No held-out errors occurred; the small test size remains a limitation.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Test set is small (16 source videos, four signers); observed percentages have high variance.",
        "- The source videos are real WLASL clips, but framing, signer behavior, and semantic/phonological similarity require human visual/sign-language review.",
        "- Confidence values are softmax scores and are not calibrated probabilities.",
        "",
    ])
    ERROR_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated WLASL V2 landmark diagnostics, representation ablation, and model benchmark."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("diagnose")
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    error_parser = commands.add_parser("error-analysis")
    error_parser.add_argument("--comparison", type=Path, default=COMPARISON_JSON)
    args = parser.parse_args()
    if args.command == "diagnose":
        result = run_landmark_diagnostics()
        print(json.dumps(result["scope"], indent=2))
    elif args.command == "benchmark":
        comparison = run_benchmark(args.epochs)
        print(json.dumps({
            "best_model": comparison["best_model"],
            "target_reached": comparison["target_reached"],
            "comparison_report": str(COMPARISON_REPORT),
            "error_report": str(ERROR_REPORT),
        }, indent=2))
    else:
        comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
        write_error_analysis(comparison)
        print(ERROR_REPORT)


if __name__ == "__main__":
    main()
