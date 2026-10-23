from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__:
    from .collection_management import (
        DEFAULT_MANIFESTS_DIR,
        DEFAULT_REVIEW_DIR,
        SESSION_PATTERN,
        SIGNER_PATTERN,
        SIGN_PATTERN,
        load_recordings,
        load_sessions,
        load_signers,
    )
    from .msl_dataset_pipeline import (
        _create_holistic_detector,
        _extract_landmarks_for_video,
        _window_starts,
        load_config,
    )
else:
    from collection_management import (
        DEFAULT_MANIFESTS_DIR,
        DEFAULT_REVIEW_DIR,
        SESSION_PATTERN,
        SIGNER_PATTERN,
        SIGN_PATTERN,
        load_recordings,
        load_sessions,
        load_signers,
    )
    from msl_dataset_pipeline import (
        _create_holistic_detector,
        _extract_landmarks_for_video,
        _window_starts,
        load_config,
    )


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
IMPORT_NAMESPACE = uuid.UUID("28d210c9-b50c-47f8-854f-11101231b15b")
IMPORT_FILENAME = "existing_recordings.csv"
REVIEW_FILENAME = "recording_review.csv"
SIGNER_REVIEW_STATES = {"CONFIRMED", "REJECTED", "UNKNOWN"}
SESSION_REVIEW_STATES = {"CONFIRMED", "REJECTED", "UNKNOWN"}
LABEL_REVIEW_STATES = {"CONFIRMED", "REJECTED", "NEEDS_REVIEW"}
LANGUAGE_REVIEW_STATES = {"MSL", "NOT_MSL", "UNKNOWN"}
QUALITY_REVIEW_STATES = {"APPROVED", "REJECTED", "NEEDS_RETAKE"}
HUMAN_REVIEW_STATES = {"APPROVED", "REJECTED", "NEEDS_REVIEW"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _infer_path_metadata(relative_path: Path) -> dict[str, str]:
    parts = relative_path.parts
    if (
        len(parts) == 4
        and SIGNER_PATTERN.fullmatch(parts[0])
        and SIGN_PATTERN.fullmatch(parts[1].upper())
        and SESSION_PATTERN.fullmatch(parts[2])
    ):
        signer, sign, session = parts[:3]
        return {
            "source_type": "controlled_layout",
            "signer_id": signer,
            "signer_identity_status": "INFERRED_UNCONFIRMED",
            "signer_inference_basis": "signer directory component",
            "sign": sign.upper(),
            "label_inference_basis": "sign directory component",
            "session_id": session,
            "session_inference_basis": "session directory component",
            "legacy_flat_layout": "false",
        }
    if (
        len(parts) == 3
        and SIGNER_PATTERN.fullmatch(parts[0])
        and SIGN_PATTERN.fullmatch(parts[1].upper())
    ):
        signer, sign = parts[:2]
        return {
            "source_type": "legacy_signer_sign_video",
            "signer_id": signer,
            "signer_identity_status": "INFERRED_UNCONFIRMED",
            "signer_inference_basis": "signer directory component",
            "sign": sign.upper(),
            "label_inference_basis": "sign directory component",
            "session_id": "UNKNOWN",
            "session_inference_basis": "not present in source path",
            "legacy_flat_layout": "true",
        }
    if len(parts) == 2 and SIGN_PATTERN.fullmatch(parts[0].upper()):
        sign = parts[0]
        return {
            "source_type": "legacy_sign_video",
            "signer_id": "UNKNOWN",
            "signer_identity_status": "UNKNOWN",
            "signer_inference_basis": "not present in source path",
            "sign": sign.upper(),
            "label_inference_basis": "sign directory component",
            "session_id": "UNKNOWN",
            "session_inference_basis": "not present in source path",
            "legacy_flat_layout": "true",
        }
    return {
        "source_type": "ambiguous_path",
        "signer_id": "UNKNOWN",
        "signer_identity_status": "UNKNOWN",
        "signer_inference_basis": "ambiguous directory structure",
        "sign": "UNKNOWN",
        "label_inference_basis": "ambiguous directory structure",
        "session_id": "UNKNOWN",
        "session_inference_basis": "not present in source path",
        "legacy_flat_layout": "false",
    }


def _existing_processed_sources(manifests_dir: Path) -> set[tuple[str, str]]:
    sources: set[tuple[str, str]] = set()
    for row in _read_csv(manifests_dir / "videos.csv"):
        if row.get("processing_status") != "PROCESSED":
            continue
        path = str(row.get("file_path", "")).strip()
        digest = str(row.get("source_sha256", "")).strip()
        if path or digest:
            sources.add((str(Path(path).resolve()) if path else "", digest))
    for row in _read_csv(manifests_dir / "sequences.csv"):
        sequence_path = Path(str(row.get("sequence_path", "")).strip())
        if row.get("processing_status") != "PROCESSED" or not sequence_path.is_file():
            continue
        path = str(row.get("file_path", "")).strip()
        digest = str(row.get("source_sha256", "")).strip()
        if path or digest:
            sources.add((str(Path(path).resolve()) if path else "", digest))
    return sources


def _technical_video_check(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {
            "video_validation_status": "VIDEO_INVALID",
            "reported_frame_count": 0,
            "decoded_frame_count": 0,
            "fps": 0.0,
            "duration_seconds": 0.0,
            "width": 0,
            "height": 0,
            "video_validation_notes": "OpenCV could not open the video.",
        }
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    decoded_frames = 0
    try:
        while True:
            decoded, _ = capture.read()
            if not decoded:
                break
            decoded_frames += 1
    finally:
        capture.release()
    problems = []
    if decoded_frames == 0:
        problems.append("No frames could be decoded.")
    if fps <= 0:
        problems.append("FPS metadata is missing or invalid.")
    if width <= 0 or height <= 0:
        problems.append("Resolution metadata is missing or invalid.")
    if reported_frames > 0 and decoded_frames != reported_frames:
        problems.append(f"Decoded {decoded_frames} frames but container reports {reported_frames}.")
    status = "VIDEO_INVALID" if problems else "READABLE"
    return {
        "video_validation_status": status,
        "reported_frame_count": reported_frames,
        "decoded_frame_count": decoded_frames,
        "fps": fps,
        "duration_seconds": decoded_frames / fps if fps > 0 else 0.0,
        "width": width,
        "height": height,
        "video_validation_notes": "; ".join(problems),
    }


def _load_registered_sources(
    manifests_dir: Path,
) -> tuple[set[str], set[str], dict[str, list[str]], set[str], set[str]]:
    known_paths: set[str] = set()
    known_hashes: set[str] = set()
    ids_by_hash: dict[str, list[str]] = defaultdict(list)
    csv_paths: set[str] = set()
    csv_hashes: set[str] = set()
    for row in load_recordings(manifests_dir):
        source_path = str(row.get("file_path", "")).strip()
        source_hash = str(row.get("sha256", "")).strip()
        if source_path:
            known_paths.add(str(Path(source_path).resolve()))
        if source_hash:
            known_hashes.add(source_hash)
            ids_by_hash[source_hash].append(str(row.get("recording_id", "UNKNOWN")))
    for row in _read_csv(manifests_dir / "recordings.csv"):
        source_path = str(row.get("file_path", "")).strip()
        source_hash = str(row.get("sha256", "")).strip()
        if source_path:
            known_paths.add(str(Path(source_path).resolve()))
        if source_hash:
            known_hashes.add(source_hash)
            ids_by_hash[source_hash].append(str(row.get("recording_id", "UNKNOWN")))
        if source_path:
            csv_paths.add(str(Path(source_path).resolve()))
        if source_hash:
            csv_hashes.add(source_hash)
    return known_paths, known_hashes, ids_by_hash, csv_paths, csv_hashes


def _processing_blockers(
    row: dict[str, Any],
    signers: dict[str, dict[str, Any]],
    sessions: dict[str, dict[str, Any]],
    vocabulary: dict[str, dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    signer_id = str(row.get("signer_id", "UNKNOWN"))
    session_id = str(row.get("session_id", "UNKNOWN"))
    sign = str(row.get("sign", "UNKNOWN"))
    signer = signers.get(signer_id, {})
    session = sessions.get(f"{signer_id}/{session_id}", {})
    vocab_entry = vocabulary.get(sign, {})
    if row.get("signer_review_status") != "CONFIRMED" or row.get("signer_identity_status") != "CONFIRMED":
        blockers.append("SIGNER_CONFIRMATION_REQUIRED")
    if not row.get("signer_confirmation_evidence", "").strip():
        blockers.append("SIGNER_CONFIRMATION_EVIDENCE_REQUIRED")
    if not signer or signer.get("status") != "ACTIVE" or signer.get("consent_status") != "GRANTED":
        blockers.append("CONSENT_NOT_GRANTED")
    if (
        row.get("session_review_status") != "CONFIRMED"
        or not session
        or session.get("status") != "ACTIVE"
    ):
        blockers.append("ACTIVE_SESSION_CONFIRMATION_REQUIRED")
    if not row.get("session_confirmation_evidence", "").strip():
        blockers.append("SESSION_CONFIRMATION_EVIDENCE_REQUIRED")
    if (
        row.get("label_review_status") != "CONFIRMED"
        or row.get("label_validation_state") != "VALIDATED"
        or sign == "UNKNOWN"
        or vocab_entry.get("language") != "MSL"
        or vocab_entry.get("status") != "ACTIVE"
        or vocab_entry.get("human_validation") != "VALIDATED"
    ):
        blockers.append("VOCABULARY_VALIDATION_REQUIRED")
    if not row.get("label_confirmation_evidence", "").strip():
        blockers.append("LABEL_CONFIRMATION_EVIDENCE_REQUIRED")
    if (
        row.get("language_review_status") != "MSL"
        or row.get("language") != "MSL"
        or row.get("language_validation_state") != "VALIDATED"
    ):
        blockers.append("MSL_LANGUAGE_CONFIRMATION_REQUIRED")
    if not row.get("language_confirmation_evidence", "").strip():
        blockers.append("LANGUAGE_CONFIRMATION_EVIDENCE_REQUIRED")
    if row.get("recording_quality_status") != "APPROVED":
        blockers.append("RECORDING_QUALITY_APPROVAL_REQUIRED")
    if row.get("human_review_status") != "APPROVED" or row.get("review_status") != "APPROVED":
        blockers.append("HUMAN_REVIEW_APPROVAL_REQUIRED")
    source = Path(str(row.get("original_path", "")))
    if (
        row.get("source_hash_changed") == "true"
        or not source.is_file()
        or _sha256(source) != row.get("sha256")
    ):
        blockers.append("SOURCE_HASH_MISMATCH")
    if row.get("video_validation_status") != "READABLE":
        blockers.append("VIDEO_NOT_READABLE")
    if (
        row.get("technical_qc_status") not in {"READY_FOR_REVIEW", "READY_FOR_PROCESSING"}
        or row.get("landmark_status") != "CHECKED"
        or int(row.get("hand_detected_frames") or 0) <= 0
        or row.get("finite_model_input") != "true"
    ):
        blockers.append("TECHNICAL_QC_FAILED")
    return blockers


def _final_status(
    row: dict[str, Any],
    signers: dict[str, dict[str, Any]],
    sessions: dict[str, dict[str, Any]],
    vocabulary: dict[str, dict[str, Any]] | None = None,
) -> str:
    if row["video_validation_status"] == "VIDEO_INVALID":
        return "VIDEO_INVALID"
    if row.get("landmark_status") == "CHECKED" and int(row.get("hand_detected_frames", 0)) == 0:
        return "LANDMARK_QUALITY_LOW"
    if row.get("technical_qc_status") == "QC_ERROR":
        return "TECHNICAL_QC_FAILED"
    if _processing_blockers(row, signers, sessions, vocabulary or {}):
        return "METADATA_INCOMPLETE"
    if row.get("metadata_confirmation_status") != "CONFIRMED":
        return "METADATA_INCOMPLETE"
    return "READY_FOR_PROCESSING"


def _write_existing_recordings_report(
    manifest_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    signers = load_signers(manifest_root)
    sessions = load_sessions(manifest_root)
    vocabulary_path = manifest_root / "sign_vocabulary.json"
    vocabulary = (
        json.loads(vocabulary_path.read_text(encoding="utf-8"))
        if vocabulary_path.is_file()
        else {}
    )
    report = [
        "# Existing Recording Audit and Human Review",
        "",
        f"- total recordings: {summary.get('total_videos', len(rows))}",
        f"- technically valid: {summary.get('videos_technically_ready', 0)}",
        f"- human approved: {summary.get('human_approved', 0)}",
        f"- human rejected: {summary.get('human_rejected', 0)}",
        f"- needs review: {summary.get('human_needs_review', 0)}",
        f"- ready for processing: {summary.get('ready_for_processing', 0)}",
        f"- blocked by missing signer: {summary.get('blocked_missing_signer', 0)}",
        f"- blocked by missing consent: {summary.get('blocked_missing_consent', 0)}",
        f"- blocked by missing session: {summary.get('blocked_missing_session', 0)}",
        f"- blocked by missing MSL validation: {summary.get('blocked_missing_msl_validation', 0)}",
        f"- blocked by review: {summary.get('blocked_by_review', 0)}",
        "",
        "Technical QC does not validate signer identity, session, label, or language.",
        "Folder-derived values remain candidates unless explicitly confirmed by a reviewer.",
        "",
        "| Recording ID | Video / path | Sign candidate | Signer | Session | Duration | FPS | Resolution | Frames | Hands % | Left % | Right % | Technical QC | Signer decision | Label decision | Language | Quality | Human review | Consent | Processing status | Blockers |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        blockers = _processing_blockers(row, signers, sessions, vocabulary)
        hand_pct = row.get("hand_detected_percentage", "")
        left_pct = row.get("left_hand_detected_percentage", "")
        right_pct = row.get("right_hand_detected_percentage", "")
        report.append(
            f"| `{row.get('recording_id', '')}` | `{row.get('filename', '')}`<br>`{row.get('original_path', '')}` "
            f"| {row.get('sign_candidate', 'UNKNOWN')} | "
            f"{row.get('signer_id', 'UNKNOWN')} ({row.get('signer_review_status', 'UNKNOWN')}) | "
            f"{row.get('session_id', 'UNKNOWN')} ({row.get('session_review_status', 'UNKNOWN')}) | "
            f"{float(row.get('duration_seconds') or 0):.2f}s | {float(row.get('fps') or 0):.2f} | "
            f"{row.get('resolution', 'UNKNOWN')} | {row.get('frame_count', 0)} | "
            f"{hand_pct}% | {left_pct}% | {right_pct}% | "
            f"{row.get('technical_qc_status', 'UNKNOWN')} | "
            f"{row.get('signer_review_status', 'UNKNOWN')} | "
            f"{row.get('label_review_status', 'NEEDS_REVIEW')} | "
            f"{row.get('language_review_status', 'UNKNOWN')} | "
            f"{row.get('recording_quality_status', 'UNKNOWN')} | "
            f"{row.get('human_review_status', 'NEEDS_REVIEW')} | "
            f"{row.get('consent_status', 'UNKNOWN')} | "
            f"{row.get('validation_status', 'UNKNOWN')} | "
            f"{', '.join(blockers) if blockers else 'none'} |"
        )
    (manifest_root / "existing_recordings_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def import_existing_recordings(
    raw_roots: list[str | Path],
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    config_path: str | Path | None = None,
    check_landmarks: bool = True,
) -> dict[str, Any]:
    roots = [Path(root).resolve() for root in raw_roots]
    if not roots:
        raise ValueError("At least one source root must be provided.")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Recording source root does not exist: {root}")
    manifest_root = Path(manifests_dir)
    config = load_config(config_path)
    signers = load_signers(manifest_root)
    sessions = load_sessions(manifest_root)
    vocabulary_path = manifest_root / "sign_vocabulary.json"
    vocabulary = (
        json.loads(vocabulary_path.read_text(encoding="utf-8"))
        if vocabulary_path.is_file()
        else {}
    )
    known_paths, known_hashes, ids_by_hash, csv_paths, csv_hashes = _load_registered_sources(manifest_root)
    processed_sources = _existing_processed_sources(manifest_root)
    previous_rows = {
        str(Path(row["original_path"]).resolve()): row
        for row in _read_csv(manifest_root / IMPORT_FILENAME)
        if row.get("original_path")
    }
    existing_reviews = {
        row.get("recording_id", ""): row
        for row in _read_csv(Path(review_dir) / REVIEW_FILENAME)
    }

    path_roots: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                path_roots[str(path.resolve())].append((root, path))

    hash_paths: dict[str, list[str]] = defaultdict(list)
    imported_rows: list[dict[str, Any]] = []
    for absolute_path, source_entries in sorted(path_roots.items()):
        root, path = source_entries[0]
        relative = path.relative_to(root)
        inferred = _infer_path_metadata(relative)
        digest = _sha256(path)
        hash_paths[digest].append(absolute_path)
        technical = _technical_video_check(path)
        processed = (absolute_path, digest) in processed_sources or any(
            source_hash == digest for _, source_hash in processed_sources if source_hash
        )
        row: dict[str, Any] = {
            "original_path": absolute_path,
            "filename": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "recording_id": str(uuid.uuid5(IMPORT_NAMESPACE, absolute_path)),
            **inferred,
            "sign_candidate": inferred["sign"],
            "signer_candidate": inferred["signer_id"],
            "session_candidate": inferred["session_id"],
            "source_type": inferred["source_type"],
            "language": "UNKNOWN",
            "language_validation_state": "UNVERIFIED",
            "label_validation_state": "UNVALIDATED" if inferred["sign"] != "UNKNOWN" else "UNKNOWN",
            "signer_review_status": "UNKNOWN",
            "session_review_status": "UNKNOWN",
            "label_review_status": "NEEDS_REVIEW",
            "language_review_status": "UNKNOWN",
            "recording_quality_status": "UNKNOWN",
            "human_review_status": "NEEDS_REVIEW",
            "metadata_confirmation_status": "PENDING",
            "signer_confirmation_evidence": "",
            "session_confirmation_evidence": "",
            "label_confirmation_evidence": "",
            "language_confirmation_evidence": "",
            "metadata_confirmed_at": "",
            "source_hash_changed": "false",
            "review_status": existing_reviews.get(
                str(uuid.uuid5(IMPORT_NAMESPACE, absolute_path)), {}
            ).get("status", "PENDING"),
            "sha256": digest,
            "duration_seconds": round(float(technical["duration_seconds"]), 6),
            "fps": round(float(technical["fps"]), 6),
            "resolution": f"{technical['width']}x{technical['height']}",
            "width": technical["width"],
            "height": technical["height"],
            "frame_count": technical["decoded_frame_count"],
            "reported_frame_count": technical["reported_frame_count"],
            "video_validation_status": technical["video_validation_status"],
            "validation_status": "PENDING_LANDMARK_QC",
            "technical_qc_status": "PENDING",
            "landmark_status": "NOT_CHECKED",
            "landmark_availability": "UNKNOWN",
            "landmark_validation_notes": "",
            "frames_processed": 0,
            "hand_detected_percentage": 0.0,
            "left_hand_detected_percentage": 0.0,
            "right_hand_detected_percentage": 0.0,
            "pose_detected_percentage": 0.0,
            "face_detected_percentage": 0.0,
            "sequence_window_count": 0,
            "model_feature_dimension": 0,
            "finite_model_input": "",
            "hand_detected_frames": "",
            "left_hand_detected_frames": "",
            "right_hand_detected_frames": "",
            "pose_detected_frames": "",
            "face_detected_frames": "",
            "processing_status": "PROCESSED" if processed else "NOT_PROCESSED",
            "processing_notes": "",
            "consent_status": "UNKNOWN",
            "session_validation_status": "UNKNOWN",
            "signer_validation_status": "UNCONFIRMED",
            "consent_status": "UNKNOWN",
            "session_validation_status": "UNKNOWN",
            "signer_validation_status": "UNCONFIRMED",
            "represented_in_recordings_csv": str(
                absolute_path in csv_paths or digest in csv_hashes
            ).lower(),
            "path_already_in_recordings_manifest": str(absolute_path in known_paths).lower(),
            "content_already_in_recordings_manifest": str(digest in known_hashes).lower(),
            "already_processed": str(processed).lower(),
            "duplicate_path": str(len(source_entries) > 1).lower(),
            "duplicate_content": "false",
            "duplicate_of_recording_ids": "",
            "notes": technical["video_validation_notes"],
            "path_roots": ";".join(str(entry[0]) for entry in source_entries),
        }
        previous = previous_rows.get(absolute_path)
        if previous and previous.get("sha256") == digest:
            for field in (
                "signer_id", "signer_identity_status", "signer_inference_basis",
                "sign", "sign_candidate", "signer_candidate", "session_candidate",
                "label_inference_basis", "session_id", "session_inference_basis",
                "language", "label_validation_state", "language_validation_state",
                "signer_review_status", "session_review_status", "label_review_status",
                "language_review_status", "recording_quality_status", "human_review_status",
                "metadata_confirmation_status", "signer_confirmation_evidence",
                "session_confirmation_evidence",
                "label_confirmation_evidence", "language_confirmation_evidence",
                "metadata_confirmed_at", "processing_status", "processing_notes",
                "consent_status", "session_validation_status", "signer_validation_status",
            ):
                if previous.get(field):
                    row[field] = previous[field]
            if not check_landmarks and previous.get("landmark_status") == "CHECKED":
                for field in (
                    "landmark_status", "landmark_availability", "landmark_validation_notes",
                    "frames_processed", "hand_detected_frames", "left_hand_detected_frames",
                    "right_hand_detected_frames", "pose_detected_frames", "face_detected_frames",
                    "hand_detected_percentage", "left_hand_detected_percentage",
                    "right_hand_detected_percentage", "pose_detected_percentage",
                    "face_detected_percentage", "sequence_window_count",
                    "model_feature_dimension", "finite_model_input",
                ):
                    if previous.get(field) != "":
                        row[field] = previous[field]
        review_record = existing_reviews.get(row["recording_id"], {})
        for field in (
            "signer_review_status", "session_review_status", "label_review_status",
            "language_review_status", "recording_quality_status", "human_review_status",
            "signer_confirmation_evidence", "session_confirmation_evidence",
            "label_confirmation_evidence",
            "language_confirmation_evidence", "reviewer", "note", "updated_at",
        ):
            if review_record.get(field):
                row[field] = review_record[field]
        if review_record.get("status"):
            row["review_status"] = review_record["status"]
        elif previous:
            row["source_hash_changed"] = "true"
            row["notes"] = "; ".join(filter(None, [
                row["notes"],
                f"Source hash changed since previous import ({previous.get('sha256', 'UNKNOWN')}).",
            ]))
        if digest in known_hashes:
            row["duplicate_content"] = "true"
            row["duplicate_of_recording_ids"] = ";".join(ids_by_hash.get(digest, []))
            row["notes"] = "; ".join(filter(None, [
                row["notes"],
                "Content hash matches an existing recording in recordings.csv/recordings.json.",
            ]))
        imported_rows.append(row)

    duplicate_groups = {
        digest: paths for digest, paths in hash_paths.items() if len(paths) > 1
    }
    by_path = {row["original_path"]: row for row in imported_rows}
    for paths in duplicate_groups.values():
        canonical_id = by_path[sorted(paths)[0]]["recording_id"]
        for source_path in paths:
            row = by_path[source_path]
            row["duplicate_content"] = "true"
            row["duplicate_of_recording_ids"] = ";".join(
                by_path[other]["recording_id"] for other in sorted(paths) if other != source_path
            )
            row["notes"] = "; ".join(filter(None, [row["notes"], f"Exact-content duplicate group; reference {canonical_id}."]))

    landmark_error = ""
    if check_landmarks:
        try:
            detector, _ = _create_holistic_detector(config)
            detector.close()
        except Exception as exc:
            landmark_error = f"{type(exc).__name__}: {exc}"
        if not landmark_error:
            for row in imported_rows:
                if row["video_validation_status"] == "VIDEO_INVALID":
                    row["landmark_status"] = "NOT_RUN_INVALID_VIDEO"
                    continue
                try:
                    features, _, presence, stats = _extract_landmarks_for_video(
                        row["original_path"],
                        config.get("face_landmark_indices", []),
                        config,
                    )
                    feature_matrix = np.asarray(features, dtype=np.float32)
                    presence_matrix = np.asarray(presence, dtype=np.float32)
                    frames_processed = int(stats.get("frame_count", len(features)))
                    if (
                        feature_matrix.ndim != 2
                        or feature_matrix.shape[0] != frames_processed
                        or not np.isfinite(feature_matrix).all()
                    ):
                        raise ValueError(
                            "Model features are malformed or contain NaN/Inf values."
                        )
                    active_groups = [
                        name for name in ("left_hand", "right_hand", "pose", "face")
                        if config.get("feature_selection", {}).get(name, False)
                    ]
                    group_indices = {name: index for index, name in enumerate(active_groups)}
                    hand_columns = [
                        group_indices[name] for name in ("left_hand", "right_hand")
                        if name in group_indices and group_indices[name] < presence_matrix.shape[1]
                    ]
                    hand_frames = (
                        int(np.any(presence_matrix[:, hand_columns] > 0, axis=1).sum())
                        if hand_columns else 0
                    )
                    left_frames = int(stats.get("left_hand_detected", 0))
                    right_frames = int(stats.get("right_hand_detected", 0))
                    denominator = max(frames_processed, 1)
                    sequence_windows = (
                        _window_starts(
                            frames_processed,
                            int(config.get("sequence_length", 30)),
                            int(config.get("stride", 15)),
                            bool(config.get("allow_padding", True)),
                        )
                        if frames_processed >= int(config.get("min_frames", 12))
                        else []
                    )
                    row.update({
                        "landmark_status": "CHECKED",
                        "landmark_availability": (
                            "HANDS_PRESENT" if hand_frames else "NO_HAND_LANDMARKS"
                        ),
                        "frames_processed": frames_processed,
                        "hand_detected_frames": hand_frames,
                        "left_hand_detected_frames": left_frames,
                        "right_hand_detected_frames": right_frames,
                        "pose_detected_frames": int(stats.get("pose_detected", 0)),
                        "face_detected_frames": int(stats.get("face_detected", 0)),
                        "hand_detected_percentage": round(100.0 * hand_frames / denominator, 3),
                        "left_hand_detected_percentage": round(100.0 * left_frames / denominator, 3),
                        "right_hand_detected_percentage": round(100.0 * right_frames / denominator, 3),
                        "pose_detected_percentage": round(
                            100.0 * int(stats.get("pose_detected", 0)) / denominator, 3
                        ),
                        "face_detected_percentage": round(
                            100.0 * int(stats.get("face_detected", 0)) / denominator, 3
                        ),
                        "sequence_window_count": len(sequence_windows),
                        "model_feature_dimension": int(feature_matrix.shape[1]),
                        "finite_model_input": "true",
                        "landmark_validation_notes": "",
                    })
                except Exception as exc:
                    row["landmark_status"] = "ERROR"
                    row["landmark_availability"] = "QC_ERROR"
                    row["finite_model_input"] = "false"
                    row["landmark_validation_notes"] = f"{type(exc).__name__}: {exc}"
    for row in imported_rows:
        if landmark_error:
            row["landmark_status"] = "UNAVAILABLE"
            row["landmark_availability"] = "MODEL_UNAVAILABLE"
            row["landmark_validation_notes"] = landmark_error
        signer = signers.get(str(row["signer_id"]), {})
        session = sessions.get(f"{row['signer_id']}/{row['session_id']}", {})
        row["consent_status"] = signer.get("consent_status", "UNKNOWN")
        row["session_validation_status"] = (
            "ACTIVE" if session.get("status") == "ACTIVE" else "UNKNOWN"
        )
        row["signer_validation_status"] = (
            row.get("signer_review_status", "UNKNOWN")
            if row.get("signer_review_status") in SIGNER_REVIEW_STATES
            else "UNKNOWN"
        )
        if row["video_validation_status"] == "VIDEO_INVALID":
            row["technical_qc_status"] = "VIDEO_INVALID"
        elif row["landmark_status"] == "ERROR":
            row["technical_qc_status"] = "QC_ERROR"
        elif row["landmark_status"] != "CHECKED":
            row["technical_qc_status"] = "MODEL_UNAVAILABLE"
        elif not int(row.get("hand_detected_frames") or 0):
            row["technical_qc_status"] = "READABLE_BUT_LOW_HAND_DETECTION"
        else:
            row["technical_qc_status"] = "READY_FOR_REVIEW"
        row["validation_status"] = _final_status(row, signers, sessions, vocabulary)

    manifest_fields = [
        "original_path", "filename", "extension", "file_size_bytes", "recording_id",
        "sign", "sign_candidate", "signer_id", "signer_candidate",
        "session_id", "session_candidate", "source_type", "language",
        "signer_identity_status", "signer_inference_basis", "label_inference_basis",
        "session_inference_basis", "legacy_flat_layout", "label_validation_state",
        "language_validation_state", "signer_review_status", "session_review_status",
        "label_review_status", "language_review_status", "recording_quality_status",
        "human_review_status", "metadata_confirmation_status",
        "signer_confirmation_evidence", "session_confirmation_evidence",
        "label_confirmation_evidence",
        "language_confirmation_evidence", "metadata_confirmed_at", "source_hash_changed",
        "consent_status", "session_validation_status", "signer_validation_status",
        "review_status", "validation_status", "technical_qc_status",
        "video_validation_status", "landmark_status", "landmark_availability",
        "landmark_validation_notes", "frames_processed", "hand_detected_percentage",
        "left_hand_detected_percentage", "right_hand_detected_percentage",
        "pose_detected_percentage", "face_detected_percentage",
        "sequence_window_count", "model_feature_dimension", "finite_model_input",
        "hand_detected_frames", "left_hand_detected_frames", "right_hand_detected_frames",
        "pose_detected_frames", "face_detected_frames", "processing_status", "sha256",
        "duration_seconds", "frame_count", "reported_frame_count", "fps", "resolution",
        "width", "height", "represented_in_recordings_csv", "already_processed",
        "processing_notes",
        "path_already_in_recordings_manifest", "content_already_in_recordings_manifest",
        "duplicate_path", "duplicate_content",
        "duplicate_of_recording_ids", "notes", "path_roots",
    ]
    _write_csv(manifest_root / IMPORT_FILENAME, manifest_fields, imported_rows)
    review_fields = [
        "recording_id", "original_path", "status", "reviewer", "note", "updated_at",
        "signer_review_status", "session_review_status", "label_review_status",
        "language_review_status", "recording_quality_status", "human_review_status",
        "signer_confirmation_evidence", "label_confirmation_evidence",
        "language_confirmation_evidence",
    ]
    review_rows = []
    for row in imported_rows:
        review = existing_reviews.get(row["recording_id"], {})
        review_rows.append({
            "recording_id": row["recording_id"],
            "original_path": row["original_path"],
            "status": review.get("status", "PENDING"),
            "reviewer": review.get("reviewer", ""),
            "note": review.get("note", ""),
            "updated_at": review.get("updated_at", ""),
            **{
                field: row.get(field, review.get(field, ""))
                for field in (
                    "signer_review_status", "session_review_status", "label_review_status",
                    "language_review_status", "recording_quality_status", "human_review_status",
                    "signer_confirmation_evidence", "label_confirmation_evidence",
                    "language_confirmation_evidence",
                )
            },
        })
    _write_csv(Path(review_dir) / REVIEW_FILENAME, review_fields, review_rows)

    grouped_signs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in imported_rows:
        grouped_signs[row["sign"]].append(row)
    review_counts = Counter(row["status"] for row in review_rows)
    duplicate_hashes = set(duplicate_groups)
    duplicate_hashes.update(
        row["sha256"] for row in imported_rows
        if row["content_already_in_recordings_manifest"] == "true"
    )
    blockers_by_recording = {
        row["recording_id"]: _processing_blockers(row, signers, sessions, vocabulary)
        for row in imported_rows
    }
    summary = {
        "total_videos": len(imported_rows),
        "readable_videos": sum(row["video_validation_status"] == "READABLE" for row in imported_rows),
        "unreadable_videos": sum(row["video_validation_status"] == "VIDEO_INVALID" for row in imported_rows),
        "duplicate_content_files": sum(row["duplicate_content"] == "true" for row in imported_rows),
        "duplicate_content_groups": len(duplicate_hashes),
        "files_already_represented_in_recordings_csv": sum(
            row["represented_in_recordings_csv"] == "true" for row in imported_rows
        ),
        "paths_already_represented_in_recordings_manifest": sum(
            row["path_already_in_recordings_manifest"] == "true" for row in imported_rows
        ),
        "duplicate_paths": sum(row["duplicate_path"] == "true" for row in imported_rows),
        "identifiable_label_candidates": sum(row["sign"] != "UNKNOWN" for row in imported_rows),
        "human_validated_labels": sum(row["label_validation_state"] == "VALIDATED" for row in imported_rows),
        "suspected_signers": sorted({row["signer_id"] for row in imported_rows if row["signer_id"] != "UNKNOWN"}),
        "confirmed_signers": sorted({
            row["signer_id"] for row in imported_rows if row["signer_identity_status"] == "CONFIRMED"
        }),
        "videos_with_unknown_signer": sum(row["signer_id"] == "UNKNOWN" for row in imported_rows),
        "videos_with_inferred_unconfirmed_signer": sum(row["signer_identity_status"] == "INFERRED_UNCONFIRMED" for row in imported_rows),
        "videos_with_unknown_session": sum(row["session_id"] == "UNKNOWN" for row in imported_rows),
        "number_of_signs": len([sign for sign in grouped_signs if sign != "UNKNOWN"]),
        "videos_per_sign": {
            sign: len(rows) for sign, rows in sorted(grouped_signs.items()) if sign != "UNKNOWN"
        },
        "videos_per_suspected_signer": dict(sorted(Counter(
            row["signer_id"] for row in imported_rows if row["signer_id"] != "UNKNOWN"
        ).items())),
        "session_count": len({row["session_id"] for row in imported_rows if row["session_id"] != "UNKNOWN"}),
        "videos_pending_review": review_counts.get("PENDING", 0),
        "videos_approved": review_counts.get("APPROVED", 0),
        "videos_rejected": review_counts.get("REJECTED", 0),
        "videos_needs_retake": review_counts.get("NEEDS_RETAKE", 0),
        "ready_for_processing": sum(row["validation_status"] == "READY_FOR_PROCESSING" for row in imported_rows),
        "human_approved": sum(row.get("human_review_status") == "APPROVED" for row in imported_rows),
        "human_rejected": sum(row.get("human_review_status") == "REJECTED" for row in imported_rows),
        "human_needs_review": sum(row.get("human_review_status") == "NEEDS_REVIEW" for row in imported_rows),
        "successfully_processed": sum(row["processing_status"] == "PROCESSED" for row in imported_rows),
        "processing_failures": sum(row["processing_status"].startswith("FAILED") for row in imported_rows),
        "validation_status_counts": dict(sorted(Counter(row["validation_status"] for row in imported_rows).items())),
        "landmark_qc_status_counts": dict(sorted(Counter(row["landmark_status"] for row in imported_rows).items())),
        "technical_qc_status_counts": dict(sorted(Counter(
            row["technical_qc_status"] for row in imported_rows
        ).items())),
        "videos_with_hand_detections": sum(
            int(row.get("hand_detected_frames") or 0) > 0 for row in imported_rows
        ),
        "videos_with_low_or_no_hand_detection": sum(
            row["technical_qc_status"] == "READABLE_BUT_LOW_HAND_DETECTION"
            for row in imported_rows
        ),
        "videos_technically_ready": sum(
            row["technical_qc_status"] in {"READY_FOR_REVIEW", "READY_FOR_PROCESSING"}
            for row in imported_rows
        ),
        "frames_processed": sum(int(row.get("frames_processed") or 0) for row in imported_rows),
        "sequence_window_count": sum(
            int(row.get("sequence_window_count") or 0) for row in imported_rows
        ),
        "blocked_missing_signer": sum(
            "SIGNER_CONFIRMATION_REQUIRED" in blockers_by_recording[row["recording_id"]]
            or "SIGNER_CONFIRMATION_EVIDENCE_REQUIRED" in blockers_by_recording[row["recording_id"]]
            for row in imported_rows
        ),
        "blocked_missing_consent": sum(
            "CONSENT_NOT_GRANTED" in blockers_by_recording[row["recording_id"]]
            for row in imported_rows
        ),
        "blocked_missing_session": sum(
            "ACTIVE_SESSION_CONFIRMATION_REQUIRED" in blockers_by_recording[row["recording_id"]]
            or "SESSION_CONFIRMATION_EVIDENCE_REQUIRED" in blockers_by_recording[row["recording_id"]]
            for row in imported_rows
        ),
        "blocked_missing_msl_validation": sum(
            "VOCABULARY_VALIDATION_REQUIRED" in blockers_by_recording[row["recording_id"]]
            or "MSL_LANGUAGE_CONFIRMATION_REQUIRED" in blockers_by_recording[row["recording_id"]]
            or "LABEL_CONFIRMATION_EVIDENCE_REQUIRED" in blockers_by_recording[row["recording_id"]]
            or "LANGUAGE_CONFIRMATION_EVIDENCE_REQUIRED" in blockers_by_recording[row["recording_id"]]
            for row in imported_rows
        ),
        "blocked_by_review": sum(
            "HUMAN_REVIEW_APPROVAL_REQUIRED" in blockers_by_recording[row["recording_id"]]
            or "RECORDING_QUALITY_APPROVAL_REQUIRED" in blockers_by_recording[row["recording_id"]]
            for row in imported_rows
        ),
        "processing_blockers": blockers_by_recording,
        "duplicate_groups": {
            digest: sorted(paths) for digest, paths in sorted(duplicate_groups.items())
        },
        "source_roots": [str(root) for root in roots],
        "holistic_asset_error": landmark_error,
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "existing_recordings_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_existing_recordings_report(manifest_root, imported_rows, summary)
    return {"records": imported_rows, "summary": summary}


def update_existing_recording_review(
    recording_id: str,
    status: str | None,
    reviewer: str,
    note: str,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    *,
    signer_review_status: str | None = None,
    confirmed_signer_id: str | None = None,
    session_review_status: str | None = None,
    confirmed_session_id: str | None = None,
    label_review_status: str | None = None,
    confirmed_sign: str | None = None,
    language_review_status: str | None = None,
    recording_quality_status: str | None = None,
    human_review_status: str | None = None,
) -> dict[str, str]:
    allowed = {"PENDING", "APPROVED", "REJECTED", "NEEDS_RETAKE"}
    normalized_status = status.upper() if status else None
    if normalized_status is not None and normalized_status not in allowed:
        raise ValueError(f"Review status must be one of {sorted(allowed)}.")
    if not reviewer.strip():
        raise ValueError("A human reviewer ID is required.")
    decisions = {
        "signer_review_status": (signer_review_status, SIGNER_REVIEW_STATES),
        "session_review_status": (session_review_status, SESSION_REVIEW_STATES),
        "label_review_status": (label_review_status, LABEL_REVIEW_STATES),
        "language_review_status": (language_review_status, LANGUAGE_REVIEW_STATES),
        "recording_quality_status": (recording_quality_status, QUALITY_REVIEW_STATES),
        "human_review_status": (human_review_status, HUMAN_REVIEW_STATES),
    }
    normalized_decisions: dict[str, str] = {}
    for field, (value, valid_values) in decisions.items():
        if value is not None:
            normalized = value.upper()
            if normalized not in valid_values:
                raise ValueError(f"{field} must be one of {sorted(valid_values)}.")
            normalized_decisions[field] = normalized
    if normalized_status is None and not normalized_decisions:
        raise ValueError("Provide a review status or at least one explicit validation decision.")
    if normalized_decisions and not note.strip():
        raise ValueError("A reviewer evidence note is required when recording explicit decisions.")
    manifest_path = Path(manifests_dir) / IMPORT_FILENAME
    rows = _read_csv(manifest_path)
    target_index = next(
        (index for index, row in enumerate(rows) if row.get("recording_id") == recording_id),
        None,
    )
    if target_index is None:
        raise ValueError(f"Imported recording {recording_id} is not in {manifest_path}.")
    target = rows[target_index]
    if normalized_decisions.get("signer_review_status") == "CONFIRMED":
        signer_id = (confirmed_signer_id or target.get("signer_id", "")).strip()
        if not SIGNER_PATTERN.fullmatch(signer_id):
            raise ValueError("A valid --confirmed-signer is required when confirming signer identity.")
        target["signer_id"] = signer_id
        target["signer_identity_status"] = "CONFIRMED"
        target["signer_confirmation_evidence"] = note.strip()
    elif normalized_decisions.get("signer_review_status") in {"REJECTED", "UNKNOWN"}:
        target["signer_id"] = "UNKNOWN"
        target["signer_identity_status"] = normalized_decisions["signer_review_status"]
        target["signer_confirmation_evidence"] = ""
    if normalized_decisions.get("session_review_status") == "CONFIRMED":
        session_id = (confirmed_session_id or target.get("session_id", "")).strip()
        if not SESSION_PATTERN.fullmatch(session_id):
            raise ValueError("A valid --confirmed-session is required when confirming a session.")
        target["session_id"] = session_id
        target["session_inference_basis"] = "human-confirmed against registered session"
        target["session_confirmation_evidence"] = note.strip()
    elif normalized_decisions.get("session_review_status") in {"REJECTED", "UNKNOWN"}:
        target["session_id"] = "UNKNOWN"
        target["session_inference_basis"] = "human review rejected or could not confirm source session"
        target["session_confirmation_evidence"] = ""
    if normalized_decisions.get("label_review_status") == "CONFIRMED":
        sign = (confirmed_sign or target.get("sign", "")).strip().upper()
        if not SIGN_PATTERN.fullmatch(sign):
            raise ValueError("A valid --confirmed-sign is required when confirming the sign label.")
        vocabulary_path = Path(manifests_dir) / "sign_vocabulary.json"
        vocabulary = (
            json.loads(vocabulary_path.read_text(encoding="utf-8"))
            if vocabulary_path.is_file()
            else {}
        )
        entry = vocabulary.get(sign, {})
        target["sign"] = sign
        target["label_review_status"] = "CONFIRMED"
        target["label_confirmation_evidence"] = note.strip()
        if (
            entry.get("language") != "MSL"
            or entry.get("status") != "ACTIVE"
            or entry.get("human_validation") != "VALIDATED"
        ):
            target["label_validation_state"] = "VOCABULARY_VALIDATION_REQUIRED"
        else:
            target["label_validation_state"] = "VALIDATED"
    elif normalized_decisions.get("label_review_status") == "REJECTED":
        target["sign"] = "UNKNOWN"
        target["label_validation_state"] = "REJECTED"
        target["label_confirmation_evidence"] = ""
    elif normalized_decisions.get("label_review_status") == "NEEDS_REVIEW":
        target["label_validation_state"] = "UNVALIDATED"
        target["label_confirmation_evidence"] = ""
    if normalized_decisions.get("language_review_status") == "MSL":
        target["language"] = "MSL"
        target["language_validation_state"] = "VALIDATED"
        target["language_confirmation_evidence"] = note.strip()
    elif normalized_decisions.get("language_review_status") == "NOT_MSL":
        target["language"] = "NOT_MSL"
        target["language_validation_state"] = "REJECTED"
        target["language_confirmation_evidence"] = note.strip()
    elif normalized_decisions.get("language_review_status") == "UNKNOWN":
        target["language"] = "UNKNOWN"
        target["language_validation_state"] = "UNVERIFIED"
        target["language_confirmation_evidence"] = ""
    target.update(normalized_decisions)
    if normalized_status is not None:
        target["review_status"] = normalized_status
        if "human_review_status" not in normalized_decisions:
            target["human_review_status"] = {
                "APPROVED": "APPROVED",
                "REJECTED": "REJECTED",
                "NEEDS_RETAKE": "NEEDS_REVIEW",
                "PENDING": "NEEDS_REVIEW",
            }[normalized_status]
    human_status = target.get("human_review_status", "NEEDS_REVIEW")
    quality_status = target.get("recording_quality_status", "UNKNOWN")
    if human_status == "APPROVED" and quality_status == "APPROVED":
        target["review_status"] = "APPROVED"
    elif human_status == "REJECTED" or quality_status == "REJECTED":
        target["review_status"] = "REJECTED"
    elif quality_status == "NEEDS_RETAKE":
        target["review_status"] = "NEEDS_RETAKE"
    elif human_status == "NEEDS_REVIEW":
        target["review_status"] = "PENDING"
    if normalized_decisions or normalized_status is not None:
        target["metadata_confirmation_status"] = "PENDING"
        target["metadata_confirmed_at"] = ""
    from datetime import datetime, timezone

    updated = {
        "recording_id": recording_id,
        "original_path": target["original_path"],
        "status": target.get("review_status", "PENDING"),
        "reviewer": reviewer.strip(),
        "note": note.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{
            field: target.get(field, "")
            for field in (
                "signer_review_status", "session_review_status", "label_review_status",
                "language_review_status", "recording_quality_status", "human_review_status",
                "signer_confirmation_evidence", "session_confirmation_evidence",
                "label_confirmation_evidence",
                "language_confirmation_evidence",
            )
        },
    }
    review_path = Path(review_dir) / REVIEW_FILENAME
    reviews = _read_csv(review_path)
    found = False
    for row in reviews:
        if row.get("recording_id") == recording_id:
            row.update(updated)
            found = True
            break
    if not found:
        reviews.append(updated)
    review_fields = list(dict.fromkeys(
        (list(reviews[0].keys()) if reviews else []) + list(updated.keys())
    ))
    signers = load_signers(manifests_dir)
    sessions = load_sessions(manifests_dir)
    vocabulary_path = Path(manifests_dir) / "sign_vocabulary.json"
    vocabulary = (
        json.loads(vocabulary_path.read_text(encoding="utf-8"))
        if vocabulary_path.is_file()
        else {}
    )
    target["consent_status"] = signers.get(target.get("signer_id", ""), {}).get(
        "consent_status", "UNKNOWN"
    )
    target["signer_validation_status"] = target.get("signer_review_status", "UNKNOWN")
    target["session_validation_status"] = (
        "ACTIVE"
        if sessions.get(
            f"{target.get('signer_id', '')}/{target.get('session_id', '')}", {}
        ).get("status") == "ACTIVE"
        else "UNKNOWN"
    )
    blockers = _processing_blockers(target, signers, sessions, vocabulary)
    if not blockers:
        rows[target_index] = target
        _write_csv(
            manifest_path,
            list(dict.fromkeys(list(rows[0].keys()) + list(target.keys()))),
            rows,
        )
        target = confirm_existing_recording_metadata(
            recording_id,
            target["signer_id"],
            target["sign"],
            target["session_id"],
            "MSL",
            target.get("signer_confirmation_evidence", ""),
            target.get("label_confirmation_evidence", ""),
            target.get("language_confirmation_evidence", ""),
            manifests_dir,
            session_evidence=target.get("session_confirmation_evidence", ""),
        )
        updated["status"] = target["review_status"]
        updated.update({
            field: target.get(field, "")
            for field in ("signer_review_status", "session_review_status", "label_review_status",
                          "language_review_status", "recording_quality_status", "human_review_status")
        })
    updated["status"] = target.get("review_status", updated["status"])
    for review_row in reviews:
        if review_row.get("recording_id") == recording_id:
            review_row.update(updated)
            break
    _write_csv(review_path, review_fields, reviews)
    target["validation_status"] = _final_status(target, signers, sessions, vocabulary)
    rows[target_index] = target
    manifest_fields = list(dict.fromkeys(
        (list(rows[0].keys()) if rows else []) + list(target.keys())
    ))
    _write_csv(manifest_path, manifest_fields, rows)
    summary_path = Path(manifests_dir) / "existing_recordings_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["videos_pending_review"] = sum(row.get("status") == "PENDING" for row in reviews)
        summary["videos_approved"] = sum(row.get("status") == "APPROVED" for row in reviews)
        summary["videos_rejected"] = sum(row.get("status") == "REJECTED" for row in reviews)
        summary["videos_needs_retake"] = sum(row.get("status") == "NEEDS_RETAKE" for row in reviews)
        summary["human_approved"] = sum(row.get("human_review_status") == "APPROVED" for row in rows)
        summary["human_rejected"] = sum(row.get("human_review_status") == "REJECTED" for row in rows)
        summary["human_needs_review"] = sum(row.get("human_review_status") == "NEEDS_REVIEW" for row in rows)
        summary["ready_for_processing"] = sum(
            row.get("validation_status") == "READY_FOR_PROCESSING" for row in rows
        )
        current_blockers = {
            row["recording_id"]: _processing_blockers(row, signers, sessions, vocabulary)
            for row in rows
        }
        summary["processing_blockers"] = current_blockers
        summary["blocked_missing_signer"] = sum(
            "SIGNER_CONFIRMATION_REQUIRED" in value
            or "SIGNER_CONFIRMATION_EVIDENCE_REQUIRED" in value
            for value in current_blockers.values()
        )
        summary["blocked_missing_consent"] = sum(
            "CONSENT_NOT_GRANTED" in value for value in current_blockers.values()
        )
        summary["blocked_missing_session"] = sum(
            "ACTIVE_SESSION_CONFIRMATION_REQUIRED" in value
            or "SESSION_CONFIRMATION_EVIDENCE_REQUIRED" in value
            for value in current_blockers.values()
        )
        summary["blocked_missing_msl_validation"] = sum(
            "VOCABULARY_VALIDATION_REQUIRED" in value
            or "MSL_LANGUAGE_CONFIRMATION_REQUIRED" in value
            or "LABEL_CONFIRMATION_EVIDENCE_REQUIRED" in value
            or "LANGUAGE_CONFIRMATION_EVIDENCE_REQUIRED" in value
            for value in current_blockers.values()
        )
        summary["blocked_by_review"] = sum(
            "HUMAN_REVIEW_APPROVAL_REQUIRED" in value
            or "RECORDING_QUALITY_APPROVAL_REQUIRED" in value
            for value in current_blockers.values()
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _write_existing_recordings_report(Path(manifests_dir), rows, summary)
    updated.update(target)
    updated["status"] = target.get("review_status", updated.get("status", "PENDING"))
    updated["processing_blockers"] = ";".join(
        _processing_blockers(target, signers, sessions, vocabulary)
    )
    return updated


def confirm_existing_recording_metadata(
    recording_id: str,
    signer_id: str,
    sign: str,
    session_id: str,
    language: str,
    signer_evidence: str,
    label_evidence: str,
    language_evidence: str,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
    *,
    session_evidence: str = "",
) -> dict[str, str]:
    manifest_root = Path(manifests_dir)
    manifest_path = manifest_root / IMPORT_FILENAME
    rows = _read_csv(manifest_path)
    row = next((item for item in rows if item.get("recording_id") == recording_id), None)
    if row is None:
        raise ValueError(f"Imported recording {recording_id} is not in {manifest_path}.")
    signer_id = signer_id.strip()
    sign = sign.strip().upper()
    session_id = session_id.strip()
    language = language.strip().upper()
    if not SIGNER_PATTERN.fullmatch(signer_id):
        raise ValueError(f"Invalid signer identifier: {signer_id!r}.")
    if not SIGN_PATTERN.fullmatch(sign):
        raise ValueError(f"Invalid sign identifier: {sign!r}.")
    if not SESSION_PATTERN.fullmatch(session_id):
        raise ValueError(f"Invalid session identifier: {session_id!r}.")
    if language != "MSL":
        raise ValueError("Only human-confirmed MSL labels can be enabled for this MSL pipeline.")
    if not all((
        signer_evidence.strip(),
        session_evidence.strip(),
        label_evidence.strip(),
        language_evidence.strip(),
    )):
        raise ValueError("Signer, session, label, and language evidence are all required.")
    source = Path(row["original_path"])
    if not source.is_file() or _sha256(source) != row.get("sha256"):
        raise ValueError("Source is missing or its SHA-256 changed; rerun the audit and review it again.")
    if row.get("source_hash_changed") == "true":
        raise ValueError("Source hash changed since import; human review is required before confirmation.")
    if row.get("video_validation_status") != "READABLE":
        raise ValueError("Only readable videos can be metadata-confirmed for preprocessing.")
    if row.get("landmark_status") != "CHECKED" or int(row.get("hand_detected_frames") or 0) == 0:
        raise ValueError("Holistic QC must confirm usable hand landmarks before preprocessing.")
    if row.get("review_status") != "APPROVED":
        raise ValueError("Recording review must be APPROVED before metadata confirmation.")
    required_human_decisions = {
        "signer_review_status": "CONFIRMED",
        "session_review_status": "CONFIRMED",
        "label_review_status": "CONFIRMED",
        "language_review_status": "MSL",
        "recording_quality_status": "APPROVED",
        "human_review_status": "APPROVED",
    }
    for field, required_value in required_human_decisions.items():
        if row.get(field) != required_value:
            raise ValueError(
                f"{field} must be explicitly set to {required_value} before processing eligibility."
            )
    for evidence_field in (
        "signer_confirmation_evidence",
        "session_confirmation_evidence",
        "label_confirmation_evidence",
        "language_confirmation_evidence",
    ):
        if not row.get(evidence_field, "").strip():
            raise ValueError(f"{evidence_field} is required before processing eligibility.")
    if row.get("sign") != sign or row.get("sign_candidate") == "":
        raise ValueError("The confirmed sign must be explicitly recorded and retain its original candidate.")
    signer = load_signers(manifest_root).get(signer_id)
    if (
        not signer
        or signer.get("status") != "ACTIVE"
        or signer.get("consent_status") != "GRANTED"
    ):
        raise ValueError(f"Signer {signer_id} must be active with currently granted consent.")
    session = load_sessions(manifest_root).get(f"{signer_id}/{session_id}")
    if not session or session.get("status") != "ACTIVE":
        raise ValueError(f"Active session {session_id} must be registered for {signer_id}.")
    vocabulary_path = manifest_root / "sign_vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8")) if vocabulary_path.is_file() else {}
    sign_entry = vocabulary.get(sign)
    if (
        not sign_entry
        or sign_entry.get("language") != "MSL"
        or sign_entry.get("status") != "ACTIVE"
        or sign_entry.get("human_validation") != "VALIDATED"
    ):
        raise ValueError(f"Sign {sign} must be human-validated in the MSL vocabulary.")
    row.update({
        "signer_id": signer_id,
        "signer_identity_status": "CONFIRMED",
        "signer_confirmation_evidence": signer_evidence.strip(),
        "session_confirmation_evidence": session_evidence.strip(),
        "sign": sign,
        "label_validation_state": "VALIDATED",
        "label_confirmation_evidence": label_evidence.strip(),
        "session_id": session_id,
        "session_inference_basis": "human-confirmed against collection records",
        "language": language,
        "language_validation_state": "VALIDATED",
        "language_confirmation_evidence": language_evidence.strip(),
        "metadata_confirmation_status": "CONFIRMED",
        "metadata_confirmed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    row["consent_status"] = signer.get("consent_status", "UNKNOWN")
    row["session_validation_status"] = session.get("status", "UNKNOWN")
    row["signer_validation_status"] = "CONFIRMED"
    blockers = _processing_blockers(row, load_signers(manifest_root), load_sessions(manifest_root), vocabulary)
    if blockers:
        row["metadata_confirmation_status"] = "PENDING"
        row["validation_status"] = "METADATA_INCOMPLETE"
        raise ValueError(
            "Recording remains blocked: " + ", ".join(blockers)
        )
    row["validation_status"] = "READY_FOR_PROCESSING"
    _write_csv(manifest_path, list(rows[0]), rows)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and safely index existing videos without moving or editing originals."
    )
    parser.add_argument("--root", action="append", help="Source folder to scan; may be supplied multiple times.")
    parser.add_argument("--config", default="config/msl_dataset.json")
    parser.add_argument("--manifests-dir")
    parser.add_argument("--review-dir")
    parser.add_argument("--skip-landmarks", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    roots = args.root or [config["raw_dataset_path"]]
    result = import_existing_recordings(
        roots,
        args.manifests_dir or config["manifests_path"],
        args.review_dir or config["review_path"],
        args.config,
        check_landmarks=not args.skip_landmarks,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
