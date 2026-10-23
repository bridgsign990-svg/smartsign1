from __future__ import annotations

import csv
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFESTS_DIR = PROJECT_ROOT / "data" / "msl_dataset" / "manifests"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data" / "msl_dataset" / "review"
SIGNER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}\Z")
SESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}\Z")
SIGN_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,39}\Z")
CONSENT_STATES = {"PENDING", "GRANTED", "REVOKED"}
REVIEW_STATES = {"PENDING", "APPROVED", "REJECTED", "NEEDS_RETAKE"}
DEFAULT_QUALITY_THRESHOLDS = {
    "min_frames": 12,
    "max_frames": 300,
    "min_fps": 10.0,
    "max_fps": 60.0,
    "min_width": 320,
    "min_height": 240,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read manifest {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def validate_identifier(value: str, kind: str) -> str:
    normalized = value.strip()
    pattern = {"signer": SIGNER_PATTERN, "session": SESSION_PATTERN, "sign": SIGN_PATTERN}[kind]
    if not pattern.fullmatch(normalized):
        raise ValueError(f"Invalid {kind} identifier: {value!r}.")
    return normalized


def signer_manifest_path(manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> Path:
    return Path(manifests_dir) / "signers.json"


def session_manifest_path(manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> Path:
    return Path(manifests_dir) / "sessions.json"


def recording_manifest_path(manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> Path:
    return Path(manifests_dir) / "recordings.json"


def load_signers(manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> dict[str, dict[str, Any]]:
    value = _read_json(signer_manifest_path(manifests_dir), {})
    if not isinstance(value, dict):
        raise ValueError("Signer manifest must be a JSON object.")
    return value


def load_sessions(manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> dict[str, dict[str, Any]]:
    value = _read_json(session_manifest_path(manifests_dir), {})
    if not isinstance(value, dict):
        raise ValueError("Session manifest must be a JSON object.")
    return value


def load_recordings(manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR) -> list[dict[str, Any]]:
    value = _read_json(recording_manifest_path(manifests_dir), [])
    if not isinstance(value, list):
        raise ValueError("Recording manifest must be a JSON array.")
    return value


def register_signer(
    signer_id: str,
    consent_status: str = "PENDING",
    consent_reference: str = "",
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    signer_id = validate_identifier(signer_id, "signer")
    consent_status = consent_status.upper()
    if consent_status not in CONSENT_STATES:
        raise ValueError(f"Consent status must be one of {sorted(CONSENT_STATES)}.")
    if consent_status == "GRANTED" and not consent_reference.strip():
        raise ValueError("A consent reference is required when recording consent as GRANTED.")
    signers = load_signers(manifests_dir)
    if signer_id in signers:
        raise ValueError(f"Signer {signer_id} is already registered; update consent explicitly instead.")
    signer = {
        "signer_id": signer_id,
        "status": "ACTIVE",
        "consent_status": consent_status,
        "consent_reference": consent_reference.strip(),
        "registered_at": _now(),
    }
    if not dry_run:
        signers[signer_id] = signer
        _write_json(signer_manifest_path(manifests_dir), signers)
        _sync_manifest_csvs(manifests_dir)
    return signer


def update_consent(
    signer_id: str,
    consent_status: str,
    consent_reference: str,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    signer_id = validate_identifier(signer_id, "signer")
    consent_status = consent_status.upper()
    if consent_status not in CONSENT_STATES:
        raise ValueError(f"Consent status must be one of {sorted(CONSENT_STATES)}.")
    if consent_status == "GRANTED" and not consent_reference.strip():
        raise ValueError("A consent reference is required when recording consent as GRANTED.")
    signers = load_signers(manifests_dir)
    if signer_id not in signers:
        raise ValueError(f"Signer {signer_id} is not registered.")
    updated = dict(signers[signer_id])
    updated.update({
        "consent_status": consent_status,
        "consent_reference": consent_reference.strip(),
        "consent_updated_at": _now(),
    })
    if not dry_run:
        signers[signer_id] = updated
        _write_json(signer_manifest_path(manifests_dir), signers)
        _sync_manifest_csvs(manifests_dir)
    return updated


def create_session(
    signer_id: str,
    session_id: str,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
    dry_run: bool = False,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    signer_id = validate_identifier(signer_id, "signer")
    session_id = validate_identifier(session_id, "session")
    signers = load_signers(manifests_dir)
    if signer_id not in signers:
        raise ValueError(f"Signer {signer_id} must be registered before creating a session.")
    if signers[signer_id].get("consent_status") != "GRANTED":
        raise ValueError(f"Signer {signer_id} has no current granted consent.")
    sessions = load_sessions(manifests_dir)
    key = f"{signer_id}/{session_id}"
    if key in sessions:
        raise ValueError(f"Session {key} is already registered.")
    metadata = metadata or {}
    session = {
        "session_id": session_id,
        "signer_id": signer_id,
        "date": metadata.get("date") or datetime.now().date().isoformat(),
        "camera": metadata.get("camera", "UNKNOWN"),
        "resolution": metadata.get("resolution", "UNKNOWN"),
        "fps": metadata.get("fps", "UNKNOWN"),
        "lighting_condition": metadata.get("lighting_condition", "UNKNOWN"),
        "background_condition": metadata.get("background_condition", "UNKNOWN"),
        "orientation": metadata.get("orientation", "UNKNOWN"),
        "notes": metadata.get("notes", ""),
        "status": "ACTIVE",
        "created_at": _now(),
    }
    if not dry_run:
        sessions[key] = session
        _write_json(session_manifest_path(manifests_dir), sessions)
        _sync_manifest_csvs(manifests_dir)
    return session


def require_recording_permission(signer_id: str, session_id: str, manifests_dir: str | Path) -> None:
    signers = load_signers(manifests_dir)
    signer = signers.get(signer_id)
    if not signer:
        raise ValueError(f"Signer {signer_id} is not registered.")
    if signer.get("status") != "ACTIVE" or signer.get("consent_status") != "GRANTED":
        raise ValueError(f"Signer {signer_id} is inactive or does not have granted consent.")
    session = load_sessions(manifests_dir).get(f"{signer_id}/{session_id}")
    if not session or session.get("status") != "ACTIVE":
        raise ValueError(f"Active session {session_id} is not registered for {signer_id}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_recording(
    video_path: str | Path,
    signer_id: str,
    sign: str,
    session_id: str,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
) -> dict[str, Any]:
    video = Path(video_path)
    signer_id = validate_identifier(signer_id, "signer")
    sign = validate_identifier(sign, "sign")
    session_id = validate_identifier(session_id, "session")
    require_recording_permission(signer_id, session_id, manifests_dir)
    if not video.is_file():
        raise FileNotFoundError(f"Recording does not exist: {video}")
    if (
        video.parent.name != session_id
        or video.parent.parent.name.upper() != sign
        or video.parent.parent.parent.name != signer_id
    ):
        raise ValueError(
            "Recording path must follow <raw>/<signer>/<sign>/<session>/<video> "
            "and match its registered metadata."
        )
    recordings = load_recordings(manifests_dir)
    resolved = str(video.resolve())
    content_hash = _sha256(video)
    if any(row.get("file_path") == resolved or row.get("sha256") == content_hash for row in recordings):
        raise ValueError(f"Duplicate recording path or content is already registered: {video}")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Video cannot be opened: {video}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _frame = cap.read()
        if not ok or frame_count < 1 or width < 1 or height < 1:
            raise ValueError(f"Video has no decodable frames or invalid dimensions: {video}")
    finally:
        cap.release()
    recording = {
        "recording_id": str(uuid.uuid4()),
        "signer_id": signer_id,
        "sign": validate_identifier(sign, "sign"),
        "session_id": session_id,
        "video_id": video.stem,
        "file_path": resolved,
        "sha256": content_hash,
        "file_size": video.stat().st_size,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "recorded_at": _now(),
        "validation_status": "PENDING_VALIDATION",
    }
    recordings.append(recording)
    _write_json(recording_manifest_path(manifests_dir), recordings)
    _sync_manifest_csvs(manifests_dir)
    return recording


def validate_recordings(
    raw_dir: str | Path,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
    allowed_extensions: set[str] | None = None,
    min_frames: int = DEFAULT_QUALITY_THRESHOLDS["min_frames"],
    max_frames: int | None = DEFAULT_QUALITY_THRESHOLDS["max_frames"],
    min_fps: float = DEFAULT_QUALITY_THRESHOLDS["min_fps"],
    max_fps: float = DEFAULT_QUALITY_THRESHOLDS["max_fps"],
    min_width: int = DEFAULT_QUALITY_THRESHOLDS["min_width"],
    min_height: int = DEFAULT_QUALITY_THRESHOLDS["min_height"],
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"MSL raw dataset directory does not exist: {root}")
    if min_frames < 1 or (max_frames is not None and max_frames < min_frames):
        raise ValueError("Frame limits must be positive and max_frames cannot be below min_frames.")
    if min_fps <= 0 or max_fps < min_fps or min_width < 1 or min_height < 1:
        raise ValueError("FPS and resolution thresholds must be positive and ordered.")
    allowed = allowed_extensions or {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    signers = load_signers(manifests_dir)
    sessions = load_sessions(manifests_dir)
    registered = load_recordings(manifests_dir)
    recordings = list(registered)
    registered_by_path = {str(Path(str(row.get("file_path", ""))).resolve()): row for row in registered if row.get("file_path")}
    known_hashes = {str(row.get("sha256")) for row in registered if row.get("sha256")}
    results: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for path in sorted(root.rglob("*")) if root.exists() else []:
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        rel = path.relative_to(root)
        issue = ""
        if len(rel.parts) != 4:
            issue = "INVALID_LAYOUT"
            signer_id = rel.parts[0] if rel.parts else ""
            sign = rel.parts[1].upper() if len(rel.parts) > 1 else ""
            session_id = rel.parts[2] if len(rel.parts) > 3 else ""
        else:
            signer_id, sign, session_id = rel.parts[0], rel.parts[1].upper(), rel.parts[2]
            if not SIGNER_PATTERN.fullmatch(signer_id) or not SIGN_PATTERN.fullmatch(sign) or not SESSION_PATTERN.fullmatch(session_id):
                issue = "INVALID_IDENTIFIER"
        content_hash = _sha256(path)
        if not issue and signer_id not in signers:
            issue = "UNREGISTERED_SIGNER"
        if not issue and (
            signers[signer_id].get("status") != "ACTIVE"
            or signers[signer_id].get("consent_status") != "GRANTED"
        ):
            issue = "CONSENT_NOT_GRANTED"
        if not issue and (
            f"{signer_id}/{session_id}" not in sessions
            or sessions[f"{signer_id}/{session_id}"].get("status") != "ACTIVE"
        ):
            issue = "UNREGISTERED_SESSION"
        existing_registration = registered_by_path.get(str(path.resolve()))
        if not issue and existing_registration and existing_registration.get("sha256") != content_hash:
            issue = "RAW_HASH_CHANGED"
        if not issue and (
            content_hash in seen_hashes
            or (content_hash in known_hashes and not (existing_registration and existing_registration.get("sha256") == content_hash))
        ):
            issue = "DUPLICATE_RECORDING"
        cap = cv2.VideoCapture(str(path))
        try:
            opened = cap.isOpened()
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
            decoded_frames = 0
            if opened:
                while True:
                    decoded, _frame = cap.read()
                    if not decoded:
                        break
                    decoded_frames += 1
        finally:
            cap.release()
        if not issue and not opened:
            issue = "VIDEO_UNREADABLE"
        if not issue and decoded_frames < min_frames:
            issue = "TOO_FEW_FRAMES"
        if not issue and (width < min_width or height < min_height):
            issue = "LOW_RESOLUTION"
        if not issue and max_frames is not None and (frames > max_frames or decoded_frames > max_frames):
            issue = "FRAME_LIMIT_EXCEEDED"
        if not issue and (fps < min_fps or fps > max_fps):
            issue = "FPS_OUT_OF_RANGE"
        if not issue and frames != decoded_frames:
            issue = "FRAME_COUNT_MISMATCH"
        warnings = []
        if fps <= 0 or fps < 5 or fps > 120:
            warnings.append("FPS unavailable or unusual.")
        if frames and decoded_frames != frames:
            warnings.append(f"Decoded {decoded_frames} frames while container metadata reports {frames}.")
        row = {
            "file_path": str(path.resolve()),
            "signer_id": signer_id,
            "sign": sign,
            "session_id": session_id,
            "sha256": content_hash,
            "frame_count": frames,
            "fps": fps,
            "width": width,
            "height": height,
            "status": issue or "VALID",
            "warnings": "; ".join(warnings),
        }
        if issue == "RAW_HASH_CHANGED":
            warnings.append("Original accepted recording changed; it remains immutable and was not re-registered.")
            row["warnings"] = "; ".join(warnings)
        results.append(row)
        seen_hashes.add(content_hash)
        if not dry_run and existing_registration:
            for registration in recordings:
                if registration.get("file_path") == str(path.resolve()):
                    registration["validation_status"] = row["status"]
                    if row["status"] == "VALID" and registration.get("sha256") == content_hash:
                        registration.update({
                            "fps": fps,
                            "frame_count": decoded_frames,
                            "width": width,
                            "height": height,
                        })
                    break
        if not dry_run and row["status"] == "VALID":
            if existing_registration and existing_registration.get("sha256") == content_hash:
                continue
            recordings.append({
                "recording_id": str(uuid.uuid4()),
                "signer_id": signer_id,
                "sign": sign,
                "session_id": session_id,
                "video_id": path.stem,
                "file_path": str(path.resolve()),
                "sha256": content_hash,
                "file_size": path.stat().st_size,
                "fps": fps,
                "frame_count": frames,
                "width": width,
                "height": height,
                "recorded_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "validation_status": "VALID",
            })
    if not dry_run:
        path = Path(manifests_dir) / "recording_validation.csv"
        _write_csv(path, list(results[0]) if results else [
            "file_path", "signer_id", "sign", "session_id", "sha256", "frame_count", "fps", "width", "height", "status"
        ], results)
        _write_json(recording_manifest_path(manifests_dir), recordings)
        _sync_manifest_csvs(manifests_dir)
    return {"checked": len(results), "valid": sum(row["status"] == "VALID" for row in results), "invalid": [row for row in results if row["status"] != "VALID"], "records": results}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sync_manifest_csvs(manifests_dir: str | Path) -> None:
    root = Path(manifests_dir)
    signers = load_signers(root)
    sessions = load_sessions(root)
    recordings = load_recordings(root)
    signer_fields = ["signer_id", "status", "consent_status", "consent_reference", "registered_at", "consent_updated_at"]
    session_fields = [
        "session_id", "signer_id", "date", "camera", "resolution", "fps",
        "lighting_condition", "background_condition", "orientation", "notes", "status", "created_at",
    ]
    recording_fields = [
        "recording_id", "signer_id", "sign", "session_id", "video_id", "file_path",
        "sha256", "file_size", "fps", "frame_count", "width", "height", "recorded_at", "validation_status",
    ]
    _write_csv(root / "signers.csv", signer_fields, list(signers.values()))
    _write_csv(root / "sessions.csv", session_fields, list(sessions.values()))
    _write_csv(root / "recordings.csv", recording_fields, recordings)


def upsert_vocabulary_sign(
    sign_id: str,
    canonical_name: str,
    display_name: str,
    description: str,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
) -> dict[str, Any]:
    sign_id = validate_identifier(sign_id.upper(), "sign")
    if not canonical_name.strip() or not display_name.strip():
        raise ValueError("Canonical and display names are required.")
    path = Path(manifests_dir) / "sign_vocabulary.json"
    entries = _read_json(path, {})
    if not isinstance(entries, dict):
        raise ValueError("Sign vocabulary must be a JSON object keyed by sign_id.")
    if sign_id in entries:
        raise ValueError(f"Sign {sign_id} already exists in the controlled vocabulary.")
    entry = {
        "sign_id": sign_id,
        "canonical_name": canonical_name.strip(),
        "display_name": display_name.strip(),
        "language": "MSL",
        "description": description.strip(),
        "status": "ACTIVE",
        "human_validation": "UNVALIDATED",
    }
    entries[sign_id] = entry
    _write_json(path, entries)
    return entry


def set_sign_validation(
    sign_id: str,
    status: str,
    reviewer: str,
    evidence_note: str,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
) -> dict[str, Any]:
    status = status.upper()
    if status not in {"UNVALIDATED", "REVIEWED", "VALIDATED"}:
        raise ValueError("Human validation must be UNVALIDATED, REVIEWED, or VALIDATED.")
    if status in {"REVIEWED", "VALIDATED"} and (not reviewer.strip() or not evidence_note.strip()):
        raise ValueError("Reviewer and evidence note are required for reviewed or validated status.")
    path = Path(manifests_dir) / "sign_vocabulary.json"
    entries = _read_json(path, {})
    if sign_id not in entries:
        raise ValueError(f"Sign {sign_id} is not registered in the controlled vocabulary.")
    entries[sign_id].update({
        "human_validation": status,
        "validation_reviewer": reviewer.strip(),
        "validation_evidence": evidence_note.strip(),
        "validation_updated_at": _now(),
    })
    _write_json(path, entries)
    return entries[sign_id]


def review_sequence(
    sequence_id: str,
    status: str,
    reviewer: str,
    note: str,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
) -> dict[str, str]:
    status = status.upper()
    if status not in REVIEW_STATES:
        raise ValueError(f"Review status must be one of {sorted(REVIEW_STATES)}.")
    if not reviewer.strip():
        raise ValueError("A human reviewer ID is required.")
    if not sequence_id.strip():
        raise ValueError("A sequence ID is required.")
    sequence_manifest = Path(manifests_dir) / "sequences.csv"
    if not sequence_manifest.is_file():
        raise FileNotFoundError(f"Sequence manifest not found: {sequence_manifest}")
    with sequence_manifest.open("r", newline="", encoding="utf-8") as source:
        if not any(row.get("sequence_id") == sequence_id for row in csv.DictReader(source)):
            raise ValueError(f"Sequence {sequence_id} is not present in the sequence manifest.")
    path = Path(review_dir) / "review_manifest.csv"
    rows = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    updated = {"sequence_id": sequence_id, "status": status, "reviewer": reviewer.strip(), "note": note.strip(), "updated_at": _now()}
    for row in rows:
        if row.get("sequence_id") == sequence_id:
            row.update(updated)
            break
    else:
        rows.append(updated)
    _write_csv(path, list(updated), rows)
    return updated


def collection_status(
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
    dry_run: bool = False,
    targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    signers = load_signers(manifests_dir)
    sessions = load_sessions(manifests_dir)
    recordings = load_recordings(manifests_dir)
    vocabulary = _read_json(Path(manifests_dir) / "sign_vocabulary.json", {})
    failures = _read_json(Path(manifests_dir) / "failures.json", [])
    imported_path = Path(manifests_dir) / "existing_recordings.csv"
    imported = []
    if imported_path.exists():
        with imported_path.open("r", newline="", encoding="utf-8") as source:
            imported = list(csv.DictReader(source))
    review_path = Path(review_dir) / "review_manifest.csv"
    reviews = []
    if review_path.exists():
        with review_path.open("r", newline="", encoding="utf-8") as source:
            reviews = list(csv.DictReader(source))
    recording_review_path = Path(review_dir) / "recording_review.csv"
    recording_reviews = []
    if recording_review_path.exists():
        with recording_review_path.open("r", newline="", encoding="utf-8") as source:
            recording_reviews = list(csv.DictReader(source))
    priorities = collection_priorities(recordings, vocabulary, targets or {})
    review_by_id = {row.get("recording_id", ""): row for row in recording_reviews}
    recordings_needing_review = [
        {
            "recording_id": row.get("recording_id", ""),
            "sign": row.get("sign", "UNKNOWN"),
            "original_path": row.get("original_path", ""),
        }
        for row in imported
        if review_by_id.get(row.get("recording_id", ""), {}).get("status", "PENDING")
        in {"PENDING", "NEEDS_RETAKE"}
    ]
    technical_failures = [
        {
            "recording_id": row.get("recording_id", ""),
            "original_path": row.get("original_path", ""),
            "status": row.get("validation_status", ""),
            "notes": row.get("notes", ""),
        }
        for row in imported
        if row.get("validation_status") == "VIDEO_INVALID"
    ]
    registered_paths = {str(row.get("file_path", "")) for row in recordings}
    combined_recordings = recordings + [
        row for row in imported if str(row.get("original_path", "")) not in registered_paths
    ]
    imported_review_counts = _count_by(recording_reviews, "status")
    split_signers: dict[str, dict[str, set[str]]] = {}
    for split_name in ("train", "validation", "test"):
        split_path = Path(manifests_dir) / f"{split_name}.csv"
        if not split_path.is_file():
            continue
        with split_path.open("r", newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                sign = str(row.get("sign", "")).upper()
                signer_id = str(row.get("signer_id", ""))
                if sign and signer_id:
                    split_signers.setdefault(sign, {}).setdefault(split_name, set()).add(signer_id)
    sequence_rows = []
    sequence_manifest = Path(manifests_dir) / "sequences.csv"
    if sequence_manifest.is_file():
        with sequence_manifest.open("r", newline="", encoding="utf-8") as source:
            sequence_rows = list(csv.DictReader(source))
    sign_status = _imported_sign_status(
        imported, recordings, vocabulary, targets or {}, split_signers, sequence_rows
    )
    split_names = ("train", "validation", "test")
    split_overlaps = []
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            shared = sorted(
                set().union(*(groups.get(left, set()) & groups.get(right, set()) for groups in split_signers.values()))
            )
            if shared:
                split_overlaps.append({"left": left, "right": right, "shared_signers": shared})
    imported_split_eligible = sum(
        row.get("signer_identity_status") == "CONFIRMED"
        and row.get("signer_id") not in {"", "UNKNOWN"}
        and row.get("session_id") not in {"", "UNKNOWN"}
        and row.get("validation_status") == "READY_FOR_PROCESSING"
        for row in imported
    )
    return {
        "dataset_version": _read_json(Path(manifests_dir) / "dataset_version.json", {"version": "0.1"}),
        "number_of_signers": len(signers),
        "number_of_suspected_imported_signers": len({
            row.get("signer_id") for row in imported
            if row.get("signer_id") not in {"", "UNKNOWN"}
        }),
        "number_of_confirmed_imported_signers": len({
            row.get("signer_id") for row in imported
            if row.get("signer_identity_status") == "CONFIRMED"
        }),
        "consent_granted_signers": sum(value.get("consent_status") == "GRANTED" for value in signers.values()),
        "number_of_sessions": len(sessions),
        "number_of_recordings": len(combined_recordings),
        "imported_recordings": len(imported),
        "readable_imported_recordings": sum(row.get("video_validation_status") == "READABLE" for row in imported),
        "invalid_imported_recordings": len(technical_failures),
        "duplicate_imported_recordings": sum(row.get("duplicate_content") == "true" for row in imported),
        "recordings_by_sign": _count_by(combined_recordings, "sign"),
        "recordings_by_signer": _count_by(recordings, "signer_id"),
        "imported_recordings_by_suspected_signer": _count_by(imported, "signer_id"),
        "total_signs": len(vocabulary),
        "validated_signs": sum(value.get("human_validation") == "VALIDATED" for value in vocabulary.values()),
        "unvalidated_signs": sum(value.get("human_validation") != "VALIDATED" for value in vocabulary.values()),
        "review_status_counts": _count_by(reviews, "status"),
        "imported_review_status_counts": imported_review_counts,
        "recordings_needing_review": recordings_needing_review,
        "technical_validation_failures": technical_failures,
        "videos_with_unknown_signer": sum(row.get("signer_id") in {"", "UNKNOWN"} for row in imported),
        "videos_with_unconfirmed_signer": sum(row.get("signer_identity_status") != "CONFIRMED" for row in imported),
        "videos_with_unknown_session": sum(row.get("session_id") in {"", "UNKNOWN"} for row in imported),
        "recordings_eligible_for_signer_split": imported_split_eligible + sum(
            row.get("signer_id") not in {"", "UNKNOWN"}
            and row.get("validation_status") == "VALID"
            for row in recordings
        ),
        "recordings_excluded_from_reliable_signer_split": len(imported) - imported_split_eligible,
        "signer_split_files_present": all(
            (Path(manifests_dir) / f"{name}.csv").is_file() for name in split_names
        ),
        "signer_leakage_detected": bool(split_overlaps),
        "signer_split_overlaps": split_overlaps,
        "sign_status": sign_status,
        "signs_ready_for_preprocessing": [
            row["sign"] for row in sign_status if row["ready_for_preprocessing"]
        ],
        "signs_ready_for_training": [
            row["sign"] for row in sign_status if row["ready_for_training"]
        ],
        "failure_count": len(failures),
        "invalid_recordings": sum(value.get("validation_status") != "VALID" for value in recordings) + len(technical_failures),
        "pending_review_sequences": sum(value.get("status") in {"PENDING", "NEEDS_RETAKE"} for value in reviews),
        "pending_review_recordings": len(recordings_needing_review),
        "collection_priorities": priorities + [
            {"sign": row["sign"], "action": row["next_action"], "gap": row["recordings_needed"]}
            for row in sign_status if row["next_action"]
        ],
        "dry_run": dry_run,
    }


def _imported_sign_status(
    imported: list[dict[str, Any]],
    recordings: list[dict[str, Any]],
    vocabulary: dict[str, dict[str, Any]],
    targets: dict[str, int],
    split_signers: dict[str, dict[str, set[str]]],
    sequence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    minimum_signers = int(targets.get("signers_per_sign", 5))
    minimum_repetitions = int(targets.get("recordings_per_signer_sign", 3))
    signs = sorted(
        ({str(row.get("sign", "")).upper() for row in imported if row.get("sign") not in {"", "UNKNOWN"}}
         | {str(row.get("sign", "")).upper() for row in recordings if row.get("sign")})
    )
    result = []
    for sign in signs:
        imported_for_sign = [row for row in imported if str(row.get("sign", "")).upper() == sign]
        registered_for_sign = [row for row in recordings if str(row.get("sign", "")).upper() == sign]
        confirmed_signers = {
            str(row.get("signer_id", ""))
            for row in registered_for_sign
            if row.get("signer_id") not in {"", "UNKNOWN"}
        }
        confirmed_signers.update(
            str(row.get("signer_id", ""))
            for row in imported_for_sign
            if row.get("signer_identity_status") == "CONFIRMED"
            and row.get("signer_id") not in {"", "UNKNOWN"}
        )
        validated_label = vocabulary.get(sign, {}).get("human_validation") == "VALIDATED"
        validated_recordings = sum(
            row.get("validation_status") == "READY_FOR_PROCESSING"
            and row.get("review_status") == "APPROVED"
            and row.get("language_validation_state") == "VALIDATED"
            for row in imported_for_sign
        ) + sum(row.get("validation_status") == "VALID" for row in registered_for_sign)
        ready_to_process = sum(
            row.get("validation_status") == "READY_FOR_PROCESSING"
            for row in imported_for_sign
        ) + sum(row.get("validation_status") == "VALID" for row in registered_for_sign)
        processed_count = sum(
            row.get("processing_status") == "PROCESSED"
            for row in imported_for_sign
        )
        enough_data = (
            len(confirmed_signers) >= minimum_signers
            and all(
                sum(row.get("signer_id") == signer for row in imported_for_sign + registered_for_sign)
                >= minimum_repetitions
                for signer in confirmed_signers
            )
        )
        sign_split_signers = split_signers.get(sign, {})
        split_names = ("train", "validation", "test")
        split_is_disjoint = not any(
            sign_split_signers.get(left, set()) & sign_split_signers.get(right, set())
            for index, left in enumerate(split_names)
            for right in split_names[index + 1:]
        )
        has_all_splits = all(sign_split_signers.get(name) for name in split_names)
        processed_sequences = [
            row for row in sequence_rows
            if str(row.get("sign", "")).upper() == sign
            and row.get("processing_status") == "PROCESSED"
        ]
        sample_counts = {
            name: sum(row.get("signer_id") in sign_split_signers.get(name, set()) for row in processed_sequences)
            for name in split_names
        }
        split_has_samples = all(sample_counts[name] > 0 for name in split_names)
        if not validated_label:
            next_action = "HUMAN_VALIDATE_SIGN_AND_LANGUAGE"
        elif len(confirmed_signers) < minimum_signers:
            next_action = "COLLECT_FROM_NEW_SIGNERS"
        elif ready_to_process == 0:
            next_action = "RESOLVE_VALIDATION_AND_CONSENT_BLOCKERS"
        elif not enough_data:
            next_action = "COLLECT_MORE_REPETITIONS"
        else:
            next_action = ""
        result.append({
            "sign": sign,
            "video_count": len(imported_for_sign) + len(registered_for_sign),
            "readable_video_count": sum(row.get("video_validation_status") == "READABLE" for row in imported_for_sign),
            "suspected_signers": sorted({
                str(row.get("signer_id")) for row in imported_for_sign
                if row.get("signer_id") not in {"", "UNKNOWN"}
            }),
            "confirmed_signer_count": len(confirmed_signers),
            "only_one_confirmed_signer": len(confirmed_signers) == 1,
            "validated_sign": validated_label,
            "validated_recording_count": validated_recordings,
            "ready_for_preprocessing_count": ready_to_process,
            "processed_recording_count": processed_count,
            "ready_for_preprocessing": ready_to_process > 0,
            "ready_for_training": bool(
                validated_label
                and enough_data
                and validated_recordings >= minimum_signers * minimum_repetitions
                and has_all_splits
                and split_is_disjoint
                and split_has_samples
            ),
            "split_signers": {
                name: sorted(sign_split_signers.get(name, set())) for name in split_names
            },
            "split_is_disjoint": split_is_disjoint,
            "split_has_samples": split_has_samples,
            "recordings_needed": max(0, minimum_signers - len(confirmed_signers)),
            "next_action": next_action,
        })
    return result


def collection_priorities(
    recordings: list[dict[str, Any]],
    vocabulary: dict[str, dict[str, Any]],
    targets: dict[str, int],
) -> list[dict[str, Any]]:
    target_signers = int(targets.get("signers_per_sign", 5))
    target_per_signer = int(targets.get("recordings_per_signer_sign", 3))
    if target_signers < 1 or target_per_signer < 1:
        raise ValueError("Collection targets must be positive integers.")
    priorities = []
    for sign_id in sorted(vocabulary):
        sign_rows = [row for row in recordings if row.get("sign") == sign_id]
        signers = sorted({str(row.get("signer_id", "")) for row in sign_rows if row.get("signer_id")})
        if len(signers) < target_signers:
            priorities.append({
                "sign": sign_id,
                "action": "ADD_NEW_SIGNER",
                "current_signers": len(signers),
                "target_signers": target_signers,
                "gap": target_signers - len(signers),
            })
            continue
        underfilled = [
            {"signer_id": signer, "recordings": sum(row.get("signer_id") == signer for row in sign_rows)}
            for signer in signers
            if sum(row.get("signer_id") == signer for row in sign_rows) < target_per_signer
        ]
        for item in underfilled:
            priorities.append({
                "sign": sign_id,
                "action": "ADD_REPETITIONS",
                **item,
                "target_recordings": target_per_signer,
                "gap": target_per_signer - item["recordings"],
            })
    return priorities


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, "") or "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def append_failure(
    recording_id: str,
    stage: str,
    error: Exception,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
) -> None:
    path = Path(manifests_dir) / "failures.json"
    failures = _read_json(path, [])
    failures.append({
        "recording_id": recording_id,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": _now(),
    })
    _write_json(path, failures)


def update_dataset_version(
    manifests_dir: str | Path,
    number_of_signers: int,
    number_of_signs: int,
    number_of_recordings: int,
    number_of_processed_sequences: int,
) -> dict[str, Any]:
    path = Path(manifests_dir) / "dataset_version.json"
    previous = _read_json(path, {"version": "0.1"})
    version = str(previous.get("version", "0.1"))
    result = {
        "version": version,
        "created_at": _now(),
        "number_of_signers": number_of_signers,
        "number_of_signs": number_of_signs,
        "number_of_recordings": number_of_recordings,
        "number_of_processed_sequences": number_of_processed_sequences,
    }
    _write_json(path, result)
    return result
