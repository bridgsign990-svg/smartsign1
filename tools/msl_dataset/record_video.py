#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2

try:
    from .collection_management import (
        DEFAULT_MANIFESTS_DIR,
        _sha256,
        register_recording,
        require_recording_permission,
        validate_identifier,
    )
except ImportError:
    from collection_management import (
        DEFAULT_MANIFESTS_DIR,
        _sha256,
        register_recording,
        require_recording_permission,
        validate_identifier,
    )


def choose_output_path(base_dir: str | Path, signer_id: str, sign: str, session_id: str, recording_number: int) -> Path:
    base = Path(base_dir)
    safe_signer = validate_identifier(signer_id, "signer")
    safe_sign = validate_identifier(sign.strip().upper(), "sign")
    safe_session = validate_identifier(session_id, "session")
    if recording_number < 1:
        raise ValueError("Recording number must be a positive integer.")
    target_dir = base / safe_signer / safe_sign / safe_session
    filename = f"video_{recording_number:03d}.mp4"
    return target_dir / filename


def record_video(
    signer_id: str,
    sign: str,
    session_id: str,
    output_dir: str | Path,
    recording_number: int | None = None,
    fps: int = 30,
    seconds: int = 5,
    camera_index: int = 0,
    manifests_dir: str | Path = DEFAULT_MANIFESTS_DIR,
) -> Path:
    signer_id = validate_identifier(signer_id, "signer")
    sign = validate_identifier(sign.strip().upper(), "sign")
    session_id = validate_identifier(session_id, "session")
    require_recording_permission(signer_id, session_id, manifests_dir)
    if recording_number is not None and recording_number < 1:
        raise ValueError("Recording number must be a positive integer.")
    if fps <= 0 or seconds <= 0:
        raise ValueError("FPS and recording duration must be positive.")
    base_dir = Path(output_dir)
    existing_numbers = []
    for recording_path in (base_dir / signer_id / sign / session_id).glob("video_*.mp4"):
        match = re.fullmatch(r"video_(\d+)\.mp4", recording_path.name, re.IGNORECASE)
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = max([recording_number or 1, *(number + 1 for number in existing_numbers)])
    target = choose_output_path(base_dir, signer_id, sign, session_id, next_number)
    while target.exists():
        next_number += 1
        target = choose_output_path(base_dir, signer_id, sign, session_id, next_number)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to access camera index {camera_index}.")

    writer = None
    started_at = None
    print(f"Previewing {sign} for signer {signer_id}; output will be {target}")
    print("Press 's' to start recording, then 'q' to stop. Recording also stops at --seconds.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped supplying frames.")
            cv2.imshow("BridgeSign MSL Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and writer is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                while True:
                    try:
                        with target.open("xb"):
                            pass
                        break
                    except FileExistsError:
                        next_number += 1
                        target = choose_output_path(base_dir, signer_id, sign, session_id, next_number)
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
                if not writer.isOpened():
                    target.unlink(missing_ok=True)
                    raise RuntimeError(f"Unable to create output video: {target}")
                started_at = time.time()
                print("Recording started.")
            if writer is not None:
                writer.write(frame)
                if started_at is not None and time.time() - started_at >= seconds:
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    if writer is None:
        raise RuntimeError("Recording was cancelled before it started.")
    if not target.is_file():
        raise RuntimeError(f"Recording output was not created: {target}")
    _validate_finished_video(target, seconds)
    registered = register_recording(target, signer_id, sign, session_id, manifests_dir)
    if registered["sha256"] != _sha256(target):
        raise RuntimeError(f"Recording changed while registering: {target}")
    return target


def _validate_finished_video(path: Path, expected_max_seconds: int) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Recorded video cannot be reopened: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        decoded = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded += 1
        duration = decoded / fps if fps > 0 else 0.0
        if decoded < 12 or frames < 12 or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Recorded video failed validation: frames={decoded}, fps={fps}, size={width}x{height}.")
        if duration > expected_max_seconds + 3:
            raise RuntimeError(f"Recorded video duration {duration:.2f}s exceeds the expected bound.")
        if decoded != frames:
            raise RuntimeError(f"Video is truncated: decoded {decoded} of {frames} reported frames.")
        return {"frames": decoded, "fps": fps, "width": width, "height": height, "duration": duration}
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record MSL training videos from the webcam.")
    parser.add_argument("--signer", required=True, help="Signer ID (example: S01)")
    parser.add_argument("--sign", required=True, help="Sign label (example: WATER)")
    parser.add_argument("--session", default="session_001", help="Session ID to attach to the recording")
    parser.add_argument("--output-dir", default="data/msl_dataset/raw", help="Base directory for raw signer data")
    parser.add_argument("--recording-number", type=int, default=1, help="Initial recording number")
    parser.add_argument("--seconds", type=int, default=5, help="Duration to record in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Video capture FPS")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    args = parser.parse_args()
    target = record_video(
        signer_id=args.signer,
        sign=args.sign,
        session_id=args.session,
        output_dir=args.output_dir,
        recording_number=args.recording_number,
        seconds=args.seconds,
        fps=args.fps,
        camera_index=args.camera_index,
        manifests_dir=args.manifests_dir,
    )
    print(f"Saved recording to: {target}")


if __name__ == "__main__":
    main()
