"""Isolated OpenHands WLASL video-recognition service."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO

import config

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "openhands_wlasl"
CHECKPOINT_PATH = MODEL_DIR / "epoch=109-step=49059.ckpt"
MODEL_CONFIG_PATH = MODEL_DIR / "lstm.yaml"
VOCABULARY_PATH = MODEL_DIR / "classes.json"
WORKER_PATH = PROJECT_DIR / "video_recognition_worker.py"

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".m4v"})


def _positive_int_setting(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


MAX_UPLOAD_BYTES = _positive_int_setting(
    "BRIDGESIGN_VIDEO_RECOGNITION_MAX_UPLOAD_BYTES",
    25 * 1024 * 1024,
)
PROCESSING_TIMEOUT_SECONDS = _positive_int_setting(
    "BRIDGESIGN_VIDEO_RECOGNITION_TIMEOUT_SECONDS",
    90,
)
MAX_VIDEO_SECONDS = _positive_int_setting(
    "BRIDGESIGN_VIDEO_RECOGNITION_MAX_SECONDS",
    10,
)


class VideoRecognitionError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _runtime_paths() -> tuple[Path, Path]:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        default_runtime_dir = Path(local_app_data) / "BridgeSign" / "OpenHandsRuntime"
    else:
        default_runtime_dir = Path.home() / ".local" / "share" / "BridgeSign" / "OpenHandsRuntime"

    python_path = Path(
        os.environ.get(
            "BRIDGESIGN_VIDEO_RECOGNITION_PYTHON",
            default_runtime_dir / "venv" / "Scripts" / "python.exe",
        )
    )
    source_dir = Path(
        os.environ.get(
            "BRIDGESIGN_OPENHANDS_SOURCE_DIR",
            default_runtime_dir / "source",
        )
    )
    return python_path, source_dir


def is_runtime_configured() -> bool:
    python_path, source_dir = _runtime_paths()
    return all(
        path.is_file()
        for path in (python_path, CHECKPOINT_PATH, MODEL_CONFIG_PATH, VOCABULARY_PATH, WORKER_PATH)
    ) and (source_dir / "openhands" / "models" / "loader.py").is_file()


def load_supported_vocabulary() -> list[str]:
    try:
        with VOCABULARY_PATH.open("r", encoding="utf-8") as source:
            vocabulary = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise VideoRecognitionError(
            "The model vocabulary file is missing or invalid.",
            503,
        ) from error

    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) != 2000
        or any(not isinstance(label, str) or not label for label in vocabulary)
    ):
        raise VideoRecognitionError("The model vocabulary file is invalid.", 503)
    return vocabulary


def save_video_upload(
    stream: BinaryIO,
    filename: str,
) -> Path:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise VideoRecognitionError(
            f"Unsupported video type. Use {supported}.",
            400,
        )

    temporary_path: Path | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix="bridgesign-recognition-",
            suffix=extension,
            dir=config.DATA_DIR,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise VideoRecognitionError(
                        f"Video exceeds the {MAX_UPLOAD_BYTES / (1024 * 1024):g} MB upload limit.",
                        413,
                    )
                temporary.write(chunk)
    except Exception:
        if temporary_path is not None:
            remove_temporary_video(temporary_path)
        raise

    if total_bytes == 0:
        remove_temporary_video(temporary_path)
        raise VideoRecognitionError("The selected video is empty.", 400)
    return temporary_path


def remove_temporary_video(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.exception("Could not remove temporary video upload %s", path)


def _worker_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise VideoRecognitionError(
        "The isolated model process returned an invalid response.",
        500,
    )


def predict_video(video_path: Path) -> dict:
    python_path, source_dir = _runtime_paths()
    if not is_runtime_configured():
        raise VideoRecognitionError(
            "The isolated OpenHands runtime is not installed. Run tools/install_video_recognition.ps1, then restart BridgeSign.",
            503,
        )

    command = [
        str(python_path),
        str(WORKER_PATH),
        "--video",
        str(video_path),
        "--source",
        str(source_dir),
        "--checkpoint",
        str(CHECKPOINT_PATH),
        "--config",
        str(MODEL_CONFIG_PATH),
        "--vocabulary",
        str(VOCABULARY_PATH),
        "--max-seconds",
        str(MAX_VIDEO_SECONDS),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=PROCESSING_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        LOGGER.warning(
            "Video recognition exceeded its %d second limit.",
            PROCESSING_TIMEOUT_SECONDS,
        )
        raise VideoRecognitionError(
            f"Processing exceeded the {PROCESSING_TIMEOUT_SECONDS} second limit. Try a shorter clip.",
            504,
        ) from error
    except OSError as error:
        LOGGER.exception("Could not start isolated OpenHands process")
        raise VideoRecognitionError(
            "The isolated model process could not be started.",
            503,
        ) from error

    try:
        payload = _worker_result(completed.stdout)
    except VideoRecognitionError:
        if completed.stderr:
            LOGGER.error("OpenHands worker failed: %s", completed.stderr[-4000:])
        raise
    if completed.returncode != 0 or payload.get("ok") is not True:
        code = payload.get("code")
        status_by_code = {
            "invalid_video": 422,
            "no_frames": 422,
            "video_too_long": 422,
            "no_hands": 422,
            "checkpoint_integrity": 503,
            "model_unavailable": 503,
        }
        if completed.stderr:
            LOGGER.error("OpenHands worker failed: %s", completed.stderr[-4000:])
        raise VideoRecognitionError(
            str(payload.get("error") or "The model could not process this video."),
            status_by_code.get(code, 500),
        )

    predictions = payload.get("predictions")
    if (
        not isinstance(predictions, list)
        or not predictions
        or len(predictions) > 5
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not isinstance(item.get("confidence"), (int, float))
            or not 0.0 <= item["confidence"] <= 1.0
            for item in predictions
        )
    ):
        raise VideoRecognitionError(
            "The model returned invalid prediction data.",
            500,
        )
    return {
        "model": "OpenHands WLASL BiLSTM",
        "task": "isolated sign recognition",
        "language": "American Sign Language (ASL)",
        "frames_processed": payload.get("frames_processed"),
        "predictions": predictions,
    }
