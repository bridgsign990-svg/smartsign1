"""CPU-only worker for the OpenHands WLASL-2000 BiLSTM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

CHECKPOINT_SHA256 = "c91551063db37f018f8246c64f1d2bfe53364d7c976377379e4d81c2d031179c"
MAX_FRAME_COUNT = 600


class WorkerError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _read_video_frames(video_path: Path, max_seconds: int) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise WorkerError("invalid_video", "The uploaded file is not a readable video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_limit = min(MAX_FRAME_COUNT, max(1, math.ceil(fps * max_seconds)))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if len(frames) >= frame_limit:
                raise WorkerError(
                    "video_too_long",
                    f"Use an isolated sign clip no longer than {max_seconds} seconds.",
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise WorkerError("no_frames", "No readable frames were found in the video.")
    return frames


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise WorkerError("model_unavailable", f"Could not load OpenHands source file: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_package_namespace(name: str, package_path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(package_path)]
    package.__package__ = name
    sys.modules[name] = package


def _require_visible_hands(confidences: np.ndarray) -> None:
    import numpy as np

    if confidences.ndim != 2 or confidences.shape[1] < 75:
        raise WorkerError("no_hands", "Hand landmarks were not available in the video.")
    if not np.any(confidences[:, 33:] > 0):
        raise WorkerError("no_hands", "No visible hands were detected in the video.")


def _load_model(
    source_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    class_count: int,
):
    import torch
    from omegaconf import OmegaConf

    if not (source_dir / "openhands" / "models" / "loader.py").is_file():
        raise WorkerError("model_unavailable", "OpenHands source is not installed.")
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise WorkerError("model_unavailable", "The OpenHands checkpoint or model config is missing.")
    if _sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise WorkerError(
            "checkpoint_integrity",
            "The model checkpoint failed its SHA-256 integrity check.",
        )

    sys.path.insert(0, str(source_dir))
    models_dir = source_dir / "openhands" / "models"
    # Avoid upstream package initializers that import unused image and transformer backbones.
    _install_package_namespace(
        "openhands.models.encoder",
        models_dir / "encoder",
    )
    _install_package_namespace(
        "openhands.models.encoder.graph",
        models_dir / "encoder" / "graph",
    )
    _install_package_namespace(
        "openhands.models.decoder",
        models_dir / "decoder",
    )
    from openhands.models.loader import get_model

    cfg = OmegaConf.load(config_path)
    model = get_model(cfg.model, in_channels=2, num_class=class_count)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise WorkerError("model_unavailable", "The checkpoint contains no model weights.")
    if state_dict and all(key.startswith("model.") for key in state_dict):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _predict(
    video_path: Path,
    source_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    vocabulary_path: Path,
    max_seconds: int,
) -> dict:
    import numpy as np
    import torch

    required_paths = (video_path, source_dir, checkpoint_path, config_path, vocabulary_path)
    if any(not path.exists() for path in required_paths):
        raise WorkerError("model_unavailable", "A required video-recognition file is missing.")

    try:
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerError("model_unavailable", "The WLASL vocabulary file is invalid.") from error
    if not isinstance(vocabulary, list) or len(vocabulary) != 2000:
        raise WorkerError("model_unavailable", "The checkpoint requires the 2,000-label WLASL vocabulary.")

    frames = _read_video_frames(video_path, max_seconds)
    pose_generator_path = source_dir / "openhands" / "datasets" / "pipelines" / "generate_pose.py"
    pose_transforms_path = source_dir / "openhands" / "datasets" / "pose_transforms.py"
    if not pose_generator_path.is_file() or not pose_transforms_path.is_file():
        raise WorkerError("model_unavailable", "OpenHands pose-processing source files are missing.")

    generator_module = _load_source_module(
        "_bridgesign_openhands_pose_generator",
        pose_generator_path,
    )
    transforms_module = _load_source_module(
        "_bridgesign_openhands_pose_transforms",
        pose_transforms_path,
    )

    generator = generator_module.MediaPipePoseGenerator(model_complexity=1)
    keypoints, confidences = generator.get_holistic_keypoints(np.asarray(frames))
    body_and_hands = np.concatenate(
        [keypoints[:, :33, :], keypoints[:, 501:, :]],
        axis=1,
    )
    hand_confidences = np.concatenate(
        [confidences[:, :33], confidences[:, 501:]],
        axis=1,
    )
    _require_visible_hands(hand_confidences)

    model_input = {
        "frames": torch.tensor(
            np.asarray(body_and_hands[:, :, :2], dtype=np.float32)
        ).permute(2, 0, 1)
    }
    model_input = transforms_module.PoseSelect(
        preset="mediapipe_holistic_minimal_27"
    )(model_input)
    model_input = transforms_module.CenterAndScaleNormalize(
        reference_points_preset="shoulder_mediapipe_holistic_minimal_27",
        scale_factor=1,
    )(model_input)

    model = _load_model(
        source_dir,
        checkpoint_path,
        config_path,
        class_count=len(vocabulary),
    )
    with torch.inference_mode():
        logits = model(model_input["frames"].unsqueeze(0))
        probabilities = torch.softmax(logits[0], dim=-1)
        values, indices = torch.topk(probabilities, k=min(5, len(vocabulary)))

    predictions = [
        {
            "label": vocabulary[int(index)],
            "confidence": round(float(value), 6),
        }
        for value, index in zip(values, indices)
    ]
    return {
        "ok": True,
        "model": "OpenHands WLASL BiLSTM",
        "task": "isolated sign recognition",
        "language": "American Sign Language (ASL)",
        "frames_processed": len(frames),
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()

    try:
        result = _predict(
            args.video,
            args.source,
            args.checkpoint,
            args.config,
            args.vocabulary,
            args.max_seconds,
        )
    except WorkerError as error:
        print(json.dumps({"ok": False, "code": error.code, "error": error.message}))
        return 2

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
