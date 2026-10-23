from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experimental_dynamic import baseline
from tools.experimental_dynamic import wlasl_v2_benchmark
from tools.msl_dataset import msl_dataset_pipeline as holistic_pipeline


DATA_ROOT = PROJECT_ROOT / "data" / "experimental" / "wlasl_v2"
SELECTION_PATH = (
    PROJECT_ROOT / "models" / "experimental" / "wlasl_v2" / "benchmark" / "best_model.json"
)
SEQUENCE_LENGTH = 30
INPUT_SIZE = 912
TARGET_FPS = 25.0
DISPLAY_CONFIDENCE_THRESHOLD = 0.45


def selected_checkpoint_path() -> Path:
    if not SELECTION_PATH.is_file():
        raise FileNotFoundError(
            f"Validation-selected benchmark model metadata is missing: {SELECTION_PATH}"
        )
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if (
        selection.get("framework") != "pytorch"
        or selection.get("representation") != "representation_v2"
        or int(selection.get("input_dimension", 0)) != INPUT_SIZE
    ):
        raise ValueError(
            "The validation-selected WLASL checkpoint must be a PyTorch representation_v2 model."
        )
    path = Path(selection["checkpoint"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_v2_model(path: str | Path) -> tuple[Any, dict[str, Any], Any]:
    model, checkpoint = wlasl_v2_benchmark.load_torch_checkpoint(path)
    metadata_path = Path(f"{path}.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    if not str(checkpoint.get("dataset", "")).startswith("WLASL v0.3"):
        raise ValueError("This camera demo accepts only a WLASL V2 checkpoint.")
    if (
        checkpoint.get("language", metadata.get("language")) != "ASL"
        or checkpoint.get("experiment_tag", metadata.get("experiment_tag"))
        != baseline.EXPERIMENT_TAG
    ):
        raise ValueError("Checkpoint metadata must identify the isolated WLASL ASL experiment.")
    if (
        checkpoint.get("sequence_length") != SEQUENCE_LENGTH
        or checkpoint.get("input_size") != INPUT_SIZE
        or checkpoint.get("representation") != "representation_v2"
    ):
        raise ValueError("Checkpoint input must be the WLASL V2 Holistic [30, 912] representation.")
    return model, checkpoint, torch.device("cpu")


def predict_sequence(model: Any, checkpoint: dict[str, Any], device: Any, sequence: np.ndarray) -> tuple[str, float]:
    if sequence.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
        raise ValueError(
            f"Expected a [30, {INPUT_SIZE}] sequence, got {sequence.shape}."
        )
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(sequence.astype(np.float32, copy=False)).unsqueeze(0).to(device),
            torch.tensor([SEQUENCE_LENGTH]),
        )
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    index_to_label = {
        index: label for label, index in checkpoint["label_to_index"].items()
    }
    index = int(np.argmax(probabilities))
    return index_to_label[index], float(probabilities[index])


def transform_live_buffer(
    features: np.ndarray,
    raw_features: np.ndarray,
    presence: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    raw_features = np.asarray(raw_features, dtype=np.float32)
    presence = np.asarray(presence, dtype=np.float32)
    if (
        features.shape != (SEQUENCE_LENGTH, 562)
        or raw_features.shape != (SEQUENCE_LENGTH, 558)
        or presence.shape != (SEQUENCE_LENGTH, 4)
    ):
        raise ValueError("Live Holistic buffers must be [30,562], [30,558], and [30,4].")
    transformed = wlasl_v2_benchmark.representation_v2(
        features,
        raw_features,
        presence,
        np.ones(SEQUENCE_LENGTH, dtype=np.float32),
        config,
    )
    if transformed.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
        raise ValueError(f"Live representation_v2 produced unexpected shape {transformed.shape}.")
    return transformed


def run_demo(camera: int, checkpoint_path: Path | None, min_confidence: float) -> None:
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1.")
    checkpoint_path = checkpoint_path or selected_checkpoint_path()
    model, checkpoint, device = load_v2_model(checkpoint_path)
    config_path = DATA_ROOT / "preprocessing_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"V2 preprocessing configuration missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    detector, backend = holistic_pipeline._create_holistic_detector(config)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        detector.close()
        raise RuntimeError(f"Could not open webcam camera index {camera}.")
    buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    raw_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    presence_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    hand_buffer: deque[bool] = deque(maxlen=SEQUENCE_LENGTH)
    feature_selection = config.get("feature_selection", {})
    feature_groups = [
        name for name in ("left_hand", "right_hand", "pose", "face")
        if feature_selection.get(name, False)
    ]
    hand_indices = [
        index for index, name in enumerate(feature_groups)
        if name in {"left_hand", "right_hand"}
    ]
    face_indices = config.get("face_landmark_indices", holistic_pipeline.DEFAULT_FACE_INDICES)
    started = time.monotonic()
    next_sample = started
    last_timestamp = -1
    label = ""
    confidence = 0.0
    status = "Show a hand to the camera"
    try:
        print("EXPERIMENTAL WLASL V2 / ASL RESEARCH/ARCHITECTURE DEMO / NOT MSL / NOT PRODUCTION")
        print(f"Validation-selected WLASL model: {checkpoint_path.resolve()}")
        print(f"Trained glosses: {len(checkpoint['label_to_index'])}")
        print("Press Q or Escape to quit. Production Live is not used.")
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames.")
            now = time.monotonic()
            timestamp_ms = max(last_timestamp + 1, int((now - started) * 1000))
            last_timestamp = timestamp_ms
            result = holistic_pipeline._detect_frame(detector, backend, frame, timestamp_ms)
            features, raw, presence = holistic_pipeline._frame_features(
                result,
                face_indices,
                feature_selection,
                config.get("normalization_method", "wrist_relative"),
            )
            if features.shape != (checkpoint["input_size"],):
                raise ValueError(
                    f"Holistic produced {features.shape} features; checkpoint expects {checkpoint['input_size']}."
                )
            if now >= next_sample:
                buffer.append(features.astype(np.float32))
                raw_buffer.append(raw.astype(np.float32))
                presence_buffer.append(presence.astype(np.float32))
                hand_buffer.append(any(
                    index < len(presence) and bool(presence[index]) for index in hand_indices
                ))
                next_sample = now + 1.0 / TARGET_FPS
                if len(buffer) == SEQUENCE_LENGTH and sum(hand_buffer) >= 8:
                    label, confidence = predict_sequence(
                        model,
                        checkpoint,
                        device,
                        transform_live_buffer(
                            np.asarray(buffer, dtype=np.float32),
                            np.asarray(raw_buffer, dtype=np.float32),
                            np.asarray(presence_buffer, dtype=np.float32),
                            config,
                        ),
                    )
                    status = "prediction" if confidence >= min_confidence else "low confidence"
                else:
                    label = ""
                    confidence = 0.0
                    status = f"Collecting frames {len(buffer)}/{SEQUENCE_LENGTH}"

            display = frame.copy()
            if label and confidence >= min_confidence:
                prediction = f"{label}  {confidence:.2f}"
            elif label:
                prediction = f"Uncertain: {label}  {confidence:.2f}"
            else:
                prediction = status
            cv2.rectangle(display, (0, 0), (display.shape[1], 88), (20, 20, 20), -1)
            cv2.putText(display, prediction, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 240, 120), 2)
            cv2.putText(display, "EXPERIMENTAL WLASL V2 / ASL - NOT MSL / NOT PRODUCTION", (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 190, 255), 1)
            cv2.imshow("BridgeSign - WLASL V2 Experimental Live", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated WLASL V2 Holistic BiLSTM webcam demo."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional explicit experimental checkpoint override; defaults to validation-selected model.",
    )
    parser.add_argument("--min-confidence", type=float, default=DISPLAY_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()
    run_demo(args.camera, args.checkpoint, args.min_confidence)


if __name__ == "__main__":
    main()
