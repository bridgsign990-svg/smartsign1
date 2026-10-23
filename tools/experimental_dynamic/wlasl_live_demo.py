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

from tools.experimental_dynamic.baseline import EXPERIMENT_TAG, load_checkpoint
from tools.msl_dataset import msl_dataset_pipeline as holistic_pipeline


DATA_ROOT = PROJECT_ROOT / "data" / "experimental_dynamic" / "wlasl"
MODEL_PATH = PROJECT_ROOT / "models" / "experimental" / "wlasl" / "best_model.pt"
SEQUENCE_LENGTH = 30
SAMPLE_RATE = 25.0


def load_wlasl_demo_model(checkpoint_path: str | Path) -> tuple[Any, dict[str, Any], Any]:
    model, checkpoint, device = load_checkpoint(checkpoint_path)
    if checkpoint.get("dataset") != "WLASL v0.3":
        raise ValueError("The webcam demo accepts only a checkpoint trained on WLASL v0.3.")
    if checkpoint.get("language") != "ASL" or checkpoint.get("experiment_tag") != EXPERIMENT_TAG:
        raise ValueError("Checkpoint metadata does not identify the isolated WLASL ASL experiment.")
    if checkpoint.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {SEQUENCE_LENGTH}-frame sequences, got {checkpoint.get('sequence_length')}."
        )
    return model, checkpoint, device


def _predict(model: Any, checkpoint: dict[str, Any], device: Any, sequence: np.ndarray) -> tuple[str, float]:
    if sequence.shape != (SEQUENCE_LENGTH, checkpoint["input_size"]):
        raise ValueError(
            f"Expected live input shape ({SEQUENCE_LENGTH}, {checkpoint['input_size']}), got {sequence.shape}."
        )
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(sequence).unsqueeze(0).to(device),
            torch.tensor([SEQUENCE_LENGTH]),
        )
        scores = torch.softmax(logits, dim=1)[0].cpu().numpy()
    index_to_label = {value: key for key, value in checkpoint["label_to_index"].items()}
    index = int(np.argmax(scores))
    return index_to_label[index], float(scores[index])


def run_demo(camera_index: int, checkpoint_path: Path, min_confidence: float) -> None:
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1.")
    model, checkpoint, device = load_wlasl_demo_model(checkpoint_path)
    feature_config_path = DATA_ROOT / "config.json"
    if not feature_config_path.is_file():
        raise FileNotFoundError(
            f"WLASL preprocessing configuration is missing: {feature_config_path}. "
            "Run wlasl_experiment.py process and train first."
        )
    config = json.loads(feature_config_path.read_text(encoding="utf-8"))
    expected_dimension = int(checkpoint["input_size"])
    detector, backend = holistic_pipeline._create_holistic_detector(config)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        detector.close()
        raise RuntimeError(f"Could not open webcam camera index {camera_index}.")
    capture.set(cv2.CAP_PROP_FPS, SAMPLE_RATE)
    history: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    hand_history: deque[bool] = deque(maxlen=SEQUENCE_LENGTH)
    face_indices = config.get("face_landmark_indices", holistic_pipeline.DEFAULT_FACE_INDICES)
    selection = config.get("feature_selection", {})
    enabled_groups = [name for name in ("left_hand", "right_hand", "pose", "face") if selection.get(name, False)]
    hand_group_indices = [i for i, name in enumerate(enabled_groups) if name in {"left_hand", "right_hand"}]
    started = time.monotonic()
    last_timestamp = -1
    next_sample_time = started
    last_prediction_time = 0.0
    label = ""
    confidence = 0.0
    status = "Show a hand to the camera"
    try:
        print("WLASL experimental webcam demo")
        print(f"Model: {checkpoint_path.resolve()}")
        print(f"Classes: {', '.join(checkpoint['label_to_index'])}")
        print("Press Q or Escape to quit. This does not use the production Live model.")
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames.")
            now = time.monotonic()
            timestamp = max(last_timestamp + 1, int((now - started) * 1000))
            last_timestamp = timestamp
            results = holistic_pipeline._detect_frame(detector, backend, frame, timestamp)
            feature_vector, _raw, presence = holistic_pipeline._frame_features(
                results,
                face_indices,
                selection,
                config.get("normalization_method", "wrist_relative"),
            )
            if feature_vector.shape[0] != expected_dimension:
                raise ValueError(
                    f"Holistic feature dimension {feature_vector.shape[0]} does not match "
                    f"the WLASL model input {expected_dimension}."
                )
            if now >= next_sample_time:
                history.append(feature_vector.astype(np.float32))
                is_hand_present = any(
                    i < len(presence) and bool(presence[i]) for i in hand_group_indices
                )
                hand_history.append(is_hand_present)
                next_sample_time = now + 1.0 / SAMPLE_RATE
                if len(history) == SEQUENCE_LENGTH and sum(hand_history) >= 8:
                    if now - last_prediction_time >= 0.2:
                        label, confidence = _predict(
                            model, checkpoint, device, np.asarray(history, dtype=np.float32)
                        )
                        last_prediction_time = now
                    status = "Prediction" if confidence >= min_confidence else "Uncertain (below threshold)"
                else:
                    label = ""
                    confidence = 0.0
                    status = "Collecting 30 frames; keep a hand visible"

            display = frame.copy()
            if not label:
                text = status
            elif confidence >= min_confidence:
                text = f"WLASL: {label}  confidence {confidence:.2f}"
            else:
                text = f"Uncertain: {label}  confidence {confidence:.2f}"
            cv2.rectangle(display, (0, 0), (display.shape[1], 82), (20, 20, 20), -1)
            cv2.putText(display, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 240, 120), 2)
            cv2.putText(display, "EXPERIMENTAL WLASL / ASL - not MSL", (16, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 190, 255), 1)
            cv2.imshow("BridgeSign - WLASL Experimental Live", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated WLASL BiLSTM with a webcam; production Live remains unchanged."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    args = parser.parse_args()
    run_demo(args.camera, args.checkpoint, args.min_confidence)


if __name__ == "__main__":
    main()
