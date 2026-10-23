from pathlib import Path

import cv2
import numpy as np

from tools.msl_dataset.msl_dataset_pipeline import (
    _create_holistic_detector,
    _extract_landmarks_for_video,
    load_config,
)


def test_configured_holistic_model_initializes_and_processes_temporary_video(tmp_path):
    config = load_config(Path("config/msl_dataset.json"))
    model_path = Path(config["holistic_model_path"])
    assert model_path.is_file()
    detector, backend = _create_holistic_detector(config)
    detector.close()
    assert backend == "tasks"

    video_path = tmp_path / "technical_smoke.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (640, 480),
    )
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
    finally:
        writer.release()

    features, raw, presence, stats = _extract_landmarks_for_video(
        video_path,
        config["face_landmark_indices"],
        config,
    )
    feature_matrix = np.asarray(features, dtype=np.float32)
    presence_matrix = np.asarray(presence, dtype=np.float32)

    assert stats["frame_count"] == 3
    assert feature_matrix.shape == (3, 562)
    assert np.isfinite(feature_matrix).all()
    assert presence_matrix.shape == (3, 4)
    assert np.isfinite(presence_matrix).all()
    assert np.isnan(np.asarray(raw, dtype=np.float32)).all()
