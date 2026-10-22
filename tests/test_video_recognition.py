from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app
import config
import video_recognition
from video_recognition_worker import WorkerError, _read_video_frames, _require_visible_hands


class FakeCapture:
    def __init__(self, frames=(), opened=True, fps=10.0):
        self.frames = list(frames)
        self.opened = opened
        self.fps = fps
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.fps

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class VideoRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_data_dir = config.DATA_DIR
        config.DATA_DIR = str(Path(self.temporary_directory.name) / "uploads")
        Path(config.DATA_DIR).mkdir()
        self.client = app.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["username"] = "video-learner"

    def tearDown(self):
        config.DATA_DIR = self.original_data_dir
        self.temporary_directory.cleanup()

    def test_page_is_authenticated_and_lists_supported_vocabulary(self):
        response = self.client.get("/video-recognition")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Video Sign Recognition", response.data)
        self.assertIn(b"American Sign Language", response.data)
        self.assertIn(b"does not translate continuous", response.data)
        self.assertIn(b"Supported vocabulary (2000 WLASL labels)", response.data)
        self.assertIn(b"<li>drink</li>", response.data)
        self.assertEqual(
            self.client.get("/static/video_recognition.js").status_code,
            200,
        )

        anonymous_client = app.app.test_client()
        self.assertEqual(anonymous_client.get("/video-recognition").status_code, 302)
        self.assertEqual(
            anonymous_client.post("/api/video-recognition/predict").status_code,
            401,
        )

    def test_valid_upload_returns_model_predictions_and_cleans_temporary_file(self):
        temporary_paths = []

        def fake_predict(video_path):
            temporary_paths.append(video_path)
            self.assertTrue(video_path.is_file())
            self.assertEqual(video_path.suffix, ".mp4")
            return {
                "model": "OpenHands WLASL BiLSTM",
                "task": "isolated sign recognition",
                "language": "American Sign Language (ASL)",
                "frames_processed": 12,
                "predictions": [{"label": "drink", "confidence": 0.589258}],
            }

        with patch("video_recognition.predict_video", side_effect=fake_predict):
            response = self.client.post(
                "/api/video-recognition/predict",
                data={"video": (io.BytesIO(b"sample video"), "sign.mp4")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["predictions"][0]["label"], "drink")
        self.assertEqual(payload["predictions"][0]["confidence"], 0.589258)
        self.assertEqual(len(temporary_paths), 1)
        self.assertFalse(temporary_paths[0].exists())

    def test_unsupported_file_type_is_rejected(self):
        response = self.client.post(
            "/api/video-recognition/predict",
            data={"video": (io.BytesIO(b"not a video"), "sign.gif")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported video type", response.get_json()["error"])
        self.assertEqual(list(Path(config.DATA_DIR).iterdir()), [])

    def test_oversized_upload_is_rejected_without_leaving_a_temporary_file(self):
        with patch.object(video_recognition, "MAX_UPLOAD_BYTES", 4):
            response = self.client.post(
                "/api/video-recognition/predict",
                data={"video": (io.BytesIO(b"12345"), "sign.mp4")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())
        self.assertEqual(list(Path(config.DATA_DIR).iterdir()), [])

    def test_request_body_limit_is_scoped_to_video_upload_route(self):
        with patch.object(video_recognition, "MAX_UPLOAD_BYTES", 4):
            response = self.client.post(
                "/api/video-recognition/predict",
                data={"video": (io.BytesIO(b"x" * (1024 * 1024 + 16)), "sign.mp4")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())
        self.assertEqual(list(Path(config.DATA_DIR).iterdir()), [])

    def test_corrupt_video_and_no_frames_are_reported(self):
        corrupt = FakeCapture(opened=False)
        with patch("cv2.VideoCapture", return_value=corrupt):
            with self.assertRaises(WorkerError) as raised:
                _read_video_frames(Path("corrupt.mp4"), 10)
        self.assertEqual(raised.exception.code, "invalid_video")
        self.assertTrue(corrupt.released)

        no_frames = FakeCapture()
        with patch("cv2.VideoCapture", return_value=no_frames):
            with self.assertRaises(WorkerError) as raised:
                _read_video_frames(Path("empty.mp4"), 10)
        self.assertEqual(raised.exception.code, "no_frames")

    def test_video_duration_and_missing_hands_are_rejected(self):
        frames = [object() for _ in range(101)]
        too_long = FakeCapture(frames=frames, fps=10.0)
        with patch("cv2.VideoCapture", return_value=too_long), patch(
            "cv2.cvtColor",
            side_effect=lambda frame, _conversion: frame,
        ):
            with self.assertRaises(WorkerError) as raised:
                _read_video_frames(Path("long.mp4"), 10)
        self.assertEqual(raised.exception.code, "video_too_long")
        self.assertTrue(too_long.released)

        no_hands = np.zeros((3, 75), dtype=float)
        with self.assertRaises(WorkerError) as raised:
            _require_visible_hands(no_hands)
        self.assertEqual(raised.exception.code, "no_hands")

    def test_model_unavailable_and_inference_failure_are_explicit(self):
        with patch.dict(
            os.environ,
            {
                "BRIDGESIGN_VIDEO_RECOGNITION_PYTHON": str(
                    Path(self.temporary_directory.name) / "missing-python.exe"
                ),
                "BRIDGESIGN_OPENHANDS_SOURCE_DIR": str(
                    Path(self.temporary_directory.name) / "missing-source"
                ),
            },
        ):
            self.assertFalse(video_recognition.is_runtime_configured())
            with self.assertRaises(video_recognition.VideoRecognitionError) as raised:
                video_recognition.predict_video(Path("video.mp4"))
        self.assertEqual(raised.exception.status_code, 503)

        runtime = Path(self.temporary_directory.name) / "runtime"
        python_path = runtime / "python.exe"
        source_dir = runtime / "source"
        (source_dir / "openhands" / "models").mkdir(parents=True)
        (source_dir / "openhands" / "models" / "loader.py").write_text("", encoding="utf-8")
        python_path.write_bytes(b"")
        completed = SimpleNamespace(
            returncode=2,
            stdout=json.dumps(
                {"ok": False, "code": "no_hands", "error": "No visible hands were detected."}
            ),
            stderr="",
        )
        with patch.dict(
            os.environ,
            {
                "BRIDGESIGN_VIDEO_RECOGNITION_PYTHON": str(python_path),
                "BRIDGESIGN_OPENHANDS_SOURCE_DIR": str(source_dir),
            },
        ), patch("video_recognition.subprocess.run", return_value=completed):
            with self.assertRaises(video_recognition.VideoRecognitionError) as raised:
                video_recognition.predict_video(Path("video.mp4"))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.message, "No visible hands were detected.")

        timeout = video_recognition.subprocess.TimeoutExpired(
            cmd="worker",
            timeout=video_recognition.PROCESSING_TIMEOUT_SECONDS,
        )
        with patch.dict(
            os.environ,
            {
                "BRIDGESIGN_VIDEO_RECOGNITION_PYTHON": str(python_path),
                "BRIDGESIGN_OPENHANDS_SOURCE_DIR": str(source_dir),
            },
        ), patch("video_recognition.subprocess.run", side_effect=timeout):
            with self.assertRaises(video_recognition.VideoRecognitionError) as raised:
                video_recognition.predict_video(Path("video.mp4"))
        self.assertEqual(raised.exception.status_code, 504)

    def test_inference_error_is_returned_and_uploaded_file_is_cleaned(self):
        temporary_paths = []

        def fail_prediction(video_path):
            temporary_paths.append(video_path)
            raise video_recognition.VideoRecognitionError("No visible hands detected.", 422)

        with patch("video_recognition.predict_video", side_effect=fail_prediction):
            response = self.client.post(
                "/api/video-recognition/predict",
                data={"video": (io.BytesIO(b"sample video"), "sign.mp4")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"], "No visible hands detected.")
        self.assertEqual(len(temporary_paths), 1)
        self.assertFalse(temporary_paths[0].exists())
