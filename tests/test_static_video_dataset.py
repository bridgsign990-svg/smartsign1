import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app
import config
import database
import static_video_dataset as video_dataset


class FakeCapture:
    def __init__(self, frame_count=5, fps=20.0):
        self.frame_count = frame_count
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == video_dataset.cv2.CAP_PROP_FPS else self.frame_count

    def read(self):
        if self.index >= self.frame_count:
            return False, None
        self.index += 1
        return True, np.zeros((32, 32, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeDetector:
    def __init__(self, has_hand=True):
        self.has_hand = has_hand

    def find_hands(self, frame, draw=False):
        points = [SimpleNamespace(x=0.2, y=0.3) for _ in range(21)]
        return frame, SimpleNamespace(hand_landmarks=[points] if self.has_hand else [])

    def get_landmarks(self, frame, hand_no=0):
        if not self.has_hand:
            return []
        return [[index, index + 1, index + 2] for index in range(21)]


class FakeExtractor:
    def extract_features(self, landmarks):
        return np.arange(42, dtype=np.float32)


class TestStaticVideoDataset(unittest.TestCase):
    def make_video(self, directory, name="sample.mp4"):
        path = Path(directory) / name
        path.write_bytes(b"mock video bytes")
        return path

    def test_samples_static_landmarks_into_provenance_staging_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self.make_video(temp_dir)
            output = Path(temp_dir) / "static_video_samples.csv"
            capture = FakeCapture()
            with patch.object(video_dataset.cv2, "VideoCapture", return_value=capture):
                result = video_dataset.extract_static_video(
                    source,
                    "WATER",
                    "S03",
                    output_path=output,
                    detector=FakeDetector(),
                    extractor=FakeExtractor(),
                )

            with output.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertEqual(result["samples_saved"], 5)
            self.assertTrue(capture.released)
            self.assertTrue(all(row["sign_label"] == "WATER" for row in rows))
            self.assertTrue(all(row["signer_id"] == "S03" for row in rows))
            self.assertEqual(len({row["source_video_sha256"] for row in rows}), 1)
            self.assertEqual(len({row["source_frame_index"] for row in rows}), 5)
            self.assertEqual(len({row["timestamp_ms"] for row in rows}), 5)
            self.assertEqual(sum(column.startswith("feature_") for column in rows[0]), 42)
            self.assertEqual(rows[0]["source_video_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertNotEqual(output.name, "dataset.csv")

    def test_no_hand_video_does_not_create_or_append_staging_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self.make_video(temp_dir)
            output = Path(temp_dir) / "static_video_samples.csv"
            with patch.object(video_dataset.cv2, "VideoCapture", return_value=FakeCapture()):
                with self.assertRaises(video_dataset.VideoImportError):
                    video_dataset.extract_static_video(
                        source,
                        "WATER",
                        "S03",
                        output_path=output,
                        detector=FakeDetector(has_hand=False),
                        extractor=FakeExtractor(),
                    )
            self.assertFalse(output.exists())

    def test_invalid_signer_and_incompatible_staging_header_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self.make_video(temp_dir)
            output = Path(temp_dir) / "static_video_samples.csv"
            with self.assertRaises(video_dataset.VideoImportError):
                video_dataset.extract_static_video(source, "WATER", "bad signer", output_path=output)
            output.write_text("old_label,feature_1\n", encoding="utf-8")
            with patch.object(video_dataset.cv2, "VideoCapture", return_value=FakeCapture()):
                with self.assertRaises(video_dataset.VideoImportError):
                    video_dataset.extract_static_video(
                        source,
                        "WATER",
                        "S03",
                        output_path=output,
                        detector=FakeDetector(),
                        extractor=FakeExtractor(),
                    )
            self.assertEqual(output.read_text(encoding="utf-8"), "old_label,feature_1\n")


class TestStaticVideoImportApi(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_data_dir = config.DATA_DIR
        config.DATA_DIR = str(Path(self.temp_dir.name) / "data")
        Path(config.DATA_DIR).mkdir()
        self.production_path = Path(config.DATA_DIR) / "dataset.csv"
        self.production_path.write_bytes(b"production dataset stays byte-identical\n")
        self.production_hash = hashlib.sha256(self.production_path.read_bytes()).hexdigest()
        database.initialize_database(
            f"sqlite:///{(Path(self.temp_dir.name) / 'video-import.sqlite3').as_posix()}",
            str(Path(self.temp_dir.name) / "empty"),
        )
        database.create_user("video-learner", "test-hash")
        with database._session() as session, session.begin():
            session.add(database.Sign(name="WATER", content_type="WORD"))
        self.client = app.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["username"] = "video-learner"

    def tearDown(self):
        if database._engine is not None:
            database._engine.dispose()
        config.DATA_DIR = self.original_data_dir
        self.temp_dir.cleanup()

    def test_catalog_options_and_import_endpoint_keep_production_dataset_untouched(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'data-tab="video-data"', page.data)
        script = self.client.get("/static/video_ingest.js?v=static-video-test")
        self.assertEqual(script.status_code, 200)

        options = self.client.get("/api/static-video/options")
        self.assertEqual(options.status_code, 200)
        self.assertIn("WATER", options.get_json()["signs"])

        temporary_paths = []

        def fake_extract(video_path, sign_label, signer_id):
            temporary_paths.append(Path(video_path))
            self.assertTrue(Path(video_path).is_file())
            self.assertEqual((sign_label, signer_id), ("WATER", "S03"))
            return {
                "sign_label": sign_label,
                "signer_id": signer_id,
                "source_video_sha256": "abc123",
                "frames_in_video": 10,
                "frames_sampled": 5,
                "sampled_frames_with_hand": 5,
                "samples_saved": 5,
                "sample_interval_ms": 50.0,
                "output_filename": "static_video_samples.csv",
                "output_path": str(Path(config.DATA_DIR) / "static_video_samples.csv"),
            }

        with patch("static_video_dataset.extract_static_video", side_effect=fake_extract):
            response = self.client.post(
                "/api/static-video/import",
                data={
                    "sign_label": "WATER",
                    "signer_id": "S03",
                    "video": (io.BytesIO(b"mock mp4 bytes"), "water.mp4"),
                },
                content_type="multipart/form-data",
            )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertEqual(payload["samples_saved"], 5)
        self.assertFalse(payload["production_dataset_modified"])
        self.assertFalse(payload["model_modified"])
        self.assertEqual(hashlib.sha256(self.production_path.read_bytes()).hexdigest(), self.production_hash)
        self.assertEqual(len(temporary_paths), 1)
        self.assertFalse(temporary_paths[0].exists())

    def test_unknown_sign_and_unsupported_extension_are_rejected(self):
        unknown = self.client.post(
            "/api/static-video/import",
            data={"sign_label": "NOT_IN_CATALOG", "signer_id": "S03", "video": (io.BytesIO(b"x"), "x.mp4")},
            content_type="multipart/form-data",
        )
        unsupported = self.client.post(
            "/api/static-video/import",
            data={"sign_label": "WATER", "signer_id": "S03", "video": (io.BytesIO(b"x"), "x.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(unknown.status_code, 400)
        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(hashlib.sha256(self.production_path.read_bytes()).hexdigest(), self.production_hash)


if __name__ == "__main__":
    unittest.main()
