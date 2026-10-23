import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app
import database
from call_room import CallRoomManager
from learning_mode import ensure_additional_alphabet_lessons, seed_starter_course, sync_avatar_catalog


class _FakeSocket:
    def __init__(self, connected=True):
        self.connected = connected
        self.sent = []

    def send(self, message):
        if not self.connected:
            raise ConnectionError("closed")
        self.sent.append(json.loads(message))


class TestCallRoomManager(unittest.TestCase):
    def test_stale_socket_does_not_keep_room_full(self):
        rooms = CallRoomManager()
        first, ghost = _FakeSocket(), _FakeSocket()
        self.assertTrue(rooms.join_room("room0001", first))
        self.assertTrue(rooms.join_room("room0001", ghost))
        ghost.connected = False  # tab crashed without a clean leave

        rejoin = _FakeSocket()
        self.assertTrue(rooms.join_room("room0001", rejoin))
        self.assertEqual(len(rooms.rooms["room0001"]), 2)
        self.assertEqual(rejoin.sent[-1], {"type": "ready", "should_offer": True})

    def test_third_live_participant_is_rejected(self):
        rooms = CallRoomManager()
        for _ in range(2):
            self.assertTrue(rooms.join_room("room0002", _FakeSocket()))
        extra = _FakeSocket()
        self.assertFalse(rooms.join_room("room0002", extra))
        self.assertEqual(extra.sent[-1]["type"], "error")

    def test_signaling_sockets_are_pinged(self):
        self.assertIn("ping_interval", app.app.config.get("SOCK_SERVER_OPTIONS", {}))


class _DbTestCase(unittest.TestCase):
    username = "hardening-user"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database.initialize_database(
            f"sqlite:///{(Path(self.temp_dir.name) / 'hardening.sqlite3').as_posix()}",
            str(Path(self.temp_dir.name) / "data"),
        )
        database.create_user(self.username, "test-hash")
        self.client = app.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["username"] = self.username

    def tearDown(self):
        if database._engine is not None:
            database._engine.dispose()
        self.temp_dir.cleanup()


class TestEmergencyPhraseLimits(_DbTestCase):
    def test_duplicate_phrase_returns_existing_entry(self):
        first = self.client.post("/api/emergency/custom", json={"phrase": "I need water"}).get_json()
        again = self.client.post("/api/emergency/custom", json={"phrase": "  i NEED water "}).get_json()
        self.assertEqual(again["id"], first["id"])
        self.assertTrue(again["duplicate"])
        custom = [key for key in self.client.get("/api/emergency/all").get_json() if key.startswith("c_")]
        self.assertEqual(len(custom), 1)

    def test_custom_phrase_count_is_capped(self):
        for index in range(app.MAX_CUSTOM_EMERGENCY_PHRASES):
            response = self.client.post("/api/emergency/custom", json={"phrase": f"Phrase {index}"})
            self.assertEqual(response.status_code, 200)
        over = self.client.post("/api/emergency/custom", json={"phrase": "One too many"})
        self.assertEqual(over.status_code, 400)


def _recorded_clip(label):
    data_path = Path(app.config.DATA_DIR) / "gesture_dataset.csv"
    if not data_path.is_file():
        return None
    with data_path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith(label + ","):
                return np.array(line.strip().split(",")[1:], dtype=np.float32).reshape(15, 42)
    return None


class TestMotionPractice(_DbTestCase):
    def setUp(self):
        super().setUp()
        seed_starter_course()
        sync_avatar_catalog()
        ensure_additional_alphabet_lessons()
        from sqlalchemy import select

        with database._session() as session:
            self.hello_id = session.scalar(
                select(database.Sign.id).where(database.Sign.name == "HELLO")
            )

    def test_word_sign_reports_motion_practice_mode(self):
        lesson = self.client.get("/api/learning/lessons/1").get_json()
        hello = next(sign for sign in lesson["signs"] if sign["name"] == "HELLO")
        if "HELLO" not in app._recognition_labels():
            self.skipTest("No gesture model with HELLO is available in models/.")
        self.assertTrue(hello["practice_available"])
        self.assertEqual(hello["practice_mode"], "motion")

    def test_motion_clip_is_scored_by_gesture_model(self):
        clip = _recorded_clip("HELLO")
        if clip is None or "HELLO" not in app._recognition_labels():
            self.skipTest("Recorded HELLO clip or gesture model unavailable.")

        frames = iter(clip)

        class _ReplayDetector:
            def find_hands(self, frame, draw=False):
                self.current = next(frames)
                return frame, SimpleNamespace(hand_landmarks=[[None] * 21])

            def get_landmarks(self, frame, hand_no=0):
                feats = self.current
                return [[i, 200 + float(feats[2 * i]) * 40, 200 + float(feats[2 * i + 1]) * 40] for i in range(21)]

        import cv2
        from feature_extractor import FeatureExtractor
        from gesture_classifier import GestureClassifier

        ok, jpeg = cv2.imencode(".jpg", np.zeros((48, 48, 3), dtype=np.uint8))
        files = [(io.BytesIO(jpeg.tobytes()), f"f{index}.jpg") for index in range(len(clip))]
        with patch.object(app, "_detector", _ReplayDetector()), \
                patch.object(app, "_extractor", FeatureExtractor()), \
                patch.object(app, "_gesture_classifier", GestureClassifier()), \
                patch.object(app, "_is_plausible_hand", lambda *args: True), \
                patch.object(app, "_load_inference_modules", lambda: None), \
                patch.dict(app._inference_status, {"ready": True}):
            response = self.client.post(
                "/api/learning/practice",
                data={"lesson_id": "1", "sign_id": str(self.hello_id), "frames": files},
                content_type="multipart/form-data",
            )
        body = response.get_json()
        self.assertEqual(response.status_code, 200, body)
        self.assertEqual(body["mode"], "motion")
        self.assertEqual(body["prediction"], "HELLO")
        self.assertTrue(body["correct"])


if __name__ == "__main__":
    unittest.main()
