import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import app
import config
import database
from classifier import Classifier
from data_collector import FeatureExtractor as CollectorFeatureExtractor
from feature_extractor import FEATURE_VECTOR_SIZE, FeatureExtractor
from image_translator import FeatureExtractor as ImageFeatureExtractor
from image_translator import ImageTranslator


class TestStaticFeaturePipeline(unittest.TestCase):
    def test_training_and_inference_share_42_feature_extractor(self):
        self.assertEqual(FEATURE_VECTOR_SIZE, 42)
        self.assertIs(CollectorFeatureExtractor, FeatureExtractor)
        self.assertIs(ImageFeatureExtractor, FeatureExtractor)

        landmarks = [[index, 100 + index, 200 + 2 * index] for index in range(21)]
        features = FeatureExtractor().extract_features(landmarks)

        self.assertEqual(features.shape, (42,))
        self.assertEqual(features[0], 0.0)
        self.assertEqual(features[1], 0.0)
        self.assertAlmostEqual(float(features[18]), 9 / np.sqrt(405), places=6)
        self.assertIsNone(FeatureExtractor().extract_features(landmarks[:20]))

    def test_classifier_rejects_wrong_feature_dimension_and_returns_mapped_label(self):
        features = [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
        labels = [0, 1, 0, 1]
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ]).fit(features, labels)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            import pickle
            model_path.write_bytes(pickle.dumps({"pipeline": pipeline, "classes": ["A", "B"]}))
            classifier = Classifier(model_path=str(model_path))

        label, confidence = classifier.predict([0.0, 0.0])
        self.assertEqual((label, confidence), ("Error", 0.0))
        label, confidence = classifier.predict([0.0, 0.0, 1.0])
        self.assertIn(label, {"A", "B"})
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class _ImageDetector:
    def __init__(self, has_hand=True):
        self.has_hand = has_hand

    def find_hands(self, frame, draw=False):
        points = [SimpleNamespace(x=0.2 + (index % 5) * 0.12, y=0.2 + (index // 5) * 0.12) for index in range(21)]
        hands = [points] if self.has_hand else []
        return frame, SimpleNamespace(hand_landmarks=hands)

    def get_landmarks(self, frame, hand_no=0):
        return [[index, 10 + index, 20 + index] for index in range(21)]


class _ImageClassifier:
    def __init__(self, confidence):
        self.confidence = confidence

    def predict_many(self, feature_rows):
        return [("A", self.confidence), ("A", self.confidence - 0.01)]


class TestStaticImageRecognition(unittest.TestCase):
    def make_translator(self, confidence=0.9, has_hand=True):
        translator = ImageTranslator.__new__(ImageTranslator)
        translator.detector = _ImageDetector(has_hand)
        translator.feature_extractor = SimpleNamespace(extract_features=lambda landmarks: np.ones(42, dtype=np.float32))
        translator.classifier = _ImageClassifier(confidence)
        return translator

    def test_image_array_abstains_below_confirmation_floor(self):
        translator = self.make_translator(0.75)
        label, confidence, _ = translator.translate_from_array(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(label, "")
        self.assertEqual(confidence, 0.75)

    def test_image_array_preserves_no_hand_and_accepts_high_confidence(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        no_hand = self.make_translator(has_hand=False).translate_from_array(frame)
        accepted = self.make_translator(0.9).translate_from_array(frame)
        self.assertEqual(no_hand[0], "No hand detected")
        self.assertEqual(accepted[0], "A")


class TestStaticInferenceEndpoints(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_data_dir = config.DATA_DIR
        database.initialize_database(
            f"sqlite:///{(Path(self.temp_dir.name) / 'recognition.sqlite3').as_posix()}",
            str(Path(self.temp_dir.name) / "empty"),
        )
        self.username = "static-recognition-test"
        database.create_user(self.username, "test-hash")
        self.client = app.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["username"] = self.username
        config.DATA_DIR = str(Path(self.temp_dir.name) / "data")
        Path(config.DATA_DIR).mkdir()
        app._image_result_cache.clear()

    def tearDown(self):
        with app._inf_lock:
            app._inference_sessions.pop(self.username, None)
        if database._engine is not None:
            database._engine.dispose()
        config.DATA_DIR = self.original_data_dir
        self.temp_dir.cleanup()

    def landmarks(self):
        return [[index, 0.2 + index * 0.01, 0.3 + index * 0.01] for index in range(21)]

    @staticmethod
    def tiny_png_bytes():
        image = Image.new("RGB", (2, 2), color="white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def test_live_endpoint_keeps_low_confidence_unconfirmed_and_preserves_response_keys(self):
        with patch.object(app, "_load_inference_modules"), \
             patch.object(app, "_inference_status", {"ready": True, "loading": False, "error": ""}), \
             patch.object(app, "_extractor", SimpleNamespace(extract_features=lambda landmarks: [0.0] * 42)), \
             patch.object(app, "_predict_static_sign", return_value=("A", 0.75)), \
             patch.object(app.tracker, "log_translation"):
            responses = [self.client.post("/api/infer_landmarks", json={
                "landmarks": self.landmarks(),
                "consecutive_threshold": 5,
            }).get_json() for _ in range(5)]

        self.assertTrue(all(response["hand_state"] == "low_confidence" for response in responses))
        self.assertTrue(all(response["label"] == "" and response["confidence"] == 0.0 for response in responses))
        self.assertEqual(responses[-1]["static_label"], "A")
        self.assertEqual(responses[-1]["consecutive"], 0)
        self.assertIn("hand_state", responses[-1])
        self.assertIn("label", responses[-1])
        self.assertIn("confidence", responses[-1])

    def test_live_endpoint_no_hand_does_not_predict(self):
        with patch.object(app, "_predict_static_sign") as predict:
            response = self.client.post("/api/infer_landmarks", json={"landmarks": []})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["hand_state"], "no_hand")
        self.assertEqual(response.get_json()["label"], "")
        predict.assert_not_called()

    def test_live_endpoint_commits_high_confidence_after_five_frames(self):
        with patch.object(app, "_load_inference_modules"), \
             patch.object(app, "_inference_status", {"ready": True, "loading": False, "error": ""}), \
             patch.object(app, "_extractor", SimpleNamespace(extract_features=lambda landmarks: [0.0] * 42)), \
             patch.object(app, "_predict_static_sign", return_value=("A", 0.85)), \
             patch.object(app.tracker, "log_translation"):
            responses = [self.client.post("/api/infer_landmarks", json={
                "landmarks": self.landmarks(),
                "consecutive_threshold": 5,
            }).get_json() for _ in range(5)]

        self.assertTrue(all(response["hand_state"] == "pending" for response in responses[:4]))
        self.assertEqual(responses[-1]["hand_state"], "recognised")
        self.assertEqual(responses[-1]["label"], "A")

    def test_image_endpoint_returns_low_confidence_instead_of_a_label(self):
        processed_paths = []

        def translate(path):
            self.assertTrue(Path(path).is_file())
            processed_paths.append(Path(path))
            return "A", 0.75, None

        with patch.object(app, "_get_image_translator", return_value=SimpleNamespace(translate=translate)):
            response = self.client.post(
                "/api/translate_image",
                data={"image": (io.BytesIO(self.tiny_png_bytes()), "sample.png", "image/png")},
                content_type="multipart/form-data",
            )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["label"], "")
        self.assertFalse(payload["no_hand"])
        self.assertTrue(payload["low_confidence"])
        self.assertEqual(payload["confidence"], "75%")
        self.assertEqual(len(processed_paths), 1)
        self.assertFalse(processed_paths[0].exists())

    def test_image_endpoint_rejects_invalid_and_oversized_uploads(self):
        invalid = self.client.post(
            "/api/translate_image",
            data={"image": (io.BytesIO(b"not an image"), "sample.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("valid", invalid.get_json()["error"])

        with patch.object(config, "IMAGE_MAX_UPLOAD_BYTES", 10):
            oversized = self.client.post(
                "/api/translate_image",
                data={"image": (io.BytesIO(self.tiny_png_bytes()), "sample.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(oversized.status_code, 413)


if __name__ == "__main__":
    unittest.main()