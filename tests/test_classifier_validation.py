import pickle
import tempfile
import unittest
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from classifier import Classifier


def _make_valid_model_payload():
    X = [
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
    y = ["A", "B", "A", "B"]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]
    )
    pipeline.fit(X, y)
    return {"pipeline": pipeline, "classes": ["A", "B"]}


class TestClassifierModelFallback(unittest.TestCase):
    def test_uses_backup_model_when_preferred_model_is_broken(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir) / "models"
            models_dir.mkdir()

            (models_dir / "sign_model_light.pkl").write_bytes(b"not a real pickle file")
            with open(models_dir / "sign_model.pkl", "wb") as f:
                pickle.dump(_make_valid_model_payload(), f)

            original_models_dir = config.MODELS_DIR
            try:
                config.MODELS_DIR = str(models_dir)
                classifier = Classifier()
                self.assertIsNotNone(classifier.pipeline)
                self.assertEqual(classifier.categories[0], "A")
                self.assertIn(classifier.predict([0.0, 0.0, 1.0])[0], {"A", "B"})
            finally:
                config.MODELS_DIR = original_models_dir

    def test_skips_unresolved_lfs_pointer_and_loads_compatible_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir) / "models"
            models_dir.mkdir()
            (models_dir / "sign_model_light.pkl").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:" + "0" * 64 + "\nsize 92000000\n",
                encoding="ascii",
            )
            with (models_dir / "sign_model.pkl").open("wb") as artifact:
                pickle.dump(_make_valid_model_payload(), artifact)

            original_models_dir = config.MODELS_DIR
            try:
                config.MODELS_DIR = str(models_dir)
                classifier = Classifier()
                self.assertIsNotNone(classifier.pipeline)
                self.assertEqual(classifier.categories[0], "A")
                self.assertIsNone(classifier.last_error)
            finally:
                config.MODELS_DIR = original_models_dir


if __name__ == "__main__":
    unittest.main()
