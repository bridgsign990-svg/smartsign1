import json
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
                self.assertTrue(classifier.model_path.endswith("sign_model.pkl"))
                # The pointer is the record of which artifact to fetch with
                # `git lfs pull`; loading must never delete it.
                self.assertTrue((models_dir / "sign_model_light.pkl").exists())
            finally:
                config.MODELS_DIR = original_models_dir


    def _write_manifest(self, models_dir, entry):
        (models_dir / "model_manifest.json").write_text(
            json.dumps({"artifacts": {"sign_model_light.pkl": entry}}), encoding="utf-8"
        )

    def _with_models_dir(self, models_dir):
        original = config.MODELS_DIR
        config.MODELS_DIR = str(models_dir)
        self.addCleanup(setattr, config, "MODELS_DIR", original)

    def test_manifest_feature_mismatch_rejects_model_and_uses_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir) / "models"
            models_dir.mkdir()
            for name in ("sign_model_light.pkl", "sign_model_v2.pkl"):
                with (models_dir / name).open("wb") as artifact:
                    pickle.dump(_make_valid_model_payload(), artifact)
            self._write_manifest(models_dir, {"labels": ["A", "B"], "n_features": 42})
            self._with_models_dir(models_dir)

            classifier = Classifier()
            self.assertIsNotNone(classifier.pipeline)
            self.assertTrue(classifier.model_path.endswith("sign_model_v2.pkl"))

    def test_manifest_label_mismatch_rejects_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir) / "models"
            models_dir.mkdir()
            with (models_dir / "sign_model_light.pkl").open("wb") as artifact:
                pickle.dump(_make_valid_model_payload(), artifact)
            self._write_manifest(models_dir, {"labels": ["B", "A"], "n_features": 3})
            self._with_models_dir(models_dir)

            classifier = Classifier()
            self.assertIsNone(classifier.pipeline)
            self.assertIn("labels do not match", classifier.last_error)

    def test_manifest_checksum_difference_is_a_warning_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir) / "models"
            models_dir.mkdir()
            with (models_dir / "sign_model_light.pkl").open("wb") as artifact:
                pickle.dump(_make_valid_model_payload(), artifact)
            self._write_manifest(
                models_dir, {"labels": ["A", "B"], "n_features": 3, "sha256": "0" * 64, "status": "production"}
            )
            self._with_models_dir(models_dir)

            classifier = Classifier()
            self.assertIsNotNone(classifier.pipeline)
            self.assertEqual(classifier.manifest_entry["status"], "production")
            self.assertTrue(any("SHA-256" in warning for warning in classifier.artifact_warnings))

    def test_v2_candidate_is_used_only_when_no_other_model_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir) / "models"
            models_dir.mkdir()
            with (models_dir / "sign_model_v2.pkl").open("wb") as artifact:
                pickle.dump(_make_valid_model_payload(), artifact)
            self._with_models_dir(models_dir)
            self.assertTrue(Classifier().model_path.endswith("sign_model_v2.pkl"))

            with (models_dir / "sign_model.pkl").open("wb") as artifact:
                pickle.dump(_make_valid_model_payload(), artifact)
            self.assertTrue(Classifier().model_path.endswith("sign_model.pkl"))

    def test_no_model_reports_unavailable_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir) / "models"
            models_dir.mkdir()
            self._with_models_dir(models_dir)
            classifier = Classifier()
            self.assertIsNone(classifier.pipeline)
            self.assertEqual(classifier.predict([0.0] * 42), ("No Model (Train First)", 0.0))


if __name__ == "__main__":
    unittest.main()
