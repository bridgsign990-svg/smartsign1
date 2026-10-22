import unittest

import numpy as np

from gesture_classifier import GestureClassifier
from gesture_feature_extractor import (
    FEATURE_CLIP,
    GESTURE_WINDOW,
    STATIC_FEATURES,
    GestureFeatureExtractor,
    mirror_sequence,
    resample_sequence,
    sanitize_sequence,
)


def _ramp_sequence():
    """x of landmark 1 moves 0 -> 1 across the window; everything else is still."""
    sequence = np.zeros((GESTURE_WINDOW, STATIC_FEATURES), dtype=np.float32)
    sequence[:, 2] = np.linspace(0.0, 1.0, GESTURE_WINDOW)
    return sequence


class TestSequenceHelpers(unittest.TestCase):
    def test_sanitize_clips_outliers_and_zeroes_non_finite(self):
        values = np.array([np.nan, np.inf, -np.inf, 1e6, -1e6, 0.5], dtype=np.float32)
        cleaned = sanitize_sequence(values)
        np.testing.assert_allclose(cleaned, [0, 0, 0, FEATURE_CLIP, -FEATURE_CLIP, 0.5])

    def test_push_frame_stores_sanitized_copy(self):
        extractor = GestureFeatureExtractor(window_size=1)
        frame = np.full(STATIC_FEATURES, 1e4, dtype=np.float32)
        extractor.push_frame(frame)
        frame[:] = 0.0
        self.assertTrue(np.all(extractor.get_raw_sequence() == FEATURE_CLIP))

    def test_mirror_flips_only_x(self):
        mirrored = mirror_sequence(_ramp_sequence() + 0.25)
        self.assertTrue(np.all(mirrored[:, 1::2] == 0.25))
        np.testing.assert_allclose(mirrored[:, 2], -(np.linspace(0, 1, GESTURE_WINDOW) + 0.25))

    def test_resample_identity_at_speed_one(self):
        sequence = _ramp_sequence()
        np.testing.assert_allclose(resample_sequence(sequence), sequence, atol=1e-6)

    def test_faster_speed_holds_end_poses(self):
        fast = resample_sequence(_ramp_sequence(), speed=3.0)
        self.assertEqual(fast.shape, (GESTURE_WINDOW, STATIC_FEATURES))
        self.assertAlmostEqual(float(fast[0, 2]), 0.0)
        self.assertAlmostEqual(float(fast[-1, 2]), 1.0)
        # The whole motion now happens in the middle third of the window.
        self.assertAlmostEqual(float(fast[3, 2]), 0.0)
        self.assertAlmostEqual(float(fast[-4, 2]), 1.0)

    def test_resample_changes_length(self):
        self.assertEqual(resample_sequence(_ramp_sequence(), length=7).shape, (7, STATIC_FEATURES))


class _FakeOnnxSession:
    """Scores class 1 by how far landmark 1 moves to the right."""

    class _Input:
        name = "input"
        shape = ["batch_size", "sequence_length", STATIC_FEATURES]

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, feeds):
        batch = feeds["input"]
        rightward = batch[:, -1, 2] - batch[:, 0, 2]
        logits = np.stack([np.zeros_like(rightward), rightward * 10.0], axis=1)
        return [logits]


class TestClassifierOrientation(unittest.TestCase):
    def _classifier(self):
        classifier = GestureClassifier.__new__(GestureClassifier)
        classifier.backend = "onnx"
        classifier.ort_session = _FakeOnnxSession()
        classifier.categories = {0: "STATIC", 1: "WAVE_RIGHT"}
        return classifier

    def test_mirrored_input_is_recognised(self):
        classifier = self._classifier()
        label, confidence = classifier.predict(_ramp_sequence())
        mirrored_label, mirrored_confidence = classifier.predict(mirror_sequence(_ramp_sequence()))
        self.assertEqual(label, "WAVE_RIGHT")
        self.assertEqual(mirrored_label, "WAVE_RIGHT")
        self.assertAlmostEqual(confidence, mirrored_confidence, places=6)

    def test_still_hand_stays_static(self):
        label, _ = self._classifier().predict(np.zeros((GESTURE_WINDOW, STATIC_FEATURES)))
        self.assertEqual(label, "STATIC")

    def test_rejects_wrong_shapes(self):
        classifier = self._classifier()
        self.assertEqual(classifier.predict(np.zeros((15, 10))), ("", 0.0))
        self.assertEqual(classifier.predict(np.zeros(100)), ("", 0.0))


if __name__ == "__main__":
    unittest.main()
