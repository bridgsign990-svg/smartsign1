import unittest
from unittest.mock import patch

import app


class FakeClassifier:
    def __init__(self):
        self.pipeline = object()

    def predict(self, features):
        return "A", 0.12

    def predict_topk(self, features, k=5):
        return [{"label": "A", "confidence": 0.12}]


class TestLiveDetectionFallback(unittest.TestCase):
    def test_low_confidence_prediction_is_not_dropped(self):
        fake = FakeClassifier()
        with patch.object(app, "_classifier", fake):
            label, conf = app._predict_static_sign([0.0] * 42)
        self.assertEqual(label, "A")
        self.assertGreater(conf, 0.0)

    def test_plausible_low_confidence_candidate_is_kept(self):
        class LowConfidenceClassifier:
            pipeline = object()

            def predict(self, features):
                return "A", 0.06

            def predict_topk(self, features, k=5):
                return [{"label": "A", "confidence": 0.06}]

        with patch.object(app, "_classifier", LowConfidenceClassifier()):
            label, conf = app._predict_static_sign([0.0] * 42)
        self.assertEqual(label, "A")
        self.assertGreaterEqual(conf, 0.06)

    def test_low_confidence_candidate_never_commits_after_repeated_frames(self):
        state = {
            "consecutive": 0,
            "prev_result_key": "",
            "last_emitted_key": "",
            "low_confidence_since": 0.0,
        }

        decisions = [
            app._update_live_confirmation(state, "A", 0.75, "handsign", now, threshold=5)
            for now in (1.0, 1.1, 1.2, 1.3, 1.4)
        ]

        self.assertTrue(all(decision["hand_state"] == "low_confidence" for decision in decisions))
        self.assertTrue(all(decision["confirmed_label"] == "" for decision in decisions))
        self.assertEqual(state["consecutive"], 0)

    def test_high_confidence_prediction_requires_five_stable_frames(self):
        state = {
            "consecutive": 0,
            "prev_result_key": "",
            "last_emitted_key": "",
            "low_confidence_since": 0.0,
        }

        decisions = [
            app._update_live_confirmation(state, "A", 0.85, "handsign", now, threshold=5)
            for now in (1.0, 1.1, 1.2, 1.3, 1.4)
        ]

        self.assertTrue(all(decision["hand_state"] == "pending" for decision in decisions[:4]))
        self.assertEqual(decisions[-1]["hand_state"], "recognised")
        self.assertEqual(decisions[-1]["confirmed_label"], "A")


if __name__ == "__main__":
    unittest.main()
