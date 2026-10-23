import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app
import gesture_classifier
from gesture_classifier import GestureClassifier


class _GestureSequence:
    def is_ready(self):
        return True

    def get_motion_magnitude(self):
        return 100.0

    def get_raw_sequence(self):
        return np.zeros((15, 42), dtype=np.float32)


class TestGestureLiveRouting(unittest.TestCase):
    def setUp(self):
        self.labels = frozenset({"HELP", "J", "Z", "STATIC"})
        self.classifier = SimpleNamespace(
            available_labels=self.labels,
            is_available=lambda: True,
            is_lstm=lambda: True,
            predict=lambda sequence: ("HELP", 0.95),
        )
        self.patcher = patch.object(app, "_gesture_classifier", self.classifier)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        # Keep artifact-selection tests independent of the real models/ folder.
        v2_patcher = patch.object(
            gesture_classifier, "ONNX_V2_MODEL_PATH", os.path.join(tempfile.gettempdir(), "missing_v2.onnx")
        )
        v2_patcher.start()
        self.addCleanup(v2_patcher.stop)

    def test_active_model_labels_are_the_only_accepted_labels(self):
        self.assertEqual(app._available_gesture_labels(), self.labels)
        word_state = {"mode": "word", "assembler": None}
        letter_state = {"mode": "letter", "assembler": None}

        self.assertEqual(app._filter_gesture_for_mode(word_state, "HELP", 0.95), ("HELP", 0.95))
        self.assertEqual(app._filter_gesture_for_mode(letter_state, "HELP", 0.95), ("", 0.0))
        self.assertEqual(app._filter_gesture_for_mode(letter_state, "J", 0.95), ("J", 0.95))
        self.assertEqual(app._filter_gesture_for_mode(word_state, "NOT_TRAINED", 0.99), ("", 0.0))
        self.assertEqual(app._filter_gesture_for_mode(word_state, "STATIC", 0.99), ("", 0.0))

    def test_new_word_label_passes_candidate_selection_and_confirmation(self):
        state = {
            "gesture_ext": _GestureSequence(),
            "mode": "word",
            "assembler": None,
            "frame_count": 5,
            "last_gesture_frame": -999,
            "cached_gesture_label": "",
            "cached_gesture_conf": 0.0,
        }

        with patch.object(app, "LIVE_GESTURES_ENABLED", True):
            label, confidence = app._predict_gesture_candidate(state)
            selected = app._select_live_output("", 0.0, label, confidence, mode="word")

        self.assertEqual((label, confidence), ("HELP", 0.95))
        self.assertEqual(selected, ("HELP", 0.95, "gesture"))

        confirmation = {
            "prev_result_key": "",
            "last_emitted_key": "",
            "consecutive": 0,
            "low_confidence_since": 0.0,
        }
        app._update_live_confirmation(confirmation, *selected[:2], selected[2], 1.0, threshold=2)
        result = app._update_live_confirmation(
            confirmation, selected[0], selected[1], selected[2], 1.1, threshold=2
        )
        self.assertEqual(result["confirmed_label"], "HELP")
        self.assertEqual(result["confirmed_source"], "gesture")

    def test_unknown_and_low_confidence_gesture_predictions_are_rejected(self):
        state = {
            "gesture_ext": _GestureSequence(),
            "mode": "word",
            "assembler": None,
            "frame_count": 5,
            "last_gesture_frame": -999,
            "cached_gesture_label": "",
            "cached_gesture_conf": 0.0,
        }
        with patch.object(app, "LIVE_GESTURES_ENABLED", True):
            self.classifier.predict = lambda sequence: ("UNKNOWN", 0.99)
            self.assertEqual(app._predict_gesture_candidate(state), ("", 0.0))

            state["last_gesture_frame"] = -999
            self.classifier.predict = lambda sequence: ("HELP", app.GESTURE_LSTM_MIN_CONF - 0.01)
            self.assertEqual(app._predict_gesture_candidate(state), ("", 0.0))

    def test_recorded_gesture_motion_below_old_gate_reaches_classifier(self):
        state = {
            "gesture_ext": _GestureSequence(),
            "mode": "word",
            "assembler": None,
            "frame_count": 5,
            "last_gesture_frame": -999,
            "cached_gesture_label": "",
            "cached_gesture_conf": 0.0,
        }
        state["gesture_ext"].get_motion_magnitude = lambda: 6.5

        with patch.object(app, "LIVE_GESTURES_ENABLED", True):
            self.assertEqual(app._predict_gesture_candidate(state), ("HELP", 0.95))

    def test_motion_letter_heuristic_rejects_conflict_instead_of_renaming(self):
        sequence = np.zeros((15, 42), dtype=np.float32)
        sequence[:, 16] = np.linspace(0.0, 0.3, 15)
        sequence[::2, 16] += 0.2
        label, confidence = app._refine_motion_letter_prediction(sequence, "J", 0.80)
        self.assertEqual((label, confidence), ("", 0.0))

    def test_confident_motion_letter_is_not_vetoed_by_path_heuristic(self):
        sequence = np.zeros((15, 42), dtype=np.float32)
        sequence[:, 16] = np.linspace(0.0, 0.3, 15)
        sequence[::2, 16] += 0.2
        self.assertEqual(
            app._refine_motion_letter_prediction(sequence, "J", 0.95), ("J", 0.95)
        )

    def test_motion_letter_beats_locked_static_letter(self):
        # J starts from the I handshape, so the static model reads I (or a weak J).
        self.assertEqual(
            app._select_live_output("I", 0.85, "J", 0.92, mode="letter"), ("J", 0.92, "gesture")
        )
        self.assertEqual(
            app._select_live_output("J", 0.79, "J", 1.0, mode="letter"), ("J", 1.0, "gesture")
        )

    def test_moving_word_gesture_needs_word_override_confidence(self):
        weak = app._select_live_output("A", 0.9, "HELP", 0.80, mode="word", moving=True)
        strong = app._select_live_output("A", 0.9, "HELP", 0.95, mode="word", moving=True)
        self.assertEqual(weak, ("A", 0.9, "handsign"))
        self.assertEqual(strong, ("HELP", 0.95, "gesture"))

    def test_confident_word_gesture_beats_confident_static_letter_in_word_mode(self):
        # PLEASE starts from an H-like handshape the static model reads at ~0.82.
        self.assertEqual(
            app._select_live_output("H", 0.82, "PLEASE", 0.99, mode="word"),
            ("PLEASE", 0.99, "gesture"),
        )
        self.assertEqual(
            app._select_live_output("H", 0.95, "PLEASE", 0.90, mode="word"),
            ("H", 0.95, "handsign"),
        )

    def test_start_pose_letter_does_not_block_word_gesture(self):
        from word_assembler import WordAssembler

        assembler = WordAssembler()
        state = {"mode": "word", "assembler": assembler}
        assembler.push_letter("B")
        self.assertEqual(app._filter_gesture_for_mode(state, "HELP", 0.80), ("HELP", 0.80))

    def test_long_held_start_pose_letter_does_not_block_word_gesture(self):
        # People hold the starting handshape before moving; the letter it reads
        # as must not lock out the word however long it was held.
        from word_assembler import WordAssembler

        assembler = WordAssembler()
        state = {"mode": "word", "assembler": assembler}
        assembler.push_letter("B")
        assembler._last_letter_ts -= 10.0
        self.assertEqual(app._filter_gesture_for_mode(state, "HELP", 0.80), ("HELP", 0.80))

    def test_spelled_letters_block_only_weak_word_gestures(self):
        from word_assembler import WordAssembler

        assembler = WordAssembler()
        state = {"mode": "word", "assembler": assembler}
        assembler.push_letter("H")
        assembler.push_letter("I")
        weak = app.GESTURE_WORD_OVERRIDE_CONF - 0.05
        self.assertEqual(app._filter_gesture_for_mode(state, "HELP", weak), ("", 0.0))
        strong = app.GESTURE_WORD_OVERRIDE_CONF
        self.assertEqual(app._filter_gesture_for_mode(state, "HELP", strong), ("HELP", strong))

    def test_word_gesture_keeps_spelled_letters_and_replaces_start_pose(self):
        from word_assembler import WordAssembler

        spelled = WordAssembler()
        spelled.push_letter("H")
        spelled.push_letter("I")
        self.assertFalse(spelled.pending_is_start_pose())
        spelled.push_word("HELP")
        self.assertEqual(spelled._words[-2:], ["HI", "HELP"])

        start_pose = WordAssembler()
        start_pose.push_letter("B")
        self.assertTrue(start_pose.pending_is_start_pose())

    def test_gesture_confirmation_uses_gesture_confidence_floor(self):
        state = {"prev_result_key": "", "last_emitted_key": "", "consecutive": 0,
                 "low_confidence_since": 0.0}
        conf = app.GESTURE_LSTM_MIN_CONF
        for i in range(app.GESTURE_CONFIRM_FRAMES):
            result = app._update_live_confirmation(state, "HELP", conf, "gesture", 1.0 + i, threshold=5)
        self.assertEqual(result["confirmed_label"], "HELP")

    def test_moving_word_gesture_still_blocked_in_letter_mode(self):
        selected = app._select_live_output("A", 0.9, "HELP", 0.99, mode="letter", moving=True)
        self.assertEqual(selected, ("A", 0.9, "handsign"))

    def test_gesture_confirmation_needs_fewer_frames_than_static(self):
        state = {"prev_result_key": "", "last_emitted_key": "", "consecutive": 0,
                 "low_confidence_since": 0.0}
        results = [
            app._update_live_confirmation(state, "HELP", 0.95, "gesture", 1.0 + i, threshold=5)
            for i in range(app.GESTURE_CONFIRM_FRAMES)
        ]
        self.assertEqual(results[-1]["confirmed_label"], "HELP")
        self.assertEqual(results[-2]["confirmed_label"], "")

    def test_static_alphabet_remains_preferred_in_letter_mode(self):
        selected = app._select_live_output(
            "A", 0.95, "HELP", 0.99, mode="letter", text_pending=False
        )
        self.assertEqual(selected, ("A", 0.95, "handsign"))

    def test_newer_trained_artifact_is_selected_before_old_onnx_export(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = {
                "onnx": os.path.join(directory, "model.onnx"),
                "metadata": os.path.join(directory, "model.json"),
                "lstm": os.path.join(directory, "model.pt"),
                "sklearn": os.path.join(directory, "model.pkl"),
            }
            for path in artifacts.values():
                with open(path, "wb") as artifact:
                    artifact.write(b"test artifact")
            os.utime(artifacts["onnx"], (10, 10))
            os.utime(artifacts["metadata"], (10, 10))
            os.utime(artifacts["sklearn"], (15, 15))
            os.utime(artifacts["lstm"], (20, 20))

            loaded = []

            def load_lstm(classifier, path):
                loaded.append(("lstm", path))
                return True

            patches = (
                patch.object(gesture_classifier, "ONNX_MODEL_PATH", artifacts["onnx"]),
                patch.object(gesture_classifier, "LSTM_META_PATH", artifacts["metadata"]),
                patch.object(gesture_classifier, "LSTM_MODEL_PATH", artifacts["lstm"]),
                patch.object(gesture_classifier, "SKLEARN_MODEL_PATH", artifacts["sklearn"]),
                patch.object(GestureClassifier, "_load_lstm", load_lstm),
                patch.object(GestureClassifier, "_load_onnx", lambda classifier, path: loaded.append(("onnx", path)) or True),
                patch.object(GestureClassifier, "_load_sklearn", lambda classifier, path: loaded.append(("sklearn", path)) or True),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                GestureClassifier()

            self.assertEqual(loaded, [("lstm", artifacts["lstm"])])

    def test_unresolved_lfs_model_pointers_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = {
                "onnx": os.path.join(directory, "model.onnx"),
                "metadata": os.path.join(directory, "model.json"),
                "lstm": os.path.join(directory, "model.pt"),
                "sklearn": os.path.join(directory, "model.pkl"),
            }
            with open(artifacts["onnx"], "wb") as artifact:
                artifact.write(b"onnx placeholder")
            with open(artifacts["metadata"], "w", encoding="utf-8") as metadata:
                metadata.write('{"classes": ["J"]}')
            pointer = (
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:" + "0" * 64 + "\nsize 270000\n"
            )
            for name in ("lstm", "sklearn"):
                with open(artifacts[name], "w", encoding="ascii") as artifact:
                    artifact.write(pointer)

            loaded = []
            patches = (
                patch.object(gesture_classifier, "ONNX_MODEL_PATH", artifacts["onnx"]),
                patch.object(gesture_classifier, "LSTM_META_PATH", artifacts["metadata"]),
                patch.object(gesture_classifier, "LSTM_MODEL_PATH", artifacts["lstm"]),
                patch.object(gesture_classifier, "SKLEARN_MODEL_PATH", artifacts["sklearn"]),
                patch.object(GestureClassifier, "_load_onnx", lambda classifier, path: loaded.append(("onnx", path)) or True),
                patch.object(GestureClassifier, "_load_lstm", lambda classifier, path: loaded.append(("lstm", path)) or True),
                patch.object(GestureClassifier, "_load_sklearn", lambda classifier, path: loaded.append(("sklearn", path)) or True),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                GestureClassifier()

            self.assertEqual(loaded, [("onnx", artifacts["onnx"])])

    def test_call_room_page_and_script_remain_available(self):
        client = app.app.test_client()
        with client.session_transaction() as browser_session:
            browser_session["username"] = "gesture-call-test"

        page = client.get("/call/room123")
        script = client.get("/static/call.js")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"call.js", page.data)
        self.assertEqual(script.status_code, 200)
        self.assertIn(b"function startCall()", script.data)

    def test_ready_reports_optional_live_gesture_availability(self):
        client = app.app.test_client()
        static_classifier = SimpleNamespace(
            pipeline=object(),
            n_features_in_=42,
            categories={0: "A"},
        )
        with (
            patch.object(app, "_load_inference_modules"),
            patch.object(app, "_inference_status", {"ready": True, "loading": False, "error": ""}),
            patch.object(app, "_classifier", static_classifier),
            patch.object(app, "_extractor", object()),
            patch.object(app, "_detector", object()),
            patch.object(app, "_gesture_classifier", None),
            patch.object(app, "LIVE_GESTURES_ENABLED", True),
        ):
            response = client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ready")
        self.assertTrue(response.get_json()["live_gestures_enabled"])
        self.assertFalse(response.get_json()["live_gestures_available"])
        self.assertEqual(response.get_json()["live_gesture_labels"], [])


if __name__ == "__main__":
    unittest.main()
