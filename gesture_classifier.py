"""
BridgeSign -- Gesture Classifier (LSTM + sklearn fallback)
============================================================
Loads the trained gesture model and exposes a
predict(gesture_features) -> (label, confidence) interface.

Supports two model backends:
  1. LSTM (gesture_lstm.pt)  -- preferred, temporal-aware
  2. sklearn (gesture_model.pkl) -- fallback for backward compat

This runs ALONGSIDE the existing static Classifier -- it does NOT
replace it. The app.py pipeline decides which result to use.
"""

import os
import json
import pickle
import numpy as np
import config
from model_artifacts import is_git_lfs_pointer, remove_stale_git_lfs_pointers, remove_unresolved_git_lfs_pointer

# Gesture window settings
from gesture_feature_extractor import (
    GESTURE_WINDOW,
    STATIC_FEATURES,
    mirror_sequence,
    resample_sequence,
    sanitize_sequence,
)

LSTM_MODEL_PATH   = os.path.join(config.MODELS_DIR, "gesture_lstm.pt")
LSTM_META_PATH    = os.path.join(config.MODELS_DIR, "gesture_lstm_meta.json")
ONNX_MODEL_PATH   = os.path.join(config.MODELS_DIR, "gesture_lstm.onnx")
ONNX_V2_MODEL_PATH = os.path.join(config.MODELS_DIR, "gesture_lstm_v2.onnx")
SKLEARN_MODEL_PATH = os.path.join(config.MODELS_DIR, "gesture_model.pkl")


class GestureClassifier:
    def __init__(self, model_path=None):
        self.backend    = None   # "onnx", "lstm", or "sklearn"
        self.model      = None
        self.pipeline   = None   # sklearn only
        self.ort_session = None  # onnx only
        self.categories = {}     # {0: 'J', 1: 'Z', ...}
        self.n_features_in_ = None
        self._device    = "cpu"
        self.model_path = None
        self.last_error = None

        stale = remove_stale_git_lfs_pointers(getattr(config, "MODELS_DIR", ""))
        if stale:
            print(f"[GestureClassifier] Removed stale Git LFS pointer artifacts: {', '.join(stale)}")

        # Artifact timestamps are not stable across clones and deployments.
        # Select the named V2 export first, then use explicit legacy fallbacks.
        # A caller-supplied path always takes precedence.
        candidates = []
        fallback_candidates = []

        def add_candidate(backend, path):
            if not os.path.isfile(path):
                return
            if is_git_lfs_pointer(path):
                if remove_unresolved_git_lfs_pointer(path):
                    print(f"[GestureClassifier] Removed stale Git LFS pointer at {path}")
                else:
                    print(
                        f"[GestureClassifier] Skipping unresolved Git LFS pointer at {path}; "
                        "the model data are unavailable."
                    )
                return
            candidates.append((backend, path))

        if model_path:
            add_candidate("sklearn", model_path)

        # This named export is the intended current model. Prefer it explicitly
        # regardless of checkout/deployment mtimes; order legacy fallbacks by
        # recency only when the V2 artifact is absent or unusable.
        add_candidate("onnx", ONNX_V2_MODEL_PATH)
        for backend, path in (
            ("onnx", ONNX_MODEL_PATH),
            ("lstm", LSTM_MODEL_PATH),
            ("sklearn", SKLEARN_MODEL_PATH),
        ):
            if os.path.isfile(path):
                fallback_candidates.append((os.path.getmtime(path), backend, path))
        fallback_candidates.sort(reverse=True)
        for _, backend, path in fallback_candidates:
            add_candidate(backend, path)

        for backend, path in candidates:
            loaded = {
                "onnx": self._load_onnx,
                "lstm": self._load_lstm,
                "sklearn": self._load_sklearn,
            }[backend](path)
            if loaded:
                self.model_path = os.path.abspath(path)
                return

        if not candidates:
            self.last_error = "No gesture model artifact found."
            print("[GestureClassifier] No gesture model found -- gesture recognition disabled.")
            print("[GestureClassifier] Run lstm_gesture_trainer.py or gesture_trainer.py.")
        else:
            print("[GestureClassifier] No usable gesture model artifact found.")

    @property
    def available_labels(self):
        """Labels supplied by the successfully loaded production model."""
        return frozenset(self.categories.values())

    def _load_lstm(self, path):
        """Load the PyTorch LSTM model."""
        try:
            import torch
            from lstm_gesture_model import GestureLSTM

            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            classes = checkpoint["classes"]
            model = GestureLSTM(
                input_size=checkpoint.get("input_size", 42),
                hidden_size=checkpoint.get("hidden_size", 64),
                num_layers=checkpoint.get("num_layers", 2),
                num_classes=len(classes),
                dropout=0.0,  # no dropout at inference
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.categories = {i: c for i, c in enumerate(classes)}
            self.model = model
            self.backend = "lstm"
            self.model_path = os.path.abspath(path)
            self.last_error = None

            print(f"[GestureClassifier] LSTM model loaded from {path}")
            print(f"[GestureClassifier] {len(classes)} gestures: {classes}")
            return True

        except Exception as e:
            self.last_error = str(e)
            print(f"[GestureClassifier] Error loading LSTM model: {e}")
            return False

    def _load_sklearn(self, path):
        """Load the legacy scikit-learn gesture model."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            pipeline = data["pipeline"]
            classes_list = data["classes"]
            if not classes_list:
                raise ValueError("Gesture model contains no class labels")
            self.pipeline = pipeline
            self.categories = {i: c for i, c in enumerate(classes_list)}
            self.n_features_in_ = getattr(pipeline, "n_features_in_", None)
            self.backend = "sklearn"
            self.model_path = os.path.abspath(path)
            self.last_error = None

            print(f"[GestureClassifier] sklearn model loaded from {path}")
            print(f"[GestureClassifier] {len(self.categories)} gestures: {classes_list}")
            return True

        except Exception as e:
            self.last_error = str(e)
            print(f"[GestureClassifier] Error loading sklearn model: {e}")
            return False

    def _load_onnx(self, path):
        try:
            import onnxruntime as ort
            ort_session = ort.InferenceSession(path)
            
            embedded_classes = ort_session.get_modelmeta().custom_metadata_map.get(
                "bridgesign.classes"
            )
            if embedded_classes:
                classes = json.loads(embedded_classes)
            else:
                with open(LSTM_META_PATH, "r") as f:
                    meta = json.load(f)
                classes = meta["classes"]
            output_shape = ort_session.get_outputs()[0].shape
            output_classes = output_shape[-1] if output_shape else None
            if isinstance(output_classes, int) and output_classes != len(classes):
                raise ValueError(
                    f"ONNX output has {output_classes} classes, metadata lists {len(classes)}"
                )
            if not classes or any(not isinstance(label, str) or not label for label in classes):
                raise ValueError("Gesture model metadata contains invalid class labels")
            input_shape = ort_session.get_inputs()[0].shape
            if len(input_shape) != 3 or input_shape[-1] != STATIC_FEATURES:
                raise ValueError(
                    f"ONNX input must be rank 3 with {STATIC_FEATURES} features; got {input_shape}"
                )
            self.categories = {i: c for i, c in enumerate(classes)}
            self.ort_session = ort_session
            self.backend = "onnx"
            self.model_path = os.path.abspath(path)
            self.last_error = None
            print(f"[GestureClassifier] ONNX model loaded from {path}")
            print(f"[GestureClassifier] {len(classes)} gestures: {classes}")
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[GestureClassifier] Error loading ONNX model: {e}")
            return False

    def is_available(self):
        """True if a gesture model is loaded and ready."""
        if self.backend == "onnx":
            return self.ort_session is not None
        if self.backend == "lstm":
            return self.model is not None
        return self.pipeline is not None

    def predict(self, gesture_features):
        """
        Parameters
        ----------
        gesture_features : np.ndarray
            For LSTM: shape (15, 42) -- raw frame sequence from the buffer
            For sklearn: shape (N,) -- flattened temporal features from
                         GestureFeatureExtractor.extract_gesture_features()

        Returns
        -------
        label      : str   -- predicted gesture label (e.g., 'J', 'Z', 'STATIC')
        confidence : float -- probability in [0, 1]
        """
        if gesture_features is None or len(gesture_features) == 0:
            return "", 0.0

        if self.backend == "onnx":
            return self._predict_onnx(gesture_features)
        elif self.backend == "lstm":
            return self._predict_lstm(gesture_features)
        elif self.backend == "sklearn":
            return self._predict_sklearn(gesture_features)
        else:
            return "", 0.0

    @staticmethod
    def _as_sequence(features):
        """Coerce (15, 42) or flat (>=630,) input into a clean (15, 42) sequence."""
        features = np.array(features, dtype=np.float32)
        if features.ndim == 1:
            # Flat input from extract_gesture_features -- take only the raw portion
            raw_len = GESTURE_WINDOW * STATIC_FEATURES  # 630
            if len(features) < raw_len:
                return None
            features = features[:raw_len].reshape(GESTURE_WINDOW, STATIC_FEATURES)
        elif features.ndim != 2 or features.shape[1] != STATIC_FEATURES:
            return None
        return sanitize_sequence(features)

    @staticmethod
    def _orientation_batch(sequence):
        """
        Stack the sequence with its X-mirror. Training captures were mirrored
        webcam frames while the browser posts raw frames (and phones switch
        between mirrored front and raw back cameras), so both are scored and
        the more confident reading wins -- the same policy as the static path.
        """
        return np.stack([sequence, mirror_sequence(sequence)]).astype(np.float32)

    def _best_of(self, probs):
        """
        Pick the reading across orientations. The wrong orientation of a real
        gesture is often read as STATIC with high confidence, so an orientation
        that reads an actual gesture is preferred; downstream confidence gates
        still reject weak readings.
        """
        readings = [
            (self.categories.get(int(np.argmax(row)), "Unknown"), float(np.max(row)))
            for row in probs
        ]
        moving = [reading for reading in readings if reading[0] != "STATIC"]
        return max(moving or readings, key=lambda reading: reading[1])

    def _predict_lstm(self, features):
        """Run inference through the LSTM model."""
        try:
            import torch

            sequence = self._as_sequence(features)
            if sequence is None:
                return "", 0.0
            X = torch.tensor(self._orientation_batch(sequence))  # (2, T, 42)

            with torch.no_grad():
                probs = torch.softmax(self.model(X), dim=1).numpy()
            return self._best_of(probs)

        except Exception as e:
            print(f"[GestureClassifier] LSTM predict error: {e}")
            return "", 0.0

    def _predict_sklearn(self, gesture_features):
        """Run inference through the sklearn pipeline (legacy)."""
        try:
            X = np.array([gesture_features], dtype=np.float32)
            proba = self.pipeline.predict_proba(X)[0]
            class_id = int(np.argmax(proba))
            confidence = float(proba[class_id])
            label = self.categories.get(class_id, "Unknown")
            return label, confidence
        except Exception as e:
            print(f"[GestureClassifier] sklearn predict error: {e}")
            return "", 0.0

    def _predict_onnx(self, features):
        try:
            sequence = self._as_sequence(features)
            if sequence is None:
                return "", 0.0

            # ONNX expects (batch_size, seq_len, input_size)
            input_meta = self.ort_session.get_inputs()[0]
            input_shape = input_meta.shape
            expected_seq_len = input_shape[1] if len(input_shape) > 1 else None
            if isinstance(expected_seq_len, int) and expected_seq_len != len(sequence):
                # Older exports may have a fixed sequence length baked in.
                sequence = resample_sequence(sequence, length=expected_seq_len)

            X = self._orientation_batch(sequence)
            expected_batch = input_shape[0] if input_shape else None
            if isinstance(expected_batch, int) and expected_batch == 1:
                logits = np.concatenate([
                    self.ort_session.run(None, {input_meta.name: X[i:i + 1]})[0]
                    for i in range(len(X))
                ])
            else:
                logits = self.ort_session.run(None, {input_meta.name: X})[0]

            logits = np.asarray(logits, dtype=np.float32)
            if logits.ndim != 2 or logits.shape != (2, len(self.categories)):
                raise ValueError(
                    f"ONNX output must have shape (2, {len(self.categories)}); got {logits.shape}"
                )
            if not np.isfinite(logits).all():
                raise ValueError("ONNX output contains non-finite values")

            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            return self._best_of(probs)
        except Exception as e:
            print(f"[GestureClassifier] ONNX predict error: {e}")
            return "", 0.0

    def is_lstm(self):
        """True if the active backend is the LSTM model or ONNX."""
        return self.backend in ["lstm", "onnx"]
