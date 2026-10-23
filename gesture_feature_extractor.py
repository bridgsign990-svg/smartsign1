"""
BridgeSign – Gesture Feature Extractor
=======================================
Captures a sliding window of hand-landmark frames and computes
temporal (motion) features for gesture recognition.

The static FeatureExtractor looks at ONE frame → shape of hand.
This module looks at N frames → movement of hand over time.

Used by gesture_classifier.py and gesture_collector.py.
"""

import numpy as np
from collections import deque

# How many frames make up one gesture sample
GESTURE_WINDOW = 15          # ~0.5s at 30fps
STATIC_FEATURES = 42        # 21 landmarks × 2 coords per frame
GESTURE_FEATURE_SIZE = GESTURE_WINDOW * STATIC_FEATURES   # 630 raw
VELOCITY_FEATURES = (GESTURE_WINDOW - 1) * STATIC_FEATURES  # 588 deltas
SUMMARY_FEATURES = 42 + 1   # displacement per landmark + total magnitude

TOTAL_GESTURE_FEATURES = GESTURE_FEATURE_SIZE + VELOCITY_FEATURES + SUMMARY_FEATURES
# = 630 + 588 + 43 = 1261 features

# Wrist-normalised coordinates of a real hand stay within a few units. Values
# far outside that come from a degenerate wrist->MCP scale (bad detection) and
# would dominate the model's input normalisation, so they are clipped.
FEATURE_CLIP = 5.0


def sanitize_sequence(sequence):
    """Return a float32 copy with non-finite values zeroed and outliers clipped."""
    seq = np.nan_to_num(np.asarray(sequence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(seq, -FEATURE_CLIP, FEATURE_CLIP)


def mirror_sequence(sequence):
    """Mirror every frame on X (left/right hand, or mirrored vs raw camera)."""
    mirrored = np.array(sequence, dtype=np.float32, copy=True)
    mirrored[..., 0::2] *= -1.0
    return mirrored


def resample_sequence(sequence, length=GESTURE_WINDOW, speed=1.0, center=None):
    """
    Resample a (frames, features) sequence onto ``length`` frames.

    ``speed`` > 1 plays the motion faster (it occupies fewer output frames and
    the ends hold the first/last pose); ``speed`` < 1 plays it slower (only the
    middle of the motion fits in the window). This mimics the same gesture
    seen at a different camera frame rate or performed at a different pace.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    n = len(seq)
    if n == 0:
        return np.zeros((length, seq.shape[-1] if seq.ndim == 2 else STATIC_FEATURES), dtype=np.float32)
    if n == 1:
        return np.repeat(seq, length, axis=0)
    if center is None:
        center = (n - 1) / 2.0
    offsets = (np.arange(length, dtype=np.float32) - (length - 1) / 2.0)
    positions = center + offsets * speed * (n - 1) / max(length - 1, 1)
    positions = np.clip(positions, 0.0, n - 1)
    lo = np.floor(positions).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = (positions - lo)[:, None]
    return (seq[lo] * (1.0 - frac) + seq[hi] * frac).astype(np.float32)


class GestureFeatureExtractor:
    """
    Maintains a sliding window of static feature vectors and produces
    a rich temporal feature vector when the window is full.
    """

    def __init__(self, window_size=GESTURE_WINDOW):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def push_frame(self, static_features):
        """
        Add one frame's static features (42-dim) to the buffer.

        Parameters
        ----------
        static_features : np.ndarray of shape (42,) or None
            Output from FeatureExtractor.extract_features().
            If None, the frame is skipped.
        """
        if static_features is not None:
            self.buffer.append(sanitize_sequence(static_features))

    def is_ready(self):
        """True when the buffer has enough frames to extract a gesture."""
        return len(self.buffer) >= self.window_size

    def get_raw_sequence(self):
        """
        Return the raw frame buffer as a (window, 42) numpy array.
        Used by the LSTM model which takes sequence input directly.
        Returns None if buffer isn't full.
        """
        if not self.is_ready():
            return None
        return np.array(list(self.buffer), dtype=np.float32)  # (15, 42)


    def get_motion_magnitude(self):
        """
        Quick check: how much total motion is in the buffer?
        Used by the decider to know if the hand is moving or static.
        Returns 0.0 if buffer isn't full.
        """
        if not self.is_ready():
            return 0.0
        frames = np.array(list(self.buffer))    # (window, 42)
        deltas = np.diff(frames, axis=0)         # (window-1, 42)
        return float(np.sum(np.abs(deltas)))

    def extract_gesture_features(self):
        """
        Extract the full temporal feature vector from the current buffer.

        Returns
        -------
        np.ndarray of shape (TOTAL_GESTURE_FEATURES,) or None if not ready.
        """
        if not self.is_ready():
            return None

        frames = np.array(list(self.buffer), dtype=np.float32)  # (15, 42)

        # 1. Raw flattened sequence (captures exact positions over time)
        raw = frames.flatten()                       # 630

        # 2. Frame-to-frame velocity (captures movement direction/speed)
        deltas = np.diff(frames, axis=0).flatten()   # 588

        # 3. Summary: displacement from first to last frame per landmark
        displacement = (frames[-1] - frames[0])      # 42
        total_mag = np.array([np.linalg.norm(displacement)])  # 1

        summary = np.concatenate([displacement, total_mag])   # 43

        return np.concatenate([raw, deltas, summary])

    def clear(self):
        """Reset the buffer (e.g., when hand disappears)."""
        self.buffer.clear()


def extract_from_sequence(frame_list):
    """
    Utility: extract gesture features from a pre-recorded list of
    static feature vectors. Used by the trainer.

    Parameters
    ----------
    frame_list : list of np.ndarray, each shape (42,), length = GESTURE_WINDOW

    Returns
    -------
    np.ndarray of shape (TOTAL_GESTURE_FEATURES,) or None
    """
    ext = GestureFeatureExtractor(window_size=len(frame_list))
    for f in frame_list:
        ext.push_frame(f)
    return ext.extract_gesture_features()
