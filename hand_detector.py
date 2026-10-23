import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import config
from model_artifacts import is_git_lfs_pointer

# Download model if not present
_MODEL_PATH = os.path.join(config.MODELS_DIR, "hand_landmarker.task")


def _ensure_model():
    """Download the hand landmarker model when the current file is missing or still a Git LFS pointer."""
    if os.path.exists(_MODEL_PATH) and not is_git_lfs_pointer(_MODEL_PATH):
        return

    import urllib.request

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    print(f"[HandDetector] Downloading hand landmarker model...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    if os.path.exists(_MODEL_PATH) and is_git_lfs_pointer(_MODEL_PATH):
        os.remove(_MODEL_PATH)

    try:
        urllib.request.urlretrieve(url, _MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Could not download the MediaPipe hand landmark model from {url}: {exc}") from exc

    if is_git_lfs_pointer(_MODEL_PATH):
        raise RuntimeError(f"Downloaded hand model path still contains a Git LFS pointer: {_MODEL_PATH}")

    print(f"[HandDetector] Model saved to {_MODEL_PATH}")

_ensure_model()

class HandDetector:
    def __init__(self,
                 mode=False,
                 max_hands=config.MAX_NUM_HANDS,
                 detection_con=config.MIN_DETECTION_CONFIDENCE,
                 track_con=config.MIN_TRACKING_CONFIDENCE):

        self.max_hands = max_hands
        self.results = None

        running_mode = mp_vision.RunningMode.IMAGE if mode else mp_vision.RunningMode.VIDEO

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con,
            min_tracking_confidence=track_con,
        )
        self._mode = running_mode
        self._options = options
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._image_detector = None  # lazily created IMAGE-mode fallback
        self._frame_ts = 0  # monotonically increasing timestamp for VIDEO mode

    def _detect_on_image(self, rgb_image):
        """Use raw image detection as a fallback when video tracking misses."""
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        if self._mode == mp_vision.RunningMode.IMAGE:
            return self._detector.detect(mp_img)
        # A VIDEO-mode landmarker rejects detect(); keep a separate IMAGE one.
        if self._image_detector is None:
            self._image_detector = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=self._options.base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=self._options.num_hands,
                    min_hand_detection_confidence=self._options.min_hand_detection_confidence,
                    min_hand_presence_confidence=self._options.min_hand_presence_confidence,
                )
            )
        return self._image_detector.detect(mp_img)

    def find_hands(self, img, draw=True):
        """Detect hands and optionally draw landmarks. Returns (img, results)."""
        if img is None or img.size == 0:
            self.results = None
            return img, self.results

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        if self._mode == mp_vision.RunningMode.VIDEO:
            self._frame_ts += 33
            try:
                self.results = self._detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    self._frame_ts,
                )
            except Exception as exc:
                print(f"[HandDetector] VIDEO detect failed: {exc}")
                self.results = None
        else:
            self.results = self._detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

        # MediaPipe video tracking is occasionally flaky in live webcam sessions, so
        # fall back to direct image detection before giving up on the frame.
        # (IMAGE mode already ran a plain detection above.)
        if (
            self._mode == mp_vision.RunningMode.VIDEO
            and (not self.results or not getattr(self.results, "hand_landmarks", None))
        ):
            try:
                self.results = self._detect_on_image(rgb)
            except Exception as exc:
                print(f"[HandDetector] IMAGE fallback failed: {exc}")
                self.results = None

        if draw and self.results and self.results.hand_landmarks:
            h, w, _ = img.shape
            for hand_lm in self.results.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                for pt in pts:
                    cv2.circle(img, pt, 5, (0, 215, 255), cv2.FILLED)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)
                ]
                for a, b in connections:
                    cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)

        return img, self.results

    def get_landmarks(self, img, hand_no=0):
        """
        Extract landmark list [id, x, y] (sub-pixel float coordinates) for a hand.

        Coordinates are not rounded: on the ~320 px frames the browser posts,
        whole-pixel rounding adds several percent of jitter relative to hand
        size, which the gesture model sees as motion.
        """
        lm_list = []
        if self.results and self.results.hand_landmarks:
            if len(self.results.hand_landmarks) > hand_no:
                h, w = img.shape[:2]
                hand_lm = self.results.hand_landmarks[hand_no]
                for id, lm in enumerate(hand_lm):
                    cx, cy = float(lm.x * w), float(lm.y * h)
                    lm_list.append([id, cx, cy])
        return lm_list
