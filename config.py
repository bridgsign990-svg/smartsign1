import os
import secrets
import tempfile

def _load_secret_key(environ=None):
    environment = os.environ if environ is None else environ
    secret_key = environment.get("SECRET_KEY", "").strip()
    if secret_key:
        return secret_key, False

    production_markers = ("FLASK_ENV", "APP_ENV", "ENVIRONMENT")
    is_production = (
        bool(environment.get("RENDER") or environment.get("RAILWAY_ENVIRONMENT"))
        or any(
            environment.get(marker, "").strip().lower() == "production"
            for marker in production_markers
        )
    )
    if is_production:
        raise RuntimeError("SECRET_KEY must be configured in production.")

    return secrets.token_urlsafe(48), True


SECRET_KEY, SECRET_KEY_IS_EPHEMERAL = _load_secret_key()
PORT       = int(os.environ.get("PORT", 5000))
APP_NAME = "BridgeSign"
APP_VERSION = "1.0.0"

# Camera Settings
CAMERA_INDEX = 0
INFER_EVERY_N_FRAMES = int(os.environ.get("BRIDGESIGN_INFER_EVERY_N_FRAMES", 2))
CONSECUTIVE_THRESHOLD = int(os.environ.get("BRIDGESIGN_CONSECUTIVE_THRESHOLD", 5))
JPEG_QUALITY = 70          # Lower = faster stream
CAMERA_MAX_INDEX = 4
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Hand-switch grace period
# When the hand disappears, wait this many seconds before treating it as
# "no hand" for word-boundary purposes.  Absorbs brief tracking drops when
# repositioning or switching hands without triggering a false word boundary.
HAND_LOST_GRACE_SEC = 0.5  # 0.5 s feels instant but covers a hand-swap

# MediaPipe Settings
MIN_DETECTION_CONFIDENCE = 0.45   # Keep live detection tolerant; plausibility gate filters obvious non-hands
MIN_TRACKING_CONFIDENCE  = 0.45
MAX_NUM_HANDS            = 1    # Single hand = half the MediaPipe work

# Candidate floor and confirmed-output floor are intentionally separate.
# The UI may display lower-confidence predictions as guesses, but does not commit them.
MIN_PREDICTION_CONFIDENCE = 0.10
MIN_CONFIRMED_STATIC_CONFIDENCE = float(
    os.environ.get("BRIDGESIGN_MIN_CONFIRMED_STATIC_CONFIDENCE", "0.8")
)

# Performance Settings
LIVE_CLIENT_MAX_DIM = int(os.environ.get("BRIDGESIGN_LIVE_CLIENT_MAX_DIM", 320))
LIVE_FRAME_SKIP = int(os.environ.get("BRIDGESIGN_LIVE_FRAME_SKIP", 1))
CLASSIFIER_CACHE_SIZE = int(os.environ.get("BRIDGESIGN_CLASSIFIER_CACHE_SIZE", 2048))
CLASSIFIER_CACHE_DECIMALS = int(os.environ.get("BRIDGESIGN_CLASSIFIER_CACHE_DECIMALS", 2))
IMAGE_MAX_DIM = int(os.environ.get("BRIDGESIGN_IMAGE_MAX_DIM", 768))
IMAGE_MAX_UPLOAD_BYTES = int(os.environ.get("BRIDGESIGN_IMAGE_MAX_UPLOAD_BYTES", 15 * 1024 * 1024))
IMAGE_TRANSLATION_CACHE_SIZE = int(os.environ.get("BRIDGESIGN_IMAGE_CACHE_SIZE", 64))
STT_CACHE_SIZE = int(os.environ.get("BRIDGESIGN_STT_CACHE_SIZE", 128))
PERF_LOG_EVERY_N = int(os.environ.get("BRIDGESIGN_PERF_LOG_EVERY_N", 60))

# Path Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_dir(path, fallback_name=None):
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except OSError as e:
        if fallback_name:
            fallback = os.path.join(tempfile.gettempdir(), fallback_name)
            os.makedirs(fallback, exist_ok=True)
            print(f"[Config] Could not write to {path!r}: {e}. Using {fallback!r}.")
            return fallback
        print(f"[Config] Could not create {path!r}: {e}.")
        return path


MODELS_DIR = _ensure_dir(os.path.join(BASE_DIR, "models"))
DATA_DIR = _ensure_dir(
    os.environ.get("BRIDGESIGN_DATA_DIR", os.path.join(BASE_DIR, "data")),
    "bridgesign-data",
)
DATASET_DIR = _ensure_dir(os.path.join(BASE_DIR, "dataset"))

# TTS Settings
TTS_RATE = 150
TTS_VOLUME = 1.0

# Colors for UI (BGR format for OpenCV)
COLOR_PRIMARY = (255, 153, 51) # Blueish
COLOR_SECONDARY = (51, 204, 255) # Orangeish
COLOR_TEXT = (255, 255, 255)
COLOR_SUCCESS = (0, 255, 0)
COLOR_WARNING = (0, 0, 255)
