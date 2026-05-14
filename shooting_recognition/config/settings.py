import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# --- Paths ---
UPLOAD_FOLDER    = BASE_DIR / "uploads"
OUTPUT_FOLDER    = BASE_DIR / "outputs" / "reports"
MODEL_PATH       = BASE_DIR / "recognition" / "weights" / "best.pt"
CALIBRATION_PATH = BASE_DIR / "calibration" / "zones.json"

# --- PostgreSQL ---
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME", "shooting_recognition")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# --- Camera ---
DEFAULT_CAMERA_INDEX = 0
RTSP_URL             = os.getenv("RTSP_URL", "rtsp://192.168.1.100/stream")
FRAME_WIDTH          = 1280
FRAME_HEIGHT         = 720

# --- Motion detection ---
MOTION_THRESHOLD   = int(os.getenv("MOTION_THRESHOLD", 5000))
MOTION_BLUR_KERNEL = 21
MOTION_DELAY_SEC   = float(os.getenv("MOTION_DELAY_SEC", 1.5))

# --- Model ---
MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", 0.50))

# Classes that carry a score directly from the model.
# If the model was trained with 10 zone classes (1–10), list them here.
# If the model only detects holes (e.g. "bulletholes"), leave this empty —
# scoring will be handled entirely by the calibration zones.
SCORE_CLASSES = {
    "10": 10, "9": 9, "8": 8, "7": 7,
    "6": 6,  "5": 5, "4": 4, "3": 3,
    "2": 2,  "1": 1,
}

# Classes that represent a bullet hole but carry no score from the model.
# Score for these is resolved purely via calibration zones.
HOLE_CLASSES = {"bulletholes", "bullet_hole", "hole", "Bulletholes"}

# --- Device (MPS → CUDA → CPU) ---
def _detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()

# --- File upload ---
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_MB     = 100

# --- Calibration ---
CALIBRATION_FRAMES = 5   # frames collected before auto-detect runs

# --- Flask ---
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
DEBUG      = os.getenv("DEBUG", "true").lower() == "true"
PORT       = int(os.getenv("PORT", 5000))
