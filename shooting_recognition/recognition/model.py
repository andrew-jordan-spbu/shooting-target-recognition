import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config.settings import DEVICE, HOLE_CLASSES, MODEL_CONFIDENCE, MODEL_PATH, SCORE_CLASSES

_model = None
_mock_mode: bool = False


def _load_model() -> None:
    global _model, _mock_mode
    if not Path(MODEL_PATH).exists():
        print(f"[WARNING] Model not found at {MODEL_PATH}. Running in mock mode.")
        _mock_mode = True
        return
    try:
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
        _model.to(DEVICE)
        _mock_mode = False
        print(f"[INFO] Model loaded on device={DEVICE}")
    except Exception as exc:
        print(f"[WARNING] Model load failed ({exc}). Running in mock mode.")
        _mock_mode = True


class TargetModel:
    """YOLOv8 inference wrapper with MPS/CUDA/CPU device selection."""

    def __init__(self) -> None:
        _load_model()

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Return list of detected bullet holes with score, box, confidence."""
        if _mock_mode:
            return self._mock_detect(frame)
        return self._real_detect(frame)

    def _real_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = _model(frame, conf=MODEL_CONFIDENCE, device=DEVICE, verbose=False)
        holes: List[Dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = result.names[cls_id]

                # Model with 10 zone classes → score comes from the model directly.
                # Model with 1 generic class (e.g. "bulletholes") → score=0 here;
                # scorer.py will resolve it via calibration zones.
                score = SCORE_CLASSES.get(cls_name, 0)
                is_hole = score > 0 or cls_name in HOLE_CLASSES

                if not is_hole:
                    continue  # skip non-hole classes (e.g. "target", "black_contour")

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                holes.append({
                    "score": score,          # 0 means "let scorer decide via zones"
                    "class": cls_name,
                    "confidence": round(float(box.conf[0]), 3),
                    "box": [x1, y1, x2, y2],
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                })
        return holes

    def _mock_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Return a single random hole for UI testing (no real model needed)."""
        h, w = frame.shape[:2]
        score = random.randint(1, 10)
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        r = 14
        return [{
            "score": score,
            "class": str(score),
            "confidence": round(random.uniform(0.70, 0.99), 3),
            "box": [cx - r, cy - r, cx + r, cy + r],
            "center": [float(cx), float(cy)],
        }]
