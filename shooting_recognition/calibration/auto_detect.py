import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import CALIBRATION_PATH


def detect_target(frame: np.ndarray) -> Optional[dict]:
    """
    Detect the outermost ring of the shooting target using Hough circles.
    Returns a zone dict (10 concentric zones) or None if detection fails.
    The returned zones are ordered from innermost (score=10) to outermost (score=1).
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    min_r = min(h, w) // 8
    max_r = min(h, w) // 2

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w) // 2,
        param1=50,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Pick the largest detected circle as the target boundary
        best = sorted(circles, key=lambda c: c[2], reverse=True)[0]
        cx, cy, outer_radius = int(best[0]), int(best[1]), int(best[2])
    else:
        # Fallback: assume target is centred in the frame
        cx, cy = w // 2, h // 2
        outer_radius = min(h, w) // 2 - 20

    zones = _build_zones(cx, cy, outer_radius)
    return {
        "cx": cx,
        "cy": cy,
        "outer_radius": outer_radius,
        "zones": zones,
        "frame_shape": [h, w],
        "auto_detected": circles is not None,
    }


def _build_zones(cx: int, cy: int, outer_radius: int) -> list:
    """Build 10 concentric scoring zones (score 10 = innermost, 1 = outermost)."""
    zones = []
    for score in range(10, 0, -1):
        zone_radius = int(outer_radius * (11 - score) / 10)
        zones.append({"score": score, "cx": cx, "cy": cy, "radius": zone_radius})
    return zones


def get_score_from_zones(x: float, y: float, zones: list) -> int:
    """
    Return the score for a point (x, y) given calibrated zones.
    Zones must be sorted innermost-first (score 10 → 1).
    """
    for zone in zones:
        dist = math.hypot(x - zone["cx"], y - zone["cy"])
        if dist <= zone["radius"]:
            return zone["score"]
    return 0


def save_zones(data: dict, path: Path = CALIBRATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_zones(path: Path = CALIBRATION_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
