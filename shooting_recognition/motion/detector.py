import time
import cv2
import numpy as np
from config.settings import MOTION_THRESHOLD, MOTION_BLUR_KERNEL, MOTION_DELAY_SEC


class MotionDetector:
    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._last_trigger: float = 0.0

    def has_changed(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (MOTION_BLUR_KERNEL, MOTION_BLUR_KERNEL), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(thresh)
        self._prev_gray = gray

        now = time.monotonic()
        if changed_pixels > MOTION_THRESHOLD and (now - self._last_trigger) > MOTION_DELAY_SEC:
            self._last_trigger = now
            return True
        return False

    def reset(self) -> None:
        self._prev_gray = None
        self._last_trigger = 0.0
