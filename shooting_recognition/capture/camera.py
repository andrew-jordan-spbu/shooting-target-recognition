import cv2
import numpy as np
from typing import Iterator
from config.settings import DEFAULT_CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraSource:
    def __init__(self, index: int = DEFAULT_CAMERA_INDEX):
        self.index = index
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "CameraSource":
        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.index}")
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()

    def get_frames(self) -> Iterator[np.ndarray]:
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Camera {self.index} is not open")
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame
