import cv2
import numpy as np
from pathlib import Path
from typing import Iterator
from config.settings import ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT


class FileSource:
    def __init__(self, path: str):
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "FileSource":
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()

    def get_frames(self) -> Iterator[np.ndarray]:
        ext = self.path.suffix.lower()
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        if ext in ALLOWED_IMAGE_EXT:
            frame = cv2.imread(str(self.path))
            if frame is None:
                raise RuntimeError(f"Cannot read image: {self.path}")
            yield frame

        elif ext in ALLOWED_VIDEO_EXT:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.path}")
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                yield frame

        else:
            raise ValueError(f"Unsupported file format: {ext}")
