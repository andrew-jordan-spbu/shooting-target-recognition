import cv2
import numpy as np
from typing import Iterator
from config.settings import RTSP_URL


class IPStreamSource:
    def __init__(self, url: str = RTSP_URL):
        self.url = url
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "IPStreamSource":
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.url}")

        # Reduce latency and avoid frame accumulation on unreliable RTSP streams.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()

    def get_frames(self) -> Iterator[np.ndarray]:
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Stream {self.url} is not open")

        failed_reads = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads >= 3:
                    break
                continue
            failed_reads = 0
            yield frame
