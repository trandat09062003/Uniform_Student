"""Đọc luồng camera RTSP / webcam / file video."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np


class CameraStream:
    """Capture đa luồng: đọc frame nền, main loop không bị block."""

    def __init__(self, source: str | int, buffer_size: int = 2):
        self.source = source
        self.buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None
        self._queue: Queue[np.ndarray] = Queue(maxsize=buffer_size)
        self._stopped = True
        self._thread: threading.Thread | None = None

    def start(self) -> "CameraStream":
        self._cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn camera: {self.source}")

        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        while not self._stopped and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            if not self._queue.full():
                self._queue.put(frame)
            else:
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
                self._queue.put(frame)

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        self._stopped = True
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()
