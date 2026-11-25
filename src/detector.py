"""Wrapper YOLO inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def upper_body_bbox(self, ratio: float = 0.55) -> tuple[float, float, float, float]:
        """Vùng thân trên của học sinh (khăn quàng, bảng tên)."""
        h = self.y2 - self.y1
        return (self.x1, self.y1, self.x2, self.y1 + h * ratio)


class UniformDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 960,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.names = self.model.names

    def predict(self, frame: np.ndarray) -> list[Detection]:
        kwargs = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model(frame, **kwargs)
        return self._parse_results(results)

    def track(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        """Nhận diện kèm bám vết (Multi-object tracking) qua ByteTrack."""
        kwargs = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "persist": persist,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model.track(frame, **kwargs)
        return self._parse_results(results)

    def _parse_results(self, results) -> list[Detection]:
        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        has_id = boxes.id is not None
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = self.names[cls_id]
            conf = float(boxes.conf[i].item())
            # Lọc ngưỡng cao hơn cho hoc_sinh (>=0.60) và áo/quần (>=0.50) để tránh nhận nhầm nền tối/cửa
            if cls_name == "hoc_sinh" and conf < max(self.confidence, 0.60):
                continue
            if cls_name in {"ao_trang", "quan_xanh"} and conf < max(self.confidence, 0.50):
                continue
            xyxy = boxes.xyxy[i].cpu().numpy()
            tid = int(boxes.id[i].item()) if has_id and boxes.id[i] is not None else None
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    track_id=tid,
                )
            )
        return detections

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Vẽ bbox lên frame (dùng ultralytics plot)."""
        kwargs = {"conf": self.confidence, "iou": self.iou, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model(frame, **kwargs)
        return results[0].plot()
