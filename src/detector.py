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

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

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
        confidence: float = 0.5,
        iou: float = 0.45,
        device: str | None = None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.names = self.model.names

    def predict(self, frame: np.ndarray) -> list[Detection]:
        kwargs = {
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model(frame, **kwargs)
        detections: list[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=self.names[cls_id],
                    confidence=conf,
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
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
