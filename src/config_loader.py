"""Load cấu hình từ .env và settings.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    camera_source: str | int
    model_path: Path
    confidence: float
    iou: float
    show_preview: bool
    upper_body_ratio: float
    overlap_iou_threshold: float
    violation_confirm_frames: int
    save_violation_images: bool
    violations_dir: Path
    classes: list[str] = field(default_factory=list)


def _parse_camera_source(raw: str) -> str | int:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return raw


def load_config(env_path: Path | None = None) -> AppConfig:
    env_path = env_path or PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(settings_path, encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    model_path = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/best_uniform_model.pt")

    return AppConfig(
        camera_source=_parse_camera_source(os.getenv("CAMERA_SOURCE", "0")),
        model_path=model_path,
        confidence=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
        iou=float(os.getenv("IOU_THRESHOLD", "0.45")),
        show_preview=os.getenv("SHOW_PREVIEW", "true").lower() in ("1", "true", "yes"),
        upper_body_ratio=float(settings.get("upper_body_ratio", 0.55)),
        overlap_iou_threshold=float(settings.get("overlap_iou_threshold", 0.1)),
        violation_confirm_frames=int(settings.get("violation_confirm_frames", 5)),
        save_violation_images=bool(settings.get("save_violation_images", True)),
        violations_dir=PROJECT_ROOT / settings.get("violations_dir", "data/violations"),
        classes=settings.get("classes", []),
    )
