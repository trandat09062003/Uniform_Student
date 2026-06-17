"""
Export model sang TensorRT (.engine) cho Jetson / GPU NVIDIA.

Chạy trên máy có CUDA:
  python scripts/export_tensorrt.py --weights models/best_uniform_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "models" / "best_uniform_model.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 (Jetson)")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Không tìm thấy: {weights}")
        return

    model = YOLO(str(weights))
    model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=0,
    )
    print("[OK] File .engine cùng thư mục với weights. Đặt MODEL_PATH trong .env trỏ tới file .engine")


if __name__ == "__main__":
    main()
