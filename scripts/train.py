"""
Huấn luyện YOLOv8/v11 trên dataset đồng phục.

Chuẩn bị dataset (YOLO format):
  dataset/
    images/train, images/val
    labels/train, labels/val

Hoặc export từ Roboflow (YOLOv8) rồi chỉnh data/dataset.yaml

Chạy:
  python scripts/train.py
  python scripts/train.py --model yolo11n.pt --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train uniform detection model")
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights (yolov8n/s/m or yolo11n.pt)")
    parser.add_argument("--data", default=str(DATA_YAML), help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="cuda device id or cpu")
    parser.add_argument("--project", default=str(ROOT / "runs" / "train"))
    parser.add_argument("--name", default="uniform_detect")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Tạo file mẫu tại {args.data} — cập nhật đường dẫn dataset trước khi train.")
        DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
        sample = """# Cập nhật path tới thư mục dataset của bạn
path: ../dataset
train: images/train
val: images/val

names:
  0: ao_trang
  1: quan_xanh
  2: khan_quang
  3: bang_ten
  4: hoc_sinh
  5: quan_sai_mau
  6: ao_sai_mau
"""
        DATA_YAML.write_text(sample, encoding="utf-8")
        print("Đã tạo data/dataset.yaml — sửa path rồi chạy lại.")
        return

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        patience=20,
        save=True,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = ROOT / "models" / "best_uniform_model.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        import shutil
        shutil.copy(best, dest)
        print(f"[OK] Đã copy weights -> {dest}")
    else:
        print(f"[OK] Train xong. Weights trong: {results.save_dir}/weights/")


if __name__ == "__main__":
    main()
