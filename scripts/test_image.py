"""
Script kiểm thử mô hình chuyên dụng cho HÌNH ẢNH (.jpg/.png/.jpeg).
Tự động nhận diện và LUÔN LUÔN LƯU kết quả gán nhãn vào thư mục data/test_results/.

Cách sử dụng:
  python scripts/test_image.py "đường_dẫn_ảnh.png"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.detector import UniformDetector
from src.violation_checker import ViolationChecker
from src.visualize import draw_violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Test YOLO11 Model with an Image")
    parser.add_argument("path", help="Đường dẫn tới file ảnh (.jpg/.png/.jpeg)")
    args = parser.parse_args()

    image_path = Path(args.path)
    if not image_path.exists():
        print(f"[LỖI] File không tồn tại: {image_path}")
        sys.exit(1)

    cfg = load_config()
    
    if not cfg.model_path.exists():
        print(f"[LỖI] Không tìm thấy file weights tại: {cfg.model_path}")
        sys.exit(1)

    print(f"[HỆ THỐNG] Đang tải mô hình từ: {cfg.model_path}")
    detector = UniformDetector(str(cfg.model_path), cfg.confidence, cfg.iou)
    checker = ViolationChecker(cfg.upper_body_ratio, cfg.overlap_iou_threshold)

    print(f"[IMAGE TEST] Đang kiểm tra ảnh: {image_path.name}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[LỖI] Không thể đọc ảnh tại: {image_path}")
        sys.exit(1)

    # Dự đoán và kiểm tra vi phạm
    detections = detector.predict(frame)
    violations = checker.check(detections)
    if not violations:
        violations = checker.check_frame_level(detections)

    # In thông tin vi phạm ra console
    print("-" * 50)
    print(f"Phát hiện {len(detections)} đối tượng.")
    if violations:
        print(f"⚠️ CẢNH BÁO: Phát hiện {len(violations)} vi phạm:")
        for idx, v in enumerate(violations, 1):
            print(f"  {idx}. [{v.violation_type.name}] {v.message}")
    else:
        print("✅ Hoàn hảo: Học sinh mặc đúng đồng phục, không có vi phạm.")
    print("-" * 50)

    # Vẽ kết quả (vẽ nhãn tiếng Việt trên đầu, không vẽ box đỏ quanh học sinh)
    annotated = draw_violations(frame, violations, detections)

    # TỰ ĐỘNG LUÔN LUÔN LƯU ảnh kết quả
    out_dir = ROOT / "data" / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"result_{image_path.name}"
    cv2.imwrite(str(save_path), annotated)
    print(f"[HỆ THỐNG] Đã tự động lưu kết quả nhận diện tại: {save_path}")

    # Hiển thị preview
    cv2.imshow(f"Test Image - {image_path.name}", annotated)
    print("Nhấn PHÍM BẤT KỲ trên bàn phím (trong cửa sổ ảnh) để đóng cửa sổ.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
