"""
Script kiểm thử mô hình với cả Ảnh (.jpg/.png) và Video (.mp4/.avi).
Không gửi email, không kết nối IoT — Tự động LUÔN LUÔN LƯU kết quả nhận diện vào thư mục data/test_results/.

Cách sử dụng:
  # Test ảnh (Tự động lưu)
  python scripts/test_media.py data/Cus_Data/test_image.jpg
  
  # Test video (Tự động lưu)
  python scripts/test_media.py data/Cus_Data/7848429174354.mp4
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


def test_image(image_path: Path, detector: UniformDetector, checker: ViolationChecker) -> None:
    print(f"\n[IMAGE TEST] Đang tải ảnh: {image_path.name}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[LỖI] Không thể đọc ảnh tại: {image_path}")
        return

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

    # Vẽ kết quả
    annotated = draw_violations(frame, violations, detections)

    # Tự động lưu ảnh kết quả
    out_dir = ROOT / "data" / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"result_{image_path.name}"
    cv2.imwrite(str(save_path), annotated)
    print(f"[HỆ THỐNG] Đã lưu kết quả tại: {save_path}")

    # Hiển thị preview
    cv2.imshow(f"Test Image - {image_path.name}", annotated)
    print("Nhấn PHÍM BẤT KỲ trên bàn phím (trong cửa sổ ảnh) để đóng cửa sổ.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_video(video_path: Path, detector: UniformDetector, checker: ViolationChecker) -> None:
    print(f"\n[VIDEO TEST] Đang tải video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở video tại: {video_path}")
        return

    # Lấy thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Tự động lưu video kết quả
    out_dir = ROOT / "data" / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"result_{video_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    print(f"[HỆ THỐNG] Đang ghi video kết quả vào: {save_path}")

    print("Nhấn phím 'q' để dừng phát video.")
    print("-" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.predict(frame)
        violations = checker.check(detections)
        if not violations:
            violations = checker.check_frame_level(detections)

        # In log vi phạm thời gian thực lên console nếu có
        for v in violations:
            print(f"⚠️ [Vi Phạm] {v.message}")

        annotated = draw_violations(frame, violations, detections)

        writer.write(annotated)

        cv2.imshow(f"Test Video - {video_path.name}", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("-" * 50)
    print(f"[VIDEO TEST] Hoàn tất và lưu video tại: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test YOLO11 Model with Image or Video")
    parser.add_argument("path", help="Đường dẫn tới file ảnh (.jpg/.png) hoặc video (.mp4/.avi)")
    args = parser.parse_args()

    media_path = Path(args.path)
    if not media_path.exists():
        print(f"[LỖI] File không tồn tại: {media_path}")
        sys.exit(1)

    cfg = load_config()
    
    if not cfg.model_path.exists():
        print(f"[LỖI] Không tìm thấy file weights tại: {cfg.model_path}")
        sys.exit(1)

    print(f"[HỆ THỐNG] Đang tải mô hình từ: {cfg.model_path}")
    detector = UniformDetector(str(cfg.model_path), cfg.confidence, cfg.iou)
    checker = ViolationChecker(cfg.upper_body_ratio, cfg.overlap_iou_threshold)

    suffix = media_path.suffix.lower()
    if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        test_image(media_path, detector, checker)
    elif suffix in [".mp4", ".avi", ".mkv", ".mov"]:
        test_video(media_path, detector, checker)
    else:
        print(f"[LỖI] Định dạng file không được hỗ trợ: {suffix}")


if __name__ == "__main__":
    main()
