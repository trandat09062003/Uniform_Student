"""
Script chuyên dụng kiểm tra luồng Camera (Webcam / Camera ngoài / RTSP / DroidCam).
Hỗ trợ độ phân giải cao 1080x1920 (dọc) hoặc 1920x1080 (ngang).

Cách sử dụng:
  # 1. Quét tìm tất cả camera đang cắm vào máy:
  python scripts/stream_test.py --scan

  # 2. Test luồng camera mặc định (đọc từ .env):
  python scripts/stream_test.py

  # 3. Test camera ngoài (Index 1 hoặc 2) với độ phân giải 1080x1920:
  python scripts/stream_test.py --source 1 --width 1080 --height 1920

  # 4. Bật sẵn mô hình AI YOLO nhận diện:
  python scripts/stream_test.py --source 1 --ai

Phím tắt trong khi xem luồng:
  [SPACE] : Bật / Tắt nhận diện AI YOLO thời gian thực
  [S]     : Chụp ảnh frame hiện tại lưu vào data/test_results/
  [Q]     : Thoát chương trình
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.camera import CameraStream
from src.config_loader import load_config
from src.detector import UniformDetector
from src.tracker_history import StudentTemporalTracker
from src.violation_checker import ViolationChecker
from src.visualize import draw_text_unicode, draw_tracking_hud, draw_violations


def scan_cameras(max_tested: int = 6) -> list[int]:
    """Quét và liệt kê các index camera hợp lệ trên hệ thống."""
    print("=" * 60)
    print("🔍 ĐANG QUÉT CÁC THIẾT BỊ CAMERA ĐANG KẾT NỐI VỚI MÁY...")
    print("=" * 60)
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  👉 Camera Index [{i}]: Mở thành công! Độ phân giải mặc định: {w}x{h} (FPS: {fps:.1f})")
            available.append(i)
            cap.release()
        else:
            print(f"  ❌ Camera Index [{i}]: Không tìm thấy thiết bị.")
    print("=" * 60)
    if available:
        print(f"✅ Đã tìm thấy {len(available)} camera: {available}")
        print("💡 Gợi ý: Nếu dùng camera ngoài USB, thường là Index 1 (hoặc 2 nếu máy có sẵn webcam).")
    else:
        print("⚠️ Không tìm thấy camera nào cắm cổng USB.")
    print("=" * 60)
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Camera Stream & Uniform AI")
    parser.add_argument("--scan", action="store_true", help="Quét tất cả camera có trên máy")
    parser.add_argument("--source", default=None, help="Nguồn camera: số 0, 1, 2... hoặc URL rtsp/http")
    parser.add_argument("--width", type=int, default=None, help="Chiều rộng mong muốn (vd: 1080)")
    parser.add_argument("--height", type=int, default=None, help="Chiều cao mong muốn (vd: 1920)")
    parser.add_argument("--ai", action="store_true", help="Bật AI nhận diện ngay từ đầu")
    args = parser.parse_args()

    if args.scan:
        scan_cameras()
        return

    cfg = load_config()

    # Xác định camera source
    source = args.source if args.source is not None else cfg.camera_source
    if isinstance(source, str) and source.strip().isdigit():
        source = int(source.strip())

    req_w = args.width or cfg.camera_width
    req_h = args.height or cfg.camera_height

    print("=" * 60)
    print("🎥 KHỞI TẠO TEST LUỒNG CAMERA")
    print(f" - Nguồn camera: {source}")
    print(f" - Độ phân giải yêu cầu: {req_w}x{req_h if req_h else 'Mặc định'}")
    print("=" * 60)

    try:
        stream = CameraStream(source, width=req_w, height=req_h).start()
    except Exception as e:
        print(f"❌ [LỖI] Không thể mở camera '{source}': {e}")
        print("💡 Hãy thử chạy lệnh sau để kiểm tra danh sách cổng camera:")
        print("   python scripts/stream_test.py --scan")
        sys.exit(1)

    # Chuẩn bị AI (Nhóm 1: Tracking + Temporal Voting)
    detector = None
    checker = None
    temporal_tracker = None
    enable_ai = args.ai
    if cfg.model_path.exists():
        detector = UniformDetector(
            str(cfg.model_path),
            confidence=cfg.confidence,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
        )
        checker = ViolationChecker(cfg.upper_body_ratio, cfg.overlap_iou_threshold)
        temporal_tracker = StudentTemporalTracker(
            min_student_height=cfg.min_student_height,
            min_track_frames=cfg.min_track_frames,
            item_confirm_count=cfg.item_confirm_count,
            require_scarf=cfg.require_scarf,
            upper_body_ratio=cfg.upper_body_ratio,
            overlap_iou_threshold=cfg.overlap_iou_threshold,
        )

    window_name = f"Camera Stream Test [{source}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False

    fps_time = time.time()
    fps = 0.0
    frame_counter = 0

    out_dir = ROOT / "data" / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n👉 CÁC PHÍM ĐIỀU KHIỂN:")
    print("  [SPACE] : Bật / Tắt AI nhận diện (Nhóm 1)")
    print("  [S]     : Chụp & lưu ảnh hiện tại")
    print("  [Q]     : Thoát\n")

    try:
        while True:
            frame = stream.read(timeout=2.0)
            if frame is None:
                print("[Camera] Đang đợi tín hiệu hình ảnh...")
                continue

            h, w = frame.shape[:2]

            # Cân chỉnh kích thước cửa sổ hiển thị vừa vặn màn hình PC
            if not window_initialized:
                if h > 850:
                    scale = 800.0 / h
                    cv2.resizeWindow(window_name, int(w * scale), 800)
                else:
                    cv2.resizeWindow(window_name, w, h)
                window_initialized = True

            frame_counter += 1
            if frame_counter % 15 == 0:
                elapsed = time.time() - fps_time
                fps = 15.0 / elapsed if elapsed > 0 else 0.0
                fps_time = time.time()

            display_frame = frame.copy()
            det_count = 0
            student_status_summary = ""

            # Xử lý AI nếu được kích hoạt (Nhóm 1)
            if enable_ai and detector and temporal_tracker:
                detections = detector.track(frame, persist=True)
                det_count = len(detections)
                histories = temporal_tracker.update(detections)
                display_frame = draw_tracking_hud(frame, detections, histories)

                # Tổng kết trạng thái học sinh
                for tid, hist in histories.items():
                    student_status_summary = f"[#{tid}] {hist.status}"

            # Vẽ thanh thông số trạng thái ở góc trên
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            ai_status_text = "AI: DANG BAT" if enable_ai else "AI: TAT (Nhan SPACE de bat)"
            ai_color = (0, 255, 0) if enable_ai else (0, 180, 255)

            cv2.putText(display_frame, f"Do phan giai: {w}x{h} | FPS: {fps:.1f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(display_frame, ai_status_text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ai_color, 2)

            if enable_ai and student_status_summary:
                display_frame = draw_text_unicode(
                    display_frame, student_status_summary, (20, 72), (0, 255, 255), font_size=15
                )
            elif enable_ai:
                cv2.putText(display_frame, f"Doi tuong: {det_count}", (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            else:
                cv2.putText(display_frame, "[S]: Chup anh | [Q]: Thoat", (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                enable_ai = not enable_ai
                print(f"[Điều khiển] Đã {'BẬT' if enable_ai else 'TẮT'} AI nhận diện.")
            elif key in (ord("s"), ord("S")):
                ts = int(time.time())
                save_path = out_dir / f"cam_test_{ts}.jpg"
                cv2.imwrite(str(save_path), display_frame)
                print(f"📸 [Chụp ảnh] Đã lưu ảnh chụp từ camera vào: {save_path}")

    except KeyboardInterrupt:
        print("\n[Hệ thống] Dừng test camera.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

