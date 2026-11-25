"""
True Form Student - Chạy real-time: Camera -> YOLO -> Nhận diện vi phạm & điểm danh OhStem.
Hoàn toàn loại bỏ tính năng gửi email để chạy mượt mà và tối giản.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2

# Thêm project root vào path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.camera import CameraStream
from src.config_loader import load_config
from src.debounce import FrameViolationCounter
from src.detector import UniformDetector
from src.visualize import draw_tracking_hud, draw_violations
from src.violation_checker import Violation, ViolationChecker
from src.ohstem_service import OhstemService
from src.tracker_history import StudentTemporalTracker

CONSOLE_COOLDOWN_SECONDS = 5.0  # Khoảng cách tối thiểu giữa 2 lần in cảnh báo cùng một lỗi
_last_printed: dict[str, float] = {}


def violation_key(v: Violation, student_index: int = 0) -> str:
    return f"{v.violation_type.value}_{student_index}"


def process_violations(
    frame,
    violations: list[Violation],
    annotated_frame,
    cfg,
    frame_counter: FrameViolationCounter,
) -> None:
    now = time.time()
    for idx, v in enumerate(violations):
        key = violation_key(v, idx)
        confirmed = frame_counter.tick(key, True)

        if not confirmed:
            continue

        # Cooldown in ra màn hình để tránh trôi log
        last = _last_printed.get(key, 0.0)
        if (now - last) < CONSOLE_COOLDOWN_SECONDS:
            continue

        save_path = None
        if cfg.save_violation_images:
            ts = int(time.time())
            save_path = cfg.violations_dir / f"{v.violation_type.name}_{ts}.jpg"
            cv2.imwrite(str(save_path), annotated_frame)

        print(f"[CẢNH BÁO] {v.message}")
        if save_path:
            print(f"[LƯU TRỮ] Đã ghi nhận ảnh bằng chứng tại: {save_path}")

        _last_printed[key] = now


def main() -> None:
    cfg = load_config()

    if not cfg.model_path.exists():
        print(f"[Lỗi] Không tìm thấy model tại: {cfg.model_path}")
        sys.exit(1)

    if cfg.save_violation_images:
        cfg.violations_dir.mkdir(parents=True, exist_ok=True)

    detector = UniformDetector(
        model_path=str(cfg.model_path),
        confidence=cfg.confidence,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
    )
    checker = ViolationChecker(
        upper_body_ratio=cfg.upper_body_ratio,
        overlap_iou_threshold=cfg.overlap_iou_threshold,
    )
    temporal_tracker = StudentTemporalTracker(
        min_student_height=cfg.min_student_height,
        min_track_frames=cfg.min_track_frames,
        item_confirm_count=cfg.item_confirm_count,
        require_scarf=cfg.require_scarf,
        upper_body_ratio=cfg.upper_body_ratio,
        overlap_iou_threshold=cfg.overlap_iou_threshold,
    )
    
    frame_counter = FrameViolationCounter(cfg.violation_confirm_frames)

    # Khởi tạo kết nối điểm danh với OhStem IoT
    enable_ohstem = os.getenv("ENABLE_OHSTEM", "false").lower() in ("1", "true", "yes")
    ohstem_service = None
    if enable_ohstem:
        ohstem_user = os.getenv("OHSTEM_USERNAME", "")
        ohstem_pass = os.getenv("OHSTEM_PASSWORD", "")
        ohstem_feed = os.getenv("OHSTEM_FEED_ID", "V1")
        if ohstem_user:
            print(f"[Ohstem] Đang kết nối tới tài khoản: {ohstem_user}...")
            ohstem_service = OhstemService(username=ohstem_user, password=ohstem_pass, feed_id=ohstem_feed)
            ohstem_service.connect()
        else:
            print("[Ohstem] Lỗi: ENABLE_OHSTEM=true nhưng OHSTEM_USERNAME để trống.")

    print(f"[OK] Khởi động mô hình: {cfg.model_path.name} (imgsz={cfg.imgsz})")
    print(f"[OK] Nguồn camera: {cfg.camera_source}")
    print(f"[OK] Cấu hình Nhóm 1: min_height={cfg.min_student_height}px, min_frames={cfg.min_track_frames}, require_scarf={cfg.require_scarf}")
    if cfg.camera_width and cfg.camera_height:
        print(f"[OK] Cấu hình độ phân giải yêu cầu: {cfg.camera_width}x{cfg.camera_height}")
    print("Nhấn phím 'q' để thoát ứng dụng.")

    stream = CameraStream(
        cfg.camera_source,
        width=cfg.camera_width,
        height=cfg.camera_height,
    ).start()

    window_name = "True Form Student - Realtime Detection"
    window_initialized = False

    fps_time = time.time()
    frame_count = 0
    last_mqtt_time = 0.0
    last_student_count = -1

    try:
        while True:
            frame = stream.read(timeout=2.0)
            if frame is None:
                print("[Camera] Mất tín hiệu frame, đang thử lại...")
                continue

            if cfg.show_preview and not window_initialized:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                h, w = frame.shape[:2]
                if h > 900:
                    scale = 800.0 / h
                    cv2.resizeWindow(window_name, int(w * scale), 800)
                window_initialized = True

            # Suy luận kèm bám vết (Multi-Object Tracking - Group 1)
            detections = detector.track(frame, persist=True)
            student_histories = temporal_tracker.update(detections)

            # Tích hợp điểm danh gửi lên OhStem (tối ưu: gửi định kỳ mỗi 3s hoặc khi sĩ số thay đổi)
            now = time.time()
            student_count = sum(1 for d in detections if d.class_name == "hoc_sinh")
            if ohstem_service and (now - last_mqtt_time >= 3.0 or student_count != last_student_count):
                ohstem_service.send_attendance_count(student_count)
                last_mqtt_time = now
                last_student_count = student_count

            # Vẽ kết quả hiển thị kèm huy hiệu đánh giá thời gian thực
            annotated = draw_tracking_hud(frame, detections, student_histories)

            # Ghi nhận cảnh báo từ các học sinh đã xác nhận vi phạm
            for tid, hist in student_histories.items():
                if hist.is_confirmed and hist.violations:
                    key = f"track_{tid}"
                    last = _last_printed.get(key, 0.0)
                    if (now - last) >= CONSOLE_COOLDOWN_SECONDS:
                        print(f"⚠️ [CẢNH BÁO NHÓM 1 - #{tid}] {hist.status}")
                        if cfg.save_violation_images:
                            ts = int(time.time())
                            save_path = cfg.violations_dir / f"VIOLATION_STUDENT_{tid}_{ts}.jpg"
                            cv2.imwrite(str(save_path), annotated)
                            print(f"[LƯU TRỮ] Đã ghi nhận ảnh vi phạm: {save_path}")
                        _last_printed[key] = now

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_time
                fps = 30 / elapsed if elapsed > 0 else 0
                fps_time = time.time()
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if cfg.show_preview:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[Hệ Thống] Người dùng yêu cầu dừng chương trình.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        if ohstem_service:
            ohstem_service.disconnect()


if __name__ == "__main__":
    main()
