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
from src.visualize import draw_violations
from src.violation_checker import Violation, ViolationChecker
from src.ohstem_service import OhstemService

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
    )
    checker = ViolationChecker(
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

    print(f"[OK] Khởi động mô hình: {cfg.model_path.name}")
    print(f"[OK] Nguồn camera: {cfg.camera_source}")
    print("Nhấn phím 'q' để thoát ứng dụng.")

    stream = CameraStream(cfg.camera_source).start()
    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            frame = stream.read(timeout=2.0)
            if frame is None:
                print("[Camera] Mất tín hiệu frame, đang thử lại...")
                continue

            detections = detector.predict(frame)

            # Tích hợp điểm danh gửi lên OhStem
            if ohstem_service:
                # Đếm số lượng nhãn học sinh
                student_count = sum(1 for d in detections if d.class_name == "hoc_sinh")
                ohstem_service.send_attendance_count(student_count)

            violations = checker.check(detections)
            if not violations:
                violations = checker.check_frame_level(detections)

            annotated = draw_violations(frame, violations, detections)

            if violations:
                process_violations(
                    frame,
                    violations,
                    annotated,
                    cfg,
                    frame_counter,
                )
            else:
                for key in list(frame_counter._counts.keys()):
                    frame_counter.tick(key, False)

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
                cv2.imshow("True Form Student - Realtime Detection", annotated)
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
