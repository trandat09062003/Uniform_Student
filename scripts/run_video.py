"""Chạy detection trên file video (không gửi email)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.detector import UniformDetector
from src.visualize import draw_violations
from src.violation_checker import ViolationChecker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Đường dẫn file .mp4")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    detector = UniformDetector(str(cfg.model_path), cfg.confidence, cfg.iou)
    checker = ViolationChecker(cfg.upper_body_ratio, cfg.overlap_iou_threshold)

    cap = cv2.VideoCapture(args.video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.predict(frame)
        violations = checker.check(detections) or checker.check_frame_level(detections)
        out = draw_violations(frame, violations, detections)
        if not args.no_show:
            cv2.imshow("Video Test", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
