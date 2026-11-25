"""Vẽ bbox và nhãn nhận diện lên frame (Hỗ trợ tiếng Việt Unicode)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.detector import Detection
from src.violation_checker import Violation

COLORS = {
    "hoc_sinh": (0, 255, 0),
    "ao_trang": (255, 255, 255),
    "quan_xanh": (255, 128, 0),
    "khan_quang": (0, 0, 255),
    "bang_ten": (255, 0, 255),
    "the_ten": (255, 0, 255),  # Map thêm the_ten giống bang_ten
    "quan_sai_mau": (0, 0, 255),
    "ao_sai_mau": (0, 0, 255),
}


def draw_text_unicode(
    img: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font_size: int = 16,
    bg_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Vẽ chữ Tiếng Việt Unicode bằng Pillow lên ảnh OpenCV (BGR) kèm tùy chọn vẽ nền."""
    # Chuyển đổi OpenCV (BGR) -> PIL (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    # Thử load font tiếng Việt trên Windows
    font = None
    for font_name in ["arial.ttf", "tahoma.ttf", "segoeui.ttf"]:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except IOError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # PIL dùng RGB nên ta chuyển BGR -> RGB
    pil_color = (color[2], color[1], color[0])
    
    if bg_color is not None:
        pil_bg = (bg_color[2], bg_color[1], bg_color[0])
        # Lấy kích thước text để vẽ hộp nền
        text_bbox = draw.textbbox(position, text, font=font)
        # Nới rộng hộp nền một chút
        padded_bbox = (text_bbox[0] - 6, text_bbox[1] - 3, text_bbox[2] + 6, text_bbox[3] + 3)
        draw.rectangle(padded_bbox, fill=pil_bg)
        # Vẽ chữ đè lên nền
        draw.text(position, text, font=font, fill=pil_color)
    else:
        # Vẽ chữ kèm shadow màu đen
        draw.text((position[0] + 1, position[1] + 1), text, font=font, fill=(0, 0, 0))
        draw.text(position, text, font=font, fill=pil_color)

    # Chuyển ngược PIL -> OpenCV (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    students = [d for d in detections if d.class_name == "hoc_sinh"]

    for d in detections:
        # Chỉ vẽ các item đồng phục nếu có ít nhất 1 học sinh và tâm item nằm trong khung học sinh
        if d.class_name != "hoc_sinh":
            if not students:
                continue
            cx, cy = d.center
            if not any(s.x1 - 15 <= cx <= s.x2 + 15 and s.y1 - 15 <= cy <= s.y2 + 15 for s in students):
                continue

        color = COLORS.get(d.class_name, (200, 200, 200))
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        
        # Đổi tên hiển thị nhãn đẹp hơn
        display_name = d.class_name
        if display_name == "the_ten" or display_name == "bang_ten":
            display_name = "bảng tên"
        elif display_name == "ao_trang":
            display_name = "áo trắng"
        elif display_name == "quan_xanh":
            display_name = "quần xanh"
        elif display_name == "khan_quang":
            display_name = "khăn quàng"
        elif display_name == "hoc_sinh":
            display_name = "học sinh"

        if d.track_id is not None and d.class_name == "hoc_sinh":
            label = f"#{d.track_id} {display_name} {d.confidence:.2f}"
        else:
            label = f"{display_name} {d.confidence:.2f}"
        
        # Vẽ nhãn bằng tiếng Việt Unicode
        out = draw_text_unicode(out, label, (x1, max(y1 - 22, 5)), color, font_size=14)
    return out


def draw_tracking_hud(
    frame: np.ndarray,
    detections: list[Detection],
    student_histories: dict,
) -> np.ndarray:
    """Vẽ nhãn nhận diện kèm huy hiệu trạng thái đánh giá đồng phục (Group 1 Voting) trên đầu học sinh."""
    out = draw_detections(frame, detections)
    students = [d for d in detections if d.class_name == "hoc_sinh"]

    for s in students:
        tid = s.track_id if s.track_id is not None else 1
        hist = student_histories.get(tid)
        if not hist:
            continue

        x1, y1, x2, y2 = map(int, s.bbox)

        # Quyết định màu sắc huy hiệu theo trạng thái
        if "ĐỦ ĐỒNG PHỤC" in hist.status:
            badge_color = (0, 255, 0)      # Xanh lá
            bg_color = (0, 100, 0)
            badge_text = f"[#{tid}] ĐỦ ĐỒNG PHỤC"
        elif "VI PHẠM" in hist.status:
            badge_color = (0, 0, 255)      # Đỏ
            bg_color = (0, 0, 120)
            badge_text = f"[#{tid}] {hist.status}"
        elif "CHƯA ĐỦ GẦN" in hist.status:
            badge_color = (180, 180, 180)  # Xám
            bg_color = (50, 50, 50)
            badge_text = f"[#{tid}] CHƯA ĐỦ GẦN"
        else:
            badge_color = (0, 255, 255)    # Vàng
            bg_color = (120, 120, 0)
            badge_text = f"[#{tid}] {hist.status}"

        # Vẽ huy hiệu nổi bật phía trên đầu học sinh
        badge_pos = (x1, max(y1 - 45, 10))
        out = draw_text_unicode(out, badge_text, badge_pos, badge_color, font_size=15, bg_color=bg_color)

    return out


def draw_violations(
    frame: np.ndarray,
    violations: list[Violation],
    detections: list[Detection] | None = None,
) -> np.ndarray:
    """
    Theo yêu cầu người dùng: Không vẽ thêm bất kỳ chữ/box cảnh báo vi phạm nào lên hình ảnh
    để tránh che khuất khuôn mặt và các chi tiết quan trọng của học sinh.
    Toàn bộ lỗi vi phạm sẽ được in ra log console thời gian thực đầy đủ.
    """
    return draw_detections(frame, detections) if detections else frame.copy()
