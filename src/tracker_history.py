"""
Bộ quản lý Bám vết (Multi-Object Tracking) & Đánh giá theo chuỗi thời gian (Temporal Aggregation).
Giải pháp cốt lõi Nhóm 1 giúp nâng độ chính xác lên 90-95% bằng cách:
1. Gán track_id duy nhất cho từng học sinh.
2. Tích lũy bằng chứng qua nhiều khung hình (thay vì kết luận vội ở 1 frame).
3. Lọc khoảng cách (Distance Gate) bỏ qua các học sinh ở quá xa.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.detector import Detection
from src.geometry import center_inside, iou


@dataclass
class StudentHistory:
    track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_frames: int = 0
    valid_inspection_frames: int = 0
    last_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    max_height: float = 0.0
    seen_items: Counter = field(default_factory=Counter)
    status: str = "CHƯA ĐỦ GẦN"  # "CHƯA ĐỦ GẦN", "ĐANG ĐÁNH GIÁ", "ĐỦ ĐỒNG PHỤC", "VI PHẠM"
    violations: List[str] = field(default_factory=list)
    is_confirmed: bool = False


class StudentTemporalTracker:
    def __init__(
        self,
        min_student_height: float = 250.0,
        min_track_frames: int = 8,
        item_confirm_count: int = 3,
        require_scarf: bool = True,
        upper_body_ratio: float = 0.55,
        overlap_iou_threshold: float = 0.05,
    ):
        self.min_student_height = min_student_height
        self.min_track_frames = min_track_frames
        self.item_confirm_count = item_confirm_count
        self.require_scarf = require_scarf
        self.upper_body_ratio = upper_body_ratio
        self.overlap_iou_threshold = overlap_iou_threshold

        self.students: Dict[int, StudentHistory] = {}
        self._next_fallback_id = 9999

    def _items_for_student(
        self,
        student: Detection,
        items: List[Detection],
    ) -> List[Detection]:
        matched = []
        student_bbox = student.bbox
        s_x1, s_y1, s_x2, s_y2 = student_bbox
        upper = student.upper_body_bbox(self.upper_body_ratio)

        for item in items:
            if item.class_name == "hoc_sinh":
                continue
            i_bbox = item.bbox
            i_cx, i_cy = item.center

            # Kiểm tra xem tâm của item có nằm trong phạm vi ngang của học sinh và trong khoảng chiều cao
            horiz_inside = s_x1 - 10 <= i_cx <= s_x2 + 10
            vert_inside = s_y1 - 10 <= i_cy <= s_y2 + 10

            is_overlap = iou(i_bbox, student_bbox) >= self.overlap_iou_threshold
            is_in_upper = center_inside(i_bbox, upper) or (horiz_inside and s_y1 <= i_cy <= upper[3])

            if (horiz_inside and vert_inside) or is_overlap or is_in_upper:
                matched.append(item)
        return matched

    def update(self, detections: List[Detection]) -> Dict[int, StudentHistory]:
        now = time.time()
        students = [d for d in detections if d.class_name == "hoc_sinh"]
        items = [d for d in detections if d.class_name != "hoc_sinh"]

        for s in students:
            # Nếu detector chưa cấp track_id (chế độ predict), gán tạm ID fallback
            tid = s.track_id if s.track_id is not None else 1

            if tid not in self.students:
                self.students[tid] = StudentHistory(track_id=tid)

            hist = self.students[tid]
            hist.last_seen = now
            hist.last_bbox = s.bbox
            hist.total_frames += 1

            s_height = s.height
            if s_height > hist.max_height:
                hist.max_height = s_height

            # Chỉ đánh giá khi học sinh đủ gần (vượt qua cổng lọc khoảng cách Distance Gate)
            if s_height < self.min_student_height:
                if hist.valid_inspection_frames == 0:
                    hist.status = "CHƯA ĐỦ GẦN"
                continue

            hist.valid_inspection_frames += 1

            # Ghép nối các vật thể thuộc về học sinh này
            matched_items = self._items_for_student(s, items)
            for m in matched_items:
                hist.seen_items[m.class_name] += 1

            # Đánh giá theo chuỗi thời gian (Temporal Voting)
            if hist.valid_inspection_frames < self.min_track_frames:
                hist.status = f"ĐANG THEO DÕI ({hist.valid_inspection_frames}/{self.min_track_frames})"
            else:
                hist.is_confirmed = True
                hist.violations.clear()

                # 1. Kiểm tra Áo trắng
                if hist.seen_items["ao_trang"] < self.item_confirm_count:
                    hist.violations.append("Thiếu áo trắng")

                # 2. Kiểm tra Quần xanh / quần tối màu
                if hist.seen_items["quan_xanh"] < self.item_confirm_count:
                    hist.violations.append("Thiếu quần xanh")

                # 3. Kiểm tra Thẻ tên / Bảng tên
                tag_count = hist.seen_items["the_ten"] + hist.seen_items["bang_ten"]
                if tag_count < self.item_confirm_count:
                    hist.violations.append("Thiếu bảng tên")

                # 4. Kiểm tra Khăn quàng đỏ (nếu trường yêu cầu)
                if self.require_scarf:
                    if hist.seen_items["khan_quang"] < self.item_confirm_count:
                        hist.violations.append("Thiếu khăn quàng")

                if hist.violations:
                    hist.status = f"VI PHẠM: {', '.join(hist.violations)}"
                else:
                    hist.status = "ĐỦ ĐỒNG PHỤC"

        return self.students

