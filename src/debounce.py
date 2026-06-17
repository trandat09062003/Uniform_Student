"""Debounce cảnh báo thời gian thực trên khung hình để tránh nhiễu/spam."""

from __future__ import annotations

from collections import defaultdict


class FrameViolationCounter:
    """Đếm frame liên tiếp có vi phạm trước khi xác nhận để tránh false positive."""

    def __init__(self, confirm_frames: int = 5):
        self.confirm_frames = confirm_frames
        self._counts: dict[str, int] = defaultdict(int)

    def tick(self, violation_key: str, is_violating: bool) -> bool:
        if is_violating:
            self._counts[violation_key] += 1
        else:
            self._counts[violation_key] = 0
        return self._counts[violation_key] >= self.confirm_frames

    def reset(self, violation_key: str) -> None:
        self._counts[violation_key] = 0
