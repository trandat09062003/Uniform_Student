"""Logic phát hiện vi phạm đồng phục."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.detector import Detection
from src.geometry import center_inside, iou


class ViolationType(str, Enum):
    MISSING_SCARF = "Thiếu khăn quàng đỏ"
    MISSING_NAME_TAG = "Thiếu bảng tên"
    WRONG_PANTS = "Sai màu quần (không phải quần xanh)"
    WRONG_SHIRT = "Sai màu áo (không phải áo trắng)"
    MISSING_UNIFORM_PARTS = "Thiếu áo trắng hoặc quần xanh"


@dataclass
class Violation:
    violation_type: ViolationType
    student_bbox: tuple[float, float, float, float]
    message: str


REQUIRED_ON_STUDENT = {"ao_trang", "quan_xanh"}
UPPER_BODY_ITEMS = {"khan_quang", "bang_ten"}
WRONG_UNIFORM = {"quan_sai_mau", "ao_sai_mau"}


class ViolationChecker:
    def __init__(
        self,
        upper_body_ratio: float = 0.55,
        overlap_iou_threshold: float = 0.1,
    ):
        self.upper_body_ratio = upper_body_ratio
        self.overlap_iou_threshold = overlap_iou_threshold

    def _items_on_student(
        self,
        student: Detection,
        items: list[Detection],
    ) -> list[Detection]:
        upper = student.upper_body_bbox(self.upper_body_ratio)
        student_bbox = student.bbox
        matched = []
        for item in items:
            if item.class_name == "hoc_sinh":
                continue
            item_bbox = item.bbox
            overlap = iou(item_bbox, student_bbox) >= self.overlap_iou_threshold
            in_upper = center_inside(item_bbox, upper) or iou(item_bbox, upper) >= self.overlap_iou_threshold
            if overlap or in_upper:
                matched.append(item)
        return matched

    def check(self, detections: list[Detection]) -> list[Violation]:
        students = [d for d in detections if d.class_name == "hoc_sinh"]
        others = [d for d in detections if d.class_name != "hoc_sinh"]

        violations: list[Violation] = []

        for student in students:
            on_student = self._items_on_student(student, others)
            names = {d.class_name for d in on_student}

            for wrong in WRONG_UNIFORM & names:
                vtype = (
                    ViolationType.WRONG_PANTS
                    if wrong == "quan_sai_mau"
                    else ViolationType.WRONG_SHIRT
                )
                violations.append(
                    Violation(
                        violation_type=vtype,
                        student_bbox=student.bbox,
                        message=vtype.value,
                    )
                )

            if "quan_sai_mau" not in names and "ao_sai_mau" not in names:
                missing = REQUIRED_ON_STUDENT - names
                if missing:
                    violations.append(
                        Violation(
                            violation_type=ViolationType.MISSING_UNIFORM_PARTS,
                            student_bbox=student.bbox,
                            message=f"Thiếu: {', '.join(sorted(missing))}",
                        )
                    )

                if "khan_quang" not in names:
                    violations.append(
                        Violation(
                            violation_type=ViolationType.MISSING_SCARF,
                            student_bbox=student.bbox,
                            message=ViolationType.MISSING_SCARF.value,
                        )
                    )
                if "bang_ten" not in names and "the_ten" not in names:
                    violations.append(
                        Violation(
                            violation_type=ViolationType.MISSING_NAME_TAG,
                            student_bbox=student.bbox,
                            message=ViolationType.MISSING_NAME_TAG.value,
                        )
                    )

        return violations

    def check_frame_level(self, detections: list[Detection]) -> list[Violation]:
        """
        Logic đơn giản khi chưa detect được bbox hoc_sinh riêng:
        có nhãn trong frame nhưng thiếu item bắt buộc.
        """
        names = {d.class_name for d in detections}
        violations: list[Violation] = []
        default_bbox = (0.0, 0.0, 0.0, 0.0)

        if "hoc_sinh" not in names:
            return violations

        if "khan_quang" not in names:
            violations.append(
                Violation(
                    violation_type=ViolationType.MISSING_SCARF,
                    student_bbox=default_bbox,
                    message=ViolationType.MISSING_SCARF.value,
                )
            )
        if "bang_ten" not in names and "the_ten" not in names:
            violations.append(
                Violation(
                    violation_type=ViolationType.MISSING_NAME_TAG,
                    student_bbox=default_bbox,
                    message=ViolationType.MISSING_NAME_TAG.value,
                )
            )
        if "quan_sai_mau" in names:
            violations.append(
                Violation(
                    violation_type=ViolationType.WRONG_PANTS,
                    student_bbox=default_bbox,
                    message=ViolationType.WRONG_PANTS.value,
                )
            )
        if "ao_sai_mau" in names:
            violations.append(
                Violation(
                    violation_type=ViolationType.WRONG_SHIRT,
                    student_bbox=default_bbox,
                    message=ViolationType.WRONG_SHIRT.value,
                )
            )
        if "quan_xanh" not in names or "ao_trang" not in names:
            violations.append(
                Violation(
                    violation_type=ViolationType.MISSING_UNIFORM_PARTS,
                    student_bbox=default_bbox,
                    message=ViolationType.MISSING_UNIFORM_PARTS.value,
                )
            )
        return violations
