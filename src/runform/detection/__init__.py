"""Detection subpackage."""

from .detector import (
    COCO_KEYPOINTS,
    BasePoseDetector,
    MockPoseDetector,
    YOLOv8PoseDetector,
)

__all__ = [
    "COCO_KEYPOINTS",
    "BasePoseDetector",
    "YOLOv8PoseDetector",
    "MockPoseDetector",
]
