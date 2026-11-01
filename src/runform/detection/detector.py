"""Pose detection wrappers and keypoint parsers for RunForm.

Supports YOLOv8-pose as the primary inference engine and includes
a mock detector for fast unit testing and offline development.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from runform.schema import Keypoint, PoseDetection

# Standard 17 COCO keypoint names in standard order (indices 0 to 16)
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class BasePoseDetector(ABC):
    """Abstract base class for all pose detection backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        """Detect human poses in an input image frame.

        Args:
            frame: BGR image array with shape (H, W, 3).

        Returns:
            List of PoseDetection objects for all detected persons.
        """
        pass


class YOLOv8PoseDetector(BasePoseDetector):
    """Ultralytics YOLOv8-Pose detector implementation."""

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        conf_threshold: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        """Initialize YOLOv8-Pose detector.

        Args:
            model_name: Model file or pre-trained name ('yolov8n-pose.pt', etc.).
            conf_threshold: Minimum detection confidence score.
            imgsz: Input inference resolution.
            device: Compute device ('cpu', 'cuda', etc.). Defaults to auto-select.
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self._model: Any = None

    def _load_model(self) -> None:
        """Lazy load the YOLO model when first inference is requested."""
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load YOLOv8 model '{self.model_name}'. "
                    "Ensure 'ultralytics' is installed and network/weights are available."
                ) from e

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        """Run YOLOv8-Pose on a single frame and extract keypoints."""
        self._load_model()

        # Run inference
        results = self._model(
            frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[PoseDetection] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, "xyxy") else []
        confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else []
        keypoints_data = result.keypoints.data.cpu().numpy() if hasattr(result.keypoints, "data") else []

        for idx, bbox in enumerate(boxes):
            conf = float(confidences[idx]) if idx < len(confidences) else 0.0
            kps_dict: dict[str, Keypoint] = {}

            if idx < len(keypoints_data):
                person_kps = keypoints_data[idx]  # Shape: (17, 2) or (17, 3)
                for kp_idx, name in enumerate(COCO_KEYPOINTS):
                    if kp_idx < len(person_kps):
                        kp_row = person_kps[kp_idx]
                        x, y = float(kp_row[0]), float(kp_row[1])
                        kp_conf = float(kp_row[2]) if len(kp_row) > 2 else conf
                        kps_dict[name] = Keypoint(x=x, y=y, confidence=kp_conf)

            detections.append(
                PoseDetection(
                    track_id=None,  # Tracking assigned in tracking module
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=conf,
                    keypoints=kps_dict,
                )
            )

        return detections


class MockPoseDetector(BasePoseDetector):
    """Deterministic mock detector for testing and headless validation."""

    def __init__(self, sample_detections: list[PoseDetection] | None = None) -> None:
        self.sample_detections = sample_detections or []

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        return self.sample_detections
