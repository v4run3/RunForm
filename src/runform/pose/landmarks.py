"""Landmark data structures and names used by the simple pose prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PoseLandmark:
    """One normalized body landmark returned by the pose detector."""

    name: str
    x: float
    y: float
    z: float | None = None
    visibility: float = 0.0


# Camera view types for view-specific measurement handling
CameraView = Literal["side", "front", "rear", "unknown"]


@dataclass
class PoseResult:
    """Pose landmarks for one detected person in one frame."""

    landmarks: dict[str, PoseLandmark] = field(default_factory=dict)
    confidence: float = 0.0
    image_width: int = 0
    image_height: int = 0
    bbox: tuple[float, float, float, float] | None = None
    camera_view: CameraView = "unknown"

    @property
    def detected(self) -> bool:
        return bool(self.landmarks)

    def get(self, name: str, min_visibility: float = 0.0) -> PoseLandmark | None:
        landmark = self.landmarks.get(name)
        if landmark is None or landmark.visibility < min_visibility:
            return None
        return landmark

    def get_landmarks_for_angle(
        self, names: list[str], min_visibility: float = 0.3
    ) -> tuple[list[PoseLandmark], list[str]]:
        """Get landmarks for angle calculation with visibility check.

        Returns:
            Tuple of (valid_landmarks, missing_landmark_names)
        """
        valid = []
        missing = []
        for name in names:
            lm = self.get(name, min_visibility=min_visibility)
            if lm is not None:
                valid.append(lm)
            else:
                missing.append(name)
        return valid, missing


POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

IMPORTANT_LANDMARKS = [
    "nose",
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
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]
