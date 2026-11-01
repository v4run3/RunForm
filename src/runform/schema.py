"""Core data contracts and types for RunForm.

Defines typed dataclasses for keypoints, detections, gait metrics,
and pipeline outputs to ensure clean separation of concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FootStrikeType(str, Enum):
    """Classification of the foot landing pattern."""

    HEEL = "heel_strike"
    MIDFOOT = "midfoot_strike"
    FOREFOOT = "forefoot_strike"
    UNKNOWN = "unknown"


class InjuryRiskLevel(str, Enum):
    """Categorical risk assessment level."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class Keypoint:
    """Individual 2D/3D joint keypoint."""

    x: float
    y: float
    confidence: float
    z: float | None = None


@dataclass
class PoseDetection:
    """Pose estimation result for a single person in a frame."""

    track_id: int | None
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    keypoints: dict[str, Keypoint] = field(default_factory=dict)


@dataclass
class BiomechanicalMetrics:
    """Calculated joint angles and kinematic measurements."""

    knee_flexion_left_deg: float | None = None
    knee_flexion_right_deg: float | None = None
    hip_angle_left_deg: float | None = None
    hip_angle_right_deg: float | None = None
    trunk_lean_deg: float | None = None
    cadence_spm: float | None = None  # steps per minute
    vertical_oscillation_proxy: float | None = None


@dataclass
class FormFeedback:
    """Actionable feedback and risk indicators."""

    risk_level: InjuryRiskLevel
    flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class FrameAnalysisResult:
    """Complete analysis output for a single video frame."""

    frame_index: int
    timestamp_sec: float
    detections: list[PoseDetection] = field(default_factory=list)
    foot_strike: dict[int, FootStrikeType] = field(default_factory=dict)  # track_id -> strike
    metrics: dict[int, BiomechanicalMetrics] = field(default_factory=dict)  # track_id -> metrics


@dataclass
class VideoAnalysisReport:
    """Aggregated report across an entire running video."""

    video_path: str
    total_frames: int
    fps: float
    duration_sec: float
    primary_strike_type: FootStrikeType
    avg_metrics: BiomechanicalMetrics
    feedback: FormFeedback
    frame_results: list[FrameAnalysisResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
