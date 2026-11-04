"""Simple MediaPipe-based pose prototype for webcam and video demos."""

from .angles import calculate_angle, calculate_pose_angles
from .detector import MediaPipePoseDetector
from .landmarks import PoseLandmark, PoseResult
from .metrics import (
    FrameMetrics,
    RunningFormMetrics,
    RunningFormSummary,
    analyze_frame_metrics,
    calculate_knee_asymmetry,
    calculate_running_form_metrics,
    summarize_frame_metrics,
    write_metrics_csv,
)

from .temporal_analysis import (
    TemporalAnalysisResult,
    TemporalFeatures,
    WindowTemporalResult,
    analyze_temporal_movement,
    calculate_temporal_features,
    smooth_frame_metrics,
)

__all__ = [
    "MediaPipePoseDetector",
    "PoseLandmark",
    "PoseResult",
    "FrameMetrics",
    "RunningFormMetrics",
    "RunningFormSummary",
    "TemporalFeatures",
    "WindowTemporalResult",
    "TemporalAnalysisResult",
    "calculate_angle",
    "calculate_pose_angles",
    "calculate_knee_asymmetry",
    "calculate_running_form_metrics",
    "analyze_frame_metrics",
    "summarize_frame_metrics",
    "write_metrics_csv",
    "calculate_temporal_features",
    "analyze_temporal_movement",
    "smooth_frame_metrics",
]
