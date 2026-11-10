"""Simple MediaPipe-based pose prototype for webcam and video demos."""

from .angles import calculate_angle, calculate_pose_angles
from .detector import MediaPipePoseDetector
from .landmarks import CameraView, PoseLandmark, PoseResult
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
from .quality import (
    AngleQualityConfig,
    MeasurementQuality,
    QualityStatus,
    ReliabilityConfig,
    calculate_reliability_score,
    calculate_trunk_lean_rom_safe,
    is_measurement_valid_for_view,
    validate_angle_plausibility,
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
    "CameraView",
    "QualityStatus",
    "MeasurementQuality",
    "AngleQualityConfig",
    "ReliabilityConfig",
    "validate_angle_plausibility",
    "is_measurement_valid_for_view",
    "calculate_trunk_lean_rom_safe",
    "calculate_reliability_score",
]
