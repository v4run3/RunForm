"""RunForm: Video-Based Running Form Analysis Using Pose Estimation."""

from .config import load_config
from .pipeline import RunFormPipeline
from .schema import (
    BiomechanicalMetrics,
    FootStrikeType,
    FormFeedback,
    FrameAnalysisResult,
    InjuryRiskLevel,
    Keypoint,
    PoseDetection,
    VideoAnalysisReport,
)

__version__ = "0.1.0"
__all__ = [
    "RunFormPipeline",
    "load_config",
    "FootStrikeType",
    "InjuryRiskLevel",
    "Keypoint",
    "PoseDetection",
    "BiomechanicalMetrics",
    "FormFeedback",
    "FrameAnalysisResult",
    "VideoAnalysisReport",
]
