"""Running-form metrics, summaries, observations, and CSV export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math
import statistics

from .angles import calculate_pose_angles
from .landmarks import PoseResult

METRIC_FIELDS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_lean",
    "knee_asymmetry",
]


@dataclass
class RunningFormMetrics:
    """Frame-level running-form measurements from 2D pose landmarks."""

    left_knee_angle: float | None = None
    right_knee_angle: float | None = None
    left_hip_angle: float | None = None
    right_hip_angle: float | None = None
    left_elbow_angle: float | None = None
    right_elbow_angle: float | None = None
    trunk_lean: float | None = None
    knee_asymmetry: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass
class FrameMetrics:
    """One row of pose-analysis data for a processed frame."""

    frame: int
    timestamp: float
    pose_detected: bool
    pose_confidence: float | None
    metrics: RunningFormMetrics

    def as_csv_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "frame": self.frame,
            "timestamp": f"{self.timestamp:.6f}",
            "pose_detected": self.pose_detected,
            "pose_confidence": "" if self.pose_confidence is None else f"{self.pose_confidence:.6f}",
        }
        for name, value in self.metrics.as_dict().items():
            row[name] = "" if value is None or not math.isfinite(value) else f"{value:.6f}"
        return row


@dataclass
class RunningFormSummary:
    """Video-level summary of basic running-form metrics."""

    frames_processed: int
    pose_detected_frames: int
    detection_rate: float
    average_left_knee_angle: float | None = None
    average_right_knee_angle: float | None = None
    average_left_hip_angle: float | None = None
    average_right_hip_angle: float | None = None
    average_trunk_lean: float | None = None
    average_knee_asymmetry: float | None = None
    observations: list[str] | None = None


def calculate_knee_asymmetry(
    left_knee_angle: float | None,
    right_knee_angle: float | None,
) -> float | None:
    """Return absolute left/right knee-angle difference in degrees."""
    if left_knee_angle is None or right_knee_angle is None:
        return None
    return abs(left_knee_angle - right_knee_angle)


def calculate_running_form_metrics(
    result: PoseResult | None,
    min_visibility: float = 0.3,
) -> RunningFormMetrics:
    """Calculate frame-level running-form metrics from a pose result."""
    angles = calculate_pose_angles(result, min_visibility=min_visibility)
    left_knee = angles.get("left_knee")
    right_knee = angles.get("right_knee")
    return RunningFormMetrics(
        left_knee_angle=left_knee,
        right_knee_angle=right_knee,
        left_hip_angle=angles.get("left_hip"),
        right_hip_angle=angles.get("right_hip"),
        left_elbow_angle=angles.get("left_elbow"),
        right_elbow_angle=angles.get("right_elbow"),
        trunk_lean=angles.get("trunk_lean"),
        knee_asymmetry=calculate_knee_asymmetry(left_knee, right_knee),
    )


def analyze_frame_metrics(
    frame_index: int,
    timestamp: float,
    result: PoseResult | None,
    min_visibility: float = 0.3,
) -> FrameMetrics:
    """Build one frame-metrics record, safely handling missing poses."""
    pose_detected = result is not None and result.detected
    confidence = result.confidence if pose_detected and result is not None else None
    return FrameMetrics(
        frame=frame_index,
        timestamp=timestamp,
        pose_detected=pose_detected,
        pose_confidence=confidence,
        metrics=calculate_running_form_metrics(result, min_visibility=min_visibility),
    )


def _clean(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _average(values: list[float | None]) -> float | None:
    cleaned = _clean(values)
    if not cleaned:
        return None
    return float(statistics.fmean(cleaned))


def build_observations(summary: RunningFormSummary) -> list[str]:
    """Create simple non-medical observations from summary metrics."""
    observations: list[str] = []
    if summary.detection_rate < 0.6:
        observations.append("Pose visibility was low; results may be less reliable.")
    if (
        summary.average_knee_asymmetry is not None
        and summary.average_knee_asymmetry >= 10.0
    ):
        observations.append("Noticeable left/right knee-angle difference detected.")
    if summary.average_trunk_lean is not None and abs(summary.average_trunk_lean) >= 15.0:
        observations.append("Increased trunk lean detected.")
    if not observations:
        observations.append("No major form deviation detected by the current basic rules.")
    return observations


def summarize_frame_metrics(frames: list[FrameMetrics]) -> RunningFormSummary:
    """Aggregate frame-level metrics into one video-level summary."""
    frames_processed = len(frames)
    pose_detected_frames = sum(1 for frame in frames if frame.pose_detected)
    detection_rate = pose_detected_frames / frames_processed if frames_processed else 0.0

    summary = RunningFormSummary(
        frames_processed=frames_processed,
        pose_detected_frames=pose_detected_frames,
        detection_rate=detection_rate,
        average_left_knee_angle=_average([f.metrics.left_knee_angle for f in frames]),
        average_right_knee_angle=_average([f.metrics.right_knee_angle for f in frames]),
        average_left_hip_angle=_average([f.metrics.left_hip_angle for f in frames]),
        average_right_hip_angle=_average([f.metrics.right_hip_angle for f in frames]),
        average_trunk_lean=_average([f.metrics.trunk_lean for f in frames]),
        average_knee_asymmetry=_average([f.metrics.knee_asymmetry for f in frames]),
    )
    summary.observations = build_observations(summary)
    return summary


def write_metrics_csv(frames: list[FrameMetrics], output_path: str | Path) -> None:
    """Write frame-level metrics to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "timestamp",
        "pose_detected",
        "pose_confidence",
        *METRIC_FIELDS,
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            writer.writerow(frame.as_csv_row())


def format_degrees(value: float | None) -> str:
    """Format an angle for terminal or overlay display."""
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.1f} deg"


def print_summary(summary: RunningFormSummary) -> None:
    """Print a clear terminal summary for a processed video."""
    print("\nRunForm Analysis")
    print("----------------")
    print(f"Frames processed: {summary.frames_processed}")
    print(f"Pose detected: {summary.pose_detected_frames} frames")
    print(f"Detection rate: {summary.detection_rate * 100.0:.1f}%")
    print("")
    print(f"Average left knee angle: {format_degrees(summary.average_left_knee_angle)}")
    print(f"Average right knee angle: {format_degrees(summary.average_right_knee_angle)}")
    print(f"Average left hip angle: {format_degrees(summary.average_left_hip_angle)}")
    print(f"Average right hip angle: {format_degrees(summary.average_right_hip_angle)}")
    print(f"Average knee asymmetry: {format_degrees(summary.average_knee_asymmetry)}")
    print(f"Average trunk lean: {format_degrees(summary.average_trunk_lean)}")
    print("")
    print("Observations:")
    for observation in summary.observations or []:
        print(f"- {observation}")
