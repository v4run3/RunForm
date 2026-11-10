"""Pose measurement quality validation, filtering, and reliability scoring.

This module provides quality control for pose-derived biomechanical measurements,
addressing issues identified in temporal analysis outputs including:
- Implausible angle values from 2D planar landmark collapse / foreshortening
- Angle wrapping artifacts (e.g., 359° trunk lean ROM across the -180/+180° boundary)
- Camera view limitations for specific measurements (sagittal vs frontal plane)
- Low landmark visibility and occlusion
- Degenerate segment lengths

NOTE: These are software-level quality checks, NOT clinical validation.
A high quality score does not guarantee correct biomechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from .landmarks import CameraView, PoseResult

# Measurement quality status
QualityStatus = Literal["valid", "implausible", "degraded", "unavailable"]


@dataclass
class MeasurementQuality:
    """Quality metadata for a single measurement."""

    status: QualityStatus
    raw_value: float | None
    filtered_value: float | None
    reason: str | None = None
    min_landmark_visibility: float | None = None


@dataclass
class AngleQualityConfig:
    """Configurable thresholds for angle quality validation."""

    # Visibility thresholds
    min_landmark_visibility: float = 0.3

    # Plausibility limits (degrees)
    # Anatomical ranges for upright human running in sagittal plane
    knee_min_plausible: float = 20.0
    knee_max_plausible: float = 185.0
    hip_min_plausible: float = 45.0
    hip_max_plausible: float = 190.0
    elbow_min_plausible: float = 20.0
    elbow_max_plausible: float = 180.0
    trunk_lean_min_plausible: float = -45.0
    trunk_lean_max_plausible: float = 45.0

    # Minimum segment length in pixels to prevent numerical collapse
    min_segment_length_px: float = 2.0


@dataclass
class ReliabilityConfig:
    """Configurable criteria for window/video reliability scoring."""

    min_valid_frames: int = 5
    min_valid_ratio: float = 0.4
    min_landmark_visibility: float = 0.3
    min_avg_landmark_visibility: float = 0.4
    min_valid_measurements_ratio: float = 0.3
    max_knee_cv: float = 0.6
    max_hip_cv: float = 0.6
    max_trunk_lean_cv: float = 1.0


DEFAULT_ANGLE_QUALITY_CONFIG = AngleQualityConfig()
DEFAULT_RELIABILITY_CONFIG = ReliabilityConfig()


def validate_angle_plausibility(
    angle_name: str,
    value: float | None,
    config: AngleQualityConfig = DEFAULT_ANGLE_QUALITY_CONFIG,
) -> tuple[QualityStatus, str | None]:
    """Validate if an angle value is biomechanically plausible.

    Args:
        angle_name: Name of the angle (e.g., "left_knee", "trunk_lean")
        value: Angle value in degrees
        config: Quality configuration thresholds

    Returns:
        Tuple of (status, reason)
    """
    if value is None or not math.isfinite(value):
        return "unavailable", "Missing or non-finite angle value"

    val = float(value)
    name_lower = angle_name.lower()

    if "knee" in name_lower:
        min_val, max_val = config.knee_min_plausible, config.knee_max_plausible
        if val < min_val or val > max_val:
            return "implausible", f"Knee angle {val:.1f}° outside plausible range [{min_val:.0f}°, {max_val:.0f}°]"
    elif "hip" in name_lower:
        min_val, max_val = config.hip_min_plausible, config.hip_max_plausible
        if val < min_val or val > max_val:
            return "implausible", f"Hip angle {val:.1f}° outside plausible range [{min_val:.0f}°, {max_val}°]"
    elif "elbow" in name_lower:
        min_val, max_val = config.elbow_min_plausible, config.elbow_max_plausible
        if val < min_val or val > max_val:
            return "implausible", f"Elbow angle {val:.1f}° outside plausible range [{min_val:.0f}°, {max_val}°]"
    elif "trunk" in name_lower:
        min_val, max_val = config.trunk_lean_min_plausible, config.trunk_lean_max_plausible
        if val < min_val or val > max_val:
            return "implausible", f"Trunk lean {val:.1f}° outside plausible range [{min_val:.0f}°, {max_val}°]"

    return "valid", None


def is_measurement_valid_for_view(
    measurement_name: str,
    camera_view: CameraView | str,
) -> tuple[bool, str | None]:
    """Check if a measurement is valid for the given camera view.

    Args:
        measurement_name: Name of the measurement
        camera_view: Camera view type ('side', 'front', 'rear', 'unknown')

    Returns:
        Tuple of (is_valid, limitation_note)
    """
    measurement_lower = measurement_name.lower()
    view = str(camera_view).lower()

    if view == "side":
        return True, None

    if view == "front":
        if "knee" in measurement_lower:
            return False, "Knee flexion/extension not measurable in front view (planar foreshortening)"
        if "hip" in measurement_lower:
            return False, "Hip flexion/extension not measurable in front view (planar foreshortening)"
        if "trunk" in measurement_lower:
            return True, "Trunk lean measures lateral tilt (sway) in front view, not forward lean"
        return True, None

    if view == "rear":
        if "knee" in measurement_lower:
            return False, "Knee flexion/extension not measurable in rear view (planar foreshortening)"
        if "hip" in measurement_lower:
            return False, "Hip flexion/extension not measurable in rear view (planar foreshortening)"
        if "trunk" in measurement_lower:
            return True, "Trunk lean measures lateral tilt (sway) in rear view, not forward lean"
        return True, None

    return True, "Camera view unknown - measurement validity uncertain"


def _unwrap_angles(angles: list[float]) -> list[float]:
    """Unwrap an angle sequence to remove ±180° phase discontinuities using cumulative phase shifts."""
    if not angles:
        return []
    unwrapped = [angles[0]]
    shift = 0.0
    for i in range(1, len(angles)):
        delta = (angles[i] + shift) - unwrapped[-1]
        while delta > 180.0:
            shift -= 360.0
            delta = (angles[i] + shift) - unwrapped[-1]
        while delta < -180.0:
            shift += 360.0
            delta = (angles[i] + shift) - unwrapped[-1]
        unwrapped.append(angles[i] + shift)
    return unwrapped


def calculate_trunk_lean_rom_safe(
    min_angle: float | None,
    max_angle: float | None,
    values: list[float | None] | None = None,
) -> float | None:
    """Calculate trunk lean ROM handling angle wrapping at ±180° boundary.

    The trunk lean calculation uses atan2() which returns values in [-180°, +180°].
    When torso detection flips or when angles cross the branch cut, raw max - min
    can erroneously report ROM close to 359° instead of the true small arc.

    Args:
        min_angle: Minimum trunk lean angle
        max_angle: Maximum trunk lean angle
        values: Optional sequence of raw angles for temporal unwrapping

    Returns:
        Corrected ROM in degrees, or None if inputs invalid
    """
    if min_angle is None or max_angle is None:
        return None

    # Normal case: no boundary wrapping detected
    if max_angle - min_angle <= 180.0:
        return max_angle - min_angle

    # Wrapping detected: values span across the -180/+180° discontinuity
    if values:
        clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if len(clean) > 1:
            unwrapped = _unwrap_angles(clean)
            return max(unwrapped) - min(unwrapped)

    # Without values sequence, use circular shortest arc
    if max_angle > 0.0 and min_angle < 0.0:
        arc1 = max_angle - min_angle
        arc2 = (180.0 - max_angle) + (180.0 + min_angle)
        return min(arc1, arc2)

    return min(max_angle - min_angle, 90.0)


def calculate_reliability_score(
    total_frames: int,
    valid_pose_frames: int,
    valid_measurements: dict[str, int],
    landmark_visibilities: dict[str, list[float]],
    measurement_stabilities: dict[str, float | None],
    config: ReliabilityConfig = DEFAULT_RELIABILITY_CONFIG,
    camera_view: str = "unknown",
    min_valid_frames: int | None = None,
    min_valid_ratio: float | None = None,
) -> tuple[float, bool, list[str]]:
    """Calculate a multi-factor reliability score and reliability determination.

    Evaluates:
    1. Frame coverage (valid pose detection ratio) - 30% weight
    2. Landmark visibility of critical joint landmarks - 25% weight
    3. Valid angle measurements ratio (valid/plausible vs missing/implausible) - 25% weight
    4. Measurement stability (coefficient of variation) - 20% weight

    Also incorporates camera viewpoint compatibility.

    Returns:
        Tuple of (reliability_score: float 0.0..1.0, is_reliable: bool, issues: list[str])
    """
    issues: list[str] = []
    threshold_frames = min_valid_frames if min_valid_frames is not None else config.min_valid_frames
    threshold_ratio = min_valid_ratio if min_valid_ratio is not None else config.min_valid_ratio

    if total_frames == 0:
        return 0.0, False, ["No frames analyzed"]

    # 1. Frame coverage (30%)
    pose_ratio = valid_pose_frames / total_frames
    frame_score = min(1.0, pose_ratio / threshold_ratio) if threshold_ratio > 0 else 1.0

    if valid_pose_frames < threshold_frames:
        issues.append(f"Insufficient valid frames ({valid_pose_frames} < {threshold_frames})")
    if pose_ratio < threshold_ratio:
        issues.append(f"Low pose detection rate ({pose_ratio:.1%} < {threshold_ratio:.1%})")

    # 2. Landmark visibility (25%)
    vis_scores: list[float] = []
    for lm_name, vis_list in landmark_visibilities.items():
        if vis_list:
            avg_vis = sum(vis_list) / len(vis_list)
            vis_scores.append(avg_vis)
            if avg_vis < config.min_landmark_visibility:
                issues.append(f"Low visibility for {lm_name} (avg={avg_vis:.2f})")

    avg_vis = sum(vis_scores) / len(vis_scores) if vis_scores else (1.0 if valid_pose_frames > 0 else 0.0)
    visibility_score = min(1.0, avg_vis / config.min_avg_landmark_visibility) if config.min_avg_landmark_visibility > 0 else 1.0

    # 3. Valid measurement ratio (25%)
    expected_measurements = valid_pose_frames * len(valid_measurements) if valid_measurements else valid_pose_frames
    total_valid = sum(valid_measurements.values())
    measurement_ratio = total_valid / expected_measurements if expected_measurements > 0 else 0.0
    measurement_score = min(1.0, measurement_ratio / config.min_valid_measurements_ratio) if config.min_valid_measurements_ratio > 0 else 1.0

    if measurement_ratio < config.min_valid_measurements_ratio:
        issues.append(f"Low valid measurement ratio ({measurement_ratio:.1%} < {config.min_valid_measurements_ratio:.1%})")

    # 4. Stability (20%)
    stability_scores: list[float] = []
    for name, cv in measurement_stabilities.items():
        if cv is not None and math.isfinite(cv):
            name_lower = name.lower()
            if "knee" in name_lower and cv > config.max_knee_cv:
                issues.append(f"High variability in {name} (CV={cv:.2f})")
                stability_scores.append(0.0)
            elif "hip" in name_lower and cv > config.max_hip_cv:
                issues.append(f"High variability in {name} (CV={cv:.2f})")
                stability_scores.append(0.0)
            elif "trunk" in name_lower and cv > config.max_trunk_lean_cv:
                issues.append(f"High variability in {name} (CV={cv:.2f})")
                stability_scores.append(0.0)
            else:
                stability_scores.append(1.0)

    stability_score = sum(stability_scores) / len(stability_scores) if stability_scores else (1.0 if valid_pose_frames > 0 else 0.0)

    # Weighted aggregate score
    score = (
        0.30 * frame_score +
        0.25 * visibility_score +
        0.25 * measurement_score +
        0.20 * stability_score
    )

    # Camera view impact
    view_str = str(camera_view).lower()
    if view_str in ("front", "rear"):
        issues.append(
            f"Camera view '{view_str}': sagittal joint angles (knee/hip flexion) are planar-foreshortened and invalid for sagittal gait analysis."
        )

    # Determine is_reliable
    is_reliable = (
        valid_pose_frames >= threshold_frames
        and pose_ratio >= threshold_ratio
        and measurement_ratio >= config.min_valid_measurements_ratio
        and view_str not in ("front", "rear")
    )

    return round(score, 4), is_reliable, issues
