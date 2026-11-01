"""Biomechanical angle and kinematic calculations for running form analysis.

Computes 2D joint angles (knee flexion, hip extension, trunk lean) from
2D body keypoints according to standard sports biomechanics definitions.
"""

from __future__ import annotations

import math
import numpy as np

from runform.schema import BiomechanicalMetrics, Keypoint, PoseDetection


def compute_angle_2d(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> float:
    """Compute the angle at vertex p2 formed by lines (p1 -> p2) and (p3 -> p2).

    Args:
        p1: (x, y) coordinates of the first point.
        p2: (x, y) coordinates of the central vertex point.
        p3: (x, y) coordinates of the third point.

    Returns:
        Angle in degrees [0, 180].
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp cosine to valid range [-1.0, 1.0] to avoid float precision domain errors
    cosine = max(-1.0, min(1.0, float(cosine)))

    return float(math.degrees(math.acos(cosine)))


def compute_trunk_lean(
    mid_shoulder: tuple[float, float],
    mid_hip: tuple[float, float],
) -> float:
    """Compute forward trunk lean angle relative to vertical axis (0 deg = upright).

    Args:
        mid_shoulder: (x, y) coordinates of the midpoint between left & right shoulders.
        mid_hip: (x, y) coordinates of the midpoint between left & right hips.

    Returns:
        Forward inclination angle in degrees. Positive indicates forward lean.
    """
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_hip[1] - mid_shoulder[1]  # In image coordinates, y increases downwards

    if dy == 0.0 and dx == 0.0:
        return 0.0

    # Angle relative to vertical (upwards)
    lean_rad = math.atan2(dx, dy)
    return float(math.degrees(lean_rad))


class BiomechanicsCalculator:
    """Calculates running kinematics and angles from pose detections."""

    def __init__(self, min_confidence: float = 0.3) -> None:
        """Initialize calculator.

        Args:
            min_confidence: Minimum keypoint confidence required for reliable angle computation.
        """
        self.min_confidence = min_confidence

    def _get_point(self, kps: dict[str, Keypoint], name: str) -> tuple[float, float] | None:
        """Extract point (x, y) if available and above confidence threshold."""
        kp = kps.get(name)
        if kp is not None and kp.confidence >= self.min_confidence:
            return (kp.x, kp.y)
        return None

    def calculate_angles(self, detection: PoseDetection) -> BiomechanicalMetrics:
        """Compute joint angles and posture metrics for a detected person.

        Args:
            detection: PoseDetection object with keypoints.

        Returns:
            BiomechanicalMetrics populated with calculated angles.
        """
        kps = detection.keypoints

        lh = self._get_point(kps, "left_hip")
        lk = self._get_point(kps, "left_knee")
        la = self._get_point(kps, "left_ankle")

        rh = self._get_point(kps, "right_hip")
        rk = self._get_point(kps, "right_knee")
        ra = self._get_point(kps, "right_ankle")

        ls = self._get_point(kps, "left_shoulder")
        rs = self._get_point(kps, "right_shoulder")

        # Left Knee Flexion (Hip - Knee - Ankle)
        left_knee_angle = compute_angle_2d(lh, lk, la) if (lh and lk and la) else None

        # Right Knee Flexion (Hip - Knee - Ankle)
        right_knee_angle = compute_angle_2d(rh, rk, ra) if (rh and rk and ra) else None

        # Left Hip Angle (Shoulder - Hip - Knee)
        left_hip_angle = compute_angle_2d(ls, lh, lk) if (ls and lh and lk) else None

        # Right Hip Angle (Shoulder - Hip - Knee)
        right_hip_angle = compute_angle_2d(rs, rh, rk) if (rs and rh and rk) else None

        # Trunk Lean Angle
        trunk_lean = None
        if ls and rs and lh and rh:
            mid_shoulder = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
            mid_hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
            trunk_lean = compute_trunk_lean(mid_shoulder, mid_hip)
        elif ls and lh:  # Monocular single side view fallback
            trunk_lean = compute_trunk_lean(ls, lh)
        elif rs and rh:
            trunk_lean = compute_trunk_lean(rs, rh)

        return BiomechanicalMetrics(
            knee_flexion_left_deg=left_knee_angle,
            knee_flexion_right_deg=right_knee_angle,
            hip_angle_left_deg=left_hip_angle,
            hip_angle_right_deg=right_hip_angle,
            trunk_lean_deg=trunk_lean,
        )
