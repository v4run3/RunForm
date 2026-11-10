"""Simple joint-angle utilities for pose landmarks."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .landmarks import PoseLandmark, PoseResult


class PointLike(Protocol):
    x: float
    y: float


def _xy(point: PoseLandmark | PointLike | tuple[float, float]) -> tuple[float, float]:
    if isinstance(point, tuple):
        return (float(point[0]), float(point[1]))
    return (float(point.x), float(point.y))


def calculate_angle(
    point_a: PoseLandmark | PointLike | tuple[float, float],
    point_b: PoseLandmark | PointLike | tuple[float, float],
    point_c: PoseLandmark | PointLike | tuple[float, float],
    min_segment_length: float = 1e-6,
) -> float | None:
    """Return the 2D angle in degrees at vertex point_b, or None if degenerate.

    Uses cosine rule: dot(ba, bc) / (|ba| * |bc|).
    Returns None when landmarks collapse onto vertex point_b or segment lengths are
    effectively zero (<= min_segment_length).
    """
    a = np.array(_xy(point_a), dtype=float)
    b = np.array(_xy(point_b), dtype=float)
    c = np.array(_xy(point_c), dtype=float)

    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))

    if norm_ba <= min_segment_length or norm_bc <= min_segment_length:
        return None

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.degrees(math.acos(cosine)))


def _pixel_point(result: PoseResult, name: str, min_visibility: float) -> tuple[float, float] | None:
    landmark = result.get(name, min_visibility=min_visibility)
    if landmark is None:
        return None
    return (landmark.x * result.image_width, landmark.y * result.image_height)


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _trunk_lean_degrees(
    shoulder: tuple[float, float],
    hip: tuple[float, float],
) -> float | None:
    """Calculate trunk lean angle relative to vertical reference axis.

    Reference axis and sign convention:
    - Reference axis: Vertical line passing upward from the hip midpoint (0, 1 in standard Cartesian).
    - Image space: Top-left origin, y increases downwards. Thus vertical upward vector is (0, -1).
    - Horizontal displacement: dx = shoulder_x - hip_x
    - Vertical displacement: dy = hip_y - shoulder_y (positive when shoulder is above hip)
    - Calculated as: math.degrees(math.atan2(dx, dy)) in range [-180°, +180°].
    - Upright posture (dx = 0, dy > 0): 0.0°.
    - Sagittal view forward lean: positive when heading right (dx > 0), negative when heading left (dx < 0).
    - Frontal view: dx measures lateral trunk sway (tilt).
    - Degenerate: returns None if shoulder and hip coincide (dx == 0 and dy == 0).
    """
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]
    if dx == 0.0 and dy == 0.0:
        return None
    return float(math.degrees(math.atan2(dx, dy)))


def calculate_pose_angles(
    result: PoseResult | None,
    min_visibility: float = 0.3,
) -> dict[str, float]:
    """Calculate common running-form angles from a pose result."""
    if result is None or not result.detected:
        return {}

    points = {
        name: _pixel_point(result, name, min_visibility)
        for name in [
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
    }
    angles: dict[str, float] = {}

    if points["left_hip"] and points["left_knee"] and points["left_ankle"]:
        angle = calculate_angle(points["left_hip"], points["left_knee"], points["left_ankle"])
        if angle is not None:
            angles["left_knee"] = angle

    if points["right_hip"] and points["right_knee"] and points["right_ankle"]:
        angle = calculate_angle(points["right_hip"], points["right_knee"], points["right_ankle"])
        if angle is not None:
            angles["right_knee"] = angle

    if points["left_shoulder"] and points["left_hip"] and points["left_knee"]:
        angle = calculate_angle(points["left_shoulder"], points["left_hip"], points["left_knee"])
        if angle is not None:
            angles["left_hip"] = angle

    if points["right_shoulder"] and points["right_hip"] and points["right_knee"]:
        angle = calculate_angle(points["right_shoulder"], points["right_hip"], points["right_knee"])
        if angle is not None:
            angles["right_hip"] = angle

    if points["left_shoulder"] and points["left_elbow"] and points["left_wrist"]:
        angle = calculate_angle(points["left_shoulder"], points["left_elbow"], points["left_wrist"])
        if angle is not None:
            angles["left_elbow"] = angle

    if points["right_shoulder"] and points["right_elbow"] and points["right_wrist"]:
        angle = calculate_angle(points["right_shoulder"], points["right_elbow"], points["right_wrist"])
        if angle is not None:
            angles["right_elbow"] = angle

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    left_hip = points["left_hip"]
    right_hip = points["right_hip"]

    tl: float | None = None
    if left_shoulder and right_shoulder and left_hip and right_hip:
        tl = _trunk_lean_degrees(
            _midpoint(left_shoulder, right_shoulder),
            _midpoint(left_hip, right_hip),
        )
    elif left_shoulder and left_hip:
        tl = _trunk_lean_degrees(left_shoulder, left_hip)
    elif right_shoulder and right_hip:
        tl = _trunk_lean_degrees(right_shoulder, right_hip)

    if tl is not None:
        angles["trunk_lean"] = tl

    return angles
