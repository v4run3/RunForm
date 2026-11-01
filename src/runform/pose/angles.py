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
) -> float:
    """Return the 2D angle in degrees at point_b."""
    a = np.array(_xy(point_a), dtype=float)
    b = np.array(_xy(point_b), dtype=float)
    c = np.array(_xy(point_c), dtype=float)

    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0

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


def _trunk_lean_degrees(shoulder: tuple[float, float], hip: tuple[float, float]) -> float:
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]
    if dx == 0.0 and dy == 0.0:
        return 0.0
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
        angles["left_knee"] = calculate_angle(
            points["left_hip"], points["left_knee"], points["left_ankle"]
        )
    if points["right_hip"] and points["right_knee"] and points["right_ankle"]:
        angles["right_knee"] = calculate_angle(
            points["right_hip"], points["right_knee"], points["right_ankle"]
        )
    if points["left_shoulder"] and points["left_hip"] and points["left_knee"]:
        angles["left_hip"] = calculate_angle(
            points["left_shoulder"], points["left_hip"], points["left_knee"]
        )
    if points["right_shoulder"] and points["right_hip"] and points["right_knee"]:
        angles["right_hip"] = calculate_angle(
            points["right_shoulder"], points["right_hip"], points["right_knee"]
        )
    if points["left_shoulder"] and points["left_elbow"] and points["left_wrist"]:
        angles["left_elbow"] = calculate_angle(
            points["left_shoulder"], points["left_elbow"], points["left_wrist"]
        )
    if points["right_shoulder"] and points["right_elbow"] and points["right_wrist"]:
        angles["right_elbow"] = calculate_angle(
            points["right_shoulder"], points["right_elbow"], points["right_wrist"]
        )

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    left_hip = points["left_hip"]
    right_hip = points["right_hip"]
    if left_shoulder and right_shoulder and left_hip and right_hip:
        angles["trunk_lean"] = _trunk_lean_degrees(
            _midpoint(left_shoulder, right_shoulder),
            _midpoint(left_hip, right_hip),
        )
    elif left_shoulder and left_hip:
        angles["trunk_lean"] = _trunk_lean_degrees(left_shoulder, left_hip)
    elif right_shoulder and right_hip:
        angles["trunk_lean"] = _trunk_lean_degrees(right_shoulder, right_hip)

    return angles
