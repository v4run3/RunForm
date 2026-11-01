"""Unit tests for biomechanics angle calculations."""

import pytest

from runform.angles.calculator import (
    BiomechanicsCalculator,
    compute_angle_2d,
    compute_trunk_lean,
)
from runform.schema import Keypoint, PoseDetection


def test_compute_angle_2d_right_angle() -> None:
    p1 = (0.0, 1.0)
    p2 = (0.0, 0.0)  # vertex
    p3 = (1.0, 0.0)
    angle = compute_angle_2d(p1, p2, p3)
    assert pytest.approx(angle, abs=1e-3) == 90.0


def test_compute_angle_2d_straight_line() -> None:
    p1 = (0.0, 2.0)
    p2 = (0.0, 1.0)  # vertex
    p3 = (0.0, 0.0)
    angle = compute_angle_2d(p1, p2, p3)
    assert pytest.approx(angle, abs=1e-3) == 180.0


def test_compute_trunk_lean_upright() -> None:
    mid_shoulder = (100.0, 50.0)  # directly above hip
    mid_hip = (100.0, 150.0)
    lean = compute_trunk_lean(mid_shoulder, mid_hip)
    assert pytest.approx(lean, abs=1e-3) == 0.0


def test_compute_trunk_lean_forward() -> None:
    mid_shoulder = (150.0, 50.0)  # 50px forward and 100px up
    mid_hip = (100.0, 150.0)
    lean = compute_trunk_lean(mid_shoulder, mid_hip)
    assert lean > 0.0  # Positive forward lean


def test_biomechanics_calculator_with_keypoints() -> None:
    kps = {
        "left_hip": Keypoint(x=100.0, y=200.0, confidence=0.9),
        "left_knee": Keypoint(x=100.0, y=300.0, confidence=0.9),
        "left_ankle": Keypoint(x=100.0, y=400.0, confidence=0.9),
        "left_shoulder": Keypoint(x=100.0, y=100.0, confidence=0.9),
        "right_shoulder": Keypoint(x=120.0, y=100.0, confidence=0.9),
        "right_hip": Keypoint(x=120.0, y=200.0, confidence=0.9),
    }
    det = PoseDetection(
        track_id=1,
        bbox=(80.0, 80.0, 150.0, 420.0),
        confidence=0.9,
        keypoints=kps,
    )
    calc = BiomechanicsCalculator()
    metrics = calc.calculate_angles(det)

    assert metrics.knee_flexion_left_deg is not None
    assert pytest.approx(metrics.knee_flexion_left_deg, abs=1e-2) == 180.0
    assert metrics.hip_angle_left_deg is not None
    assert pytest.approx(metrics.hip_angle_left_deg, abs=1e-2) == 180.0
