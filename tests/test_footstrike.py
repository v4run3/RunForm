"""Unit tests for foot strike classification and contact detection."""

import pytest

from runform.footstrike.detector import FootStrikeDetector
from runform.schema import FootStrikeType, Keypoint, PoseDetection


def test_classify_heel_strike() -> None:
    detector = FootStrikeDetector(heel_angle_threshold_deg=5.0)
    # Ankle is ahead of knee (dx = 30px forward, dy = 100px down -> angle ~16.7 deg > 5 deg)
    knee = (100.0, 200.0)
    ankle = (130.0, 300.0)
    strike = detector.classify_strike_from_shank_angle(knee, ankle, velocity_y=0.0)
    assert strike == FootStrikeType.HEEL


def test_classify_forefoot_strike() -> None:
    detector = FootStrikeDetector(forefoot_angle_threshold_deg=-2.0)
    # Ankle is behind knee (dx = -20px backward, dy = 100px down -> negative angle)
    knee = (100.0, 200.0)
    ankle = (80.0, 300.0)
    strike = detector.classify_strike_from_shank_angle(knee, ankle, velocity_y=0.0)
    assert strike == FootStrikeType.FOREFOOT


def test_classify_midfoot_strike() -> None:
    detector = FootStrikeDetector(heel_angle_threshold_deg=8.0, forefoot_angle_threshold_deg=-2.0)
    # Ankle is directly under knee (dx = 2px, dy = 100px -> angle ~1.1 deg)
    knee = (100.0, 200.0)
    ankle = (102.0, 300.0)
    strike = detector.classify_strike_from_shank_angle(knee, ankle, velocity_y=0.0)
    assert strike == FootStrikeType.MIDFOOT


def test_process_frame_detection() -> None:
    detector = FootStrikeDetector()
    kps = {
        "left_hip": Keypoint(x=100.0, y=100.0, confidence=0.9),
        "left_knee": Keypoint(x=100.0, y=200.0, confidence=0.9),
        "left_ankle": Keypoint(x=130.0, y=300.0, confidence=0.9),  # Forward -> Heel
        "right_hip": Keypoint(x=120.0, y=100.0, confidence=0.9),
        "right_knee": Keypoint(x=120.0, y=200.0, confidence=0.9),
        "right_ankle": Keypoint(x=100.0, y=300.0, confidence=0.9),  # Backward -> Forefoot
    }
    det = PoseDetection(
        track_id=1,
        bbox=(80.0, 80.0, 150.0, 320.0),
        confidence=0.9,
        keypoints=kps,
    )
    result = detector.process_frame(det, frame_idx=1)
    assert result["left"] == FootStrikeType.HEEL
    assert result["right"] == FootStrikeType.FOREFOOT
