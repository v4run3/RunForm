"""Smoke, contract, and end-to-end pipeline tests for RunForm."""

import numpy as np
import pytest

from runform.detection.detector import MockPoseDetector
from runform.pipeline import RunFormPipeline
from runform.schema import (
    FootStrikeType,
    InjuryRiskLevel,
    Keypoint,
    PoseDetection,
)


def test_pipeline_instantiates_with_default_config() -> None:
    pipeline = RunFormPipeline()
    assert "RunForm" in pipeline.config["app"]["title"]


def test_pipeline_accepts_custom_config() -> None:
    pipeline = RunFormPipeline(config={"detection": {"conf": 0.7}})
    assert pipeline.config["detection"]["conf"] == 0.7


def test_analyze_frame_end_to_end() -> None:
    # Build sample mock detection
    kps = {
        "left_hip": Keypoint(x=100.0, y=100.0, confidence=0.9),
        "left_knee": Keypoint(x=100.0, y=200.0, confidence=0.9),
        "left_ankle": Keypoint(x=130.0, y=300.0, confidence=0.9),
        "right_hip": Keypoint(x=120.0, y=100.0, confidence=0.9),
        "right_knee": Keypoint(x=120.0, y=200.0, confidence=0.9),
        "right_ankle": Keypoint(x=100.0, y=300.0, confidence=0.9),
    }
    sample_det = PoseDetection(
        track_id=None,
        bbox=(80.0, 80.0, 150.0, 320.0),
        confidence=0.9,
        keypoints=kps,
    )
    mock_detector = MockPoseDetector(sample_detections=[sample_det])
    pipeline = RunFormPipeline(detector=mock_detector)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, annotated = pipeline.analyze_frame(dummy_frame, frame_idx=0, draw_overlay=True)

    assert len(result.detections) == 1
    assert result.detections[0].track_id == 1
    assert result.foot_strike[1] == FootStrikeType.HEEL
    assert result.metrics[1].knee_flexion_left_deg is not None
    assert annotated is not None
    assert annotated.shape == (480, 640, 3)


def test_analyze_frame_uses_supplied_fps_for_timestamp() -> None:
    mock_detector = MockPoseDetector(sample_detections=[])
    pipeline = RunFormPipeline(detector=mock_detector)
    dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)

    result, _ = pipeline.analyze_frame(dummy_frame, frame_idx=30, fps=60.0)

    assert result.timestamp_sec == pytest.approx(0.5)

