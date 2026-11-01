"""Tests for ML-ready RunForm feature aggregation."""

from pathlib import Path

import pytest

from runform.features.extractor import extract_video_features, select_primary_track_id, write_feature_csv
from runform.schema import (
    BiomechanicalMetrics,
    FootStrikeType,
    FrameAnalysisResult,
    Keypoint,
    PoseDetection,
)


def _frame(
    idx: int,
    left_knee: float | None,
    right_knee: float | None,
    strike: FootStrikeType,
    confidence: float = 0.9,
) -> FrameAnalysisResult:
    detection = PoseDetection(
        track_id=1,
        bbox=(0.0, 0.0, 100.0, 100.0),
        confidence=confidence,
        keypoints={
            "left_hip": Keypoint(0.0, 0.0, confidence),
            "left_knee": Keypoint(0.0, 1.0, confidence),
            "left_ankle": Keypoint(0.0, 2.0, confidence),
        },
    )
    return FrameAnalysisResult(
        frame_index=idx,
        timestamp_sec=idx / 60.0,
        detections=[detection],
        foot_strike={1: strike},
        metrics={
            1: BiomechanicalMetrics(
                knee_flexion_left_deg=left_knee,
                knee_flexion_right_deg=right_knee,
                hip_angle_left_deg=150.0 + idx,
                hip_angle_right_deg=145.0 + idx,
                trunk_lean_deg=5.0 + idx,
            )
        },
    )




def _multi_runner_frame(idx: int, include_track_1: bool = True) -> FrameAnalysisResult:
    detections = []
    metrics = {}
    strikes = {}
    if include_track_1:
        detections.append(
            PoseDetection(
                track_id=1,
                bbox=(0.0, 0.0, 100.0, 100.0),
                confidence=0.9,
                keypoints={"left_knee": Keypoint(0.0, 1.0, 0.9)},
            )
        )
        metrics[1] = BiomechanicalMetrics(
            knee_flexion_left_deg=100.0 + idx,
            knee_flexion_right_deg=90.0 + idx,
            hip_angle_left_deg=120.0,
            hip_angle_right_deg=110.0,
            trunk_lean_deg=4.0,
        )
        strikes[1] = FootStrikeType.HEEL

    detections.append(
        PoseDetection(
            track_id=2,
            bbox=(200.0, 0.0, 300.0, 100.0),
            confidence=0.5,
            keypoints={"left_knee": Keypoint(0.0, 1.0, 0.5)},
        )
    )
    metrics[2] = BiomechanicalMetrics(
        knee_flexion_left_deg=170.0,
        knee_flexion_right_deg=160.0,
        hip_angle_left_deg=170.0,
        hip_angle_right_deg=160.0,
        trunk_lean_deg=20.0,
    )
    strikes[2] = FootStrikeType.FOREFOOT

    return FrameAnalysisResult(
        frame_index=idx,
        timestamp_sec=idx / 30.0,
        detections=detections,
        foot_strike=strikes,
        metrics=metrics,
    )

def test_feature_extractor_aggregates_angles_and_asymmetry() -> None:
    features = extract_video_features(
        [
            _frame(0, 160.0, 150.0, FootStrikeType.HEEL),
            _frame(1, 140.0, 130.0, FootStrikeType.MIDFOOT),
        ],
        video_id="sample.mp4",
        fps=60.0,
    )

    assert features.knee_left_mean == pytest.approx(150.0)
    assert features.knee_left_min == pytest.approx(140.0)
    assert features.knee_left_max == pytest.approx(160.0)
    assert features.knee_asymmetry_mean == pytest.approx(10.0)
    assert features.hip_left_mean == pytest.approx(150.5)
    assert features.trunk_lean_mean == pytest.approx(5.5)


def test_feature_extractor_handles_missing_values_and_insufficient_pose() -> None:
    frame = _frame(0, None, None, FootStrikeType.UNKNOWN, confidence=0.1)
    features = extract_video_features([frame], video_id="low-confidence.mp4", fps=30.0)

    assert features.knee_left_mean is None
    assert features.knee_asymmetry_mean is None
    assert features.unknown_ratio == pytest.approx(1.0)
    assert features.pose_confidence_mean == pytest.approx(0.1)
    assert features.valid_pose_frame_ratio == pytest.approx(0.0)


def test_feature_extractor_aggregates_foot_strike_ratios_and_confidence() -> None:
    features = extract_video_features(
        [
            _frame(0, 160.0, 150.0, FootStrikeType.HEEL, confidence=0.8),
            _frame(1, 161.0, 151.0, FootStrikeType.FOREFOOT, confidence=0.6),
            _frame(2, 162.0, 152.0, FootStrikeType.FOREFOOT, confidence=0.4),
        ],
        video_id="strikes.mp4",
        fps=30.0,
    )

    assert features.heel_ratio == pytest.approx(1 / 3)
    assert features.forefoot_ratio == pytest.approx(2 / 3)
    assert features.midfoot_ratio == pytest.approx(0.0)
    assert features.dominant_foot_strike == FootStrikeType.FOREFOOT.value
    assert features.pose_confidence_mean == pytest.approx(0.6)


def test_feature_extractor_uses_frame_timestamps_to_infer_fps() -> None:
    features = extract_video_features(
        [
            _frame(0, 160.0, 150.0, FootStrikeType.HEEL),
            _frame(1, 161.0, 151.0, FootStrikeType.HEEL),
        ],
        video_id="timed.mp4",
    )

    assert features.fps == pytest.approx(60.0)
    assert features.duration_sec == pytest.approx(2 / 60)


def test_feature_csv_export_format(tmp_path: Path) -> None:
    features = extract_video_features(
        [_frame(0, 160.0, 150.0, FootStrikeType.HEEL)],
        video_id="export.mp4",
        fps=30.0,
    )
    output = tmp_path / "features.csv"

    write_feature_csv([features], output)

    text = output.read_text(encoding="utf-8")
    assert "video_id,knee_left_mean" in text
    assert "export.mp4" in text
    assert "injury" not in text.lower()


def test_feature_extractor_uses_single_primary_runner_track() -> None:
    frames = [
        _multi_runner_frame(0, include_track_1=True),
        _multi_runner_frame(1, include_track_1=True),
        _multi_runner_frame(2, include_track_1=True),
        _multi_runner_frame(3, include_track_1=False),
    ]

    assert select_primary_track_id(frames) == 2

    features = extract_video_features(frames, video_id="multi-runner.mp4", fps=30.0)

    assert features.knee_left_mean == pytest.approx(170.0)
    assert features.knee_right_mean == pytest.approx(160.0)
    assert features.trunk_lean_mean == pytest.approx(20.0)
    assert features.forefoot_ratio == pytest.approx(1.0)
    assert features.heel_ratio == pytest.approx(0.0)
    assert features.pose_confidence_mean == pytest.approx(0.5)


def test_feature_extractor_can_use_explicit_primary_track_id() -> None:
    frames = [
        _multi_runner_frame(0, include_track_1=True),
        _multi_runner_frame(1, include_track_1=True),
    ]

    features = extract_video_features(
        frames,
        video_id="explicit-primary.mp4",
        fps=30.0,
        primary_track_id=1,
    )

    assert features.knee_left_mean == pytest.approx(100.5)
    assert features.knee_right_mean == pytest.approx(90.5)
    assert features.trunk_lean_mean == pytest.approx(4.0)
    assert features.heel_ratio == pytest.approx(1.0)
    assert features.forefoot_ratio == pytest.approx(0.0)
    assert features.pose_confidence_mean == pytest.approx(0.9)
