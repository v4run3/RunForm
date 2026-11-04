"""Tests for Milestone 1: Temporal Movement Analysis module."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

import pytest

from runform.pose.metrics import FrameMetrics, RunningFormMetrics, RunningFormSummary
from runform.pose.service import summary_to_dict
from runform.pose.temporal_analysis import (
    TemporalAnalysisResult,
    TemporalFeatures,
    WindowTemporalResult,
    analyze_temporal_movement,
    calculate_temporal_features,
    smooth_frame_metrics,
)


def _make_frame(
    frame: int,
    timestamp: float,
    left_knee: float | None = 150.0,
    right_knee: float | None = 150.0,
    left_hip: float | None = 160.0,
    right_hip: float | None = 160.0,
    trunk_lean: float | None = 10.0,
    pose_detected: bool = True,
    confidence: float | None = 0.9,
) -> FrameMetrics:
    asym = abs(left_knee - right_knee) if left_knee is not None and right_knee is not None else None
    return FrameMetrics(
        frame=frame,
        timestamp=timestamp,
        pose_detected=pose_detected,
        pose_confidence=confidence if pose_detected else None,
        metrics=RunningFormMetrics(
            left_knee_angle=left_knee if pose_detected else None,
            right_knee_angle=right_knee if pose_detected else None,
            left_hip_angle=left_hip if pose_detected else None,
            right_hip_angle=right_hip if pose_detected else None,
            trunk_lean=trunk_lean if pose_detected else None,
            knee_asymmetry=asym if pose_detected else None,
        ),
    )


def test_temporal_features_known_expected_values() -> None:
    """Verify mean, std, min, max, ROM, and bilateral differences on synthetic known values."""
    frames = [
        _make_frame(0, 0.0, left_knee=140.0, right_knee=150.0, left_hip=160.0, right_hip=170.0, trunk_lean=10.0),
        _make_frame(1, 0.1, left_knee=160.0, right_knee=130.0, left_hip=150.0, right_hip=160.0, trunk_lean=14.0),
        _make_frame(2, 0.2, left_knee=150.0, right_knee=140.0, left_hip=170.0, right_hip=150.0, trunk_lean=12.0),
    ]

    features = calculate_temporal_features(frames, min_valid_frames=2, min_valid_ratio=0.5)

    # Knee Left: [140, 160, 150]
    assert features.knee_left_mean == pytest.approx(150.0)
    assert features.knee_left_min == pytest.approx(140.0)
    assert features.knee_left_max == pytest.approx(160.0)
    assert features.knee_left_rom == pytest.approx(20.0)
    expected_lk_std = statistics.pstdev([140.0, 160.0, 150.0])
    assert features.knee_left_std == pytest.approx(expected_lk_std)

    # Knee Right: [150, 130, 140]
    assert features.knee_right_mean == pytest.approx(140.0)
    assert features.knee_right_min == pytest.approx(130.0)
    assert features.knee_right_max == pytest.approx(150.0)
    assert features.knee_right_rom == pytest.approx(20.0)
    assert features.knee_right_std == pytest.approx(expected_lk_std)

    # Bilateral Knee Diff (left - right): [-10, 30, 10]
    expected_diff_mean = statistics.fmean([-10.0, 30.0, 10.0])
    expected_diff_std = statistics.pstdev([-10.0, 30.0, 10.0])
    assert features.knee_diff_mean == pytest.approx(expected_diff_mean)
    assert features.knee_diff_std == pytest.approx(expected_diff_std)

    # Asymmetry (|left - right|): [10, 30, 10]
    assert features.knee_asymmetry_mean == pytest.approx(statistics.fmean([10.0, 30.0, 10.0]))
    assert features.knee_asymmetry_std == pytest.approx(statistics.pstdev([10.0, 30.0, 10.0]))

    # Hip Left: [160, 150, 170]
    assert features.hip_left_mean == pytest.approx(160.0)
    assert features.hip_left_rom == pytest.approx(20.0)

    # Hip Right: [170, 160, 150]
    assert features.hip_right_mean == pytest.approx(160.0)
    assert features.hip_right_rom == pytest.approx(20.0)

    # Trunk Lean: [10, 14, 12]
    assert features.trunk_lean_mean == pytest.approx(12.0)
    assert features.trunk_lean_min == pytest.approx(10.0)
    assert features.trunk_lean_max == pytest.approx(14.0)
    assert features.trunk_lean_rom == pytest.approx(4.0)
    assert features.trunk_lean_std == pytest.approx(statistics.pstdev([10.0, 14.0, 12.0]))

    # Quality metrics
    assert features.total_frames == 3
    assert features.valid_pose_frames == 3
    assert features.valid_pose_frame_ratio == pytest.approx(1.0)
    assert features.missing_measurements == 0
    assert features.is_reliable is True


def test_missing_and_invalid_measurements() -> None:
    """Verify statistics cleanly ignore missing/invalid frames and count missing slots."""
    frames = [
        _make_frame(0, 0.0, left_knee=150.0, right_knee=150.0, pose_detected=True, confidence=0.8),
        # Frame with undetected pose
        _make_frame(1, 0.1, left_knee=None, right_knee=None, pose_detected=False, confidence=None),
        # Frame with missing right knee
        _make_frame(2, 0.2, left_knee=170.0, right_knee=None, pose_detected=True, confidence=0.9),
    ]

    features = calculate_temporal_features(frames, min_valid_frames=2, min_valid_ratio=0.5)

    assert features.total_frames == 3
    assert features.valid_pose_frames == 2
    assert features.valid_pose_frame_ratio == pytest.approx(2 / 3)
    assert features.pose_confidence_mean == pytest.approx(0.85)

    # Left knee has [150.0, 170.0]
    assert features.knee_left_mean == pytest.approx(160.0)
    assert features.knee_left_min == pytest.approx(150.0)
    assert features.knee_left_max == pytest.approx(170.0)
    assert features.knee_left_rom == pytest.approx(20.0)

    # Right knee only has [150.0]
    assert features.knee_right_mean == pytest.approx(150.0)
    assert features.knee_right_std == pytest.approx(0.0)
    assert features.knee_right_rom == pytest.approx(0.0)

    # Knee diff only exists on frame 0
    assert features.knee_diff_mean == pytest.approx(0.0)
    assert features.knee_diff_std == pytest.approx(0.0)

    # Missing slots:
    # Frame 1 (undetected): 5 missing
    # Frame 2 (missing right knee): 1 missing
    assert features.missing_measurements == 6
    assert features.is_reliable is True


def test_insufficient_valid_frames() -> None:
    """Verify short or mostly invalid recordings are marked is_reliable=False without crashing."""
    frames = [
        _make_frame(0, 0.0, left_knee=150.0, pose_detected=True),
        _make_frame(1, 0.1, pose_detected=False),
        _make_frame(2, 0.2, pose_detected=False),
        _make_frame(3, 0.3, pose_detected=False),
    ]

    features = calculate_temporal_features(frames, min_valid_frames=3, min_valid_ratio=0.5)
    assert features.total_frames == 4
    assert features.valid_pose_frames == 1
    assert features.valid_pose_frame_ratio == pytest.approx(0.25)
    assert features.is_reliable is False


def test_empty_sequence_graceful_handling() -> None:
    """Empty list of frames produces clean empty features without errors."""
    features = calculate_temporal_features([])
    assert features.total_frames == 0
    assert features.valid_pose_frames == 0
    assert features.knee_left_mean is None
    assert features.knee_left_rom is None
    assert features.is_reliable is False

    res = analyze_temporal_movement([], window_sec=2.0)
    assert res.total_frames == 0
    assert len(res.windows) == 0
    assert res.video_features.knee_left_mean is None


def test_short_recording_window_generation() -> None:
    """Short recording (< window_sec) generates 1 window covering the entire duration."""
    frames = [
        _make_frame(i, i * 0.1, left_knee=140.0 + i * 2, right_knee=140.0)
        for i in range(5)  # 0.0s to 0.4s
    ]
    res = analyze_temporal_movement(frames, window_sec=2.0)
    assert res.total_frames == 5
    assert len(res.windows) == 1
    assert res.windows[0].window_index == 0
    assert res.windows[0].total_frames == 5
    assert res.windows[0].features.knee_left_mean is not None


def test_different_window_durations() -> None:
    """Verify configurable window durations (1s, 2s, 3s)."""
    # 6.0 seconds of video at 10 fps (60 frames)
    frames = [
        _make_frame(i, i * 0.1, left_knee=145.0 + math.sin(i * 0.5) * 15.0)
        for i in range(60)
    ]

    res_1s = analyze_temporal_movement(frames, window_sec=1.0)
    assert len(res_1s.windows) >= 6

    res_2s = analyze_temporal_movement(frames, window_sec=2.0)
    assert len(res_2s.windows) >= 3

    res_3s = analyze_temporal_movement(frames, window_sec=3.0)
    assert len(res_3s.windows) >= 2


def test_optional_smoothing_preprocessing() -> None:
    """Verify smoothing reduces high-frequency noise when enabled and leaves frames raw when disabled."""
    frames = [
        _make_frame(0, 0.0, left_knee=140.0),
        _make_frame(1, 0.1, left_knee=170.0),  # Sudden spike
        _make_frame(2, 0.2, left_knee=140.0),
    ]

    # Without smoothing (window_size=1)
    raw_frames = smooth_frame_metrics(frames, window_size=1)
    assert raw_frames[1].metrics.left_knee_angle == 170.0

    # With smoothing (window_size=3)
    smoothed = smooth_frame_metrics(frames, window_size=3)
    assert len(smoothed) == 3
    # Centered window around index 1: (140 + 170 + 140) / 3 = 150.0
    assert smoothed[1].metrics.left_knee_angle == pytest.approx(150.0)


def test_csv_and_json_export(tmp_path: Path) -> None:
    """Verify CSV and JSON serialization for window-level and video-level results."""
    frames = [
        _make_frame(i, i * 0.5, left_knee=140.0 + i * 5, right_knee=150.0 - i * 5)
        for i in range(10)  # 0.0s to 4.5s
    ]
    result = analyze_temporal_movement(frames, window_sec=2.0)

    # JSON export
    json_path = tmp_path / "temporal_summary.json"
    result.write_summary_json(json_path)
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "video_features" in data
    assert "windows" in data
    assert len(data["windows"]) > 0

    # CSV export
    csv_path = tmp_path / "temporal_windows.csv"
    result.write_windows_csv(csv_path, video_id="test_run")
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == len(result.windows)
        assert "window_index" in rows[0]
        assert "knee_left_rom" in rows[0]
        assert "video_id" in rows[0]
        assert rows[0]["video_id"] == "test_run"


def test_api_summary_backward_compatibility() -> None:
    """Ensure summary_to_dict preserves all existing keys and format."""
    summary = RunningFormSummary(
        frames_processed=100,
        pose_detected_frames=95,
        detection_rate=0.95,
        average_left_knee_angle=158.4,
        average_right_knee_angle=160.1,
        average_left_hip_angle=162.0,
        average_right_hip_angle=161.5,
        average_trunk_lean=8.2,
        average_knee_asymmetry=2.5,
        observations=["Good symmetry."],
    )
    d = summary_to_dict(summary)
    expected_keys = {
        "frames_processed",
        "pose_detected_frames",
        "detection_rate",
        "average_left_knee_angle",
        "average_right_knee_angle",
        "average_left_hip_angle",
        "average_right_hip_angle",
        "average_trunk_lean",
        "average_knee_asymmetry",
        "observations",
    }
    assert set(d.keys()) == expected_keys
    assert d["frames_processed"] == 100
    assert d["detection_rate"] == 0.95
    assert d["average_left_knee_angle"] == 158.4
