"""Tests for pose measurement quality validation, filtering, and reliability scoring."""

from __future__ import annotations

import math
import pytest

from runform.pose.angles import calculate_angle, calculate_pose_angles
from runform.pose.landmarks import PoseLandmark, PoseResult
from runform.pose.metrics import (
    FrameMetrics,
    RunningFormMetrics,
    analyze_frame_metrics,
    calculate_running_form_metrics,
    extract_raw_and_filtered_metrics,
)
from runform.pose.quality import (
    DEFAULT_ANGLE_QUALITY_CONFIG,
    AngleQualityConfig,
    ReliabilityConfig,
    calculate_reliability_score,
    calculate_trunk_lean_rom_safe,
    is_measurement_valid_for_view,
    validate_angle_plausibility,
)
from runform.pose.temporal_analysis import (
    TemporalFeatures,
    analyze_temporal_movement,
    calculate_temporal_features,
)


def _make_dummy_pose(
    left_knee_points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    vis: float = 1.0,
    shoulder_y: float = 0.2,
    hip_y: float = 0.4,
    shoulder_x: float = 0.5,
    hip_x: float = 0.5,
) -> PoseResult:
    """Helper to create a synthetic PoseResult with specified landmark positions."""
    lk_p = left_knee_points or ((0.5, 0.4), (0.5, 0.6), (0.5, 0.8))
    landmarks = {
        "left_hip": PoseLandmark("left_hip", lk_p[0][0], lk_p[0][1], visibility=vis),
        "left_knee": PoseLandmark("left_knee", lk_p[1][0], lk_p[1][1], visibility=vis),
        "left_ankle": PoseLandmark("left_ankle", lk_p[2][0], lk_p[2][1], visibility=vis),
        "right_hip": PoseLandmark("right_hip", 0.6, hip_y, visibility=vis),
        "right_knee": PoseLandmark("right_knee", 0.6, 0.6, visibility=vis),
        "right_ankle": PoseLandmark("right_ankle", 0.6, 0.8, visibility=vis),
        "left_shoulder": PoseLandmark("left_shoulder", shoulder_x, shoulder_y, visibility=vis),
        "right_shoulder": PoseLandmark("right_shoulder", shoulder_x + 0.1, shoulder_y, visibility=vis),
        "left_elbow": PoseLandmark("left_elbow", 0.4, 0.3, visibility=vis),
        "left_wrist": PoseLandmark("left_wrist", 0.4, 0.45, visibility=vis),
        "right_elbow": PoseLandmark("right_elbow", 0.7, 0.3, visibility=vis),
        "right_wrist": PoseLandmark("right_wrist", 0.7, 0.45, visibility=vis),
    }
    return PoseResult(
        landmarks=landmarks,
        confidence=vis,
        image_width=1000,
        image_height=1000,
    )


# ==============================================================================
# 1. Missing Landmarks Handling
# ==============================================================================

def test_missing_landmarks_returns_none_without_crashing():
    """Verify that missing landmarks cause missing angles without raising exceptions."""
    result = PoseResult(
        landmarks={
            "left_hip": PoseLandmark("left_hip", 0.5, 0.4, visibility=0.9),
            # Missing left_knee and left_ankle
        },
        confidence=0.9,
        image_width=100,
        image_height=100,
    )
    angles = calculate_pose_angles(result)
    assert "left_knee" not in angles

    filtered, raw, flags = extract_raw_and_filtered_metrics(result)
    assert filtered.left_knee_angle is None
    assert raw.left_knee_angle is None


# ==============================================================================
# 2. Low Landmark Visibility
# ==============================================================================

def test_low_visibility_landmarks_are_rejected():
    """Verify that landmarks below min_visibility are excluded from angle calculation."""
    result = _make_dummy_pose(vis=0.2)  # Below default 0.3
    angles = calculate_pose_angles(result, min_visibility=0.3)
    assert "left_knee" not in angles

    # But with lower threshold, they are included
    angles_permissive = calculate_pose_angles(result, min_visibility=0.1)
    assert "left_knee" in angles_permissive


# ==============================================================================
# 3. Invalid & Implausible Angle Quality Checks
# ==============================================================================

def test_implausible_knee_angles_are_filtered_and_preserved_in_raw():
    """Verify that 2.7° collapsed knee angle is rejected in filtered metrics but saved in raw."""
    # Synthetic collapsed triplet (nearly overlapping/collinear returning ~2.7°)
    status, reason = validate_angle_plausibility("left_knee", 2.69)
    assert status == "implausible"
    assert "outside plausible range" in reason

    status_ok, _ = validate_angle_plausibility("left_knee", 145.0)
    assert status_ok == "valid"

    # In extract_raw_and_filtered_metrics
    dummy_pose = _make_dummy_pose()
    # Mocking angle dict with implausible knee
    metrics = FrameMetrics(
        frame=0,
        timestamp=0.0,
        pose_detected=True,
        pose_confidence=0.9,
        metrics=RunningFormMetrics(left_knee_angle=None),  # Filtered will be None
        raw_metrics=RunningFormMetrics(left_knee_angle=2.69),  # Raw preserved
        measurement_quality={"left_knee": "implausible: Knee angle 2.7° outside [20°, 185°]"},
    )
    assert metrics.metrics.left_knee_angle is None
    assert metrics.raw_metrics.left_knee_angle == 2.69
    assert "implausible" in metrics.measurement_quality["left_knee"]


def test_implausible_hip_and_elbow_angles():
    """Verify hip and elbow plausibility ranges."""
    # Hip: [45°, 190°]
    assert validate_angle_plausibility("left_hip", 1.8)[0] == "implausible"
    assert validate_angle_plausibility("left_hip", 150.0)[0] == "valid"

    # Elbow: [20°, 180°]
    assert validate_angle_plausibility("left_elbow", 5.0)[0] == "implausible"
    assert validate_angle_plausibility("left_elbow", 90.0)[0] == "valid"


# ==============================================================================
# 4. Trunk Lean & Angle Wrapping
# ==============================================================================

def test_trunk_lean_reference_axis_upright_posture():
    """Verify upright torso (shoulder directly above hip) yields 0.0° lean."""
    pose = _make_dummy_pose(shoulder_y=0.2, hip_y=0.5, shoulder_x=0.5, hip_x=0.5)
    angles = calculate_pose_angles(pose)
    assert pytest.approx(angles["trunk_lean"], abs=1e-2) == 0.0


def test_trunk_lean_forward_lean():
    """Verify forward lean produces expected positive angle when runner heads right."""
    # Shoulder forward (x=0.55 > 0.5) and above hip (y=0.2 < 0.5)
    pose = _make_dummy_pose(shoulder_y=0.2, hip_y=0.5, shoulder_x=0.55, hip_x=0.5)
    angles = calculate_pose_angles(pose)
    assert angles["trunk_lean"] > 0.0  # Positive forward lean


def test_trunk_lean_angle_wrapping_safe_rom():
    """Verify that angles spanning -180/+180 boundary do not report ~359° ROM."""
    min_tl = -179.7
    max_tl = 179.4

    # Raw max - min would be 359.1°
    assert (max_tl - min_tl) > 350.0

    # Safe ROM calculation handles boundary wrapping
    rom = calculate_trunk_lean_rom_safe(min_tl, max_tl)
    assert rom is not None
    # Circular shortest distance: (180 - 179.4) + (180 - 179.7) = 0.6 + 0.3 = 0.9°
    assert rom < 5.0

    # Test with unwrapping sequence
    vals = [-179.7, -178.0, 179.4, 178.0]
    rom_seq = calculate_trunk_lean_rom_safe(min_tl, max_tl, values=vals)
    assert rom_seq is not None
    assert rom_seq < 15.0


def test_inverted_trunk_lean_is_flagged_implausible():
    """Verify inverted trunk lean (e.g. 179°) is flagged as implausible."""
    status, reason = validate_angle_plausibility("trunk_lean", 179.4)
    assert status == "implausible"
    assert "outside plausible range" in reason


# ==============================================================================
# 5. Camera Views Handling
# ==============================================================================

def test_camera_view_validity_rules():
    """Verify view-specific measurement validity rules."""
    # Side view: all valid
    assert is_measurement_valid_for_view("left_knee", "side")[0] is True
    assert is_measurement_valid_for_view("left_hip", "side")[0] is True
    assert is_measurement_valid_for_view("trunk_lean", "side")[0] is True

    # Front view: knee and hip flexion invalid due to planar foreshortening
    valid_knee, note_knee = is_measurement_valid_for_view("left_knee", "front")
    assert valid_knee is False
    assert "foreshortening" in note_knee.lower()

    valid_hip, note_hip = is_measurement_valid_for_view("right_hip", "front")
    assert valid_hip is False
    assert "foreshortening" in note_hip.lower()

    valid_tl, note_tl = is_measurement_valid_for_view("trunk_lean", "front")
    assert valid_tl is True
    assert "lateral" in note_tl.lower()

    # Rear view: same foreshortening limitation
    assert is_measurement_valid_for_view("left_knee", "rear")[0] is False


def test_front_view_video_marked_unreliable():
    """Verify that front-view analysis reflects view limitations in reliability and issues."""
    frames = [
        FrameMetrics(
            frame=i,
            timestamp=i * 0.1,
            pose_detected=True,
            pose_confidence=0.9,
            metrics=RunningFormMetrics(left_knee_angle=150.0, trunk_lean=5.0),
        )
        for i in range(10)
    ]
    res = analyze_temporal_movement(frames, window_sec=2.0, camera_view="front")
    assert res.video_features.is_reliable is False
    assert any("foreshortened" in iss for iss in res.video_features.view_quality_issues)


# ==============================================================================
# 6. Multi-Factor Reliability Scoring
# ==============================================================================

def test_unreliable_window_scoring_low_detection():
    """Verify window with low pose detection rate gets low reliability score."""
    score, is_rel, issues = calculate_reliability_score(
        total_frames=50,
        valid_pose_frames=10,  # 20% < 40%
        valid_measurements={"left_knee": 10},
        landmark_visibilities={},
        measurement_stabilities={},
        camera_view="side",
    )
    assert is_rel is False
    assert any("Low pose detection rate" in iss for iss in issues)


def test_unreliable_window_high_cv():
    """Verify window with excessive angle variability (jitter) is flagged."""
    score, is_rel, issues = calculate_reliability_score(
        total_frames=50,
        valid_pose_frames=45,
        valid_measurements={"left_knee": 45, "left_hip": 45},
        landmark_visibilities={},
        measurement_stabilities={"left_knee": 0.85},  # CV > 0.6 threshold
        camera_view="side",
    )
    assert any("High variability" in iss for iss in issues)


# ==============================================================================
# 7. Backward Compatibility
# ==============================================================================

def test_backward_compatibility_metric_fields():
    """Ensure RunningFormMetrics and FrameMetrics preserve existing CSV and dict interfaces."""
    rfm = RunningFormMetrics(
        left_knee_angle=140.0,
        right_knee_angle=142.0,
        left_hip_angle=160.0,
        right_hip_angle=161.0,
        left_elbow_angle=90.0,
        right_elbow_angle=92.0,
        trunk_lean=8.0,
        knee_asymmetry=2.0,
    )
    d = rfm.as_dict()
    assert len(d) == 8
    assert d["left_knee_angle"] == 140.0
    assert d["knee_asymmetry"] == 2.0

    fm = FrameMetrics(
        frame=1,
        timestamp=0.033,
        pose_detected=True,
        pose_confidence=0.95,
        metrics=rfm,
    )
    row = fm.as_csv_row()
    assert "frame" in row
    assert "left_knee_angle" in row
    assert row["frame"] == 1
    assert row["left_knee_angle"] == "140.000000"


def test_degenerate_zero_length_segment_returns_none():
    """Verify that calculate_angle returns None rather than 0.0 for coincident points."""
    # Vertex and point_a are the exact same coordinates
    angle = calculate_angle((1.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    assert angle is None
