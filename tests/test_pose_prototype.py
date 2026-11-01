import csv

from runform.pose.angles import calculate_angle, calculate_pose_angles
from runform.pose.landmarks import PoseLandmark, PoseResult
from runform.pose.metrics import (
    FrameMetrics,
    RunningFormMetrics,
    analyze_frame_metrics,
    calculate_knee_asymmetry,
    summarize_frame_metrics,
    write_metrics_csv,
)


def _sample_pose() -> PoseResult:
    return PoseResult(
        image_width=100,
        image_height=100,
        confidence=0.95,
        landmarks={
            "left_hip": PoseLandmark("left_hip", 0.5, 0.4, visibility=1.0),
            "left_knee": PoseLandmark("left_knee", 0.5, 0.6, visibility=1.0),
            "left_ankle": PoseLandmark("left_ankle", 0.5, 0.8, visibility=1.0),
            "right_hip": PoseLandmark("right_hip", 0.7, 0.4, visibility=1.0),
            "right_knee": PoseLandmark("right_knee", 0.7, 0.6, visibility=1.0),
            "right_ankle": PoseLandmark("right_ankle", 0.7, 0.8, visibility=1.0),
            "left_shoulder": PoseLandmark("left_shoulder", 0.5, 0.2, visibility=1.0),
            "right_shoulder": PoseLandmark("right_shoulder", 0.7, 0.2, visibility=1.0),
            "left_elbow": PoseLandmark("left_elbow", 0.4, 0.35, visibility=1.0),
            "left_wrist": PoseLandmark("left_wrist", 0.4, 0.5, visibility=1.0),
            "right_elbow": PoseLandmark("right_elbow", 0.8, 0.35, visibility=1.0),
            "right_wrist": PoseLandmark("right_wrist", 0.8, 0.5, visibility=1.0),
        },
    )


def test_calculate_angle_right_angle():
    assert round(calculate_angle((0, 1), (0, 0), (1, 0))) == 90


def test_calculate_pose_angles_from_landmarks():
    angles = calculate_pose_angles(_sample_pose())

    assert round(angles["left_knee"]) == 180
    assert round(angles["right_knee"]) == 180
    assert round(angles["trunk_lean"]) == 0


def test_knee_asymmetry_calculation():
    assert calculate_knee_asymmetry(145.0, 152.5) == 7.5
    assert calculate_knee_asymmetry(None, 152.5) is None


def test_missing_landmarks_do_not_crash():
    result = PoseResult(
        image_width=100,
        image_height=100,
        confidence=0.5,
        landmarks={"left_knee": PoseLandmark("left_knee", 0.5, 0.6, visibility=1.0)},
    )

    frame = analyze_frame_metrics(frame_index=0, timestamp=0.0, result=result)

    assert frame.pose_detected is True
    assert frame.metrics.left_knee_angle is None
    assert frame.metrics.knee_asymmetry is None


def test_trunk_lean_calculation():
    result = _sample_pose()
    result.landmarks["left_shoulder"] = PoseLandmark("left_shoulder", 0.6, 0.2, visibility=1.0)
    result.landmarks["right_shoulder"] = PoseLandmark("right_shoulder", 0.8, 0.2, visibility=1.0)

    angles = calculate_pose_angles(result)

    assert round(angles["trunk_lean"]) == 27


def test_summary_calculation():
    frames = [
        FrameMetrics(
            frame=0,
            timestamp=0.0,
            pose_detected=True,
            pose_confidence=0.9,
            metrics=RunningFormMetrics(
                left_knee_angle=140.0,
                right_knee_angle=150.0,
                left_hip_angle=160.0,
                right_hip_angle=158.0,
                trunk_lean=5.0,
                knee_asymmetry=10.0,
            ),
        ),
        FrameMetrics(
            frame=1,
            timestamp=0.04,
            pose_detected=False,
            pose_confidence=None,
            metrics=RunningFormMetrics(),
        ),
    ]

    summary = summarize_frame_metrics(frames)

    assert summary.frames_processed == 2
    assert summary.pose_detected_frames == 1
    assert summary.detection_rate == 0.5
    assert summary.average_left_knee_angle == 140.0
    assert summary.average_knee_asymmetry == 10.0
    assert "Pose visibility was low" in summary.observations[0]


def test_csv_generation(tmp_path):
    frames = [
        FrameMetrics(
            frame=7,
            timestamp=0.28,
            pose_detected=True,
            pose_confidence=0.88,
            metrics=RunningFormMetrics(left_knee_angle=141.0, knee_asymmetry=None),
        )
    ]
    output = tmp_path / "metrics.csv"

    write_metrics_csv(frames, output)

    with output.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows[0]["frame"] == "7"
    assert rows[0]["pose_detected"] == "True"
    assert rows[0]["left_knee_angle"] == "141.000000"
    assert rows[0]["knee_asymmetry"] == ""
