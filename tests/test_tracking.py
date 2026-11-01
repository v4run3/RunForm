"""Unit tests for tracking and 1-Euro keypoint filtering."""

import pytest

from runform.schema import Keypoint, PoseDetection
from runform.tracking.tracker import OneEuroFilter, RunnerTracker, compute_iou


def test_compute_iou_identical_boxes() -> None:
    box = (10.0, 10.0, 50.0, 50.0)
    assert pytest.approx(compute_iou(box, box)) == 1.0


def test_compute_iou_disjoint_boxes() -> None:
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box1, box2) == 0.0


def test_one_euro_filter_smoothing() -> None:
    f = OneEuroFilter(freq=30.0)
    val1 = f.filter(100.0)
    assert val1 == 100.0
    # Noisy jump
    val2 = f.filter(110.0)
    assert 100.0 < val2 < 110.0  # Filter attenuates abrupt spike


def test_runner_tracker_lifecycle() -> None:
    tracker = RunnerTracker(iou_threshold=0.3, max_missed_frames=2)

    det1 = PoseDetection(
        track_id=None,
        bbox=(10.0, 10.0, 60.0, 100.0),
        confidence=0.9,
        keypoints={"left_knee": Keypoint(x=30.0, y=50.0, confidence=0.9)},
    )
    res1 = tracker.update([det1], frame_idx=0)
    assert len(res1) == 1
    assert res1[0].track_id == 1

    # Frame 2: Runner moves slightly
    det2 = PoseDetection(
        track_id=None,
        bbox=(12.0, 10.0, 62.0, 100.0),
        confidence=0.9,
        keypoints={"left_knee": Keypoint(x=32.0, y=50.0, confidence=0.9)},
    )
    res2 = tracker.update([det2], frame_idx=1)
    assert len(res2) == 1
    assert res2[0].track_id == 1  # Retains stable track ID
