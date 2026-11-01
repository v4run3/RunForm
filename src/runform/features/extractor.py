"""Aggregate RunForm frame analysis into one ML-ready row per video.

This module intentionally does not train or load an injury prediction model. It
exports pose-derived, 2D video features that can later be joined with a real
labelled dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math
import statistics

from runform.schema import FootStrikeType, FrameAnalysisResult, PoseDetection


@dataclass
class VideoFeatureVector:
    """One row of pose-derived features for future model training."""

    video_id: str
    knee_left_mean: float | None = None
    knee_left_std: float | None = None
    knee_left_min: float | None = None
    knee_left_max: float | None = None
    knee_right_mean: float | None = None
    knee_right_std: float | None = None
    knee_right_min: float | None = None
    knee_right_max: float | None = None
    knee_asymmetry_mean: float | None = None
    knee_asymmetry_std: float | None = None
    hip_left_mean: float | None = None
    hip_left_std: float | None = None
    hip_left_min: float | None = None
    hip_left_max: float | None = None
    hip_right_mean: float | None = None
    hip_right_std: float | None = None
    hip_right_min: float | None = None
    hip_right_max: float | None = None
    trunk_lean_mean: float | None = None
    trunk_lean_std: float | None = None
    trunk_lean_min: float | None = None
    trunk_lean_max: float | None = None
    heel_ratio: float | None = None
    midfoot_ratio: float | None = None
    forefoot_ratio: float | None = None
    unknown_ratio: float | None = None
    dominant_foot_strike: str | None = None
    cadence_spm: float | None = None
    pose_confidence_mean: float | None = None
    valid_pose_frame_ratio: float | None = None
    fps: float | None = None
    duration_sec: float | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a CSV/API-friendly dictionary."""
        return asdict(self)


def _clean(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _stats(values: list[float | None]) -> tuple[float | None, float | None, float | None, float | None]:
    cleaned = _clean(values)
    if not cleaned:
        return None, None, None, None
    mean = float(statistics.fmean(cleaned))
    std = float(statistics.pstdev(cleaned)) if len(cleaned) > 1 else 0.0
    return mean, std, float(min(cleaned)), float(max(cleaned))


def _ratio(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return count / total


def _detection_track_id(detection: PoseDetection) -> int:
    return detection.track_id if detection.track_id is not None else 0


def select_primary_track_id(frame_results: list[FrameAnalysisResult]) -> int | None:
    """Select the runner track visible in the most frames.

    RunnerTracker assigns explicit integer track IDs. For one video-level feature
    row, we use a single primary track instead of blending multiple runners.
    Ties are resolved by the lowest track ID for deterministic exports.
    """
    visible_frame_counts: dict[int, int] = {}
    for frame in frame_results:
        frame_track_ids = {_detection_track_id(detection) for detection in frame.detections}
        frame_track_ids.update(frame.metrics.keys())
        frame_track_ids.update(frame.foot_strike.keys())
        for track_id in frame_track_ids:
            visible_frame_counts[track_id] = visible_frame_counts.get(track_id, 0) + 1

    if not visible_frame_counts:
        return None
    return min(
        visible_frame_counts,
        key=lambda track_id: (-visible_frame_counts[track_id], track_id),
    )


def _frame_pose_confidence(frame: FrameAnalysisResult, primary_track_id: int) -> float | None:
    confidences: list[float] = []
    for detection in frame.detections:
        if _detection_track_id(detection) == primary_track_id:
            confidences.extend(kp.confidence for kp in detection.keypoints.values())
    if not confidences:
        return None
    return float(statistics.fmean(confidences))


def extract_video_features(
    frame_results: list[FrameAnalysisResult],
    video_id: str,
    fps: float | None = None,
    min_pose_confidence: float = 0.3,
    primary_track_id: int | None = None,
) -> VideoFeatureVector:
    """Build one feature row from frame-level RunForm results.

    The returned features are based on one selected runner track from 2D keypoints
    in ordinary monocular video. Absolute distances and medically labelled injury
    targets are not inferred here.
    """
    selected_track_id = primary_track_id
    if selected_track_id is None:
        selected_track_id = select_primary_track_id(frame_results)

    knee_left: list[float | None] = []
    knee_right: list[float | None] = []
    hip_left: list[float | None] = []
    hip_right: list[float | None] = []
    trunk: list[float | None] = []
    asymmetry: list[float | None] = []
    pose_confidences: list[float | None] = []
    strike_counts = {
        FootStrikeType.HEEL: 0,
        FootStrikeType.MIDFOOT: 0,
        FootStrikeType.FOREFOOT: 0,
        FootStrikeType.UNKNOWN: 0,
    }

    for frame in frame_results:
        if selected_track_id is None:
            pose_confidences.append(None)
            continue

        metrics = frame.metrics.get(selected_track_id)
        if metrics is not None:
            knee_left.append(metrics.knee_flexion_left_deg)
            knee_right.append(metrics.knee_flexion_right_deg)
            hip_left.append(metrics.hip_angle_left_deg)
            hip_right.append(metrics.hip_angle_right_deg)
            trunk.append(metrics.trunk_lean_deg)

            if metrics.knee_flexion_left_deg is not None and metrics.knee_flexion_right_deg is not None:
                asymmetry.append(abs(metrics.knee_flexion_left_deg - metrics.knee_flexion_right_deg))

        if selected_track_id in frame.foot_strike:
            strike = frame.foot_strike[selected_track_id]
            strike_counts[strike if strike in strike_counts else FootStrikeType.UNKNOWN] += 1

        pose_confidences.append(_frame_pose_confidence(frame, selected_track_id))

    knee_left_mean, knee_left_std, knee_left_min, knee_left_max = _stats(knee_left)
    knee_right_mean, knee_right_std, knee_right_min, knee_right_max = _stats(knee_right)
    knee_asymmetry_mean, knee_asymmetry_std, _, _ = _stats(asymmetry)
    hip_left_mean, hip_left_std, hip_left_min, hip_left_max = _stats(hip_left)
    hip_right_mean, hip_right_std, hip_right_min, hip_right_max = _stats(hip_right)
    trunk_mean, trunk_std, trunk_min, trunk_max = _stats(trunk)
    pose_confidence_mean, _, _, _ = _stats(pose_confidences)

    valid_pose_frames = sum(
        1 for conf in pose_confidences if conf is not None and conf >= min_pose_confidence
    )
    total_frames = len(frame_results)
    total_strikes = sum(strike_counts.values())
    dominant_strike = max(strike_counts, key=strike_counts.get) if total_strikes else None

    inferred_fps = fps
    if inferred_fps is None and len(frame_results) > 1:
        elapsed = frame_results[-1].timestamp_sec - frame_results[0].timestamp_sec
        inferred_fps = (len(frame_results) - 1) / elapsed if elapsed > 0 else None

    duration = None
    if frame_results:
        duration = frame_results[-1].timestamp_sec
        if inferred_fps:
            duration += 1.0 / inferred_fps

    return VideoFeatureVector(
        video_id=video_id,
        knee_left_mean=knee_left_mean,
        knee_left_std=knee_left_std,
        knee_left_min=knee_left_min,
        knee_left_max=knee_left_max,
        knee_right_mean=knee_right_mean,
        knee_right_std=knee_right_std,
        knee_right_min=knee_right_min,
        knee_right_max=knee_right_max,
        knee_asymmetry_mean=knee_asymmetry_mean,
        knee_asymmetry_std=knee_asymmetry_std,
        hip_left_mean=hip_left_mean,
        hip_left_std=hip_left_std,
        hip_left_min=hip_left_min,
        hip_left_max=hip_left_max,
        hip_right_mean=hip_right_mean,
        hip_right_std=hip_right_std,
        hip_right_min=hip_right_min,
        hip_right_max=hip_right_max,
        trunk_lean_mean=trunk_mean,
        trunk_lean_std=trunk_std,
        trunk_lean_min=trunk_min,
        trunk_lean_max=trunk_max,
        heel_ratio=_ratio(strike_counts[FootStrikeType.HEEL], total_strikes),
        midfoot_ratio=_ratio(strike_counts[FootStrikeType.MIDFOOT], total_strikes),
        forefoot_ratio=_ratio(strike_counts[FootStrikeType.FOREFOOT], total_strikes),
        unknown_ratio=_ratio(strike_counts[FootStrikeType.UNKNOWN], total_strikes),
        dominant_foot_strike=dominant_strike.value if dominant_strike else None,
        # TODO: derive cadence only after validated foot-event/running-cycle detection exists.
        cadence_spm=None,
        pose_confidence_mean=pose_confidence_mean,
        valid_pose_frame_ratio=_ratio(valid_pose_frames, total_frames),
        fps=inferred_fps,
        duration_sec=duration,
    )


def write_feature_csv(
    rows: list[VideoFeatureVector],
    output_path: str | Path,
) -> None:
    """Write feature vectors to CSV without adding labels or target columns."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(VideoFeatureVector(video_id="").as_dict().keys())

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
