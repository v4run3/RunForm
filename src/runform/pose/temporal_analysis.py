"""Temporal movement-feature extraction from frame-level pose measurements.

This module aggregates sequences of pose measurements (from MediaPipe Pose) into
temporal movement statistics across full recordings and configurable time windows.
These extracted features are designed for future machine-learning training and
biomechanical evaluation without rule-based injury classification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence
import csv
import json
import math
import statistics

from .metrics import FrameMetrics, RunningFormMetrics, calculate_knee_asymmetry

MONITORED_METRIC_FIELDS = (
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "trunk_lean",
)


@dataclass
class TemporalFeatures:
    """Statistical summary of running form movement over a sequence of frames."""

    # Pose quality & coverage
    total_frames: int = 0
    valid_pose_frames: int = 0
    valid_pose_frame_ratio: float = 0.0
    pose_confidence_mean: float | None = None
    missing_measurements: int = 0
    is_reliable: bool = False

    # A. Knee measurements
    knee_left_mean: float | None = None
    knee_left_std: float | None = None
    knee_left_min: float | None = None
    knee_left_max: float | None = None
    knee_left_rom: float | None = None

    knee_right_mean: float | None = None
    knee_right_std: float | None = None
    knee_right_min: float | None = None
    knee_right_max: float | None = None
    knee_right_rom: float | None = None

    knee_diff_mean: float | None = None
    knee_diff_std: float | None = None
    knee_asymmetry_mean: float | None = None
    knee_asymmetry_std: float | None = None

    # B. Hip measurements
    hip_left_mean: float | None = None
    hip_left_std: float | None = None
    hip_left_min: float | None = None
    hip_left_max: float | None = None
    hip_left_rom: float | None = None

    hip_right_mean: float | None = None
    hip_right_std: float | None = None
    hip_right_min: float | None = None
    hip_right_max: float | None = None
    hip_right_rom: float | None = None

    # C. Trunk measurements
    trunk_lean_mean: float | None = None
    trunk_lean_std: float | None = None
    trunk_lean_min: float | None = None
    trunk_lean_max: float | None = None
    trunk_lean_rom: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert features to a dictionary with rounded floats for clean JSON."""
        data = asdict(self)
        for key, val in data.items():
            if isinstance(val, float):
                data[key] = round(val, 4)
        return data


@dataclass
class WindowTemporalResult:
    """Temporal feature results for a specific time window."""

    window_index: int
    start_time: float
    end_time: float
    total_frames: int
    valid_frames: int
    is_reliable: bool
    features: TemporalFeatures

    def as_dict(self) -> dict[str, Any]:
        """Dictionary representation suitable for JSON serialization."""
        return {
            "window_index": self.window_index,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "total_frames": self.total_frames,
            "valid_frames": self.valid_frames,
            "is_reliable": self.is_reliable,
            "features": self.features.as_dict(),
        }

    def as_csv_row(self) -> dict[str, Any]:
        """Flattened row dictionary for CSV export."""
        row: dict[str, Any] = {
            "window_index": self.window_index,
            "start_time": f"{self.start_time:.3f}",
            "end_time": f"{self.end_time:.3f}",
            "total_frames": self.total_frames,
            "valid_frames": self.valid_frames,
            "is_reliable": self.is_reliable,
        }
        for k, v in self.features.as_dict().items():
            if k in ("total_frames", "valid_pose_frames", "is_reliable"):
                continue
            if v is None:
                row[k] = ""
            elif isinstance(v, float):
                row[k] = f"{v:.4f}"
            else:
                row[k] = v
        return row


@dataclass
class TemporalAnalysisResult:
    """Overall result of temporal analysis including whole-video and window features."""

    video_features: TemporalFeatures
    windows: list[WindowTemporalResult] = field(default_factory=list)
    window_duration_sec: float = 2.0
    step_duration_sec: float = 2.0
    total_frames: int = 0
    fps: float | None = None
    duration_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary of temporal analysis."""
        return {
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2) if self.fps is not None else None,
            "duration_sec": round(self.duration_sec, 2) if self.duration_sec is not None else None,
            "window_duration_sec": self.window_duration_sec,
            "step_duration_sec": self.step_duration_sec,
            "total_windows": len(self.windows),
            "reliable_windows": sum(1 for w in self.windows if w.is_reliable),
            "video_features": self.video_features.as_dict(),
            "windows": [w.as_dict() for w in self.windows],
        }

    def to_video_csv_row(self, video_id: str = "") -> dict[str, Any]:
        """Single-row feature record representing the entire video."""
        row: dict[str, Any] = {
            "video_id": video_id,
            "fps": f"{self.fps:.2f}" if self.fps is not None else "",
            "duration_sec": f"{self.duration_sec:.2f}" if self.duration_sec is not None else "",
        }
        for k, v in self.video_features.as_dict().items():
            if v is None:
                row[k] = ""
            elif isinstance(v, float):
                row[k] = f"{v:.4f}"
            else:
                row[k] = v
        return row

    def to_window_csv_rows(self, video_id: str = "") -> list[dict[str, Any]]:
        """Flattened window records for multi-row CSV export."""
        rows: list[dict[str, Any]] = []
        for w in self.windows:
            w_row = w.as_csv_row()
            if video_id:
                w_row = {"video_id": video_id, **w_row}
            rows.append(w_row)
        return rows

    def write_windows_csv(self, output_path: str | Path, video_id: str = "") -> None:
        """Write window-level feature records to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_window_csv_rows(video_id=video_id)
        if not rows:
            # Write header with dummy window
            dummy = WindowTemporalResult(0, 0.0, 0.0, 0, 0, False, TemporalFeatures())
            dummy_row = dummy.as_csv_row()
            if video_id:
                dummy_row = {"video_id": video_id, **dummy_row}
            fieldnames = list(dummy_row.keys())
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_summary_json(self, output_path: str | Path) -> None:
        """Save JSON summary to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _clean_finite(values: Sequence[float | None]) -> list[float]:
    """Filter out None, NaN, and Inf values."""
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _compute_stats(
    values: Sequence[float | None],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Calculate mean, std, min, max, and range of motion (ROM) for a numeric series."""
    cleaned = _clean_finite(values)
    if not cleaned:
        return None, None, None, None, None
    mean_val = float(statistics.fmean(cleaned))
    std_val = float(statistics.pstdev(cleaned)) if len(cleaned) > 1 else 0.0
    min_val = float(min(cleaned))
    max_val = float(max(cleaned))
    rom_val = max_val - min_val
    return mean_val, std_val, min_val, max_val, rom_val


def calculate_temporal_features(
    frames: Sequence[FrameMetrics],
    min_valid_frames: int = 3,
    min_valid_ratio: float = 0.4,
) -> TemporalFeatures:
    """Calculate temporal movement features across a sequence of frame measurements."""
    total_frames = len(frames)
    if total_frames == 0:
        return TemporalFeatures(
            total_frames=0,
            valid_pose_frames=0,
            valid_pose_frame_ratio=0.0,
            pose_confidence_mean=None,
            missing_measurements=0,
            is_reliable=False,
        )

    valid_pose_frames = sum(1 for f in frames if f.pose_detected)
    valid_pose_frame_ratio = valid_pose_frames / total_frames

    confidences = _clean_finite([f.pose_confidence for f in frames if f.pose_detected])
    pose_confidence_mean = float(statistics.fmean(confidences)) if confidences else None

    # Count missing measurements across the 5 monitored joint channels
    missing_count = 0
    for f in frames:
        if not f.pose_detected:
            missing_count += len(MONITORED_METRIC_FIELDS)
        else:
            for field_name in MONITORED_METRIC_FIELDS:
                val = getattr(f.metrics, field_name, None)
                if val is None or not math.isfinite(float(val)):
                    missing_count += 1

    is_reliable = (valid_pose_frames >= min_valid_frames) and (
        valid_pose_frame_ratio >= min_valid_ratio
    )

    # Knee measurements
    left_knee_vals = [f.metrics.left_knee_angle for f in frames if f.pose_detected]
    lk_mean, lk_std, lk_min, lk_max, lk_rom = _compute_stats(left_knee_vals)

    right_knee_vals = [f.metrics.right_knee_angle for f in frames if f.pose_detected]
    rk_mean, rk_std, rk_min, rk_max, rk_rom = _compute_stats(right_knee_vals)

    # Bilateral knee difference & asymmetry
    # Pairwise difference is calculated for frames where both knees are valid
    knee_diffs: list[float] = []
    knee_asymmetries: list[float] = []
    for f in frames:
        if not f.pose_detected:
            continue
        lk = f.metrics.left_knee_angle
        rk = f.metrics.right_knee_angle
        if lk is not None and rk is not None and math.isfinite(lk) and math.isfinite(rk):
            diff = lk - rk
            knee_diffs.append(diff)
            knee_asymmetries.append(abs(diff))

    diff_mean, diff_std, _, _, _ = _compute_stats(knee_diffs)
    asym_mean, asym_std, _, _, _ = _compute_stats(knee_asymmetries)

    # Hip measurements
    left_hip_vals = [f.metrics.left_hip_angle for f in frames if f.pose_detected]
    lh_mean, lh_std, lh_min, lh_max, lh_rom = _compute_stats(left_hip_vals)

    right_hip_vals = [f.metrics.right_hip_angle for f in frames if f.pose_detected]
    rh_mean, rh_std, rh_min, rh_max, rh_rom = _compute_stats(right_hip_vals)

    # Trunk measurements
    trunk_vals = [f.metrics.trunk_lean for f in frames if f.pose_detected]
    tl_mean, tl_std, tl_min, tl_max, tl_rom = _compute_stats(trunk_vals)

    return TemporalFeatures(
        total_frames=total_frames,
        valid_pose_frames=valid_pose_frames,
        valid_pose_frame_ratio=valid_pose_frame_ratio,
        pose_confidence_mean=pose_confidence_mean,
        missing_measurements=missing_count,
        is_reliable=is_reliable,
        # Knees
        knee_left_mean=lk_mean,
        knee_left_std=lk_std,
        knee_left_min=lk_min,
        knee_left_max=lk_max,
        knee_left_rom=lk_rom,
        knee_right_mean=rk_mean,
        knee_right_std=rk_std,
        knee_right_min=rk_min,
        knee_right_max=rk_max,
        knee_right_rom=rk_rom,
        knee_diff_mean=diff_mean,
        knee_diff_std=diff_std,
        knee_asymmetry_mean=asym_mean,
        knee_asymmetry_std=asym_std,
        # Hips
        hip_left_mean=lh_mean,
        hip_left_std=lh_std,
        hip_left_min=lh_min,
        hip_left_max=lh_max,
        hip_left_rom=lh_rom,
        hip_right_mean=rh_mean,
        hip_right_std=rh_std,
        hip_right_min=rh_min,
        hip_right_max=rh_max,
        hip_right_rom=rh_rom,
        # Trunk
        trunk_lean_mean=tl_mean,
        trunk_lean_std=tl_std,
        trunk_lean_min=tl_min,
        trunk_lean_max=tl_max,
        trunk_lean_rom=tl_rom,
    )


def smooth_frame_metrics(
    frames: Sequence[FrameMetrics],
    window_size: int = 1,
) -> list[FrameMetrics]:
    """Apply an optional moving average over continuous angle series.

    When window_size <= 1, returns the original frame records unmodified.
    When window_size > 1, computes a centered moving average over finite values
    without altering original frame structures or smoothing away discrete events.
    """
    if window_size <= 1 or len(frames) <= 1:
        return list(frames)

    half_win = window_size // 2
    n = len(frames)

    def _smooth_series(raw_vals: list[float | None]) -> list[float | None]:
        smoothed: list[float | None] = []
        for i in range(n):
            if raw_vals[i] is None or not math.isfinite(raw_vals[i]):
                smoothed.append(None)
                continue
            start_idx = max(0, i - half_win)
            end_idx = min(n, i + half_win + 1)
            window_slice = [v for v in raw_vals[start_idx:end_idx] if v is not None and math.isfinite(v)]
            if window_slice:
                smoothed.append(float(statistics.fmean(window_slice)))
            else:
                smoothed.append(None)
        return smoothed

    lk_smooth = _smooth_series([f.metrics.left_knee_angle for f in frames])
    rk_smooth = _smooth_series([f.metrics.right_knee_angle for f in frames])
    lh_smooth = _smooth_series([f.metrics.left_hip_angle for f in frames])
    rh_smooth = _smooth_series([f.metrics.right_hip_angle for f in frames])
    tl_smooth = _smooth_series([f.metrics.trunk_lean for f in frames])
    le_smooth = _smooth_series([f.metrics.left_elbow_angle for f in frames])
    re_smooth = _smooth_series([f.metrics.right_elbow_angle for f in frames])

    smoothed_frames: list[FrameMetrics] = []
    for i, f in enumerate(frames):
        lk = lk_smooth[i]
        rk = rk_smooth[i]
        asym = calculate_knee_asymmetry(lk, rk)
        new_metrics = RunningFormMetrics(
            left_knee_angle=lk,
            right_knee_angle=rk,
            left_hip_angle=lh_smooth[i],
            right_hip_angle=rh_smooth[i],
            left_elbow_angle=le_smooth[i],
            right_elbow_angle=re_smooth[i],
            trunk_lean=tl_smooth[i],
            knee_asymmetry=asym,
        )
        smoothed_frames.append(
            FrameMetrics(
                frame=f.frame,
                timestamp=f.timestamp,
                pose_detected=f.pose_detected,
                pose_confidence=f.pose_confidence,
                metrics=new_metrics,
            )
        )
    return smoothed_frames


def analyze_temporal_movement(
    frames: Sequence[FrameMetrics],
    window_sec: float = 2.0,
    step_sec: float | None = None,
    min_valid_frames: int = 3,
    min_valid_ratio: float = 0.4,
    smooth_window: int = 1,
    fps: float | None = None,
) -> TemporalAnalysisResult:
    """Analyze a sequence of frame measurements using full-video and window-based temporal analysis.

    Parameters
    ----------
    frames : Sequence[FrameMetrics]
        Frame-level measurements from MediaPipe pose detector.
    window_sec : float, default 2.0
        Duration of each temporal analysis window in seconds (e.g. 1.0, 2.0, 3.0).
    step_sec : float | None, default None
        Step duration between window starts. Defaults to window_sec (non-overlapping).
    min_valid_frames : int, default 3
        Minimum valid pose frames required for a window or sequence to be considered reliable.
    min_valid_ratio : float, default 0.4
        Minimum fraction of valid pose frames required for reliability.
    smooth_window : int, default 1
        Optional rolling average window size (1 = raw, no smoothing).
    fps : float | None, default None
        Video frame rate. Inferred from timestamps if omitted.
    """
    total_frames = len(frames)
    win_sec = max(0.1, float(window_sec))
    step = float(step_sec) if step_sec is not None and step_sec > 0 else win_sec

    if total_frames == 0:
        empty_features = TemporalFeatures(
            total_frames=0,
            valid_pose_frames=0,
            valid_pose_frame_ratio=0.0,
            pose_confidence_mean=None,
            missing_measurements=0,
            is_reliable=False,
        )
        return TemporalAnalysisResult(
            video_features=empty_features,
            windows=[],
            window_duration_sec=win_sec,
            step_duration_sec=step,
            total_frames=0,
            fps=fps,
            duration_sec=0.0,
        )

    # Optional smoothing
    processed_frames = smooth_frame_metrics(frames, window_size=smooth_window)

    # Resolve timestamps and duration
    timestamps = [float(f.timestamp) for f in processed_frames]
    min_t = timestamps[0]
    max_t = timestamps[-1]

    # In case timestamps were non-monotonic or zero (e.g. synthetic test frames)
    if max_t <= min_t and total_frames > 1:
        eff_fps = fps if fps and fps > 0 else 30.0
        timestamps = [f.frame / eff_fps for f in processed_frames]
        min_t = timestamps[0]
        max_t = timestamps[-1]
    else:
        if fps is None and total_frames > 1 and max_t > min_t:
            fps = (total_frames - 1) / (max_t - min_t)

    duration_sec = max(0.0, max_t - min_t)

    # Calculate overall video-level features
    video_features = calculate_temporal_features(
        processed_frames,
        min_valid_frames=min_valid_frames,
        min_valid_ratio=min_valid_ratio,
    )

    # Generate temporal windows
    windows: list[WindowTemporalResult] = []
    window_idx = 0
    curr_start = min_t

    # Loop through time windows
    while curr_start <= max_t:
        curr_end = curr_start + win_sec
        # Find frames in window [curr_start, curr_end)
        # For the final window edge, allow inclusive <= max_t
        if curr_end >= max_t:
            window_frames = [
                f for f, t in zip(processed_frames, timestamps) if curr_start <= t <= max_t
            ]
        else:
            window_frames = [
                f for f, t in zip(processed_frames, timestamps) if curr_start <= t < curr_end
            ]

        win_features = calculate_temporal_features(
            window_frames,
            min_valid_frames=min_valid_frames,
            min_valid_ratio=min_valid_ratio,
        )

        valid_in_win = sum(1 for f in window_frames if f.pose_detected)

        windows.append(
            WindowTemporalResult(
                window_index=window_idx,
                start_time=curr_start - min_t,
                end_time=min(max_t, curr_end) - min_t,
                total_frames=len(window_frames),
                valid_frames=valid_in_win,
                is_reliable=win_features.is_reliable,
                features=win_features,
            )
        )

        window_idx += 1
        curr_start += step
        if curr_start > max_t:
            break

    # If duration is very short and no window was generated, guarantee at least 1 window
    if not windows:
        win_features = calculate_temporal_features(
            processed_frames,
            min_valid_frames=min_valid_frames,
            min_valid_ratio=min_valid_ratio,
        )
        windows.append(
            WindowTemporalResult(
                window_index=0,
                start_time=0.0,
                end_time=duration_sec,
                total_frames=total_frames,
                valid_frames=sum(1 for f in processed_frames if f.pose_detected),
                is_reliable=win_features.is_reliable,
                features=win_features,
            )
        )

    return TemporalAnalysisResult(
        video_features=video_features,
        windows=windows,
        window_duration_sec=win_sec,
        step_duration_sec=step,
        total_frames=total_frames,
        fps=fps,
        duration_sec=duration_sec,
    )
