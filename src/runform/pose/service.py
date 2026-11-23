"""Reusable services for MediaPipe running-form analysis in API and CLI code."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import MediaPipePoseDetector
from .drawing import draw_pose_overlay
from .encoding import convert_to_browser_mp4, ensure_supported_input_video
from .enhancement import (
    compute_frame_quality,
    enhance_frame,
)
from .metrics import (
    FrameMetrics,
    RunningFormSummary,
    analyze_frame_metrics,
    summarize_frame_metrics,
    write_metrics_csv,
)
from .quality import AngleQualityConfig, ReliabilityConfig
from .temporal_analysis import (
    TemporalAnalysisResult,
    analyze_temporal_movement,
)


@dataclass
class PoseVideoAnalysisResult:
    """Paths and summary returned after analyzing a video."""

    frames: list[FrameMetrics]
    summary: RunningFormSummary
    output_path: Path | None
    csv_path: Path | None
    fps: float
    width: int
    height: int
    temporal_analysis: TemporalAnalysisResult | None = None
    temporal_csv_path: Path | None = None
    quality_info: dict[str, object] | None = None


def summary_to_dict(summary: RunningFormSummary) -> dict[str, object]:
    """Convert a summary dataclass into an API-friendly dictionary."""
    return {
        "frames_processed": summary.frames_processed,
        "pose_detected_frames": summary.pose_detected_frames,
        "detection_rate": round(summary.detection_rate, 4),
        "average_left_knee_angle": _round_or_none(summary.average_left_knee_angle),
        "average_right_knee_angle": _round_or_none(summary.average_right_knee_angle),
        "average_left_hip_angle": _round_or_none(summary.average_left_hip_angle),
        "average_right_hip_angle": _round_or_none(summary.average_right_hip_angle),
        "average_trunk_lean": _round_or_none(summary.average_trunk_lean),
        "average_knee_asymmetry": _round_or_none(summary.average_knee_asymmetry),
        "observations": summary.observations or [],
    }


def frame_metrics_to_dict(frame: FrameMetrics) -> dict[str, object]:
    """Convert one frame metrics object into an API-friendly dictionary."""
    return {
        "frame": frame.frame,
        "timestamp": frame.timestamp,
        "pose_detected": frame.pose_detected,
        "pose_confidence": frame.pose_confidence,
        **frame.metrics.as_dict(),
    }


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 1)


def analyze_pose_frame_image(
    frame: np.ndarray,
    frame_index: int = 0,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    min_landmark_visibility: float = 0.3,
    filter_implausible: bool = True,
    quality_config: AngleQualityConfig | None = None,
    enhancement_filter: str = "none",
    enhancement_strength: float = 1.0,
    enhancement_strengths: dict[str, float] | None = None,
) -> tuple[FrameMetrics, np.ndarray]:
    """Analyze one BGR frame and return metrics plus annotated image."""
    detector = MediaPipePoseDetector(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    try:
        proc_frame = frame
        if enhancement_filter and enhancement_filter != "none":
            proc_frame = enhance_frame(
                frame,
                filter_type=enhancement_filter,
                strength=enhancement_strength,
                strengths=enhancement_strengths,
            )

        result = detector.detect(proc_frame)
        metrics = analyze_frame_metrics(
            frame_index=frame_index,
            timestamp=time.time(),
            result=result,
            min_visibility=min_landmark_visibility,
            filter_implausible=filter_implausible,
            quality_config=quality_config,
        )
        annotated = draw_pose_overlay(
            proc_frame,
            result,
            metrics=metrics.metrics,
            min_visibility=min_landmark_visibility,
        )
        return metrics, annotated
    finally:
        detector.close()


def encode_frame_as_data_url(frame: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR frame as a browser-friendly JPEG data URL."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def analyze_pose_video_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    temporal_csv_path: str | Path | None = None,
    window_sec: float = 2.0,
    step_sec: float | None = None,
    smooth_window: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    min_landmark_visibility: float = 0.3,
    max_frames: int = 0,
    camera_view: str = "unknown",
    filter_implausible: bool = True,
    quality_config: AngleQualityConfig | None = None,
    reliability_config: ReliabilityConfig | None = None,
    enhancement_filter: str = "none",
    enhancement_strength: float = 1.0,
    enhancement_strengths: dict[str, float] | None = None,
) -> PoseVideoAnalysisResult:
    """Analyze a video with MediaPipe, compute temporal movement features, and optionally save overlay video/CSVs."""
    input_path = Path(input_path)
    output = Path(output_path) if output_path is not None else None
    csv_output = Path(csv_path) if csv_path is not None else None
    temporal_csv_output = Path(temporal_csv_path) if temporal_csv_path is not None else None

    usable_input, is_temp = ensure_supported_input_video(input_path)

    try:
        cap = cv2.VideoCapture(str(usable_input))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: cv2.VideoWriter | None = None
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not create output video: {output}")

        detector = MediaPipePoseDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        frame_records: list[FrameMetrics] = []
        initial_blur_scores: list[float] = []
        enhanced_blur_scores: list[float] = []
        enhancement_filters = (
            enhancement_filter
            if isinstance(enhancement_filter, (list, tuple))
            else str(enhancement_filter).split(",")
        )
        enhancement_filters = [value.strip() for value in enhancement_filters if value.strip()]
        has_enhancement = any(value not in ("none", "off") for value in enhancement_filters)

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_index = len(frame_records)

                # Track quality diagnostics sample (every 5 frames or first 20 frames)
                if frame_index % 5 == 0 or frame_index < 20:
                    q_init = compute_frame_quality(frame)
                    initial_blur_scores.append(q_init.blur_score)

                if has_enhancement:
                    proc_frame = enhance_frame(
                        frame,
                        filter_type=enhancement_filter,
                        strength=enhancement_strength,
                        strengths=enhancement_strengths,
                    )
                    if frame_index % 5 == 0 or frame_index < 20:
                        q_enh = compute_frame_quality(proc_frame)
                        enhanced_blur_scores.append(q_enh.blur_score)
                else:
                    proc_frame = frame

                result = detector.detect(proc_frame)
                frame_metrics = analyze_frame_metrics(
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    result=result,
                    min_visibility=min_landmark_visibility,
                    filter_implausible=filter_implausible,
                    quality_config=quality_config,
                )
                frame_records.append(frame_metrics)

                if writer is not None:
                    annotated = draw_pose_overlay(
                        proc_frame,
                        result,
                        metrics=frame_metrics.metrics,
                        min_visibility=min_landmark_visibility,
                    )
                    writer.write(annotated)

                if max_frames > 0 and len(frame_records) >= max_frames:
                    break
        finally:
            detector.close()
            cap.release()
            if writer is not None:
                writer.release()

        if output is not None and output.exists():
            convert_to_browser_mp4(output, output)

        if csv_output is not None:
            write_metrics_csv(frame_records, csv_output)

        temporal_result = analyze_temporal_movement(
            frame_records,
            window_sec=window_sec,
            step_sec=step_sec,
            smooth_window=smooth_window,
            fps=fps,
            camera_view=camera_view,
            quality_config=quality_config,
            reliability_config=reliability_config,
        )

        if temporal_csv_output is not None:
            temporal_result.write_windows_csv(temporal_csv_output)

        summary = summarize_frame_metrics(frame_records)

        avg_init_blur = float(np.mean(initial_blur_scores)) if initial_blur_scores else 0.0
        avg_enh_blur = float(np.mean(enhanced_blur_scores)) if enhanced_blur_scores else avg_init_blur
        sharpness_gain_pct = 0.0
        if avg_init_blur > 0:
            sharpness_gain_pct = round(((avg_enh_blur - avg_init_blur) / avg_init_blur) * 100.0, 1)

        quality_info = {
            "enhancement_filter": enhancement_filter,
            "enhancement_strength": enhancement_strength,
            "enhancement_strengths": enhancement_strengths or {},
            "applied": has_enhancement,
            "initial_blur_score": round(avg_init_blur, 2),
            "enhanced_blur_score": round(avg_enh_blur, 2),
            "sharpness_improvement_pct": sharpness_gain_pct,
        }

        return PoseVideoAnalysisResult(
            frames=frame_records,
            summary=summary,
            output_path=output,
            csv_path=csv_output,
            fps=fps,
            width=width,
            height=height,
            temporal_analysis=temporal_result,
            temporal_csv_path=temporal_csv_output,
            quality_info=quality_info,
        )
    finally:
        if is_temp and usable_input.exists():
            try:
                usable_input.unlink()
            except OSError:
                pass
