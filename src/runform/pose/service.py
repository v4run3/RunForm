"""Reusable services for MediaPipe running-form analysis in API and CLI code."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from .detector import MediaPipePoseDetector
from .drawing import draw_pose_overlay
from .encoding import convert_to_browser_mp4, ensure_supported_input_video
from .metrics import (
    FrameMetrics,
    RunningFormSummary,
    analyze_frame_metrics,
    summarize_frame_metrics,
    write_metrics_csv,
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
) -> tuple[FrameMetrics, np.ndarray]:
    """Analyze one BGR frame and return metrics plus annotated image."""
    detector = MediaPipePoseDetector(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    try:
        result = detector.detect(frame)
        metrics = analyze_frame_metrics(
            frame_index=frame_index,
            timestamp=time.time(),
            result=result,
            min_visibility=min_landmark_visibility,
        )
        annotated = draw_pose_overlay(
            frame,
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
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    min_landmark_visibility: float = 0.3,
    max_frames: int = 0,
) -> PoseVideoAnalysisResult:
    """Analyze a video with MediaPipe and optionally save overlay video/CSV."""
    input_path = Path(input_path)
    output = Path(output_path) if output_path is not None else None
    csv_output = Path(csv_path) if csv_path is not None else None

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

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_index = len(frame_records)
                result = detector.detect(frame)
                frame_metrics = analyze_frame_metrics(
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    result=result,
                    min_visibility=min_landmark_visibility,
                )
                frame_records.append(frame_metrics)

                if writer is not None:
                    annotated = draw_pose_overlay(
                        frame,
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

        summary = summarize_frame_metrics(frame_records)
        return PoseVideoAnalysisResult(
            frames=frame_records,
            summary=summary,
            output_path=output,
            csv_path=csv_output,
            fps=fps,
            width=width,
            height=height,
        )
    finally:
        if is_temp and usable_input.exists():
            try:
                usable_input.unlink()
            except OSError:
                pass
