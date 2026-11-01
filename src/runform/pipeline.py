"""End-to-end RunForm pipeline orchestrator.

Integrates pose detection, multi-runner tracking, biomechanical angle computation,
foot strike classification, and injury risk assessment into a unified API.
"""

from __future__ import annotations

from typing import Any, Callable
import cv2
import numpy as np

from runform.angles.calculator import BiomechanicsCalculator
from runform.config import load_config
from runform.detection.detector import BasePoseDetector, YOLOv8PoseDetector
from runform.feedback.advisor import RiskAdvisor
from runform.features.extractor import extract_video_features
from runform.footstrike.detector import FootStrikeDetector
from runform.schema import (
    BiomechanicalMetrics,
    FootStrikeType,
    FrameAnalysisResult,
    PoseDetection,
    VideoAnalysisReport,
)
from runform.pose.encoding import convert_to_browser_mp4
from runform.tracking.tracker import RunnerTracker
from runform.utils.viz import draw_pose_overlay


class RunFormPipeline:
    """Orchestrates detection, tracking, angle and feedback modules."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        detector: BasePoseDetector | None = None,
    ) -> None:
        """Initialize RunForm pipeline.

        Args:
            config: Optional config dict; defaults to ``configs/default.yaml``.
            detector: Optional custom BasePoseDetector instance (e.g. MockPoseDetector for testing).
        """
        self.config = config if config is not None else load_config()

        det_cfg = self.config.get("detection", {})
        if detector is not None:
            self.detector = detector
        else:
            self.detector = YOLOv8PoseDetector(
                model_name=det_cfg.get("model", "yolov8n-pose.pt"),
                conf_threshold=float(det_cfg.get("conf", 0.5)),
                imgsz=int(det_cfg.get("imgsz", 640)),
            )

        track_cfg = self.config.get("tracking", {})
        self.tracker = RunnerTracker(
            max_missed_frames=int(track_cfg.get("reid_buffer_frames", 30)),
        )

        self.calculator = BiomechanicsCalculator()
        self.strike_detector = FootStrikeDetector()
        self.advisor = RiskAdvisor()

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        fps: float = 30.0,
        draw_overlay: bool = False,
    ) -> tuple[FrameAnalysisResult, np.ndarray | None]:
        """Analyze a single video frame.

        Args:
            frame: Input BGR image (H, W, 3).
            frame_idx: Current frame index.
            fps: Video frame rate used for timestamp calculation.
            draw_overlay: Whether to return an annotated image with pose/angle overlay.

        Returns:
            Tuple of (FrameAnalysisResult, annotated_frame_or_None).
        """
        raw_detections = self.detector.detect(frame)
        tracked_detections = self.tracker.update(raw_detections, frame_idx=frame_idx)

        strikes_map: dict[int, FootStrikeType] = {}
        metrics_map: dict[int, BiomechanicalMetrics] = {}

        for det in tracked_detections:
            t_id = det.track_id if det.track_id is not None else 0

            metrics = self.calculator.calculate_angles(det)
            metrics_map[t_id] = metrics

            strike_res = self.strike_detector.process_frame(det, frame_idx=frame_idx)
            primary_strike = strike_res.get("left") or strike_res.get("right") or FootStrikeType.UNKNOWN
            strikes_map[t_id] = primary_strike

        result = FrameAnalysisResult(
            frame_index=frame_idx,
            timestamp_sec=frame_idx / fps,
            detections=tracked_detections,
            foot_strike=strikes_map,
            metrics=metrics_map,
        )

        annotated_frame = None
        if draw_overlay:
            annotated_frame = draw_pose_overlay(
                frame=frame,
                detections=tracked_detections,
                metrics_map=metrics_map,
                strikes_map=strikes_map,
            )

        return result, annotated_frame

    def analyze_video(
        self,
        video_path: str,
        output_annotated_path: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> VideoAnalysisReport:
        """Process an entire video file and generate a comprehensive gait analysis report.

        Args:
            video_path: Path to input video file.
            output_annotated_path: Optional path to save annotated output video.
            progress_callback: Optional callback receiving float progress (0.0 to 1.0).

        Returns:
            VideoAnalysisReport containing aggregated kinematics and risk feedback.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file at '{video_path}'")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: cv2.VideoWriter | None = None
        if output_annotated_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_annotated_path, fourcc, fps, (width, height))

        frame_results: list[FrameAnalysisResult] = []
        knee_left_vals: list[float] = []
        knee_right_vals: list[float] = []
        trunk_lean_vals: list[float] = []
        strike_counts: dict[FootStrikeType, int] = {
            FootStrikeType.HEEL: 0,
            FootStrikeType.MIDFOOT: 0,
            FootStrikeType.FOREFOOT: 0,
        }

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res, annotated = self.analyze_frame(
                frame,
                frame_idx=frame_idx,
                fps=fps,
                draw_overlay=(writer is not None),
            )
            frame_results.append(res)

            if writer is not None and annotated is not None:
                writer.write(annotated)

            for metrics in res.metrics.values():
                if metrics.knee_flexion_left_deg is not None:
                    knee_left_vals.append(metrics.knee_flexion_left_deg)
                if metrics.knee_flexion_right_deg is not None:
                    knee_right_vals.append(metrics.knee_flexion_right_deg)
                if metrics.trunk_lean_deg is not None:
                    trunk_lean_vals.append(metrics.trunk_lean_deg)

            for strike in res.foot_strike.values():
                if strike in strike_counts:
                    strike_counts[strike] += 1

            frame_idx += 1
            if progress_callback:
                progress_callback(min(1.0, frame_idx / total_frames))

        cap.release()
        if writer is not None:
            writer.release()

        if output_annotated_path is not None and Path(output_annotated_path).exists():
            convert_to_browser_mp4(output_annotated_path, output_annotated_path)

        avg_knee_l = float(np.mean(knee_left_vals)) if knee_left_vals else None
        avg_knee_r = float(np.mean(knee_right_vals)) if knee_right_vals else None
        avg_trunk = float(np.mean(trunk_lean_vals)) if trunk_lean_vals else None
        primary_strike = (
            max(strike_counts, key=strike_counts.get)
            if any(strike_counts.values())
            else FootStrikeType.UNKNOWN
        )

        avg_metrics = BiomechanicalMetrics(
            knee_flexion_left_deg=avg_knee_l,
            knee_flexion_right_deg=avg_knee_r,
            trunk_lean_deg=avg_trunk,
        )
        feedback = self.advisor.evaluate(avg_metrics, strike_type=primary_strike)

        report = VideoAnalysisReport(
            video_path=video_path,
            total_frames=frame_idx,
            fps=fps,
            duration_sec=frame_idx / fps,
            primary_strike_type=primary_strike,
            avg_metrics=avg_metrics,
            feedback=feedback,
            frame_results=frame_results,
        )
        report.metadata["ml_features"] = extract_video_features(
            frame_results=frame_results,
            video_id=video_path,
            fps=fps,
        ).as_dict()
        return report
