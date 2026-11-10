"""Run pose detection and basic running-form analysis on a saved video file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .detector import MediaPipePoseDetector
from .drawing import draw_pose_overlay
from .encoding import convert_to_browser_mp4
from .metrics import (
    FrameMetrics,
    analyze_frame_metrics,
    print_summary,
    summarize_frame_metrics,
    write_metrics_csv,
)


from .temporal_analysis import analyze_temporal_movement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunForm video pose detection and running-form analysis.")
    parser.add_argument("video_path", type=Path, help="Path to the input video.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for annotated MP4 output.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional frame-level metrics CSV output.")
    parser.add_argument("--temporal-csv", type=Path, default=None, help="Optional window-level temporal features CSV output.")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Window duration for temporal analysis (seconds).")
    parser.add_argument("--step-sec", type=float, default=None, help="Step duration between windows (seconds).")
    parser.add_argument("--smooth-window", type=int, default=1, help="Optional smoothing window size (1 = no smoothing).")
    parser.add_argument("--no-display", action="store_true", help="Process without opening a display window.")
    parser.add_argument("--camera-view", choices=["side", "front", "rear", "unknown"], default="unknown", help="Optional camera viewpoint metadata.")
    parser.add_argument("--no-filter-implausible", action="store_true", help="Disable biomechanical plausibility filtering.")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--model-path", type=Path, default=None, help="Optional MediaPipe .task model path.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-landmark-visibility", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for quick tests.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = args.video_path

    if not video_path.exists():
        print(f"Error: video file does not exist: {video_path}")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video file: {video_path}")
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_delay_ms = max(1, int(round(1000.0 / fps)))

    writer: cv2.VideoWriter | None = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            print(f"Error: could not create output video: {args.output}")
            return 1

    print(f"Video: {video_path}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")

    detector = MediaPipePoseDetector(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
        model_path=args.model_path,
    )

    frame_records: list[FrameMetrics] = []
    start_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_index = len(frame_records)
            timestamp = frame_index / fps
            result = detector.detect(frame)
            frame_metrics = analyze_frame_metrics(
                frame_index=frame_index,
                timestamp=timestamp,
                result=result,
                min_visibility=args.min_landmark_visibility,
                filter_implausible=not args.no_filter_implausible,
            )
            frame_records.append(frame_metrics)

            elapsed = time.perf_counter() - start_time
            live_fps = frame_index / elapsed if elapsed > 0 else 0.0
            annotated = draw_pose_overlay(
                frame,
                result,
                metrics=frame_metrics.metrics,
                fps=live_fps,
                min_visibility=args.min_landmark_visibility,
            )

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("RunForm Pose - Video", annotated)
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    break

            if args.max_frames > 0 and len(frame_records) >= args.max_frames:
                break
    finally:
        detector.close()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if args.output is not None and args.output.exists():
        convert_to_browser_mp4(args.output, args.output)

    summary = summarize_frame_metrics(frame_records)
    print_summary(summary)

    temporal_result = analyze_temporal_movement(
        frame_records,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        smooth_window=args.smooth_window,
        fps=fps,
        camera_view=args.camera_view,
    )
    print(f"\nTemporal Movement Analysis ({len(temporal_result.windows)} windows):")
    tf = temporal_result.video_features
    if tf.knee_left_mean is not None and tf.knee_left_rom is not None:
        print(f"  Left knee ROM: {tf.knee_left_rom:.1f} deg (mean: {tf.knee_left_mean:.1f} deg, std: {tf.knee_left_std:.1f} deg)")
    if tf.knee_right_mean is not None and tf.knee_right_rom is not None:
        print(f"  Right knee ROM: {tf.knee_right_rom:.1f} deg (mean: {tf.knee_right_mean:.1f} deg, std: {tf.knee_right_std:.1f} deg)")
    if tf.trunk_lean_mean is not None and tf.trunk_lean_rom is not None:
        print(f"  Trunk lean ROM: {tf.trunk_lean_rom:.1f} deg (mean: {tf.trunk_lean_mean:.1f} deg, std: {tf.trunk_lean_std:.1f} deg)")

    if args.csv is not None:
        write_metrics_csv(frame_records, args.csv)
        print(f"\nSaved frame metrics CSV: {args.csv}")
    if args.temporal_csv is not None:
        temporal_result.write_windows_csv(args.temporal_csv)
        print(f"Saved temporal windows CSV: {args.temporal_csv}")
    if args.output is not None:
        print(f"Saved annotated video: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
