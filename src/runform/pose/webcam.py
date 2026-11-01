"""Run live webcam pose detection with a simple OpenCV window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .detector import MediaPipePoseDetector
from .drawing import draw_pose_overlay
from .metrics import analyze_frame_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunForm live webcam pose detection demo.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to open.")
    parser.add_argument("--width", type=int, default=0, help="Optional capture width.")
    parser.add_argument("--height", type=int, default=0, help="Optional capture height.")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--model-path", type=Path, default=None, help="Optional MediaPipe .task model path.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-landmark-visibility", type=float, default=0.3)
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirrored webcam display.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: could not open webcam with index {args.camera_index}.")
        return 1

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = MediaPipePoseDetector(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
        model_path=args.model_path,
    )

    fps = 0.0
    frame_index = 0
    previous_time = time.perf_counter()
    print("Starting RunForm webcam pose detection. Press Q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: could not read a webcam frame.")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            result = detector.detect(frame)
            frame_metrics = analyze_frame_metrics(
                frame_index=frame_index,
                timestamp=time.perf_counter(),
                result=result,
                min_visibility=args.min_landmark_visibility,
            )

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
            previous_time = now

            annotated = draw_pose_overlay(
                frame,
                result,
                metrics=frame_metrics.metrics,
                fps=fps,
                min_visibility=args.min_landmark_visibility,
            )
            cv2.imshow("RunForm Pose - Webcam", annotated)

            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
