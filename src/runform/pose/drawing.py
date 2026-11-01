"""OpenCV drawing helpers for skeleton and RunForm prototype text."""

from __future__ import annotations

import cv2
import numpy as np

from .landmarks import IMPORTANT_LANDMARKS, SKELETON_CONNECTIONS, PoseLandmark, PoseResult
from .metrics import RunningFormMetrics, format_degrees


def _to_pixel(landmark: PoseLandmark, width: int, height: int) -> tuple[int, int]:
    x = int(round(landmark.x * width))
    y = int(round(landmark.y * height))
    return (x, y)


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.6,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _metrics_from_angles(angles: dict[str, float]) -> RunningFormMetrics:
    left_knee = angles.get("left_knee")
    right_knee = angles.get("right_knee")
    knee_asymmetry = None
    if left_knee is not None and right_knee is not None:
        knee_asymmetry = abs(left_knee - right_knee)
    return RunningFormMetrics(
        left_knee_angle=left_knee,
        right_knee_angle=right_knee,
        left_hip_angle=angles.get("left_hip"),
        right_hip_angle=angles.get("right_hip"),
        left_elbow_angle=angles.get("left_elbow"),
        right_elbow_angle=angles.get("right_elbow"),
        trunk_lean=angles.get("trunk_lean"),
        knee_asymmetry=knee_asymmetry,
    )


def draw_pose_overlay(
    frame: np.ndarray,
    result: PoseResult | None,
    metrics: RunningFormMetrics | None = None,
    fps: float | None = None,
    min_visibility: float = 0.3,
    angles: dict[str, float] | None = None,
) -> np.ndarray:
    """Draw skeleton, key landmarks, and simple RunForm metrics on a frame."""
    output = frame.copy()
    height, width = output.shape[:2]
    if metrics is None:
        metrics = _metrics_from_angles(angles or {})
    detected = result is not None and result.detected

    if detected and result is not None:
        for start_name, end_name in SKELETON_CONNECTIONS:
            start = result.get(start_name, min_visibility=min_visibility)
            end = result.get(end_name, min_visibility=min_visibility)
            if start is None or end is None:
                continue
            cv2.line(
                output,
                _to_pixel(start, width, height),
                _to_pixel(end, width, height),
                (0, 220, 255),
                2,
            )

        for name in IMPORTANT_LANDMARKS:
            landmark = result.get(name, min_visibility=min_visibility)
            if landmark is None:
                continue
            cv2.circle(output, _to_pixel(landmark, width, height), 4, (0, 255, 0), -1)
            cv2.circle(output, _to_pixel(landmark, width, height), 6, (0, 0, 0), 1)

        if result.bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in result.bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), (80, 200, 255), 1)

    panel_height = 242
    overlay = output.copy()
    cv2.rectangle(overlay, (8, 8), (350, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, output, 0.38, 0, output)

    _put_text(output, "RUNFORM", (22, 34), color=(80, 220, 255), scale=0.75)
    _put_text(output, f"Pose: {'YES' if detected else 'NO'}", (22, 62))
    if fps is not None:
        _put_text(output, f"FPS: {fps:.1f}", (205, 62))

    rows = [
        ("Left Knee", metrics.left_knee_angle),
        ("Right Knee", metrics.right_knee_angle),
        ("Knee Difference", metrics.knee_asymmetry),
        ("Left Hip", metrics.left_hip_angle),
        ("Right Hip", metrics.right_hip_angle),
        ("Trunk Lean", metrics.trunk_lean),
    ]
    y = 94
    for label, value in rows:
        _put_text(output, f"{label}: {format_degrees(value)}", (22, y))
        y += 24

    return output
