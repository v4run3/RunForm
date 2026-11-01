"""Visualization utilities for drawing skeletons, angles, and risk badges on video frames."""

from __future__ import annotations

import cv2
import numpy as np

from runform.schema import (
    BiomechanicalMetrics,
    FootStrikeType,
    FormFeedback,
    InjuryRiskLevel,
    PoseDetection,
)

# Standard COCO skeleton connections: (joint_name_1, joint_name_2, color_bgr)
SKELETON_PAIRS = [
    ("nose", "left_eye", (255, 200, 0)),
    ("left_eye", "left_ear", (255, 200, 0)),
    ("nose", "right_eye", (0, 200, 255)),
    ("right_eye", "right_ear", (0, 200, 255)),
    ("left_shoulder", "right_shoulder", (200, 200, 200)),
    ("left_shoulder", "left_elbow", (0, 255, 0)),
    ("left_elbow", "left_wrist", (0, 255, 0)),
    ("right_shoulder", "right_elbow", (0, 165, 255)),
    ("right_elbow", "right_wrist", (0, 165, 255)),
    ("left_shoulder", "left_hip", (0, 255, 0)),
    ("right_shoulder", "right_hip", (0, 165, 255)),
    ("left_hip", "right_hip", (200, 200, 200)),
    ("left_hip", "left_knee", (0, 255, 0)),
    ("left_knee", "left_ankle", (0, 255, 0)),
    ("right_hip", "right_knee", (0, 165, 255)),
    ("right_knee", "right_ankle", (0, 165, 255)),
]


def draw_pose_overlay(
    frame: np.ndarray,
    detections: list[PoseDetection],
    metrics_map: dict[int, BiomechanicalMetrics] | None = None,
    strikes_map: dict[int, FootStrikeType] | None = None,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """Draw skeleton, joint angles, and track IDs onto an image frame.

    Args:
        frame: Original BGR image array (H, W, 3).
        detections: List of PoseDetection objects.
        metrics_map: Mapping of track_id to BiomechanicalMetrics.
        strikes_map: Mapping of track_id to primary FootStrikeType.
        min_confidence: Minimum confidence to draw keypoints.

    Returns:
        Annotated BGR frame array.
    """
    canvas = frame.copy()
    metrics_map = metrics_map or {}
    strikes_map = strikes_map or {}

    for det in detections:
        kps = det.keypoints
        track_id = det.track_id if det.track_id is not None else 0

        # Draw skeleton limbs
        for p1_name, p2_name, color in SKELETON_PAIRS:
            kp1 = kps.get(p1_name)
            kp2 = kps.get(p2_name)
            if kp1 and kp2 and kp1.confidence >= min_confidence and kp2.confidence >= min_confidence:
                pt1 = (int(kp1.x), int(kp1.y))
                pt2 = (int(kp2.x), int(kp2.y))
                cv2.line(canvas, pt1, pt2, color, 3, cv2.LINE_AA)

        # Draw joint keypoints
        for name, kp in kps.items():
            if kp.confidence >= min_confidence:
                pt = (int(kp.x), int(kp.y))
                cv2.circle(canvas, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Bounding Box & Track ID
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(
            canvas,
            f"Runner #{track_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Display Live Angle & Strike Info near knee/ankle
        m = metrics_map.get(track_id)
        if m and m.knee_flexion_left_deg is not None:
            lk = kps.get("left_knee")
            if lk and lk.confidence >= min_confidence:
                cv2.putText(
                    canvas,
                    f"{m.knee_flexion_left_deg:.0f} deg",
                    (int(lk.x) + 10, int(lk.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        strike = strikes_map.get(track_id)
        if strike and strike != FootStrikeType.UNKNOWN:
            la = kps.get("left_ankle")
            if la and la.confidence >= min_confidence:
                cv2.putText(
                    canvas,
                    str(strike.value).replace("_", " ").upper(),
                    (int(la.x) + 10, int(la.y) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (50, 220, 50),
                    2,
                    cv2.LINE_AA,
                )

    return canvas
