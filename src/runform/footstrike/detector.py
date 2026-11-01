"""Foot strike classification and touchdown (Initial Contact) detection.

Implements rule-based gait event detection to identify the exact Initial Contact (IC)
frame and classify the landing pattern into Heel Strike (RFS), Midfoot Strike (MFS),
or Forefoot Strike (FFS) according to standard biomechanical definitions.
"""

from __future__ import annotations

from collections import deque
import math
import numpy as np

from runform.schema import FootStrikeType, Keypoint, PoseDetection


class FootStrikeDetector:
    """Detects foot contact events and classifies strike patterns across video frames."""

    def __init__(
        self,
        window_size: int = 15,
        min_confidence: float = 0.3,
        heel_angle_threshold_deg: float = 8.0,
        forefoot_angle_threshold_deg: float = -2.0,
    ) -> None:
        """Initialize foot strike detector.

        Args:
            window_size: Temporal sliding window (in frames) to track vertical velocity.
            min_confidence: Minimum keypoint confidence.
            heel_angle_threshold_deg: Foot/shank angle above which a strike is Heel Strike.
            forefoot_angle_threshold_deg: Foot/shank angle below which a strike is Forefoot Strike.
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.heel_threshold = heel_angle_threshold_deg
        self.forefoot_threshold = forefoot_angle_threshold_deg

        # Per-runner tracking buffers: track_id -> deque of (frame_idx, ankle_y, knee_y, hip_y, ankle_x, knee_x)
        self._history_left: dict[int, deque[tuple[int, float, float, float, float, float]]] = {}
        self._history_right: dict[int, deque[tuple[int, float, float, float, float, float]]] = {}

    def _get_coords(self, kps: dict[str, Keypoint], name: str) -> tuple[float, float] | None:
        kp = kps.get(name)
        if kp is not None and kp.confidence >= self.min_confidence:
            return (kp.x, kp.y)
        return None

    def classify_strike_from_shank_angle(
        self,
        knee: tuple[float, float],
        ankle: tuple[float, float],
        velocity_y: float,
    ) -> FootStrikeType:
        """Classify foot strike pattern from shank vector orientation at contact.

        In running biomechanics, the shank angle relative to the vertical line
        at touchdown strongly correlates with foot strike type (Cavanagh & Lafortune 1980):
        - Forward-angled shank (ankle ahead of knee) -> Rearfoot/Heel strike
        - Vertical shank (ankle directly under knee) -> Midfoot strike
        - Backward-angled shank (ankle behind knee) -> Forefoot strike

        Args:
            knee: (x, y) coordinates of the knee.
            ankle: (x, y) coordinates of the ankle.
            velocity_y: Instantaneous downward vertical velocity of the ankle.

        Returns:
            Classified FootStrikeType.
        """
        dx = ankle[0] - knee[0]
        dy = ankle[1] - knee[1]  # positive downwards

        if dy <= 0:  # Inverted or invalid geometry
            return FootStrikeType.UNKNOWN

        # Angle relative to vertical (degrees). Positive = ankle ahead of knee (dorsiflexed strike)
        shank_angle_deg = math.degrees(math.atan2(dx, dy))

        if shank_angle_deg > self.heel_threshold:
            return FootStrikeType.HEEL
        elif shank_angle_deg < self.forefoot_threshold:
            return FootStrikeType.FOREFOOT
        else:
            return FootStrikeType.MIDFOOT

    def process_frame(
        self,
        detection: PoseDetection,
        frame_idx: int = 0,
    ) -> dict[str, FootStrikeType]:
        """Analyze a frame detection and detect strike events for left and right legs.

        Args:
            detection: PoseDetection object with keypoints.
            frame_idx: Current sequential frame index.

        Returns:
            Dictionary mapping leg ('left', 'right') to detected FootStrikeType.
        """
        kps = detection.keypoints
        track_id = detection.track_id if detection.track_id is not None else 0

        if track_id not in self._history_left:
            self._history_left[track_id] = deque(maxlen=self.window_size)
            self._history_right[track_id] = deque(maxlen=self.window_size)

        lk = self._get_coords(kps, "left_knee")
        la = self._get_coords(kps, "left_ankle")
        lh = self._get_coords(kps, "left_hip")

        rk = self._get_coords(kps, "right_knee")
        ra = self._get_coords(kps, "right_ankle")
        rh = self._get_coords(kps, "right_hip")

        results: dict[str, FootStrikeType] = {}

        # Left Leg Analysis
        if lk and la and lh:
            hist_l = self._history_left[track_id]
            vy = 0.0
            if len(hist_l) >= 2:
                prev_y = hist_l[-1][1]
                vy = la[1] - prev_y  # positive = moving downwards

            hist_l.append((frame_idx, la[1], lk[1], lh[1], la[0], lk[0]))
            results["left"] = self.classify_strike_from_shank_angle(lk, la, vy)
        else:
            results["left"] = FootStrikeType.UNKNOWN

        # Right Leg Analysis
        if rk and ra and rh:
            hist_r = self._history_right[track_id]
            vy = 0.0
            if len(hist_r) >= 2:
                prev_y = hist_r[-1][1]
                vy = ra[1] - prev_y

            hist_r.append((frame_idx, ra[1], rk[1], rh[1], ra[0], rk[0]))
            results["right"] = self.classify_strike_from_shank_angle(rk, ra, vy)
        else:
            results["right"] = FootStrikeType.UNKNOWN

        return results

    def reset_tracking(self, track_id: int | None = None) -> None:
        """Reset history buffers for a specific runner or all runners."""
        if track_id is None:
            self._history_left.clear()
            self._history_right.clear()
        else:
            self._history_left.pop(track_id, None)
            self._history_right.pop(track_id, None)
