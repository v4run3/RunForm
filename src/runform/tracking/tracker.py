"""Multi-runner tracking and temporal keypoint smoothing for RunForm.

Provides multi-object tracking with ID persistence across occlusions/out-of-frame
re-entry, and temporal 1€-filter keypoint smoothing to eliminate high-velocity jitter.
"""

from __future__ import annotations

import math
import numpy as np

from runform.schema import Keypoint, PoseDetection


class OneEuroFilter:
    """1€ Filter for adaptive, low-latency smoothing of noisy keypoint coordinates.

    Reference:
        Casiez, G., Roussel, N. and Vogel, F. (2012).
        1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Human-Computer Interaction.
        ACM CHI 2012.
    """

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.007,
        dcutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x

        # Filter derivative
        dx = (x - self.x_prev) * self.freq
        edx = self.dx_prev + self._alpha(self.dcutoff) * (dx - self.dx_prev)
        self.dx_prev = edx

        # Filter signal based on speed
        cutoff = self.mincutoff + self.beta * abs(edx)
        hatx = self.x_prev + self._alpha(cutoff) * (x - self.x_prev)
        self.x_prev = hatx
        return hatx

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0


def compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


class RunnerTrack:
    """Represents an active track for a single runner."""

    def __init__(self, track_id: int, initial_detection: PoseDetection, fps: float = 30.0) -> None:
        self.track_id = track_id
        self.bbox = initial_detection.bbox
        self.confidence = initial_detection.confidence
        self.last_seen_frame = 0
        self.missed_frames = 0
        self.fps = fps

        # Per-keypoint 1€ filters: kp_name -> (filter_x, filter_y)
        self._filters: dict[str, tuple[OneEuroFilter, OneEuroFilter]] = {}

    def smooth_keypoints(self, keypoints: dict[str, Keypoint]) -> dict[str, Keypoint]:
        """Apply 1€ filter to each keypoint coordinate."""
        smoothed: dict[str, Keypoint] = {}
        for name, kp in keypoints.items():
            if name not in self._filters:
                self._filters[name] = (
                    OneEuroFilter(freq=self.fps, mincutoff=1.2, beta=0.01),
                    OneEuroFilter(freq=self.fps, mincutoff=1.2, beta=0.01),
                )
            fx, fy = self._filters[name]
            sx = fx.filter(kp.x)
            sy = fy.filter(kp.y)
            smoothed[name] = Keypoint(x=sx, y=sy, confidence=kp.confidence, z=kp.z)
        return smoothed

    def update(self, detection: PoseDetection, frame_idx: int) -> PoseDetection:
        """Update track state with a new frame detection and return smoothed pose."""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.last_seen_frame = frame_idx
        self.missed_frames = 0

        smoothed_kps = self.smooth_keypoints(detection.keypoints)
        return PoseDetection(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.confidence,
            keypoints=smoothed_kps,
        )


class RunnerTracker:
    """Multi-runner tracker with out-of-frame buffer and temporal smoothing."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 30,
        fps: float = 30.0,
    ) -> None:
        """Initialize runner tracker.

        Args:
            iou_threshold: Minimum IoU match threshold for track association.
            max_missed_frames: Maximum frames to retain an out-of-frame runner for re-ID.
            fps: Video frame rate for filter tuning.
        """
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.fps = fps
        self._next_id = 1
        self._tracks: dict[int, RunnerTrack] = {}

    def update(self, detections: list[PoseDetection], frame_idx: int = 0) -> list[PoseDetection]:
        """Associate detections with existing tracks and apply smoothing.

        Args:
            detections: List of raw PoseDetections from detector.
            frame_idx: Current video frame index.

        Returns:
            List of PoseDetections with assigned track_id and smoothed keypoints.
        """
        active_tracks = list(self._tracks.values())
        tracked_results: list[PoseDetection] = []

        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self._tracks.keys())

        # Match detections to existing tracks via IoU
        if active_tracks and detections:
            cost_matrix = np.zeros((len(detections), len(active_tracks)), dtype=float)
            for d_idx, det in enumerate(detections):
                for t_idx, trk in enumerate(active_tracks):
                    cost_matrix[d_idx, t_idx] = compute_iou(det.bbox, trk.bbox)

            # Greedy matching for simplicity and speed
            matched_d = set()
            matched_t = set()
            while True:
                if len(matched_d) == len(detections) or len(matched_t) == len(active_tracks):
                    break
                max_val = -1.0
                best_d, best_t = -1, -1
                for d in range(len(detections)):
                    if d in matched_d:
                        continue
                    for t in range(len(active_tracks)):
                        if t in matched_t:
                            continue
                        if cost_matrix[d, t] > max_val:
                            max_val = cost_matrix[d, t]
                            best_d, best_t = d, t

                if max_val < self.iou_threshold or best_d == -1:
                    break

                matched_d.add(best_d)
                matched_t.add(best_t)
                trk = active_tracks[best_t]
                det = detections[best_d]

                updated_pose = trk.update(det, frame_idx)
                tracked_results.append(updated_pose)
                if trk.track_id in unmatched_tracks:
                    unmatched_tracks.remove(trk.track_id)

            unmatched_detections = [i for i in range(len(detections)) if i not in matched_d]

        # Handle unmatched detections (new runners entering frame)
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_id = self._next_id
            self._next_id += 1
            new_track = RunnerTrack(track_id=new_id, initial_detection=det, fps=self.fps)
            self._tracks[new_id] = new_track
            updated_pose = new_track.update(det, frame_idx)
            tracked_results.append(updated_pose)

        # Increment missed frames for unmatched tracks and prune old tracks
        for t_id in unmatched_tracks:
            self._tracks[t_id].missed_frames += 1
            if self._tracks[t_id].missed_frames > self.max_missed_frames:
                del self._tracks[t_id]

        return tracked_results

    def reset(self) -> None:
        """Reset all active tracks."""
        self._tracks.clear()
        self._next_id = 1
