"""MediaPipe Pose detector wrapper for RunForm's simple prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import numpy as np

from .landmarks import IMPORTANT_LANDMARKS, POSE_LANDMARK_NAMES, PoseLandmark, PoseResult

DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DEFAULT_TASK_MODEL_PATH = Path("data/models/pose_landmarker_lite.task")


class MediaPipePoseDetector:
    """CPU-friendly single-person pose detector based on MediaPipe Pose."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        static_image_mode: bool = False,
        model_path: str | Path | None = None,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.static_image_mode = static_image_mode
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_TASK_MODEL_PATH
        self._pose: Any = None
        self._mp: Any = None
        self._backend = ""

    def _ensure_task_model(self) -> Path:
        if self.model_path.exists():
            return self.model_path

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe pose model to {self.model_path} ...")
        try:
            urlretrieve(DEFAULT_TASK_MODEL_URL, self.model_path)
        except (OSError, URLError) as exc:
            raise RuntimeError(
                "Could not download the MediaPipe pose model. "
                f"Download it manually from {DEFAULT_TASK_MODEL_URL} and save it as {self.model_path}."
            ) from exc
        return self.model_path

    def _ensure_model(self) -> None:
        if self._pose is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Install it with: python -m pip install mediapipe"
            ) from exc

        self._mp = mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._backend = "solutions"
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            return

        if not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
            raise RuntimeError("This MediaPipe install does not provide Pose solutions or Tasks vision APIs.")

        self._backend = "tasks"
        model_path = self._ensure_task_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _build_result(
        self,
        raw_landmarks: list[Any],
        image_width: int,
        image_height: int,
    ) -> PoseResult | None:
        landmarks: dict[str, PoseLandmark] = {}
        xs: list[float] = []
        ys: list[float] = []

        for index, raw_landmark in enumerate(raw_landmarks):
            if index >= len(POSE_LANDMARK_NAMES):
                continue
            name = POSE_LANDMARK_NAMES[index]
            visibility = float(getattr(raw_landmark, "visibility", 0.0))
            presence = float(getattr(raw_landmark, "presence", visibility))
            confidence = max(visibility, presence)
            landmark = PoseLandmark(
                name=name,
                x=float(raw_landmark.x),
                y=float(raw_landmark.y),
                z=float(getattr(raw_landmark, "z", 0.0)),
                visibility=confidence,
            )
            landmarks[name] = landmark
            if confidence >= 0.2:
                xs.append(float(raw_landmark.x))
                ys.append(float(raw_landmark.y))

        if not landmarks:
            return None

        confidence_values = [
            landmarks[name].visibility for name in IMPORTANT_LANDMARKS if name in landmarks
        ]
        confidence = float(np.mean(confidence_values)) if confidence_values else 0.0

        bbox = None
        if xs and ys:
            x1 = max(0.0, min(xs)) * image_width
            y1 = max(0.0, min(ys)) * image_height
            x2 = min(1.0, max(xs)) * image_width
            y2 = min(1.0, max(ys)) * image_height
            bbox = (x1, y1, x2, y2)

        return PoseResult(
            landmarks=landmarks,
            confidence=confidence,
            image_width=image_width,
            image_height=image_height,
            bbox=bbox,
        )

    def detect(self, frame_bgr: np.ndarray) -> PoseResult | None:
        """Detect one person's pose in a BGR OpenCV frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        self._ensure_model()
        image_height, image_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        if self._backend == "solutions":
            results = self._pose.process(frame_rgb)
            if not results.pose_landmarks:
                return None
            return self._build_result(results.pose_landmarks.landmark, image_width, image_height)

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._pose.detect(image)
        if not results.pose_landmarks:
            return None
        return self._build_result(results.pose_landmarks[0], image_width, image_height)

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self) -> "MediaPipePoseDetector":
        self._ensure_model()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
