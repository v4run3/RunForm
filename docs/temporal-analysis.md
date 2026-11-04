# Temporal Movement Analysis (Milestone 1)

## Overview

Milestone 1 implements **Temporal Movement Analysis** for the RunForm running form assessment pipeline. 

The primary objective of this module is to extract time-series kinematics and statistical movement features from consecutive 2D body pose landmarks (MediaPipe Pose). These extracted features serve as objective, multi-frame inputs for downstream machine learning model training without relying on subjective or hard-coded injury thresholds.

---

## 1. What is Temporal Movement Analysis?

Running is an inherently cyclical and dynamic movement. A single snapshot or frame (e.g., foot strike, midstance, or toe-off) only captures instantaneous joint positions. Single-frame snapshots suffer from:
- **Phase ambiguity**: A single frame cannot convey whether a joint is extending, flexing, or maintaining constant velocity.
- **Occlusion and jitter sensitivity**: Measurement noise in one frame can cause significant outliers.
- **Lack of dynamic context**: True biomechanical running patterns (such as range of motion, stride consistency, and bilateral symmetry) only manifest over time across complete movement cycles.

**Temporal Movement Analysis** computes continuous and windowed statistical properties over a sequence of frames, converting raw per-frame pose angles into robust metrics that capture:
- Central tendency (mean angles)
- Joint variability and movement stability (standard deviations)
- Extremes and amplitude (minimum, maximum, range of motion)
- Bilateral asymmetry dynamics (left vs. right differences and their variations over time)
- Data quality and landmark tracking reliability

---

## 2. Calculated Temporal Features

The module extracts features across three anatomical regions and overall pose quality:

### A. Knee Kinematics
- **`knee_left_mean` / `knee_right_mean`**: Mean knee flexion/extension angle across valid frames.
- **`knee_left_std` / `knee_right_std`**: Population standard deviation of knee flexion/extension over time (movement variability).
- **`knee_left_min` / `knee_right_min`**: Minimum knee angle (peak flexion during swing or loading).
- **`knee_left_max` / `knee_right_max`**: Maximum knee angle (peak extension).
- **`knee_left_rom` / `knee_right_rom`**: Knee range of motion (`max - min`).
- **`knee_diff_mean` / `knee_diff_std`**: Mean and standard deviation of pairwise signed left/right differences `(left_knee - right_knee)`.
- **`knee_asymmetry_mean` / `knee_asymmetry_std`**: Mean and standard deviation of absolute difference `|left_knee - right_knee|`.

### B. Hip Kinematics
- **`hip_left_mean` / `hip_right_mean`**: Mean hip angle over time.
- **`hip_left_std` / `hip_right_std`**: Standard deviation of hip flexion/extension angles.
- **`hip_left_min` / `hip_left_max`**: Minimum and maximum hip angles.
- **`hip_right_min` / `hip_right_max`**: Minimum and maximum hip angles.
- **`hip_left_rom` / `hip_right_rom`**: Hip range of motion (`max - min`).

### C. Trunk Kinematics
- **`trunk_lean_mean`**: Mean forward/backward inclination of the torso relative to vertical.
- **`trunk_lean_std`**: Torso posture variation over time (stability indicator).
- **`trunk_lean_min` / `trunk_lean_max`**: Minimum and maximum forward lean angles.
- **`trunk_lean_rom`**: Total trunk range of motion.

### D. Pose Quality & Coverage
- **`total_frames`**: Number of video frames in the window or recording.
- **`valid_pose_frames`**: Number of frames where body landmarks were detected.
- **`valid_pose_frame_ratio`**: Ratio of detected frames to total frames (`valid / total`).
- **`pose_confidence_mean`**: Average landmark visibility/presence confidence from MediaPipe.
- **`missing_measurements`**: Integer count of missing or non-finite joint angle slots across the evaluated frames.
- **`is_reliable`**: Boolean flag indicating whether sufficient valid frames and tracking coverage exist (`valid_pose_frames >= min_valid_frames` and `valid_pose_frame_ratio >= min_valid_ratio`).

---

## 3. Window-Based Temporal Analysis

Running videos often span multiple seconds or minutes, during which running cadence, fatigue, or speed may vary. The temporal analysis module supports partitioning video streams into configurable time windows:

- Configurable duration: e.g., 1-second, 2-second (default), or 3-second windows (`window_sec`).
- Configurable step duration: e.g., non-overlapping tumbling windows (`step_sec = window_sec`) or overlapping sliding windows (e.g., `step_sec = 1.0` with `window_sec = 2.0`).
- No complete-stride assumption: Windows are strictly time-based slices and do not assume fixed stride cycles or require gait-phase segmentation.
- Robust boundary handling: Short recordings (< `window_sec`) automatically evaluate as a single window containing all available frames without throwing boundary errors.

Each window produces:
```json
{
  "window_index": 0,
  "start_time": 0.0,
  "end_time": 2.0,
  "total_frames": 60,
  "valid_frames": 58,
  "is_reliable": true,
  "features": { ... }
}
```

---

## 4. Smoothing and Missing Data Handling

### Inherent vs. Additional Smoothing
- **MediaPipe Pose Detector**: MediaPipe's underlying landmark tracking model (`smooth_landmarks=True`) already applies internal temporal filtering to 3D/2D landmark coordinates across frames.
- **Optional Measurement Smoothing**: To prevent duplicate smoothing or dampening real biomechanical movement extremes, smoothing is **disabled by default (`smooth_window=1`)**. When enabled (`smooth_window > 1`), a centered moving-average filter smooths only finite numeric measurements while preserving raw frame records for evaluation.

### Missing Data Handling
- Frames where no pose is detected (`pose_detected == False`) are omitted from angle statistics.
- If individual landmarks are obscured or drop below visibility thresholds, their respective angle channels evaluate to `None`.
- Statistical aggregations (`mean`, `std`, `min`, `max`, `rom`) only operate over finite numbers.
- If fewer than 2 valid observations exist for a channel, `std` evaluates to `0.0` (or `None` if 0 observations) without raising `ZeroDivisionError`.
- Missing angle slots increment the `missing_measurements` counter.
- Windows falling below reliability thresholds are clearly tagged with `is_reliable = False`.

---

## 5. Machine Learning Suitability

This module intentionally does **not** predict injury or assign rule-based injury risk labels. Instead, it exports structured, objective feature vectors suitable for:

1. **Tabular Machine Learning**: Training classifiers or regressors (e.g., XGBoost, Random Forests, Multi-Layer Perceptrons) by pairing video-level or window-level feature rows with external ground-truth labels from biomechanical running injury datasets.
2. **Time-Series Analysis**: Training sequence models (e.g., LSTMs, Transformers, 1D Temporal Convolutions) on windowed sequences of movement features.
3. **Multi-View Gait Analysis**: Joining side-view, front-view, and rear-view kinematics into comprehensive gait profiles.

### Export Formats:
- **JSON API Response**: Complete hierarchical data including whole-video statistics and per-window breakdowns.
- **Window-Level CSV (`runform_temporal_<id>.csv`)**: Multi-row tabular export where each row corresponds to one time window.
- **Video-Level CSV Row**: Single flattened row summarizing the entire trial for dataset creation.

---

## 6. Current Limitations

1. **2D Plane Projection**: Angles are calculated in the 2D camera image plane. Out-of-plane rotations (e.g., pelvic drop from a side view) cannot be measured without calibrated multi-camera setups or 3D pose lifting.
2. **Stride Cycle Agnostic**: Time windows are fixed-duration slices rather than stride-synchronized intervals. Future milestones will introduce stride segmentation to calculate stance-phase and swing-phase sub-metrics.
3. **Subject Identification**: The MediaPipe pipeline currently tracks the primary detected runner in the video.
