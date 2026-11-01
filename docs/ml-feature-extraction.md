# RunForm ML Feature Extraction

RunForm currently performs running-form analysis from ordinary monocular video. It uses YOLOv8-Pose detections, tracker IDs, 2D joint-angle calculations, rule-based foot-strike classification, and a rule-based risk advisor. No injury prediction model has been trained.

## Current Pipeline

1. Video frames are read with OpenCV.
2. YOLOv8-Pose produces COCO 17-keypoint detections.
3. The tracker assigns runner IDs across frames.
4. `BiomechanicsCalculator` computes 2D knee angle, 2D hip angle, and 2D trunk lean where keypoints are confident enough.
5. `FootStrikeDetector` classifies frame-level strike pattern from shank orientation.
6. `RiskAdvisor` produces rule-based running-form feedback.
7. `runform.features.extractor` selects one primary runner track, then aggregates that track into one feature row per video for future ML work.

## ML-Ready Features

These features are currently exported by `VideoFeatureVector`:

- `knee_left_mean`, `knee_left_std`, `knee_left_min`, `knee_left_max`
- `knee_right_mean`, `knee_right_std`, `knee_right_min`, `knee_right_max`
- `knee_asymmetry_mean`, `knee_asymmetry_std`
- `hip_left_mean`, `hip_left_std`, `hip_left_min`, `hip_left_max`
- `hip_right_mean`, `hip_right_std`, `hip_right_min`, `hip_right_max`
- `trunk_lean_mean`, `trunk_lean_std`, `trunk_lean_min`, `trunk_lean_max`
- `heel_ratio`, `midfoot_ratio`, `forefoot_ratio`, `unknown_ratio` as backward-compatible column names for frame-level foot-strike classification proportions
- `dominant_foot_strike`
- `pose_confidence_mean`, `valid_pose_frame_ratio`
- `fps`, `duration_sec`
- `cadence_spm`, currently exported as unavailable because validated step/contact event timing is not yet implemented. TODO: derive this later only from validated or explicitly defined running-cycle/foot-event detection.

## Reliability Notes

Relatively reliable from side-view monocular video:

- 2D knee, hip, and trunk angle summaries when pose confidence is high.
- Left/right 2D angle asymmetry when both sides are visible enough.
- Pose confidence summaries and valid-frame ratio.

Experimental or estimated:

- The `heel_ratio`, `midfoot_ratio`, `forefoot_ratio`, and `unknown_ratio` fields are frame-level foot-strike classification proportions. They are retained for CSV compatibility and should not be interpreted as clinically validated foot-strike event ratios.
- Dominant strike pattern, because camera angle and shoe occlusion can affect classification.

Not currently reliable without calibration or lab equipment:

- Absolute stride length in meters.
- Ground reaction force, loading rate, joint moments, and joint power.
- Validated stance/contact timing from force plates or pressure insoles.

## Injury Prediction Status

RunForm must not claim to diagnose an injury. The defensible future target is injury-associated biomechanical risk, and that requires a real labelled dataset, careful cohort definition, train/validation/test splitting, cross-validation, and transparent evaluation. The current system has no injury-labelled dataset and no trained injury-risk ML model.

## CSV Export Example

```python
from runform.features import extract_video_features, write_feature_csv

features = extract_video_features(report.frame_results, video_id=report.video_path, fps=report.fps)
write_feature_csv([features], "data/features/runform_features.csv")
```

Or from the command line:

```bash
python -m runform.features.cli path/to/video.mp4 --output data/features/runform_features.csv
```

Example columns:

```text
video_id,knee_left_mean,knee_left_std,knee_left_min,knee_left_max,knee_right_mean,...
```

There is intentionally no `injury` target column until real labelled data exists.



