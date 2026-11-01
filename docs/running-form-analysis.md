# RunForm Basic Running-Form Analysis

This prototype performs basic running-form analysis from 2D pose landmarks. It is not a medical diagnostic system and does not predict injuries.

## What Pose Detection Does

RunForm uses MediaPipe pose detection to find one visible person in each webcam or video frame. The detector returns normalized 2D body landmarks such as shoulders, elbows, wrists, hips, knees, ankles, heels, and foot-index points. These landmarks are then drawn as a skeleton overlay and passed into simple angle calculations.

## Detected Joints

The prototype focuses on the landmarks needed for a clear running-form demo:

- nose
- left and right shoulder
- left and right elbow
- left and right wrist
- left and right hip
- left and right knee
- left and right ankle
- left and right heel
- left and right foot index

If a landmark is missing or has low visibility, any metric that depends on it is left unavailable for that frame.

## Knee Angle

Knee angle is calculated as a 2D angle at the knee joint:

- Left knee: left hip -> left knee -> left ankle
- Right knee: right hip -> right knee -> right ankle

The middle point is the joint being measured. A straighter leg gives a larger angle, while a more flexed knee gives a smaller angle.

## Hip Angle

Hip angle is calculated as a 2D angle at the hip joint:

- Left hip: left shoulder -> left hip -> left knee
- Right hip: right shoulder -> right hip -> right knee

This is a simple 2D approximation from camera landmarks, not a full biomechanical lab measurement.

## Trunk Lean

Trunk lean is calculated from the line between the shoulder midpoint and hip midpoint. The angle is measured relative to the vertical direction in the image. If both sides are not visible, the prototype falls back to the visible shoulder and hip on one side.

## Knee Asymmetry

Knee asymmetry is the absolute difference between left and right knee angles:

```text
knee_asymmetry = abs(left_knee_angle - right_knee_angle)
```

If either knee angle is unavailable, knee asymmetry is also unavailable for that frame.

## Frame-Level Metrics

For every processed frame, RunForm stores:

- frame
- timestamp
- pose_detected
- pose_confidence
- left_knee_angle
- right_knee_angle
- left_hip_angle
- right_hip_angle
- left_elbow_angle
- right_elbow_angle
- trunk_lean
- knee_asymmetry

These can optionally be exported to CSV from the video command.

## Video Summary

After video processing, RunForm prints averages for the available frame-level values:

- average left knee angle
- average right knee angle
- average left hip angle
- average right hip angle
- average trunk lean
- average knee asymmetry
- pose detection rate

Missing values are ignored in averages.

## Simple Observations

The observation layer is intentionally basic and rule-based:

- Low pose detection rate produces: "Pose visibility was low; results may be less reliable."
- High average knee asymmetry produces: "Noticeable left/right knee-angle difference detected."
- Large trunk lean produces: "Increased trunk lean detected."
- If no rule is triggered: "No major form deviation detected by the current basic rules."

These are observations about visible 2D movement patterns. They are not injury predictions and should not be treated as medical advice.

## Limitations

- The analysis is based on ordinary 2D video, so camera angle strongly affects measurements.
- Loose clothing, occlusion, poor lighting, and fast motion can reduce landmark reliability.
- MediaPipe detects visible landmarks, not true internal joint centers.
- The prototype currently prefers one detected person and does not perform multi-runner tracking.
- The observations are simple thresholds intended for a prototype demonstration.
- This module is separate from the existing YOLO-based RunForm pipeline and does not connect to RiskAdvisor yet.
