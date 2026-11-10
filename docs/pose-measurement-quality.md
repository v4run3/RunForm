# RunForm — Pose Measurement Quality & Reliability Specification

> [!WARNING]
> **Biomechanical Validation Disclaimer**: The methods described in this document are **software-level quality control, filtering, and numerical plausibility checks**. They do NOT constitute clinical validation or medical diagnosis. A high reliability score indicates consistent computer-vision tracking and geometrically plausible values within 2D planar projection; it does **not** guarantee injury-free or clinically optimal running form.

---

## 1. Overview & Objectives

In early prototype outputs of the RunForm temporal analysis pipeline, several measurement anomalies were detected in non-sagittal views:
- **`RUN_FRONT_01`**: Knee minimum angle collapsed to **2.7°**, hip minimum collapsed to **1.8°**, and trunk lean range reported **359.1°**.
- **`RUN_REAR_01`**: Extremely low knee minimums (**3.5°**), hip minimums (**0.5°**), and extreme negative trunk lean values (**-137.2°**).

These errors originated from:
1. **Planar foreshortening in 2D**: Out-of-plane joint motion projected onto a 2D camera sensor collapses limb segments toward collinearity or coincident points.
2. **Angle wrapping across the $\pm 180^\circ$ branch cut**: Inverted torso detections or lateral sway crossing the `atan2` discontinuity resulted in catastrophic ROM artifacts ($\approx 360^\circ$).
3. **Unvalidated raw measurements**: Landmark triplets were computed blindly without anatomical plausibility checks.
4. **Single-factor reliability**: Reliability previously checked only whether MediaPipe reported *any* person detection, ignoring joint-level data quality.

This specification documents the comprehensive measurement-quality assurance framework implemented to solve these issues.

---

## 2. Audited Joint Angle Calculations

### Joint Landmark Triplets

| Measurement | Vertex | Proximal Landmark | Distant Landmark | Anatomical Motion (Side View) |
|---|---|---|---|---|
| `left_knee_angle` | `left_knee` | `left_hip` | `left_ankle` | Knee flexion / extension |
| `right_knee_angle` | `right_knee` | `right_hip` | `right_ankle` | Knee flexion / extension |
| `left_hip_angle` | `left_hip` | `left_shoulder` | `left_knee` | Hip flexion / extension |
| `right_hip_angle` | `right_hip` | `right_shoulder` | `right_knee` | Hip flexion / extension |
| `left_elbow_angle` | `left_elbow` | `left_shoulder` | `left_wrist` | Elbow flexion / extension |
| `right_elbow_angle` | `right_elbow` | `right_shoulder` | `right_wrist` | Elbow flexion / extension |
| `trunk_lean` | Hip midpoint | Shoulder midpoint | Vertical Axis | Forward pitch / backward lean |

### Degenerate Segment Handling

The 2D joint angle at vertex $B$ between proximal point $A$ and distant point $C$ is computed using the vector dot product:

$$\cos(\theta) = \frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \, \|\vec{BC}\|}$$

If either $\|\vec{BA}\| \le \epsilon$ or $\|\vec{BC}\| \le \epsilon$ ($\epsilon = 10^{-6}$):
- Previously, the function returned `0.0°`. This falsely converted a degenerate division-by-zero into a valid 0-degree angle.
- **Fixed**: The function now returns `None`. Missing/collapsed limbs are treated as missing values, avoiding zero-angle pollution.

---

## 3. Trunk Lean Calculation & Boundary Wrapping

### Reference Axis & Sign Convention

Trunk lean is computed relative to the upright vertical reference axis:
- **Reference axis**: A vertical line passing upward from the hip midpoint (`mid_hip` to `mid_hip + (0, -1)` in image space).
- **Coordinate offsets**:
  - Horizontal displacement: $dx = \text{shoulder}_x - \text{hip}_x$
  - Vertical displacement: $dy = \text{hip}_y - \text{shoulder}_y$ (positive when shoulder is above hip)
- **Lean angle**: $\theta = \text{degrees}(\text{atan2}(dx, dy))$ in $[-180^\circ, +180^\circ]$.
- **Upright posture**: $dx = 0, dy > 0 \implies \theta = 0.0^\circ$.
- **Sagittal (side) view**:
  - When the runner moves from left to right: forward lean has $dx > 0 \implies \theta > 0$.
  - When the runner moves from right to left: forward lean has $dx < 0 \implies \theta < 0$.
- **Frontal (front/rear) view**:
  - $dx$ measures lateral torso tilt (sway), NOT sagittal forward pitch.

### Solving the 359° Angle Wrapping Artifact

When a runner's torso detection degrades or temporarily inverts ($dy \le 0$):
- $\text{atan2}(+0.1, -10) \approx +179.4^\circ$
- $\text{atan2}(-0.1, -10) \approx -179.7^\circ$

Calculating naive range of motion as $\max - \min$ yielded:
$$\text{ROM} = 179.4^\circ - (-179.7^\circ) = 359.1^\circ$$

#### The Solution:
1. **Plausibility Filtering**: In upright human running, trunk lean is constrained to $[-45^\circ, +45^\circ]$. Values outside this range represent inverted or hallucinated poses and are filtered to `None`.
2. **Safe ROM Calculation (`calculate_trunk_lean_rom_safe`)**:
   - If $\max - \min \le 180^\circ$, compute direct difference.
   - If $\max - \min > 180^\circ$, unwrap the continuous temporal sequence across the $\pm 180^\circ$ boundary or take the shortest circular arc:
     $$\text{Arc} = \min\big(\max - \min, \, (180 - \max) + (180 + \min)\big)$$

---

## 4. Biomechanical Plausibility Quality Control

### Anatomical Plausibility Ranges

| Measurement Channel | Plausible Range | Rationale / Failure Mode |
|---|---|---|
| `left_knee_angle`, `right_knee_angle` | $[20.0^\circ, 185.0^\circ]$ | Below 20° is anatomically impossible in running (tibial collapse). Angles > 185° indicate unnatural hyperextension or inverted detection. |
| `left_hip_angle`, `right_hip_angle` | $[45.0^\circ, 190.0^\circ]$ | Below 45° implies thigh compressed against torso (impossible in running stance/swing). |
| `trunk_lean` | $[-45.0^\circ, +45.0^\circ]$ | Running trunk lean typically ranges from $0^\circ$ to $20^\circ$ (up to $30^\circ$ in acceleration). Values $> 45^\circ$ represent inverted or falling poses. |
| `left_elbow_angle`, `right_elbow_angle` | $[20.0^\circ, 180.0^\circ]$ | Anatomical range for elbow flexion/extension. |

### Preserving Raw vs. Filtered Measurements

- **`frame.metrics`**: Holds quality-filtered, plausible angles. Implausible measurements evaluate to `None`.
- **`frame.raw_metrics`**: Preserves raw geometric measurements before filtering for debugging and research evaluation.
- **`frame.measurement_quality`**: Documents specific reasons for any rejected angle (e.g., `"implausible: Knee angle 2.7° outside [20°, 185°]"`).
- **No silent zero-replacement**: Missing values strictly remain `None`.

---

## 5. Camera Viewpoint Compatibility Matrix

| Measurement | Side (Sagittal) View | Front View | Rear View | Unknown View |
|---|---|---|---|---|
| **Knee Flexion / Extension** | ✅ **Valid & Accurate** | ❌ **Invalid** (planar foreshortening collapses 2D projection) | ❌ **Invalid** (planar foreshortening collapses 2D projection) | ⚠️ Uncertain |
| **Hip Flexion / Extension** | ✅ **Valid & Accurate** | ❌ **Invalid** (planar foreshortening) | ❌ **Invalid** (planar foreshortening) | ⚠️ Uncertain |
| **Trunk Lean** | ✅ **Measures Forward Pitch** | ⚠️ **Measures Lateral Sway** | ⚠️ **Measures Lateral Sway** | ⚠️ Uncertain |
| **Bilateral Knee Asymmetry** | ✅ **Meaningful** | ❌ **Invalid** | ❌ **Invalid** | ⚠️ Uncertain |

> [!IMPORTANT]
> The pipeline supports explicit camera-view metadata (`side`, `front`, `rear`, `unknown`).
> In front and rear views, knee and hip flexion features are automatically flagged with `view_quality_issues` and marked `is_reliable = False` for sagittal gait analysis. Measurements from different viewpoints are never combined without explicit distinction.

---

## 6. Multi-Factor Reliability Scoring

Reliability is evaluated across four weighted dimensions:

$$\text{Score} = 0.30 \times S_{\text{frame}} + 0.25 \times S_{\text{vis}} + 0.25 \times S_{\text{meas}} + 0.20 \times S_{\text{stab}}$$

1. **Frame Coverage ($S_{\text{frame}}$)** (30%):
   - Evaluates $\text{valid\_pose\_frames} / \text{total\_frames}$. Must meet `min_valid_ratio` (default 0.40).
2. **Landmark Visibility ($S_{\text{vis}}$)** (25%):
   - Average MediaPipe visibility scores across hips, knees, ankles, and shoulders.
3. **Valid Measurement Ratio ($S_{\text{meas}}$)** (25%):
   - Proportion of expected joint angles that passed plausibility checks.
4. **Measurement Stability ($S_{\text{stab}}$)** (20%):
   - Evaluates coefficient of variation ($\text{CV} = \text{std} / |\text{mean}|$) to detect limb swapping or erratic landmark jitter.

A sequence is marked `is_reliable = True` ONLY when:
- $\text{valid\_pose\_frames} \ge \text{threshold\_frames}$
- $\text{valid\_pose\_ratio} \ge \text{threshold\_ratio}$
- $\text{measurement\_ratio} \ge \text{min\_valid\_measurements\_ratio}$
- Camera view is compatible (`camera_view not in ("front", "rear")`)
