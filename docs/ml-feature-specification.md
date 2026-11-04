# RunForm ML Feature Specification & Dataset Compatibility Guide

**Version:** 1.0 (Milestone 1 Audit & ML Preparation)  
**Status:** Approved for Dataset Sourcing & Feature Harmonization  
**Scope:** MediaPipe Pose 2D Temporal Kinematics Pipeline  

---

## 1. Executive Summary

RunForm's long-term goal is to build an objective, machine learning-driven running-form assessment and injury-associated risk analysis system. To prevent reliance on arbitrary, manually hard-coded clinical rules, **Milestone 1** implemented an automated **Temporal Movement Analysis** engine.

This document serves as the foundational **Feature Specification and Dataset Compatibility Contract**:
1. It rigorously catalogues all 2D kinematic and pose-quality features currently extracted by RunForm.
2. It documents empirical measurement behavior observed across real running video benchmarks (`RUN_SIDE_01`, `RUN_SIDE_02`, `RUN_FRONT_01`, `RUN_REAR_01`).
3. It outlines mathematical differences between 2D monocular video kinematics and clinical 3D optical motion capture.
4. It specifies the requirements an external research dataset must satisfy to be ingested by the RunForm ML pipeline.
5. It defines an ML-ready tabular schema with empty label placeholders for future model training.

> [!IMPORTANT]
> This milestone strictly audits and specifies feature representations. No injury classification model has been trained, and no injury labels have been synthesized or fabricated.

---

## 2. Complete Feature Catalogue

RunForm calculates features at three hierarchical levels:
- **Frame-Level:** Instantaneous 2D joint angles and landmark visibility measured per frame.
- **Window-Level:** Aggregated statistical distributions computed across fixed-duration time slices (default: 2.0 seconds).
- **Video-Level:** Full-recording kinematic summaries for high-level trial indexing.

### 2.1 Frame-Level Measurements

| Feature Name | Level | Unit | Mathematical Method | Interpretation of Higher Values | Missing-Value Handling | Viewpoint Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `frame` | Frame | int | Sequential index `[0, N-1]` | Later temporal position in video. | Always present. | None. |
| `timestamp` | Frame | s | Frame timestamp in seconds: `frame / fps`. | Elapsed trial time. | Inferred from FPS if metadata missing. | None. |
| `pose_detected` | Frame | bool | MediaPipe landmark detection state (`len(landmarks) > 0`). | Landmark presence verified. | `False` if runner out-of-frame or occluded. | High occlusion in crowded views. |
| `pose_confidence` | Frame | `[0, 1]` | Arithmetic mean of landmark visibility across 17 primary body joints. | Higher tracker certainty; fewer landmark occlusions. | `None` when `pose_detected == False`. | Drops in low-light, blur, or loose clothing. |
| `left_knee_angle` | Frame | deg | 2D angle between 3 points: `left_hip` $\rightarrow$ `left_knee` $\rightarrow$ `left_ankle`. $\theta = \arccos\left(\frac{\mathbf{ba} \cdot \mathbf{bc}}{\|\mathbf{ba}\| \|\mathbf{bc}\|}\right)$. | Straighter leg / knee extension ($180^\circ$ = straight). Lower = deeper knee flexion. | `None` if hip, knee, or ankle visibility $< 0.3$. | Valid in **sagittal (side)** view only. Severe planar foreshortening in front/rear views. |
| `right_knee_angle` | Frame | deg | 2D angle between `right_hip` $\rightarrow$ `right_knee` $\rightarrow$ `right_ankle`. | Straighter leg / knee extension. | `None` if landmark visibility $< 0.3$. | Valid in sagittal view. Distorted in frontal view. |
| `left_hip_angle` | Frame | deg | 2D angle between `left_shoulder` $\rightarrow$ `left_hip` $\rightarrow$ `left_knee`. | More open hip extension ($180^\circ$ = straight torso-thigh). Lower = hip flexion. | `None` if shoulder, hip, or knee visibility $< 0.3$. | Valid in sagittal view. Coronal abduction cannot be measured in 2D side view. |
| `right_hip_angle` | Frame | deg | 2D angle between `right_shoulder` $\rightarrow$ `right_hip` $\rightarrow$ `right_knee`. | More open hip extension. | `None` if landmark visibility $< 0.3$. | Valid in sagittal view. |
| `left_elbow_angle` | Frame | deg | 2D angle between `left_shoulder` $\rightarrow$ `left_elbow` $\rightarrow$ `left_wrist`. | Straighter arm. Lower = tighter arm carriage flexion. | `None` if arm landmarks occluded. | Secondary metric; susceptible to arm-swing crossing. |
| `right_elbow_angle` | Frame | deg | 2D angle between `right_shoulder` $\rightarrow$ `right_elbow` $\rightarrow$ `right_wrist`. | Straighter arm. | `None` if arm landmarks occluded. | Secondary metric. |
| `trunk_lean` | Frame | deg | $\arctan2(\Delta x, \Delta y)$ between torso midpoint (shoulders) and pelvis midpoint (hips). | In side view: positive = forward lean (if running rightwards), negative = backward lean. | `None` if shoulder and hip midpoints cannot be resolved. | Meaning depends on runner's heading direction. In front/rear views, represents lateral torso tilt (lateral lean). |
| `knee_asymmetry` | Frame | deg | Absolute difference: $\| \text{left\_knee\_angle} - \text{right\_knee\_angle} \|$. | Greater instantaneous bilateral knee asymmetry. | `None` if either knee angle is `None`. | Requires both legs simultaneously visible. |

---

### 2.2 Window-Level & Video-Level Temporal Features

Calculated over consecutive time windows (e.g. 2.0-second intervals) or across the full video sequence.

#### A. Knee Movement Features

| Feature Name | Level | Unit | Mathematical Method | Biomechanical Interpretation | Missing-Value Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `knee_left_mean` | Window / Video | deg | $\bar{\theta} = \frac{1}{N} \sum_{i=1}^N \theta_{L,i}$ across valid frames. | Central knee flexion/extension angle. | `None` if 0 valid knee frames exist. |
| `knee_left_std` | Window / Video | deg | Population standard deviation $\sigma = \sqrt{\frac{1}{N} \sum (\theta_{L,i} - \bar{\theta})^2}$. | Dynamic knee variability / articulation amplitude. | `0.0` if $N=1$; `None` if $N=0$. |
| `knee_left_min` | Window / Video | deg | $\min(\theta_{L})$. | Peak knee flexion during swing or early loading response. | `None` if 0 valid knee frames. |
| `knee_left_max` | Window / Video | deg | $\max(\theta_{L})$. | Peak knee extension (terminal swing / initial contact). | `None` if 0 valid knee frames. |
| `knee_left_rom` | Window / Video | deg | Range of Motion: $\max(\theta_L) - \min(\theta_L)$. | Total angular excursion of the knee through movement cycles. | `None` if 0 valid knee frames. |
| `knee_right_mean` | Window / Video | deg | Mean of valid right knee angles. | Central right knee angle. | `None` if 0 valid knee frames. |
| `knee_right_std` | Window / Video | deg | Standard deviation of right knee angles. | Right knee movement variability. | `0.0` if $N=1$; `None` if $N=0$. |
| `knee_right_min` | Window / Video | deg | Minimum right knee angle. | Peak right knee flexion. | `None` if 0 valid knee frames. |
| `knee_right_max` | Window / Video | deg | Maximum right knee angle. | Peak right knee extension. | `None` if 0 valid knee frames. |
| `knee_right_rom` | Window / Video | deg | Range of Motion: $\max(\theta_R) - \min(\theta_R)$. | Total angular excursion of right knee. | `None` if 0 valid knee frames. |
| `knee_diff_mean` | Window / Video | deg | $\frac{1}{M} \sum (\theta_{L,i} - \theta_{R,i})$ for frames where both knees are valid. | Directional bilateral bias ($>0 \implies$ left leg straighter than right). | `None` if no pairwise frames exist. |
| `knee_diff_std` | Window / Video | deg | Standard deviation of signed pairwise difference $(\theta_L - \theta_R)$. | Bilateral phase coordination variance across gait cycles. | `0.0` if $M=1$; `None` if $M=0$. |
| `knee_asymmetry_mean` | Window / Video | deg | $\frac{1}{M} \sum \|\theta_{L,i} - \theta_{R,i}\|$. | Magnitude of bilateral knee asymmetry. High values indicate asymmetric gait. | `None` if no pairwise frames exist. |
| `knee_asymmetry_std` | Window / Video | deg | Standard deviation of absolute knee difference. | Temporal fluctuation in bilateral asymmetry. | `0.0` if $M=1$; `None` if $M=0$. |

#### B. Hip Movement Features

| Feature Name | Level | Unit | Mathematical Method | Biomechanical Interpretation | Missing-Value Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `hip_left_mean` | Window / Video | deg | Mean of valid left hip angles. | Average thigh-torso alignment. | `None` if 0 valid hip frames. |
| `hip_left_std` | Window / Video | deg | Standard deviation of left hip angle. | Dynamic hip flexion/extension variability. | `0.0` if $N=1$; `None` if $N=0$. |
| `hip_left_min` | Window / Video | deg | Minimum left hip angle. | Maximum forward hip flexion during swing phase. | `None` if 0 valid hip frames. |
| `hip_left_max` | Window / Video | deg | Maximum left hip angle. | Maximum trailing hip extension at late stance / push-off. | `None` if 0 valid hip frames. |
| `hip_left_rom` | Window / Video | deg | Range of Motion: $\max(\text{hip}_L) - \min(\text{hip}_L)$. | Hip mobility amplitude in the sagittal plane. | `None` if 0 valid hip frames. |
| `hip_right_mean` | Window / Video | deg | Mean of valid right hip angles. | Average right thigh-torso alignment. | `None` if 0 valid hip frames. |
| `hip_right_std` | Window / Video | deg | Standard deviation of right hip angle. | Right hip flexion/extension variability. | `0.0` if $N=1$; `None` if $N=0$. |
| `hip_right_min` | Window / Video | deg | Minimum right hip angle. | Maximum forward right hip flexion. | `None` if 0 valid hip frames. |
| `hip_right_max` | Window / Video | deg | Maximum right hip angle. | Maximum trailing right hip extension. | `None` if 0 valid hip frames. |
| `hip_right_rom` | Window / Video | deg | Range of Motion: $\max(\text{hip}_R) - \min(\text{hip}_R)$. | Right hip mobility amplitude. | `None` if 0 valid hip frames. |

#### C. Trunk Movement Features

| Feature Name | Level | Unit | Mathematical Method | Biomechanical Interpretation | Missing-Value Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `trunk_lean_mean` | Window / Video | deg | Mean of valid trunk lean angles. | Sustained posture orientation. In side view: sustained forward/backward pitch. | `None` if 0 valid trunk frames. |
| `trunk_lean_std` | Window / Video | deg | Standard deviation of trunk lean angles. | Upper body dynamic stability. High values indicate trunk bobbing or sway. | `0.0` if $N=1$; `None` if $N=0$. |
| `trunk_lean_min` | Window / Video | deg | Minimum trunk inclination angle. | Peak backward inclination (or reduced forward lean). | `None` if 0 valid trunk frames. |
| `trunk_lean_max` | Window / Video | deg | Maximum trunk inclination angle. | Peak forward trunk inclination. | `None` if 0 valid trunk frames. |
| `trunk_lean_rom` | Window / Video | deg | Range of Motion: $\max(\text{trunk}) - \min(\text{trunk})$. | Total torso oscillation amplitude during movement. | `None` if 0 valid trunk frames. |

#### D. Pose Quality & Window Reliability

| Feature Name | Level | Unit | Mathematical Method | Biomechanical Interpretation | Missing-Value Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `total_frames` | Window / Video | count | Number of video frames in window or video. | Temporal window sample size ($T = \text{frames} / \text{fps}$). | Always present ($\ge 0$). |
| `valid_pose_frames`| Window / Video | count | Frames with `pose_detected == True`. | Observed tracking coverage. | Always present ($\ge 0$). |
| `valid_pose_frame_ratio` | Window / Video | ratio | $\frac{\text{valid\_pose\_frames}}{\text{total\_frames}}$. | Fraction of window with tracked body landmarks. | `0.0` if `total_frames == 0`. |
| `pose_confidence_mean` | Window / Video | `[0, 1]` | Average of frame-level `pose_confidence` over detected frames. | Tracker reliability across the window. | `None` if 0 valid frames. |
| `missing_measurements` | Window / Video | count | Integer count of missing values across 5 monitored channels ($5 \times \text{frames}$). | Direct measure of occlusions, dropouts, or tracker failure. | Always present ($\ge 0$). |
| `is_reliable` | Window / Video | bool | `valid_pose_frames >= 3` AND `valid_pose_frame_ratio >= 0.40`. | Quality gate for ML ingestion. Windows with `False` should be masked or dropped. | Evaluates to `False` on empty or noisy windows. |

---

## 3. Empirical Benchmark & Viewpoint Audit

To evaluate how these features behave in real conditions, the actual pipeline was executed against four reference running videos across different camera viewpoints.

### Benchmark Execution Summary

| Video ID | Camera View | Total Frames | FPS | Duration | Valid Frames | Valid Ratio | Avg Conf | Missing Slots | Reliability | Left Knee ROM | Right Knee ROM | Trunk Lean ROM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `RUN_SIDE_01` | Sagittal (Side) | 136 | 25.0 | 5.40s | 80 | 58.8% | 0.998 | 280 | **True** | $93.2^\circ$ | $124.0^\circ$ | $26.8^\circ$ |
| `RUN_SIDE_02` | Sagittal (Side) | 484 | 25.0 | 19.32s | 484 | 100.0% | 0.999 | 0 | **True** | $120.3^\circ$ | $118.8^\circ$ | $17.6^\circ$ |
| `RUN_FRONT_01`| Coronal (Front) | 732 | 25.0 | 29.24s | 108 | 14.8% | 0.931 | 3,143 | **False** | $177.3^\circ$ | $174.1^\circ$ | $359.1^\circ$ |
| `RUN_REAR_01` | Coronal (Rear) | 190 | 24.0 | 7.88s | 69 | 36.3% | 0.995 | 605 | **False** | $176.5^\circ$ | $175.9^\circ$ | $159.6^\circ$ |

### Key Viewpoint Findings:

1. **Sagittal Side-View Stability:**
   - Both `RUN_SIDE_01` and `RUN_SIDE_02` achieved high tracking quality and were marked `is_reliable = True`.
   - Measured knee ranges of motion ($93^\circ$ to $124^\circ$) and hip ranges of motion ($54^\circ$ to $80^\circ$) closely align with physiological running biomechanics in the sagittal plane.
   - Bilateral knee asymmetry variation ($\sigma \approx 22^\circ$) reflects normal alternating running gait cycles.

2. **Frontal and Rear View Degeneracy in 2D:**
   - **Severe Planar Foreshortening:** In front and rear camera views, the knee flexion/extension axis is perpendicular to the camera sensor. As a runner bends their knee backward during swing, the 2D projected distance between hip, knee, and ankle shrinks, causing the 2D angle formula to collapse toward $0^\circ$ or $180^\circ$ (notice minimum knee angle dropping to $2.69^\circ$ in `RUN_FRONT_01` vs $58.8^\circ$ in `RUN_SIDE_02`).
   - **Trunk Angle Discontinuity:** Torso lean in front view measures lateral trunk lean (side tilt) rather than forward pitch. Lateral arm crossing and chest rotation triggered severe angle flips (trunk ROM $359.1^\circ$ in `RUN_FRONT_01`).
   - **Tracking Dropouts:** Runner movement toward/away from the camera causes high perspective scale changes, foot self-occlusion, and camera boundary clipping, resulting in poor valid frame ratios ($14.8\%$ and $36.3\%$).

> [!WARNING]
> **Viewpoint Rule:** Features extracted from Front or Rear camera angles **cannot** be mixed with Side (sagittal) features without explicit multi-view normalization or separate model heads. 2D sagittal angles measure flexion/extension; 2D frontal angles measure abduction/adduction or perspective collapse.

---

## 4. Research Dataset Compatibility Requirements

To train valid machine learning models that generalize to real runners without learning spurious lab-specific artifacts, target research datasets must meet the following structural requirements:

### 4.1 2D Video vs 3D Motion-Capture Measurements
- **Do Not Conflate 2D with 3D:** Clinical biomechanics datasets often use optical motion capture (e.g., Vicon, Qualisys) with 3D retro-reflective markers to compute 3D Cardan/Euler angles in local anatomic coordinate frames.
- **Why They Differ:** MediaPipe outputs 2D pixel coordinates projected onto a camera plane. 2D projection angles omit out-of-plane rotation, internal/external tibia rotation, and out-of-plane hip abduction.
- **Ingestion Requirement:** To ensure feature parity, the research dataset **must supply raw RGB video recordings** (or 2D keypoint coordinate streams) so that RunForm's exact detector and feature pipeline can extract the training inputs. Using pre-computed 3D joint angles directly as training inputs for a 2D deployment pipeline will cause severe distribution shift and model failure.

### 4.2 Time-Series & Temporal Continuity Requirements
- **Continuous Sequences Required:** Datasets that only provide discrete scalar snapshots (e.g., "knee angle at foot strike = $158^\circ$") are **incompatible** with temporal analysis.
- **Minimum Duration:** Continuous trials of at least 3.0 to 10.0 seconds per recording (yielding at least two to five complete 2.0-second analysis windows).
- **Sampling Rate:** Known frame rate (minimum 25–30 FPS; 60–120 FPS preferred for high-speed treadmill analysis).

### 4.3 Participant Identification & Grouping
- **`participant_id` Is Mandatory:** Every trial must have an anonymized runner ID.
- **Preventing Data Leakage:** Because a single runner produces dozens of 2.0-second temporal windows, cross-validation **must be grouped by participant** (e.g., `GroupKFold` or Leave-One-Subject-Out). Randomly shuffling window records across train and test sets leads to massive data leakage, where the model merely memorizes individual clothing, body aspect ratios, or running style.

### 4.4 Labeling Standards (Retrospective vs Prospective)
To train an injury-associated ML model without rule-based thresholds, datasets must supply verified labels:
- **Retrospective Labels:** History of running-related injury (RRI) in the past 6–12 months, or currently symptomatic vs asymptomatic status (e.g., Patellofemoral Pain Syndrome, Iliotibial Band Syndrome, Achilles Tendinopathy, Plantar Fasciitis, Medial Tibial Stress Syndrome).
- **Prospective Labels (Gold Standard):** Baseline running video collected while healthy, followed by 3 to 12 months of longitudinal injury tracking with recorded running exposure (hours or mileage).
- **Form Labels:** Verified ground truth for foot strike pattern (Rearfoot, Midfoot, Forefoot), measured cadence, or overstriding index.

### 4.5 Camera Specifications
- Clear documentation of camera placement: Viewpoint (`side_left`, `side_right`, `front`, `rear`), height ($\approx 0.8 - 1.0\text{ m}$ from ground), distance ($\approx 2.5 - 4.0\text{ m}$), and treadmill vs outdoor setting.

---

## 5. ML-Ready Tabular Schema

This schema represents the uniform row structure for tabular machine learning pipelines. 

Every record corresponds to **one temporal window** extracted from a running trial. 

> [!IMPORTANT]
> All `label_*` fields remain **empty (`None` / empty string)** until verified labels from an approved research dataset are joined by `participant_id` and `recording_id`. No labels are fabricated.

### Schema Definition

```
# Primary Keys & Metadata
participant_id             : string (Unique runner identifier)
recording_id               : string (Trial / video file identifier)
camera_view                : string ('side_left' | 'side_right' | 'front' | 'rear')
environment                : string ('treadmill' | 'overground_track' | 'road')
speed_kph                  : float  (Running speed if recorded, e.g. 10.5)

# Window Temporal Bounds
window_index               : int    (Sequential window counter: 0, 1, 2, ...)
window_start               : float  (Window start time in seconds from recording start)
window_end                 : float  (Window end time in seconds from recording start)
window_duration_sec        : float  (Target window duration, default 2.0)

# Pose Quality & Reliability Gate
total_frames               : int    (Total frames processed in window)
valid_frames               : int    (Frames where pose landmarks were detected)
valid_pose_frame_ratio     : float  (valid_frames / total_frames)
pose_confidence_mean       : float  (Average landmark detection confidence [0, 1])
missing_measurements       : int    (Count of missing joint angle slots)
is_reliable                : bool   (Quality gate: valid_frames >= 3 and ratio >= 0.40)

# Knee Temporal Kinematics
knee_left_mean             : float  (degrees)
knee_left_std              : float  (degrees)
knee_left_min              : float  (degrees)
knee_left_max              : float  (degrees)
knee_left_rom              : float  (degrees)
knee_right_mean            : float  (degrees)
knee_right_std             : float  (degrees)
knee_right_min             : float  (degrees)
knee_right_max             : float  (degrees)
knee_right_rom             : float  (degrees)
knee_diff_mean             : float  (degrees, signed left - right)
knee_diff_std              : float  (degrees, signed left - right variation)
knee_asymmetry_mean        : float  (degrees, absolute |left - right|)
knee_asymmetry_std         : float  (degrees, absolute asymmetry variation)

# Hip Temporal Kinematics
hip_left_mean              : float  (degrees)
hip_left_std               : float  (degrees)
hip_left_min               : float  (degrees)
hip_left_max               : float  (degrees)
hip_left_rom               : float  (degrees)
hip_right_mean             : float  (degrees)
hip_right_std              : float  (degrees)
hip_right_min              : float  (degrees)
hip_right_max              : float  (degrees)
hip_right_rom              : float  (degrees)

# Trunk Temporal Kinematics
trunk_lean_mean            : float  (degrees)
trunk_lean_std             : float  (degrees)
trunk_lean_min             : float  (degrees)
trunk_lean_max             : float  (degrees)
trunk_lean_rom             : float  (degrees)

# Ground-Truth Labels (REMAIN EMPTY UNTIL VERIFIED DATASET JOIN)
label_injury_status        : string (EMPTY - e.g., 'healthy' | 'injured' | 'subsequent_injury')
label_injury_type          : string (EMPTY - e.g., 'patellofemoral' | 'itbs' | 'achilles')
label_injury_side          : string (EMPTY - e.g., 'left' | 'right' | 'bilateral')
label_injury_onset_weeks   : float  (EMPTY - prospective weeks until injury onset)
label_footstrike           : string (EMPTY - 'rearfoot' | 'midfoot' | 'forefoot')
label_cadence_spm          : float  (EMPTY - verified steps per minute)
```

---

## 6. Generated Actual Sample Artifacts

The feature extraction pipeline was executed against all available benchmark videos. The resulting verified output files are located in:

1. **Video-Level Temporal Features CSV:**  
   `data/features/sample_temporal_video_features.csv`  
   *Contains 4 full-trial feature rows (one for each video) capturing overall trial dynamics.*

2. **Window-Level Temporal Features CSV:**  
   `data/features/sample_temporal_window_features.csv`  
   *Contains 32 window rows across the 4 videos, sliced into 2.0-second intervals with complete kinematics and reliability flags.*

3. **Hierarchical JSON Summary:**  
   `data/features/sample_temporal_summary.json`  
   *Contains nested JSON structures including frames processed, detection rates, video-level feature dictionaries, and window arrays.*

---

## 7. Next Steps for Milestone 2

1. **Dataset Identification:** Locate publicly available running biomechanics video datasets (e.g., OpenCap running subsets, public treadmill gait datasets) that supply synchronized side-view video or 2D landmark sequences with clinical runner injury metadata.
2. **Batch Feature Ingestion:** Run the RunForm temporal feature extraction pipeline across all dataset videos to populate the tabular schema.
3. **Subject-Grouped Split:** Partition data into train/validation/test sets grouped strictly by `participant_id`.
4. **Baseline Modeling:** Train baseline models (e.g., Ridge, Random Forest, XGBoost) on window features to evaluate whether temporal kinematics predict gait characteristics before advancing to complex sequence models.

