# 🏃 RunForm — Video-Based Running Form Analysis Using Pose Estimation

> **Thesis Workflow & Execution Plan**
>
> - **Programme:** M.Tech Computer Engineering (Batch 2025–2027)
> - **Student:** Varun Bhonslay (Roll No. 16030725019)
> - **Guide:** Dr. Nirmala Baloorkar
> - **Department Thrust Area:** Intelligent System & Data Processing
> - **Institution:** K.J. Somaiya College of Engineering, Somaiya Vidyavihar University

---

## 1. Project Overview

**RunForm** is an open-source, video-based running form analysis system. It takes a simple
video of someone running (pre-recorded or from a phone/laptop webcam) and automatically:

1. Detects the runner's body joints in every frame (**pose estimation**)
2. Classifies the **foot strike pattern** — heel-strike, mid-foot, or fore-foot
3. Measures key **biomechanical angles** (knee flexion, hip rotation, trunk lean)
4. **Tracks the runner** across occlusions and out-of-frame re-entry
5. Detects **multiple runners simultaneously** in the same video
6. Generates simple, actionable **form feedback**

---

## 2. Technical Stack

| Category | Choice |
|----------|--------|
| Language | Python 3.9+ & TypeScript / JavaScript (React) |
| Vision / I/O | OpenCV 4.x |
| Pose estimation (YOLO) | Ultralytics YOLOv8 (pose) |
| Pose estimation (MediaPipe) | Google MediaPipe Pose |
| Deep learning | PyTorch |
| Backend API | FastAPI, Uvicorn |
| Frontend UI | React 19, Vite, Lucide Icons |
| Datasets | COCO, PoseTrack21, Human3.6M |

---

## 3. Dual Pipeline Architecture

RunForm provides **two separate pose analysis pipelines**:

### YOLO Pipeline (existing)
- Uses YOLOv8-pose for multi-runner detection and tracking
- Foot strike classification, knee/hip angles, trunk lean
- Risk-level feedback via the `RiskAdvisor` module
- Available via **Upload** and **Camera** tabs in the frontend

### MediaPipe Pipeline (new prototype)
- Uses **Google MediaPipe Pose** for single-person 2D landmark detection
- Computes: knee angles (L/R), hip angles (L/R), trunk lean, knee asymmetry
- Provides simple non-medical **observations** (not injury predictions)
- Available via the **Pose** (video upload) and **Pose Cam** (live webcam) tabs
- **Does not** replace or modify the existing YOLO pipeline

> **Disclaimer:** RunForm provides 2D movement/form observations and is not a
> medical diagnostic or injury-prediction system.

---

## 4. Project Structure

```
runform/
├── configs/                  # model + app config (YAML)
├── data/                     # dataset storage
├── frontend/                 # React + Vite web client (desktop & mobile)
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoUploadView.jsx      # YOLO video upload
│   │   │   ├── LiveCameraView.jsx       # YOLO live camera
│   │   │   ├── PoseAnalysisView.jsx     # MediaPipe video analysis (NEW)
│   │   │   ├── PoseCameraView.jsx       # MediaPipe live camera (NEW)
│   │   │   ├── AnalysisHistoryView.jsx
│   │   │   ├── BiomechanicsGuideView.jsx
│   │   │   └── AuthView.jsx
│   │   ├── App.jsx
│   │   └── App.css
│   └── package.json
├── src/runform/
│   ├── detection/            # YOLOv8-pose wrapper
│   ├── tracking/             # 1€ filter & multi-runner tracker
│   ├── angles/               # 2D joint kinematics calculator
│   ├── footstrike/           # Strike classification module
│   ├── feedback/             # Biomechanical risk advisor
│   ├── pose/                 # MediaPipe prototype (NEW)
│   │   ├── detector.py       # MediaPipe pose detector
│   │   ├── angles.py         # Joint angle calculations
│   │   ├── metrics.py        # Running-form metrics & observations
│   │   ├── drawing.py        # Skeleton overlay rendering
│   │   ├── service.py        # Video/frame analysis service
│   │   ├── webcam.py         # Terminal webcam demo
│   │   └── landmarks.py      # Landmark data structures
│   ├── pipeline.py           # End-to-end YOLO pipeline orchestrator
│   ├── api.py                # FastAPI REST server (YOLO + MediaPipe)
│   └── schema.py             # Domain data models & types
├── app/
│   └── main.py               # Backend server entrypoint
├── tests/                    # Unit & contract tests (pytest)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 5. API Endpoints

### Existing YOLO Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze-video` | YOLO video analysis (auth required) |
| POST | `/api/analyze-frame` | YOLO single-frame analysis |
| GET | `/api/analyses` | List saved analyses |
| GET | `/api/analyses/{id}/video` | Download annotated video |
| GET | `/api/analyses/{id}/report` | Download JSON report |

### New MediaPipe Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/pose-analysis` | MediaPipe video analysis (no auth) |
| GET | `/api/pose-video/{id}` | Serve annotated pose video |
| POST | `/api/pose-analysis-frame` | MediaPipe single-frame analysis |

### MediaPipe Summary Metrics
- `detection_rate` — percentage of frames with pose detected
- `average_left_knee_angle` / `average_right_knee_angle`
- `average_left_hip_angle` / `average_right_hip_angle`
- `average_trunk_lean`
- `average_knee_asymmetry`
- `observations` — list of simple non-medical observations

---

## 6. Running the Application

### 1. Start the FastAPI Backend
```powershell
python app/main.py
```
*API will run at `http://localhost:8000` (docs at `http://localhost:8000/docs`).*

### 2. Start the React Frontend
```powershell
cd frontend
npm run dev
```
*Web client will open at `http://localhost:5173`.*

---

## 7. Running Tests
```powershell
python -m pytest
```

---

## 8. Exam Demonstration Guide

1. **Start the backend:** `python app/main.py`
2. **Start the frontend:** `cd frontend && npm run dev`
3. **Open the browser** at `http://localhost:5173`
4. **Register/login** with any email and password
5. **Demonstrate YOLO analysis:** Go to the **Upload** tab, upload `data/videos/RUN_SIDE_01.mp4`, click "Review form"
6. **Demonstrate MediaPipe analysis:** Go to the **Pose** tab, upload the same video, click "Analyze form"
7. **Demonstrate live webcam:** Go to the **Pose Cam** tab, click "Start camera", show yourself moving — skeleton and metrics appear in real-time
8. **Show the API docs:** Open `http://localhost:8000/docs` to show all endpoints
9. **Run tests:** `python -m pytest -v` (all 37 tests should pass)

> The **Pose** and **Pose Cam** tabs demonstrate the MediaPipe-based prototype.
> The **Upload** and **Camera** tabs demonstrate the existing YOLO-based pipeline.
> Both pipelines coexist independently.

