"""FastAPI Backend Server for RunForm.

Provides REST endpoints for account-based running-form analysis, saved results,
and media/report downloads.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import tempfile
import time
from typing import Any
import uuid

import cv2
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel

from runform.pipeline import RunFormPipeline
from runform.pose.service import (
    analyze_pose_frame_image,
    analyze_pose_video_file,
    encode_frame_as_data_url,
    frame_metrics_to_dict,
    summary_to_dict,
)
from runform.schema import (
    BiomechanicalMetrics,
    FootStrikeType,
    VideoAnalysisReport,
)

app = FastAPI(
    title="RunForm API",
    description="Video-Based Running Gait Analysis and Injury Prevention API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RunFormPipeline | None = None
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "runform.db"
MEDIA_DIR = DATA_DIR / "user_media"
POSE_MEDIA_DIR = DATA_DIR / "pose_media"
SECRET_PATH = DATA_DIR / ".auth_secret"
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7

DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
POSE_MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def get_pipeline() -> RunFormPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RunFormPipeline()
    return _pipeline


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                feedback_json TEXT NOT NULL,
                video_path TEXT NOT NULL,
                report_path TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )


init_db()


def get_secret() -> bytes:
    if not SECRET_PATH.exists():
        SECRET_PATH.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    return SECRET_PATH.read_text(encoding="utf-8").strip().encode("utf-8")


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    rounds = 210_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), rounds)
    return f"pbkdf2_sha256${rounds}${salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, rounds, salt, expected = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(rounds))
        return hmac.compare_digest(digest.hex(), expected)
    except ValueError:
        return False


def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
        "nonce": secrets.token_urlsafe(12),
    }
    payload_b64 = b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(get_secret(), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return f"{payload_b64}.{b64url_encode(signature)}"


def read_token(token: str) -> str:
    try:
        payload_b64, signature_b64 = token.split(".", 1)
        expected = hmac.new(get_secret(), payload_b64.encode("utf-8"), hashlib.sha256).digest()
        if not hmac.compare_digest(b64url_decode(signature_b64), expected):
            raise ValueError("bad signature")
        payload = json.loads(b64url_decode(payload_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        return str(payload["sub"])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session") from exc


def current_user(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Login required")
    user_id = read_token(authorization.removeprefix("Bearer ").strip())
    with db() as conn:
        user = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


class AuthRequest(BaseModel):
    name: str | None = None
    email: str
    password: str


class FrameAnalysisRequest(BaseModel):
    image_base64: str
    frame_idx: int = 0


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "RunForm API"}


@app.post("/api/auth/register")
def register(payload: AuthRequest) -> dict[str, Any]:
    email = payload.email.strip().lower()
    name = (payload.name or email.split("@", 1)[0]).strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    user_id = str(uuid.uuid4())
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, name, email, hash_password(payload.password), int(time.time())),
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists") from exc

    return {"token": create_token(user_id), "user": {"id": user_id, "name": name, "email": email}}


@app.post("/api/auth/login")
def login(payload: AuthRequest) -> dict[str, Any]:
    email = payload.email.strip().lower()
    with db() as conn:
        user = conn.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return {
        "token": create_token(user["id"]),
        "user": {"id": user["id"], "name": user["name"], "email": user["email"]},
    }


@app.get("/api/auth/me")
def me(user: dict[str, str] = Depends(current_user)) -> dict[str, str]:
    return user


@app.post("/api/analyze-frame")
def analyze_frame_endpoint(payload: FrameAnalysisRequest) -> dict[str, Any]:
    try:
        header_split = payload.image_base64.split(",")
        encoded_data = header_split[-1]
        img_bytes = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        pipeline = get_pipeline()
        result, annotated = pipeline.analyze_frame(
            frame=frame,
            frame_idx=payload.frame_idx,
            draw_overlay=True,
        )

        annotated_b64 = ""
        if annotated is not None:
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        first_metrics = next(iter(result.metrics.values()), BiomechanicalMetrics())
        first_strike = next(iter(result.foot_strike.values()), FootStrikeType.UNKNOWN)
        feedback = pipeline.advisor.evaluate(first_metrics, strike_type=first_strike)

        return {
            "frame_index": result.frame_index,
            "timestamp_sec": result.timestamp_sec,
            "annotated_image": annotated_b64,
            "primary_strike": first_strike.value,
            "metrics": {
                "knee_flexion_left": first_metrics.knee_flexion_left_deg,
                "knee_flexion_right": first_metrics.knee_flexion_right_deg,
                "trunk_lean": first_metrics.trunk_lean_deg,
            },
            "feedback": {
                "risk_level": feedback.risk_level.value,
                "flags": feedback.flags,
                "recommendations": feedback.recommendations,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/analyze-video")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    user: dict[str, str] = Depends(current_user),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    analysis_id = str(uuid.uuid4())
    user_dir = MEDIA_DIR / user["id"] / analysis_id
    user_dir.mkdir(parents=True, exist_ok=True)
    input_path = user_dir / f"source_{Path(file.filename).name}"
    output_path = user_dir / "annotated.mp4"
    report_path = user_dir / "report.json"

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        pipeline = get_pipeline()
        report: VideoAnalysisReport = pipeline.analyze_video(
            video_path=str(input_path),
            output_annotated_path=str(output_path),
        )

        summary = {
            "total_frames": report.total_frames,
            "fps": report.fps,
            "duration_sec": round(report.duration_sec, 2),
            "primary_strike_type": report.primary_strike_type.value,
            "avg_knee_flexion_left": round(report.avg_metrics.knee_flexion_left_deg, 1) if report.avg_metrics.knee_flexion_left_deg else None,
            "avg_knee_flexion_right": round(report.avg_metrics.knee_flexion_right_deg, 1) if report.avg_metrics.knee_flexion_right_deg else None,
            "avg_trunk_lean": round(report.avg_metrics.trunk_lean_deg, 1) if report.avg_metrics.trunk_lean_deg else None,
        }
        feedback = {
            "risk_level": report.feedback.risk_level.value,
            "flags": report.feedback.flags,
            "recommendations": report.feedback.recommendations,
        }
        ml_features = report.metadata.get("ml_features", {})
        response = {
            "analysis_id": analysis_id,
            "video_id": analysis_id,
            "annotated_video_url": f"/api/analyses/{analysis_id}/video",
            "report_download_url": f"/api/analyses/{analysis_id}/report",
            "summary": summary,
            "feedback": feedback,
            "ml_features": ml_features,
        }
        report_path.write_text(json.dumps(response, indent=2), encoding="utf-8")

        with db() as conn:
            conn.execute(
                """
                INSERT INTO analyses
                (id, user_id, filename, created_at, source_type, summary_json, feedback_json, video_path, report_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    user["id"],
                    Path(file.filename).name,
                    int(time.time()),
                    "upload",
                    json.dumps(summary),
                    json.dumps(feedback),
                    str(output_path),
                    str(report_path),
                ),
            )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e
    finally:
        if input_path.exists():
            try:
                os.remove(input_path)
            except OSError:
                pass


@app.get("/api/analyses")
def list_analyses(user: dict[str, str] = Depends(current_user)) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, created_at, source_type, summary_json, feedback_json
            FROM analyses
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user["id"],),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "source_type": row["source_type"],
            "summary": json.loads(row["summary_json"]),
            "feedback": json.loads(row["feedback_json"]),
            "annotated_video_url": f"/api/analyses/{row['id']}/video",
            "report_download_url": f"/api/analyses/{row['id']}/report",
        }
        for row in rows
    ]


def get_owned_analysis(analysis_id: str, user: dict[str, str]) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user["id"]),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return row


@app.get("/api/analyses/{analysis_id}/video")
def get_analysis_video(analysis_id: str, user: dict[str, str] = Depends(current_user)):
    row = get_owned_analysis(analysis_id, user)
    video_path = Path(row["video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"runform_{analysis_id}.mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/analyses/{analysis_id}/report")
def get_analysis_report(analysis_id: str, user: dict[str, str] = Depends(current_user)):
    row = get_owned_analysis(analysis_id, user)
    report_path = Path(row["report_path"])
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=report_path, media_type="application/json", filename=f"runform_{analysis_id}.json")


@app.get("/api/video/{video_id}")
def get_processed_video(video_id: str, user: dict[str, str] = Depends(current_user)):
    return get_analysis_video(video_id, user)


# ---------------------------------------------------------------------------
# MediaPipe Pose Analysis Endpoints (separate from YOLO pipeline)
# ---------------------------------------------------------------------------

@app.post("/api/pose-analysis")
async def pose_analysis_endpoint(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Run MediaPipe-based running-form analysis on an uploaded video.

    Returns summary metrics, observations, and a URL to the annotated video.
    This is independent of the existing YOLO pipeline.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    analysis_id = str(uuid.uuid4())
    pose_dir = POSE_MEDIA_DIR / analysis_id
    pose_dir.mkdir(parents=True, exist_ok=True)
    input_path = pose_dir / f"source_{Path(file.filename).name}"
    output_path = pose_dir / "annotated.mp4"

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = analyze_pose_video_file(
            input_path=str(input_path),
            output_path=str(output_path),
        )

        summary = summary_to_dict(result.summary)
        video_url = f"/api/pose-video/{analysis_id}"
        return {
            "analysis_id": analysis_id,
            "video_id": analysis_id,
            "video_url": video_url,
            "annotated_video_url": video_url,
            "metrics": summary,
            "observations": summary.get("observations", []),
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose analysis failed: {e}") from e
    finally:
        if input_path.exists():
            try:
                os.remove(input_path)
            except OSError:
                pass


@app.get("/api/pose-video/{analysis_id}")
def get_pose_video(analysis_id: str):
    """Serve the annotated video produced by the MediaPipe pose analysis."""
    video_path = POSE_MEDIA_DIR / analysis_id / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found")
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"runform_pose_{analysis_id}.mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.post("/api/pose-analysis-frame")
def pose_analysis_frame_endpoint(payload: FrameAnalysisRequest) -> dict[str, Any]:
    """Analyze a single frame using MediaPipe pose detection.

    Returns annotated image, per-frame metrics, and pose detection status.
    This is independent of the existing YOLO pipeline.
    """
    try:
        header_split = payload.image_base64.split(",")
        encoded_data = header_split[-1]
        img_bytes = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        frame_metrics, annotated = analyze_pose_frame_image(
            frame,
            frame_index=payload.frame_idx,
        )

        annotated_b64 = encode_frame_as_data_url(annotated) if annotated is not None else ""
        metrics_dict = frame_metrics_to_dict(frame_metrics)

        return {
            "frame_index": payload.frame_idx,
            "annotated_image": annotated_b64,
            "pose_detected": frame_metrics.pose_detected,
            "metrics": metrics_dict,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
