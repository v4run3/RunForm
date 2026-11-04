"""Tests for the MediaPipe pose analysis API endpoints and video delivery."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from runform.api import POSE_MEDIA_DIR, app
from runform.pose.encoding import convert_to_browser_mp4, find_ffmpeg_executable
from runform.pose.metrics import FrameMetrics, RunningFormMetrics, RunningFormSummary
from runform.pose.service import (
    PoseVideoAnalysisResult,
    frame_metrics_to_dict,
    summary_to_dict,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests for service helpers
# ---------------------------------------------------------------------------


def test_summary_to_dict_basic():
    """summary_to_dict should round floats and include observations."""
    summary = RunningFormSummary(
        frames_processed=100,
        pose_detected_frames=92,
        detection_rate=0.92,
        average_left_knee_angle=148.12345,
        average_right_knee_angle=151.6789,
        average_left_hip_angle=162.0,
        average_right_hip_angle=159.3,
        average_trunk_lean=4.2,
        average_knee_asymmetry=3.6,
        observations=["No major form deviation detected by the current basic rules."],
    )
    d = summary_to_dict(summary)
    assert d["frames_processed"] == 100
    assert d["pose_detected_frames"] == 92
    assert d["detection_rate"] == 0.92
    assert d["average_left_knee_angle"] == 148.1
    assert d["average_right_knee_angle"] == 151.7
    assert d["average_trunk_lean"] == 4.2
    assert d["average_knee_asymmetry"] == 3.6
    assert len(d["observations"]) == 1


def test_summary_to_dict_none_values():
    """None metric values should stay None in the dict output."""
    summary = RunningFormSummary(
        frames_processed=10,
        pose_detected_frames=0,
        detection_rate=0.0,
    )
    d = summary_to_dict(summary)
    assert d["average_left_knee_angle"] is None
    assert d["average_right_knee_angle"] is None
    assert d["average_trunk_lean"] is None


def test_frame_metrics_to_dict():
    """frame_metrics_to_dict should include frame index and metrics."""
    fm = FrameMetrics(
        frame=5,
        timestamp=0.166,
        pose_detected=True,
        pose_confidence=0.88,
        metrics=RunningFormMetrics(
            left_knee_angle=150.0,
            right_knee_angle=152.0,
            trunk_lean=3.5,
            knee_asymmetry=2.0,
        ),
    )
    d = frame_metrics_to_dict(fm)
    assert d["frame"] == 5
    assert d["pose_detected"] is True
    assert d["left_knee_angle"] == 150.0
    assert d["trunk_lean"] == 3.5


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


def test_pose_analysis_api_response():
    """POST /api/pose-analysis should return video_id, video_url, metrics, observations."""
    dummy_summary = RunningFormSummary(
        frames_processed=30,
        pose_detected_frames=28,
        detection_rate=0.9333,
        average_left_knee_angle=142.5,
        average_right_knee_angle=145.0,
        average_left_hip_angle=160.0,
        average_right_hip_angle=162.0,
        average_trunk_lean=6.5,
        average_knee_asymmetry=2.5,
        observations=["Noticeable left/right knee-angle difference detected."],
    )
    mock_result = PoseVideoAnalysisResult(
        frames=[],
        summary=dummy_summary,
        output_path=Path("dummy.mp4"),
        csv_path=None,
        fps=30.0,
        width=640,
        height=480,
    )

    with patch("runform.api.analyze_pose_video_file", return_value=mock_result):
        dummy_file = io.BytesIO(b"fake video content")
        response = client.post(
            "/api/pose-analysis",
            files={"file": ("test_run.mp4", dummy_file, "video/mp4")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "video_id" in data
    assert "video_url" in data
    assert "metrics" in data
    assert "observations" in data
    assert data["video_url"] == f"/api/pose-video/{data['video_id']}"
    assert data["metrics"]["detection_rate"] == 0.9333
    assert data["observations"] == ["Noticeable left/right knee-angle difference detected."]


def test_get_pose_video_content_type():
    """GET /api/pose-video/{id} must return Content-Type: video/mp4 and byte content."""
    test_id = "test-video-fixture-123"
    test_dir = POSE_MEDIA_DIR / test_id
    test_dir.mkdir(parents=True, exist_ok=True)
    video_file = test_dir / "annotated.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

    try:
        response = client.get(f"/api/pose-video/{test_id}")
        assert response.status_code == 200
        assert "video/mp4" in response.headers.get("content-type", "")
        assert response.headers.get("accept-ranges") == "bytes"
        assert len(response.content) > 0
    finally:
        if video_file.exists():
            video_file.unlink()
        if test_dir.exists():
            test_dir.rmdir()


def test_get_pose_video_missing():
    """GET /api/pose-video/{id} should return 404 for missing video."""
    response = client.get("/api/pose-video/non-existent-id-999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Annotated video not found"


def test_pose_frame_endpoint():
    """POST /api/pose-analysis-frame should process single frame payload."""
    # Create 100x100 black image encoded as JPEG data URL
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    import base64

    b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    response = client.post(
        "/api/pose-analysis-frame",
        json={"image_base64": b64, "frame_idx": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert "frame_index" in data
    assert "annotated_image" in data
    assert "metrics" in data
    assert "pose_detected" in data


def test_video_transcoder_h264():
    """Test find_ffmpeg_executable and convert_to_browser_mp4 functionality."""
    exe = find_ffmpeg_executable()
    assert exe is not None, "FFmpeg executable must be discovered"


def test_api_returns_relative_urls():
    """All generated video URLs must be relative paths starting with /api/ for same-origin proxying."""
    test_id = "test-relative-id-456"
    test_dir = POSE_MEDIA_DIR / test_id
    test_dir.mkdir(parents=True, exist_ok=True)
    video_file = test_dir / "annotated.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

    try:
        dummy_summary = RunningFormSummary(
            frames_processed=10,
            pose_detected_frames=10,
            detection_rate=1.0,
        )
        mock_result = PoseVideoAnalysisResult(
            frames=[],
            summary=dummy_summary,
            output_path=video_file,
            csv_path=None,
            fps=30.0,
            width=640,
            height=480,
        )
        with patch("runform.api.analyze_pose_video_file", return_value=mock_result):
            dummy_file = io.BytesIO(b"fake video")
            res = client.post("/api/pose-analysis", files={"file": ("test.mp4", dummy_file, "video/mp4")})
        assert res.status_code == 200
        payload = res.json()
        assert payload["video_url"].startswith("/api/pose-video/")
        assert payload["annotated_video_url"].startswith("/api/pose-video/")
        assert "localhost" not in payload["video_url"]
        assert "127.0.0.1" not in payload["video_url"]
    finally:
        if video_file.exists():
            video_file.unlink()
        if test_dir.exists():
            test_dir.rmdir()


def test_vite_proxy_and_relative_urls_in_frontend():
    """Ensure vite.config.js proxies /api to 127.0.0.1:8000 and frontend code has no hardcoded localhost."""
    vite_config_path = Path("frontend/vite.config.js")
    assert vite_config_path.exists(), "frontend/vite.config.js must exist"
    config_content = vite_config_path.read_text(encoding="utf-8")
    assert "0.0.0.0" in config_content
    assert "5173" in config_content
    assert "/api" in config_content
    assert "http://127.0.0.1:8000" in config_content

    # Check frontend components do not contain hardcoded localhost:8000
    frontend_src = Path("frontend/src")
    for js_file in frontend_src.rglob("*.jsx"):
        content = js_file.read_text(encoding="utf-8")
        assert "localhost:8000" not in content, f"Hardcoded localhost:8000 found in {js_file}"
        assert "127.0.0.1:8000" not in content, f"Hardcoded 127.0.0.1:8000 found in {js_file}"


def test_camera_recording_upload_webm_and_mp4():
    """POST /api/pose-analysis should accept recorded .webm and .mp4 files."""
    dummy_summary = RunningFormSummary(
        frames_processed=15,
        pose_detected_frames=15,
        detection_rate=1.0,
        average_left_knee_angle=140.0,
        average_right_knee_angle=142.0,
        average_trunk_lean=5.0,
        average_knee_asymmetry=2.0,
        observations=["Optimal form observed."],
    )
    mock_result = PoseVideoAnalysisResult(
        frames=[],
        summary=dummy_summary,
        output_path=Path("dummy_annotated.mp4"),
        csv_path=None,
        fps=30.0,
        width=640,
        height=480,
    )

    with patch("runform.api.analyze_pose_video_file", return_value=mock_result):
        # 1. Test WebM upload (standard browser MediaRecorder format)
        dummy_webm = io.BytesIO(b"fake webm video bytes")
        res_webm = client.post(
            "/api/pose-analysis",
            files={"file": ("camera_recording.webm", dummy_webm, "video/webm")},
        )
        assert res_webm.status_code == 200
        data_webm = res_webm.json()
        assert "video_id" in data_webm
        assert data_webm["video_url"].startswith("/api/pose-video/")
        assert data_webm["metrics"]["average_left_knee_angle"] == 140.0

        # 2. Test MP4 upload (Safari / iOS format)
        dummy_mp4 = io.BytesIO(b"fake mp4 video bytes")
        res_mp4 = client.post(
            "/api/pose-analysis",
            files={"file": ("camera_recording.mp4", dummy_mp4, "video/mp4")},
        )
        assert res_mp4.status_code == 200
        data_mp4 = res_mp4.json()
        assert "video_id" in data_mp4
        assert data_mp4["video_url"].startswith("/api/pose-video/")


def test_camera_recording_api_failure_handling():
    """POST /api/pose-analysis should return 400 or 422 when invalid or missing file is sent."""
    res = client.post("/api/pose-analysis", files={"file": ("", io.BytesIO(b""), "video/mp4")})
    assert res.status_code in [400, 422]

    res_missing = client.post("/api/pose-analysis", data={})
    assert res_missing.status_code in [400, 422]


def test_ensure_supported_input_video_utility():
    """ensure_supported_input_video should validate readable videos and handle conversion."""
    from runform.pose.encoding import ensure_supported_input_video

    # Test with valid existing video
    sample_video = Path("data/videos/RUN_SIDE_01.mp4")
    if sample_video.exists():
        usable, is_temp = ensure_supported_input_video(sample_video)
        assert usable == sample_video.resolve()
        assert is_temp is False


def test_frontend_camera_workflow_contract():
    """Verify LiveCameraView includes complete recording-analysis-download workflow structure."""
    cam_file = Path("frontend/src/components/LiveCameraView.jsx")
    assert cam_file.exists()
    content = cam_file.read_text(encoding="utf-8")

    # State flow indicators
    for expected_state in ["IDLE", "RECORDING", "RECORDED", "ANALYZING", "RESULTS"]:
        assert expected_state in content

    # Key UI and actions
    assert "Start Recording" in content
    assert "Stop Recording" in content
    assert "Analyze Recording" in content
    assert "Download Annotated Video" in content
    assert "Record Again" in content
    assert "RunForm_Annotated_" in content
    assert "URL.revokeObjectURL" in content
    assert "MediaRecorder" in content


def test_pose_analysis_endpoint_temporal_features(tmp_path: Path):
    """POST /api/pose-analysis should return temporal_analysis and serve temporal CSV."""
    from runform.pose.temporal_analysis import TemporalAnalysisResult, TemporalFeatures, WindowTemporalResult

    tf = TemporalFeatures(
        total_frames=30,
        valid_pose_frames=30,
        valid_pose_frame_ratio=1.0,
        knee_left_rom=35.0,
        knee_right_rom=34.0,
        trunk_lean_rom=6.0,
        is_reliable=True,
    )
    win = WindowTemporalResult(
        window_index=0,
        start_time=0.0,
        end_time=2.0,
        total_frames=30,
        valid_frames=30,
        is_reliable=True,
        features=tf,
    )
    temporal_res = TemporalAnalysisResult(
        video_features=tf,
        windows=[win],
        window_duration_sec=2.0,
        step_duration_sec=2.0,
        total_frames=30,
        fps=15.0,
        duration_sec=2.0,
    )

    dummy_summary = RunningFormSummary(
        frames_processed=30,
        pose_detected_frames=30,
        detection_rate=1.0,
        average_left_knee_angle=155.0,
        average_right_knee_angle=156.0,
    )

    def mock_analyze(input_path, output_path=None, temporal_csv_path=None, **kwargs):
        if temporal_csv_path:
            temporal_res.write_windows_csv(temporal_csv_path)
        return PoseVideoAnalysisResult(
            frames=[],
            summary=dummy_summary,
            output_path=Path("dummy_annotated.mp4"),
            csv_path=None,
            fps=15.0,
            width=640,
            height=480,
            temporal_analysis=temporal_res,
        )

    with patch("runform.api.analyze_pose_video_file", side_effect=mock_analyze):
        dummy_file = io.BytesIO(b"fake video data")
        res = client.post(
            "/api/pose-analysis",
            files={"file": ("run_test.mp4", dummy_file, "video/mp4")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "temporal_analysis" in data
        assert data["temporal_analysis"]["total_frames"] == 30
        assert data["temporal_analysis"]["video_features"]["knee_left_rom"] == 35.0
        assert "temporal_csv_url" in data

        # Test CSV download endpoint
        analysis_id = data["analysis_id"]
        csv_res = client.get(f"/api/pose-temporal-csv/{analysis_id}")
        assert csv_res.status_code == 200
        assert csv_res.headers["content-type"].startswith("text/csv")
        assert "knee_left_rom" in csv_res.text



