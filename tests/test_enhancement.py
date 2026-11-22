"""Unit tests for video quality enhancement, unblur filters, and diagnostics."""

import base64
import io
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from runform.api import app
from runform.pose.enhancement import (
    EnhancementFilter,
    apply_auto_enhancement,
    apply_contrast_clahe,
    apply_denoise,
    apply_low_light_boost,
    apply_unblur,
    compute_frame_quality,
    enhance_frame,
)
from runform.pose.metrics import RunningFormSummary
from runform.pose.service import PoseVideoAnalysisResult

client = TestClient(app)


@pytest.fixture
def sharp_image() -> np.ndarray:
    """Create a high-contrast pattern image with sharp edges."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[20:80, 20:80] = (255, 255, 255)
    img[120:180, 120:180] = (255, 255, 255)
    cv2.line(img, (0, 0), (200, 200), (0, 255, 0), 3)
    cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
    return img


@pytest.fixture
def blurred_image(sharp_image: np.ndarray) -> np.ndarray:
    """Create a heavily blurred version of the sharp image."""
    return cv2.GaussianBlur(sharp_image, (21, 21), sigmaX=7.0)


@pytest.fixture
def dark_image() -> np.ndarray:
    """Create an underexposed low-light image."""
    return np.full((150, 150, 3), 30, dtype=np.uint8)


def test_compute_frame_quality_sharp_vs_blur(sharp_image: np.ndarray, blurred_image: np.ndarray):
    """compute_frame_quality should report significantly higher blur_score for sharp image."""
    sharp_q = compute_frame_quality(sharp_image)
    blur_q = compute_frame_quality(blurred_image)

    assert sharp_q.blur_score > blur_q.blur_score
    assert blur_q.is_blurry is True
    assert isinstance(sharp_q.to_dict(), dict)
    assert "blur_score" in sharp_q.to_dict()


def test_compute_frame_quality_low_light(dark_image: np.ndarray):
    """compute_frame_quality should detect low brightness and flag is_low_light."""
    q = compute_frame_quality(dark_image)
    assert q.is_low_light is True
    assert q.brightness < 50.0


def test_compute_frame_quality_empty_frame():
    """Empty or None frames should not raise exceptions."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    q = compute_frame_quality(empty)
    assert q.is_blurry is True
    assert q.brightness == 0.0


def test_apply_unblur_increases_sharpness(blurred_image: np.ndarray):
    """apply_unblur should noticeably increase Laplacian variance on blurred images."""
    init_score = compute_frame_quality(blurred_image).blur_score
    unblurred = apply_unblur(blurred_image, strength=1.5)
    new_score = compute_frame_quality(unblurred).blur_score

    assert unblurred.shape == blurred_image.shape
    assert unblurred.dtype == np.uint8
    assert new_score > init_score, f"Expected higher blur score, got {new_score} vs {init_score}"


def test_apply_contrast_clahe_boosts_contrast():
    """apply_contrast_clahe should improve contrast on flat low-contrast frames."""
    flat = np.full((120, 120, 3), 100, dtype=np.uint8)
    flat[30:90, 30:90] = 110  # very low contrast (std dev is tiny)

    init_contrast = compute_frame_quality(flat).contrast
    enhanced = apply_contrast_clahe(flat, clip_limit=3.0)
    new_contrast = compute_frame_quality(enhanced).contrast

    assert enhanced.shape == flat.shape
    assert new_contrast > init_contrast


def test_apply_low_light_boost_brightens(dark_image: np.ndarray):
    """apply_low_light_boost should increase mean luminance."""
    init_bright = compute_frame_quality(dark_image).brightness
    brightened = apply_low_light_boost(dark_image, gamma=1.5, gain=1.2)
    new_bright = compute_frame_quality(brightened).brightness

    assert new_bright > init_bright


def test_apply_denoise():
    """apply_denoise should filter out pixel noise."""
    noisy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    denoised = apply_denoise(noisy, strength=1.2)
    assert denoised.shape == noisy.shape
    assert denoised.dtype == np.uint8


def test_apply_auto_enhancement_runs(blurred_image: np.ndarray, dark_image: np.ndarray):
    """apply_auto_enhancement should adjust both blurry and dark frames safely."""
    enh1 = apply_auto_enhancement(blurred_image)
    assert enh1.shape == blurred_image.shape

    enh2 = apply_auto_enhancement(dark_image)
    assert enh2.shape == dark_image.shape
    assert compute_frame_quality(enh2).brightness > compute_frame_quality(dark_image).brightness


def test_enhance_frame_dispatcher(sharp_image: np.ndarray):
    """enhance_frame should route correctly across all filter types."""
    for f in [
        EnhancementFilter.NONE,
        EnhancementFilter.AUTO,
        EnhancementFilter.UNBLUR,
        EnhancementFilter.CONTRAST,
        EnhancementFilter.LOW_LIGHT,
        EnhancementFilter.DENOISE,
        "unknown_filter",
    ]:
        res = enhance_frame(sharp_image, filter_type=f, strength=1.0)
        assert res.shape == sharp_image.shape
        assert res.dtype == np.uint8


def test_pose_enhance_preview_endpoint(blurred_image: np.ndarray):
    """POST /api/pose-enhance-preview should return enhanced preview and metrics."""
    _, buf = cv2.imencode(".jpg", blurred_image)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    res = client.post(
        "/api/pose-enhance-preview",
        json={"image_base64": b64, "filter_type": "unblur", "strength": 1.2},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["filter_type"] == "unblur"
    assert "enhanced_image" in data
    assert data["enhanced_image"].startswith("data:image/jpeg;base64,")
    assert "initial_quality" in data
    assert "enhanced_quality" in data
    assert "sharpness_improvement_pct" in data


def test_pose_analysis_endpoint_supports_enhancement_params():
    """POST /api/pose-analysis should accept enhancement_filter and enhancement_strength."""
    dummy_summary = RunningFormSummary(
        frames_processed=10,
        pose_detected_frames=10,
        detection_rate=1.0,
    )
    mock_result = PoseVideoAnalysisResult(
        frames=[],
        summary=dummy_summary,
        output_path=Path("dummy_annotated.mp4"),
        csv_path=None,
        fps=30.0,
        width=640,
        height=480,
        quality_info={
            "enhancement_filter": "unblur",
            "enhancement_strength": 1.2,
            "applied": True,
            "initial_blur_score": 45.2,
            "enhanced_blur_score": 68.4,
            "sharpness_improvement_pct": 51.3,
        },
    )

    with patch("runform.api.analyze_pose_video_file", return_value=mock_result):
        dummy_file = io.BytesIO(b"fake video")
        res = client.post(
            "/api/pose-analysis",
            files={"file": ("run_test.mp4", dummy_file, "video/mp4")},
            data={"enhancement_filter": "unblur", "enhancement_strength": "1.2"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "quality_info" in data
        assert data["quality_info"]["enhancement_filter"] == "unblur"
        assert data["quality_info"]["sharpness_improvement_pct"] == 51.3
