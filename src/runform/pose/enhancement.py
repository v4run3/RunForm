"""Video quality enhancement and pre-processing filters for RunForm.

Provides high-performance, edge-preserving image processing routines to improve
pose estimation accuracy on challenging running videos (motion blur, low light,
poor contrast, sensor noise).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import cv2
import numpy as np


class EnhancementFilter(str, enum.Enum):
    """Supported video enhancement filter types."""

    NONE = "none"
    AUTO = "auto"
    UNBLUR = "unblur"
    CONTRAST = "contrast"
    LOW_LIGHT = "low_light"
    DENOISE = "denoise"


@dataclass
class QualityMetrics:
    """Quantitative image quality measurements for a frame."""

    blur_score: float  # Variance of Laplacian (higher = sharper, < 100 often blurry)
    is_blurry: bool
    brightness: float  # Mean luminance (0-255)
    contrast: float  # Standard deviation of luminance
    is_low_light: bool
    is_low_contrast: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "blur_score": round(self.blur_score, 2),
            "is_blurry": self.is_blurry,
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "is_low_light": self.is_low_light,
            "is_low_contrast": self.is_low_contrast,
        }


def compute_frame_quality(frame: np.ndarray) -> QualityMetrics:
    """Analyze quality metrics of an input frame (blur, brightness, contrast)."""
    if frame is None or frame.size == 0:
        return QualityMetrics(
            blur_score=0.0,
            is_blurry=True,
            brightness=0.0,
            contrast=0.0,
            is_low_light=True,
            is_low_contrast=True,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # 1. Sharpness / Blur score via variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    is_blurry = blur_score < 110.0

    # 2. Brightness & contrast via luminance statistics
    mean_val, std_val = cv2.meanStdDev(gray)
    brightness = float(mean_val[0][0])
    contrast = float(std_val[0][0])

    is_low_light = brightness < 75.0
    is_low_contrast = contrast < 35.0

    return QualityMetrics(
        blur_score=blur_score,
        is_blurry=is_blurry,
        brightness=brightness,
        contrast=contrast,
        is_low_light=is_low_light,
        is_low_contrast=is_low_contrast,
    )


def apply_unblur(
    frame: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Unblur and sharpen frame using unsharp masking with edge-preserving blending.

    Args:
        frame: BGR input image.
        strength: Sharpening intensity multiplier (0.1 to 2.5, default 1.0).
    """
    if frame is None or frame.size == 0 or strength <= 0.0:
        return frame

    # Clamp strength
    strength = max(0.1, min(strength, 3.0))

    # Unsharp mask: sharpened = original + strength * (original - blurred)
    # Using Gaussian blur kernel proportional to resolution
    h, w = frame.shape[:2]
    k_size = 5 if min(h, w) >= 480 else 3
    blurred = cv2.GaussianBlur(frame, (k_size, k_size), sigmaX=1.5)

    # Weighted add to sharpen without color clipping artifacts
    sharpened = cv2.addWeighted(frame, 1.0 + strength * 0.8, blurred, -strength * 0.8, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_contrast_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on luminance.

    Preserves color hues and saturation while boosting local contrast for joint definition.
    """
    if frame is None or frame.size == 0:
        return frame

    if len(frame.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(frame)

    # Convert to LAB color space to operate solely on L (luminance) channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_low_light_boost(
    frame: np.ndarray,
    gamma: float = 1.35,
    gain: float = 1.15,
) -> np.ndarray:
    """Lift shadow details in underexposed videos using gamma correction and CLAHE."""
    if frame is None or frame.size == 0:
        return frame

    # Build gamma lookup table for O(1) pixel transformation
    inv_gamma = 1.0 / max(0.2, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 * gain for i in np.arange(0, 256)]).astype("uint8")
    table = np.clip(table, 0, 255).astype("uint8")

    brightened = cv2.LUT(frame, table)
    # Apply mild CLAHE to retain local edge contrast
    return apply_contrast_clahe(brightened, clip_limit=1.8, tile_grid_size=(8, 8))


def apply_denoise(
    frame: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Denoise video frame using edge-preserving bilateral filtering."""
    if frame is None or frame.size == 0 or strength <= 0.0:
        return frame

    d = 7 if strength <= 1.0 else 9
    sigma_color = int(35 * strength)
    sigma_space = int(35 * strength)

    return cv2.bilateralFilter(frame, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_auto_enhancement(
    frame: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Automatically assess frame metrics and selectively apply required enhancements."""
    if frame is None or frame.size == 0:
        return frame

    metrics = compute_frame_quality(frame)
    output = frame.copy()

    # If underexposed or dim
    if metrics.is_low_light:
        output = apply_low_light_boost(output, gamma=1.25 * strength, gain=1.1)

    # If low contrast
    if metrics.is_low_contrast and not metrics.is_low_light:
        output = apply_contrast_clahe(output, clip_limit=2.0 * strength)

    # If blurry or soft
    if metrics.is_blurry:
        output = apply_unblur(output, strength=1.1 * strength)
    else:
        # Mild sharpening to clarify pose contours
        output = apply_unblur(output, strength=0.4 * strength)

    return output


def enhance_frame(
    frame: np.ndarray,
    filter_type: str | EnhancementFilter = EnhancementFilter.NONE,
    strength: float = 1.0,
) -> np.ndarray:
    """Enhance a frame using the specified filter mode.

    Args:
        frame: BGR image.
        filter_type: 'none', 'auto', 'unblur', 'contrast', 'low_light', 'denoise'.
        strength: Multiplier (0.1 to 2.5).

    Returns:
        Enhanced BGR frame.
    """
    if frame is None or frame.size == 0:
        return frame

    f_type = str(filter_type).lower().strip()

    if f_type in ("none", "off", "", "null"):
        return frame

    if f_type in ("unblur", "sharpen", "deblur"):
        return apply_unblur(frame, strength=strength)

    if f_type in ("contrast", "clahe"):
        clip_limit = max(1.0, min(2.5 * strength, 5.0))
        return apply_contrast_clahe(frame, clip_limit=clip_limit)

    if f_type in ("low_light", "brighten", "night"):
        gamma = 1.0 + (0.4 * strength)
        return apply_low_light_boost(frame, gamma=gamma)

    if f_type in ("denoise", "smooth"):
        return apply_denoise(frame, strength=strength)

    if f_type in ("auto", "adaptive"):
        return apply_auto_enhancement(frame, strength=strength)

    # Default fallback
    return frame
