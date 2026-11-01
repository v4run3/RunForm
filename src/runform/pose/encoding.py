"""Video transcoding and browser-compatibility utilities for RunForm.

Ensures generated annotated videos are encoded using the universal H.264
(avc1) video codec with yuv420p pixel format and faststart flags, allowing
seamless playback across Chrome, Firefox, Safari, and Edge.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess

logger = logging.getLogger(__name__)


def find_ffmpeg_executable() -> str | None:
    """Locate a valid ffmpeg executable from imageio-ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return str(exe)
    except Exception:
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    return None


def convert_to_browser_mp4(
    input_path: str | Path,
    output_path: str | Path | None = None,
    crf: int = 22,
) -> Path:
    """Convert a video to browser-compatible H.264 MP4 with yuv420p pixel format.

    Args:
        input_path: Path to the input video (e.g. OpenCV mp4v output).
        output_path: Destination path. If None or same as input, transcode in-place.
        crf: Constant rate factor (default 22 for high visual quality).

    Returns:
        Path to the verified output file.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    target_path = input_path if output_path is None else Path(output_path).resolve()

    ffmpeg_exe = find_ffmpeg_executable()
    if not ffmpeg_exe:
        logger.warning(
            "FFmpeg executable not found. Video at %s cannot be transcoded to H.264. "
            "Keeping original OpenCV output.",
            input_path,
        )
        if target_path != input_path:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, target_path)
        return target_path

    same_file = input_path == target_path
    temp_target = (
        target_path.with_name(f".tmp_{target_path.stem}_{os.getpid()}.mp4")
        if same_file
        else target_path
    )
    temp_target.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(temp_target),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        if same_file:
            shutil.move(str(temp_target), str(target_path))
        return target_path
    except subprocess.CalledProcessError as err:
        stderr_msg = err.stderr.decode(errors="replace") if err.stderr else str(err)
        logger.error("FFmpeg transcoding failed for %s: %s", input_path, stderr_msg)
        if temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass
        if target_path != input_path and not target_path.exists():
            shutil.copy2(input_path, target_path)
        return target_path


def ensure_supported_input_video(input_path: str | Path) -> tuple[Path, bool]:
    """Ensure the input video can be read by OpenCV.

    If OpenCV cannot open the file or read frames (for example, non-standard WebM
    from MediaRecorder or unsupported container formats), transcode it to a
    standard temporary MP4 using FFmpeg if available.

    Returns:
        tuple[Path, bool]: (usable_path, is_temporary)
    """
    import cv2

    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    # Test if OpenCV can open and read at least one valid frame
    can_read = False
    cap = cv2.VideoCapture(str(input_path))
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                can_read = True
    cap.release()

    if can_read:
        return input_path, False

    # Try format conversion with FFmpeg
    ffmpeg_exe = find_ffmpeg_executable()
    if not ffmpeg_exe:
        raise ValueError(
            f"Video format ({input_path.suffix or 'unknown'}) could not be decoded by OpenCV, "
            "and FFmpeg is not available to convert the format."
        )

    converted_path = input_path.with_name(f"{input_path.stem}_converted_{os.getpid()}.mp4")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(converted_path),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        cap2 = cv2.VideoCapture(str(converted_path))
        if not cap2.isOpened():
            cap2.release()
            raise ValueError(f"Converted video for {input_path.name} cannot be read by OpenCV.")
        cap2.release()
        return converted_path, True
    except subprocess.CalledProcessError as err:
        stderr_msg = err.stderr.decode(errors="replace") if err.stderr else str(err)
        logger.error("FFmpeg input conversion failed: %s", stderr_msg)
        if converted_path.exists():
            try:
                converted_path.unlink()
            except OSError:
                pass
        raise ValueError(f"Failed to decode video file '{input_path.name}': {stderr_msg}") from err

