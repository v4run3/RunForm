"""Configuration loading for RunForm.

Reads a YAML config (see ``configs/default.yaml``) into a dict, falling
back to sane defaults so the package never crashes without a config file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - config loading degrades gracefully
    yaml = None

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

DEFAULTS: dict[str, Any] = {
    "detection": {"model": "yolov8n-pose.pt", "conf": 0.5, "imgsz": 640},
    "footstrike": {"method": "rule_based", "min_stance_frames": 3},
    "angles": {"knee_flexion_risk_deg": 160, "hip_rotation_risk_deg": 15},
    "tracking": {"reid_buffer_frames": 30},
    "feedback": {"rules_path": "configs/risk_rules.yaml"},
    "app": {"title": "RunForm — Running Form Analysis", "icon": "\U0001F3C3"},
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from ``path`` (or the default YAML) merged over DEFAULTS.

    Args:
        path: Optional path to a YAML config file.

    Returns:
        Nested config dict. Every top-level section from :data:`DEFAULTS`
        is guaranteed to be present.
    """
    merged = dict(DEFAULTS)

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if yaml is not None and config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        merged.update({k: v for k, v in loaded.items() if v is not None})

    return merged