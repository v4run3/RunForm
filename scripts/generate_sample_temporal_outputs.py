"""Generate actual sample temporal feature outputs from existing running videos with quality validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from runform.pose.service import analyze_pose_video_file

VIDEO_CONFIGS = [
    {"path": "data/videos/RUN_SIDE_01.mp4", "camera_view": "side"},
    {"path": "data/videos/RUN_SIDE_02.mp4", "camera_view": "side"},
    {"path": "data/videos/RUN_FRONT_01.mp4", "camera_view": "front"},
    {"path": "data/videos/RUN_REAR_01.mp4", "camera_view": "rear"},
]

OUTPUT_DIRS = [
    Path("data/features"),
    Path("data/output"),
]
for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def main():
    video_rows = []
    all_window_rows = []
    json_summaries = {}

    for item in VIDEO_CONFIGS:
        video_path = Path(item["path"])
        camera_view = item["camera_view"]

        if not video_path.exists():
            print(f"Warning: {video_path} not found, skipping.")
            continue

        video_id = video_path.stem
        print(f"Processing {video_id} ({video_path}) [camera_view={camera_view}] ...")
        res = analyze_pose_video_file(
            input_path=video_path,
            output_path=None,
            window_sec=2.0,
            step_sec=2.0,
            camera_view=camera_view,
            filter_implausible=True,
        )

        temporal_res = res.temporal_analysis
        if temporal_res is None:
            print(f"No temporal analysis returned for {video_id}")
            continue

        v_row = temporal_res.to_video_csv_row(video_id=video_id)
        video_rows.append(v_row)

        w_rows = temporal_res.to_window_csv_rows(video_id=video_id)
        all_window_rows.extend(w_rows)

        json_summaries[video_id] = {
            "camera_view": camera_view,
            "summary": {
                "frames_processed": res.summary.frames_processed,
                "pose_detected_frames": res.summary.pose_detected_frames,
                "detection_rate": round(res.summary.detection_rate, 4),
            },
            "temporal_analysis": temporal_res.to_dict(),
        }

    # Write files to both output directories
    for out_dir in OUTPUT_DIRS:
        video_csv_path = out_dir / "sample_temporal_video_features.csv"
        window_csv_path = out_dir / "sample_temporal_window_features.csv"
        summary_json_path = out_dir / "sample_temporal_summary.json"

        # Write video-level CSV
        if video_rows:
            fieldnames = list(video_rows[0].keys())
            with video_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(video_rows)
            print(f"Saved video-level CSV: {video_csv_path} ({len(video_rows)} rows)")

        # Write window-level CSV
        if all_window_rows:
            fieldnames = list(all_window_rows[0].keys())
            with window_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_window_rows)
            print(f"Saved window-level CSV: {window_csv_path} ({len(all_window_rows)} rows)")

        # Write JSON summary
        with summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(json_summaries, f, indent=2)
        print(f"Saved JSON summary: {summary_json_path}")


if __name__ == "__main__":
    main()
