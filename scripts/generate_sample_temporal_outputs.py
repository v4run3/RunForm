"""Generate actual sample temporal feature outputs from existing running videos."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from runform.pose.service import analyze_pose_video_file

VIDEO_FILES = [
    "data/videos/RUN_SIDE_01.mp4",
    "data/videos/RUN_SIDE_02.mp4",
    "data/videos/RUN_FRONT_01.mp4",
    "data/videos/RUN_REAR_01.mp4",
]

OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_CSV_PATH = OUTPUT_DIR / "sample_temporal_video_features.csv"
WINDOW_CSV_PATH = OUTPUT_DIR / "sample_temporal_window_features.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "sample_temporal_summary.json"


def main():
    video_rows = []
    all_window_rows = []
    json_summaries = {}

    for video_rel in VIDEO_FILES:
        video_path = Path(video_rel)
        if not video_path.exists():
            print(f"Warning: {video_path} not found, skipping.")
            continue

        video_id = video_path.stem
        print(f"Processing {video_id} ({video_path}) ...")
        res = analyze_pose_video_file(
            input_path=video_path,
            output_path=None,
            window_sec=2.0,
            step_sec=2.0,
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
            "summary": {
                "frames_processed": res.summary.frames_processed,
                "pose_detected_frames": res.summary.pose_detected_frames,
                "detection_rate": round(res.summary.detection_rate, 4),
            },
            "temporal_analysis": temporal_res.to_dict(),
        }

    # Write video-level CSV
    if video_rows:
        fieldnames = list(video_rows[0].keys())
        with VIDEO_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(video_rows)
        print(f"Saved video-level CSV: {VIDEO_CSV_PATH} ({len(video_rows)} rows)")

    # Write window-level CSV
    if all_window_rows:
        fieldnames = list(all_window_rows[0].keys())
        with WINDOW_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_window_rows)
        print(f"Saved window-level CSV: {WINDOW_CSV_PATH} ({len(all_window_rows)} rows)")

    # Write JSON summary
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(json_summaries, f, indent=2)
    print(f"Saved JSON summary: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()

