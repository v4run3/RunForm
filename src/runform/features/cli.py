"""Command-line utility for exporting RunForm video features to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from runform.features.extractor import VideoFeatureVector, write_feature_csv
from runform.pipeline import RunFormPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one RunForm ML feature row per video.")
    parser.add_argument("videos", nargs="+", help="Input running videos to analyze.")
    parser.add_argument(
        "--output",
        default="data/features/runform_features.csv",
        help="Destination CSV path. Defaults to data/features/runform_features.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = RunFormPipeline()
    rows = []

    for video_path in args.videos:
        report = pipeline.analyze_video(video_path=video_path)
        features = report.metadata.get("ml_features")
        if features is None:
            raise RuntimeError(f"No ML features were generated for {video_path}")
        rows.append(VideoFeatureVector(**features))

    write_feature_csv(rows, Path(args.output))



if __name__ == "__main__":
    main()

