from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .collection_management import validate_recordings
    from .msl_dataset_pipeline import load_config
else:
    from collection_management import validate_recordings
    from msl_dataset_pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate consent, layout, duplicates, and video readability.")
    parser.add_argument("--config", default="config/msl_dataset.json")
    parser.add_argument("--raw-dir")
    parser.add_argument("--manifests-dir")
    parser.add_argument("--min-frames", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--min-fps", type=float)
    parser.add_argument("--max-fps", type=float)
    parser.add_argument("--min-width", type=int)
    parser.add_argument("--min-height", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    quality = config["quality_thresholds"]
    result = validate_recordings(
        args.raw_dir or config["raw_dataset_path"],
        args.manifests_dir or config["manifests_path"],
        min_frames=args.min_frames if args.min_frames is not None else quality["min_frames"],
        max_frames=args.max_frames if args.max_frames is not None else quality["max_frames"],
        min_fps=args.min_fps if args.min_fps is not None else quality["min_fps"],
        max_fps=args.max_fps if args.max_fps is not None else quality["max_fps"],
        min_width=args.min_width if args.min_width is not None else quality["min_width"],
        min_height=args.min_height if args.min_height is not None else quality["min_height"],
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    if result["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
