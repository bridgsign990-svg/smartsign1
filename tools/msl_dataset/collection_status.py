from __future__ import annotations

import argparse
import json

if __package__:
    from .collection_management import collection_status
    from .msl_dataset_pipeline import load_config
else:
    from collection_management import collection_status
    from msl_dataset_pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MSL collection balance, review, and validation status.")
    parser.add_argument("--config", default="config/msl_dataset.json")
    parser.add_argument("--manifests-dir")
    parser.add_argument("--review-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(collection_status(
        args.manifests_dir or config["manifests_path"],
        args.review_dir or config["review_path"],
        args.dry_run,
        targets=config["collection_targets"],
    ), indent=2))


if __name__ == "__main__":
    main()
