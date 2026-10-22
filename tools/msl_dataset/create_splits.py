#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

if __package__:
    from .msl_dataset_pipeline import generate_split_manifest, load_config
else:
    from msl_dataset_pipeline import generate_split_manifest, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create signer-separated train/validation/test splits.")
    parser.add_argument("--config", type=str, default="config/msl_dataset.json", help="Path to MSL dataset config")
    parser.add_argument("--manifests-dir", type=str, default="data/msl_dataset/manifests", help="Manifests directory")
    args = parser.parse_args()
    config = load_config(args.config)
    result = generate_split_manifest(args.manifests_dir, config=config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
