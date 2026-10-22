#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

if __package__:
    from .msl_dataset_pipeline import validate_dataset
else:
    from msl_dataset_pipeline import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated MSL dataset manifests and outputs.")
    parser.add_argument("--manifests-dir", type=str, default="data/msl_dataset/manifests", help="Manifests directory")
    args = parser.parse_args()
    result = validate_dataset(args.manifests_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
