#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

if __package__:
    from .msl_dataset_pipeline import generate_dataset_report
else:
    from msl_dataset_pipeline import generate_dataset_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the MSL dataset summary and markdown report.")
    parser.add_argument("--manifests-dir", type=str, default="data/msl_dataset/manifests", help="Manifests directory")
    args = parser.parse_args()
    report_path = generate_dataset_report(args.manifests_dir)
    print(json.dumps({"report_path": report_path}, indent=2))


if __name__ == "__main__":
    main()
