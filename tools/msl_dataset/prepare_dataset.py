#!/usr/bin/env python3
from __future__ import annotations

import argparse

if __package__:
    from .msl_dataset_pipeline import prepare_msl_dataset
else:
    from msl_dataset_pipeline import prepare_msl_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan and process raw MSL videos into landmark sequences.")
    parser.add_argument("--config", type=str, default="config/msl_dataset.json", help="Path to MSL dataset config")
    parser.add_argument("--raw-dir", type=str, default=None, help="Optional override for raw dataset path")
    parser.add_argument("--dry-run", action="store_true", help="Print the processing plan without writing outputs")
    args = parser.parse_args()
    result = prepare_msl_dataset(
        raw_dataset_path=args.raw_dir,
        config_path=args.config,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(result)
    else:
        print("MSL dataset preparation complete.")


if __name__ == "__main__":
    main()
