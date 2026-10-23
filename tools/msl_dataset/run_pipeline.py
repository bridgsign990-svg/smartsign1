#!/usr/bin/env python3
from __future__ import annotations

import argparse

if __package__:
    from .msl_dataset_pipeline import (
        generate_dataset_report,
        generate_split_manifest,
        load_config,
        prepare_msl_dataset,
        validate_dataset,
    )
else:
    from msl_dataset_pipeline import (
        generate_dataset_report,
        generate_split_manifest,
        load_config,
        prepare_msl_dataset,
        validate_dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BridgeSign MSL dataset pipeline.")
    parser.add_argument("--config", type=str, default="config/msl_dataset.json")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Plan preparation without writing outputs")
    args = parser.parse_args()
    tasks = [args.prepare, args.validate, args.split, args.report, args.all]
    if not any(tasks):
        args.all = True
    config = load_config(args.config)
    manifests_dir = config.get("manifests_path", "data/msl_dataset/manifests")
    if args.prepare or args.all:
        result = prepare_msl_dataset(
            raw_dataset_path=args.raw_dir,
            config_path=args.config,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(json.dumps(result, indent=2))
    if args.validate or args.all:
        print("Validation result:")
        print(validate_dataset(manifests_dir))
    if args.split or args.all:
        generate_split_manifest(manifests_dir, config=config)
    if args.report or args.all:
        report_path = generate_dataset_report(manifests_dir, config=config)
        print(f"Dataset report written to {report_path}")


if __name__ == "__main__":
    main()
