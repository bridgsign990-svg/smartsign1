#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .msl_dataset_pipeline import build_review_html
else:
    from msl_dataset_pipeline import build_review_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple HTML QC report for an MSL sequence.")
    parser.add_argument("--sequence", type=str, required=True, help="Path to the sequence .npz file")
    parser.add_argument("--output", type=str, default=None, help="Optional HTML output path")
    args = parser.parse_args()
    sequence_path = Path(args.sequence)
    html = build_review_html(sequence_path)
    output_path = Path(args.output) if args.output else sequence_path.with_suffix(".html")
    output_path.write_text(html, encoding="utf-8")
    print(json.dumps({"output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
