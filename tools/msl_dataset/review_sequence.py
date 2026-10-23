from __future__ import annotations

import argparse
import json

from .collection_management import review_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a human decision for one generated sequence.")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--status", required=True, choices=("PENDING", "APPROVED", "REJECTED", "NEEDS_RETAKE"))
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--review-dir", default="data/msl_dataset/review")
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    args = parser.parse_args()
    print(json.dumps(review_sequence(
        args.sequence_id, args.status, args.reviewer, args.note, args.review_dir, args.manifests_dir
    ), indent=2))


if __name__ == "__main__":
    main()
