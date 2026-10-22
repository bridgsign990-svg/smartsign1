from __future__ import annotations

import argparse
import json

from .collection_management import set_sign_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Record human validation of an MSL vocabulary entry.")
    parser.add_argument("--sign-id", required=True)
    parser.add_argument("--status", required=True, choices=("UNVALIDATED", "REVIEWED", "VALIDATED"))
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--evidence-note", default="")
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    args = parser.parse_args()
    print(json.dumps(set_sign_validation(
        args.sign_id, args.status, args.reviewer, args.evidence_note, args.manifests_dir
    ), indent=2))


if __name__ == "__main__":
    main()
