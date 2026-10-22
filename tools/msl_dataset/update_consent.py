from __future__ import annotations

import argparse
import json

from .collection_management import update_consent


def main() -> None:
    parser = argparse.ArgumentParser(description="Update a registered signer's consent state.")
    parser.add_argument("--signer", required=True)
    parser.add_argument("--consent-status", required=True, choices=("PENDING", "GRANTED", "REVOKED"))
    parser.add_argument("--consent-reference", default="")
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(update_consent(
        args.signer, args.consent_status, args.consent_reference, args.manifests_dir, args.dry_run
    ), indent=2))


if __name__ == "__main__":
    main()
