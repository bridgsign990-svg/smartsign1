from __future__ import annotations

import argparse
import json

from .collection_management import register_signer


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an MSL signer without collecting unnecessary personal data.")
    parser.add_argument("--signer", required=True)
    parser.add_argument("--consent-status", choices=("PENDING", "GRANTED"), default="PENDING")
    parser.add_argument("--consent-reference", default="", help="Non-identifying reference to the separately stored consent record.")
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = register_signer(
        args.signer,
        consent_status=args.consent_status,
        consent_reference=args.consent_reference,
        manifests_dir=args.manifests_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
