from __future__ import annotations

import argparse
import json

from .collection_management import upsert_vocabulary_sign


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a manually defined MSL entry to the controlled vocabulary.")
    parser.add_argument("--sign-id", required=True)
    parser.add_argument("--canonical-name", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    args = parser.parse_args()
    print(json.dumps(upsert_vocabulary_sign(
        args.sign_id, args.canonical_name, args.display_name, args.description, args.manifests_dir
    ), indent=2))


if __name__ == "__main__":
    main()
