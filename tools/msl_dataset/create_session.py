from __future__ import annotations

import argparse
import json

from .collection_management import create_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a collection session for a consenting signer.")
    parser.add_argument("--signer", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--date", default="")
    parser.add_argument("--camera", default="UNKNOWN")
    parser.add_argument("--resolution", default="UNKNOWN")
    parser.add_argument("--fps", default="UNKNOWN")
    parser.add_argument("--lighting-condition", default="UNKNOWN")
    parser.add_argument("--background-condition", default="UNKNOWN")
    parser.add_argument("--orientation", default="UNKNOWN")
    parser.add_argument("--notes", default="")
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    metadata = {
        "date": args.date,
        "camera": args.camera,
        "resolution": args.resolution,
        "fps": args.fps,
        "lighting_condition": args.lighting_condition,
        "background_condition": args.background_condition,
        "orientation": args.orientation,
        "notes": args.notes,
    }
    print(json.dumps(create_session(
        args.signer, args.session, args.manifests_dir, args.dry_run, metadata
    ), indent=2))


if __name__ == "__main__":
    main()
