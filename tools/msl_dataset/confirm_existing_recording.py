from __future__ import annotations

import argparse
import json

if __package__:
    from .import_existing_recordings import confirm_existing_recording_metadata
else:
    from import_existing_recordings import confirm_existing_recording_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm imported video metadata after human review and consent checks."
    )
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--signer", required=True)
    parser.add_argument("--sign", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--signer-evidence", required=True)
    parser.add_argument("--session-evidence", required=True)
    parser.add_argument("--label-evidence", required=True)
    parser.add_argument("--language-evidence", required=True)
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    args = parser.parse_args()
    result = confirm_existing_recording_metadata(
        args.recording_id,
        args.signer,
        args.sign,
        args.session,
        args.language,
        args.signer_evidence,
        args.label_evidence,
        args.language_evidence,
        args.manifests_dir,
        session_evidence=args.session_evidence,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
