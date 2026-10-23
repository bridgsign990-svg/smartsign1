from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from .import_existing_recordings import update_existing_recording_review
else:
    from import_existing_recordings import update_existing_recording_review


def _read_rows(manifests_dir: str) -> list[dict[str, str]]:
    path = Path(manifests_dir) / "existing_recordings.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Existing-recording manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _display_recording(row: dict[str, str], index: int, total: int) -> None:
    print(f"\n[{index}/{total}] {row.get('recording_id', '')}")
    for label, field in (
        ("Video", "filename"),
        ("Path", "original_path"),
        ("Sign candidate (not validated)", "sign_candidate"),
        ("Signer ID / candidate", "signer_id"),
        ("Session ID", "session_id"),
        ("Duration seconds", "duration_seconds"),
        ("FPS", "fps"),
        ("Resolution", "resolution"),
        ("Frames", "frame_count"),
        ("Hand detection %", "hand_detected_percentage"),
        ("Left hand %", "left_hand_detected_percentage"),
        ("Right hand %", "right_hand_detected_percentage"),
        ("Technical QC", "technical_qc_status"),
        ("Signer validation", "signer_validation_status"),
        ("Consent", "consent_status"),
        ("Session validation", "session_validation_status"),
        ("Label validation", "label_validation_state"),
        ("Language validation", "language_validation_state"),
        ("Quality decision", "recording_quality_status"),
        ("Human review", "human_review_status"),
        ("Processing eligibility", "validation_status"),
    ):
        print(f"{label}: {row.get(field) or 'UNKNOWN'}")


def _prompt_decision(label: str, current: str, choices: tuple[str, ...]) -> str | None:
    options = "/".join(choices)
    answer = input(f"{label} [{current or 'unset'}] ({options}; Enter keeps): ").strip().upper()
    if not answer:
        return None
    if answer not in choices:
        raise ValueError(f"{label} must be one of {options}.")
    return answer


def _interactive_review(args: argparse.Namespace) -> None:
    rows = _read_rows(args.manifests_dir)
    if args.recording_id:
        rows = [row for row in rows if row.get("recording_id") == args.recording_id]
        if not rows:
            raise ValueError(f"Recording {args.recording_id} was not found.")
    for index, row in enumerate(rows, start=1):
        _display_recording(row, index, len(rows))
        input("Open/play the video at the path above, then press Enter to enter decisions...")
        signer_decision = _prompt_decision(
            "Signer", row.get("signer_review_status", "UNKNOWN"),
            ("CONFIRMED", "REJECTED", "UNKNOWN"),
        )
        session_decision = _prompt_decision(
            "Session", row.get("session_review_status", "UNKNOWN"),
            ("CONFIRMED", "REJECTED", "UNKNOWN"),
        )
        label_decision = _prompt_decision(
            "Sign label", row.get("label_review_status", "NEEDS_REVIEW"),
            ("CONFIRMED", "REJECTED", "NEEDS_REVIEW"),
        )
        language_decision = _prompt_decision(
            "Language", row.get("language_review_status", "UNKNOWN"),
            ("MSL", "NOT_MSL", "UNKNOWN"),
        )
        quality_decision = _prompt_decision(
            "Recording quality", row.get("recording_quality_status", "UNKNOWN"),
            ("APPROVED", "REJECTED", "NEEDS_RETAKE"),
        )
        human_decision = _prompt_decision(
            "Human review", row.get("human_review_status", "NEEDS_REVIEW"),
            ("APPROVED", "REJECTED", "NEEDS_REVIEW"),
        )
        if not any((
            signer_decision,
            session_decision,
            label_decision,
            language_decision,
            quality_decision,
            human_decision,
        )):
            print("No decision changed; existing review record preserved.")
            continue
        confirmed_signer = None
        if signer_decision == "CONFIRMED":
            confirmed_signer = input(
                f"Confirmed registered signer ID [{row.get('signer_candidate', '')}]: "
            ).strip() or row.get("signer_candidate", "")
        confirmed_session = None
        if session_decision == "CONFIRMED":
            confirmed_session = input(
                f"Confirmed registered session ID [{row.get('session_id', '')}]: "
            ).strip() or row.get("session_id", "")
        confirmed_sign = None
        if label_decision == "CONFIRMED":
            confirmed_sign = input(
                f"Confirmed vocabulary sign [{row.get('sign_candidate', '')}]: "
            ).strip() or row.get("sign_candidate", "")
        note = input("Evidence/reviewer note (required for confirmations): ").strip()
        result = update_existing_recording_review(
            row["recording_id"],
            None,
            args.reviewer,
            note,
            args.manifests_dir,
            args.review_dir,
            signer_review_status=signer_decision,
            confirmed_signer_id=confirmed_signer,
            session_review_status=session_decision,
            confirmed_session_id=confirmed_session,
            label_review_status=label_decision,
            confirmed_sign=confirmed_sign,
            language_review_status=language_decision,
            recording_quality_status=quality_decision,
            human_review_status=human_decision,
        )
        print(json.dumps(result, indent=2))


def _save_decisions(args: argparse.Namespace) -> dict[str, str]:
    return update_existing_recording_review(
        args.recording_id,
        args.status,
        args.reviewer,
        args.note,
        args.manifests_dir,
        args.review_dir,
        signer_review_status=args.signer_decision,
        confirmed_signer_id=args.confirmed_signer,
        session_review_status=args.session_decision,
        confirmed_session_id=args.confirmed_session,
        label_review_status=args.label_decision,
        confirmed_sign=args.confirmed_sign,
        language_review_status=args.language_decision,
        recording_quality_status=args.quality_decision,
        human_review_status=args.human_review_decision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review imported recordings sequentially or save explicit human validation decisions."
    )
    parser.add_argument("--recording-id", help="Review one recording; omit for all records with --interactive.")
    parser.add_argument("--list", action="store_true", help="Print QC and current human status for all recordings.")
    parser.add_argument("--interactive", action="store_true", help="Review videos sequentially; Enter preserves each current decision.")
    parser.add_argument("--status", choices=("PENDING", "APPROVED", "REJECTED", "NEEDS_RETAKE"))
    parser.add_argument("--reviewer", help="Non-identifying reviewer ID.")
    parser.add_argument("--note", default="")
    parser.add_argument("--signer-decision", choices=("CONFIRMED", "REJECTED", "UNKNOWN"))
    parser.add_argument("--confirmed-signer")
    parser.add_argument("--session-decision", choices=("CONFIRMED", "REJECTED", "UNKNOWN"))
    parser.add_argument("--confirmed-session")
    parser.add_argument("--label-decision", choices=("CONFIRMED", "REJECTED", "NEEDS_REVIEW"))
    parser.add_argument("--confirmed-sign")
    parser.add_argument("--language-decision", choices=("MSL", "NOT_MSL", "UNKNOWN"))
    parser.add_argument("--quality-decision", choices=("APPROVED", "REJECTED", "NEEDS_RETAKE"))
    parser.add_argument("--human-review-decision", choices=("APPROVED", "REJECTED", "NEEDS_REVIEW"))
    parser.add_argument("--manifests-dir", default="data/msl_dataset/manifests")
    parser.add_argument("--review-dir", default="data/msl_dataset/review")
    args = parser.parse_args()

    if args.list:
        rows = _read_rows(args.manifests_dir)
        for index, row in enumerate(rows, start=1):
            _display_recording(row, index, len(rows))
        return
    if args.interactive:
        if not args.reviewer:
            parser.error("--reviewer is required with --interactive.")
        _interactive_review(args)
        return
    if not args.recording_id or not args.reviewer:
        parser.error("--recording-id and --reviewer are required unless --list is used.")
    result = _save_decisions(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
