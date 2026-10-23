import cv2
import os
import time
import argparse
import csv
import math
import uuid
from datetime import datetime, timezone
import config
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor, FEATURE_VECTOR_SIZE

try:
    from word_signs import SIGN_TIPS, display_name
except ImportError:
    SIGN_TIPS    = {}
    display_name = lambda s: s.replace("_", " ")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BridgeSign Data Collector — collect training samples for letters and word signs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_collector.py --sign A                        # collect letter A (100 samples)
  python data_collector.py --sign HELP                     # collect word sign HELP (200 samples)
  python data_collector.py --sign B --samples 150

  # ── Batch mode (collect multiple signs in one session) ──────────
  python data_collector.py --batch J K M N P Q R S T U V X Y --samples 400
  python data_collector.py --batch A B C --samples 500
        """
    )
    parser.add_argument("--sign",    type=str, default=None,
                        help="Single sign label to collect. Use A-Z for letters, or a word like HELP.")
    parser.add_argument("--batch",   type=str, nargs="+", default=None,
                        help="Batch mode: list of signs to collect sequentially in one session.")
    parser.add_argument("--samples", type=int, default=None,
                        help="Target total samples per sign (default: 100 for letters, 200 for words). "
                             "In batch mode, existing samples are counted and only the remaining are collected.")
    parser.add_argument("--folder",  type=str, default=config.DATA_DIR,
                        help="Folder to save dataset.csv into.")
    parser.add_argument("--v2", action="store_true",
                        help="Collect static A-Z except motion letters J/Z to dataset_v2.csv with provenance.")
    parser.add_argument("--signer-id", type=str, default=None,
                        help="Signer identifier required with --v2.")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Collection session ID with --v2 (generated if omitted).")
    parser.add_argument("--device", type=str, default="camera-0",
                        help="Camera/source identifier to store with --v2 samples.")
    parser.add_argument("--priority-extra", type=int, default=200,
                        help="Additional total-dataset samples for B,M,P,Q,S,U,V,Z in V2 mode.")
    parser.add_argument("--session-samples", type=int, default=None,
                        help="Per-letter target for this signer/session (default V2 batch: up to 60).")
    parser.add_argument("--session-priority-extra", type=int, default=0,
                        help="Extra per-session target for B,M,P,Q,S,U,V,Z when --session-samples is set "
                             "(default V2 batch cap: 40).")
    args = parser.parse_args()

    if args.sign is None and args.batch is None:
        parser.error("You must specify either --sign or --batch.")
    if args.sign and args.batch:
        parser.error("Use --sign for a single sign or --batch for multiple, not both.")
    if args.v2:
        if not args.signer_id or not args.signer_id.strip():
            parser.error("--signer-id is required with --v2.")
        if args.session_id is not None and not args.session_id.strip():
            parser.error("--session-id cannot be blank.")
        if args.priority_extra < 0:
            parser.error("--priority-extra cannot be negative.")
        if args.samples is not None and args.samples < 1:
            parser.error("--samples must be positive.")
        if args.session_samples is not None and args.session_samples < 1:
            parser.error("--session-samples must be positive.")
        if args.session_priority_extra < 0:
            parser.error("--session-priority-extra cannot be negative.")

    return args


def is_letter_sign(sign: str) -> bool:
    """True if sign is a single alphabet letter."""
    return len(sign) == 1 and sign.upper().isalpha()


def count_existing_samples(save_path: str) -> dict:
    """Count existing samples per label in the dataset CSV."""
    counts = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    label = parts[0]
                    counts[label] = counts.get(label, 0) + 1
    return counts


V2_PRIORITY_LETTERS = frozenset({"B", "M", "P", "Q", "S", "U", "V"})
V2_MOTION_LETTERS = frozenset({"J", "Z"})
V2_VARIATION_PROMPTS = (
    "Keep the hand fully visible; vary its position slightly.",
    "Keep the sign clear; move a little nearer or farther from camera.",
    "Keep all fingertips in frame; try a small natural hand rotation.",
    "Keep the hand clear; vary background or lighting if practical.",
    "Keep the sign shape correct; allow natural signer variation.",
)
V2_FEATURE_FIELDS = [f"feature_{index:02d}" for index in range(FEATURE_VECTOR_SIZE)]
V2_DATASET_FIELDS = [
    "label", "signer_id", "session_id", "timestamp_utc", "source", "device",
    "frame_width", "frame_height", *V2_FEATURE_FIELDS,
]


def count_v2_samples(save_path: str, session_id: str | None = None) -> dict:
    """Count rows in the versioned dataset without touching dataset.csv."""
    counts = {}
    if not os.path.exists(save_path):
        return counts
    with open(save_path, "r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != V2_DATASET_FIELDS:
            raise ValueError(
                f"V2 dataset has an unexpected header: {save_path}. "
                "The collector will not rewrite it."
            )
        for row in reader:
            label = (row.get("label") or "").strip().upper()
            if label and (session_id is None or row.get("session_id") == session_id):
                counts[label] = counts.get(label, 0) + 1
    return counts


def _v2_capture_problem(detector, frame, extractor):
    """Return a rejection reason or a finite, valid 42-value feature vector."""
    results = detector.results
    hands = getattr(results, "hand_landmarks", None) if results else None
    if not hands:
        return "No hand detected.", None
    if len(hands) != 1:
        return "Exactly one visible hand is required.", None

    landmarks = hands[0]
    if len(landmarks) != 21:
        return "A complete 21-landmark hand was not detected.", None
    coords = [(float(point.x), float(point.y)) for point in landmarks]
    if not all(math.isfinite(value) for point in coords for value in point):
        return "Landmarks contain a non-finite coordinate.", None
    if any(x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0 for x, y in coords):
        return "Landmarks fall outside the camera frame.", None

    min_x = min(x for x, _ in coords)
    max_x = max(x for x, _ in coords)
    min_y = min(y for _, y in coords)
    max_y = max(y for _, y in coords)
    if max(max_x - min_x, max_y - min_y) < 0.12:
        return "Hand is too small in the frame; move closer.", None
    if min_x < 0.015 or min_y < 0.015 or max_x > 0.985 or max_y > 0.985:
        return "Hand is too close to the frame edge; reposition it.", None

    pixel_landmarks = detector.get_landmarks(frame, hand_no=0)
    features = extractor.extract_features(pixel_landmarks)
    if features is None or len(features) != FEATURE_VECTOR_SIZE:
        return "Feature extraction did not produce all 42 features.", None
    values = [float(value) for value in features]
    if not all(math.isfinite(value) for value in values):
        return "Extracted features contain a non-finite value.", None
    return None, values


def _append_v2_sample(save_path: str, sign: str, features: list[float],
                      metadata: dict, frame_shape) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.exists(save_path)
    if file_exists:
        with open(save_path, "r", newline="", encoding="utf-8-sig") as stream:
            if next(csv.reader(stream), None) != V2_DATASET_FIELDS:
                raise ValueError(f"V2 dataset header changed during collection: {save_path}")

    height, width = frame_shape[:2]
    row = {
        "label": sign,
        "signer_id": metadata["signer_id"],
        "session_id": metadata["session_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source": metadata["source"],
        "device": metadata["device"],
        "frame_width": width,
        "frame_height": height,
    }
    row.update(dict(zip(V2_FEATURE_FIELDS, features)))
    with open(save_path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=V2_DATASET_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def draw_overlay(frame, sign: str, count: int, target: int,
                 recording: bool, tip: str, existing: int = 0,
                 v2_mode: bool = False):
    """Draw a clean HUD onto the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    label_display = display_name(sign)

    if recording:
        # Progress bar background
        bar_w = int((count / target) * (w - 40)) if target > 0 else 0
        cv2.rectangle(frame, (20, h - 28), (w - 20, h - 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 28), (20 + bar_w, h - 10), (0, 200, 80), -1)

        total_label = f"  (total: {existing + count})" if existing > 0 else ""
        cv2.putText(frame, f"Recording: {count}/{target}{total_label}",
                    (20, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 80), 2)

        # Pulsing REC indicator
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (w - 28, 24), 10, (0, 0, 220), -1)
            cv2.putText(frame, "REC", (w - 60, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
    else:
        # Idle instructions
        instruction = ("Press [S] once to save a sample  [Q] to quit" if v2_mode
                       else "Press [S] to record  [Q] to quit")
        cv2.putText(frame, f"Sign: {label_display}  |  {instruction}",
                    (20, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Sign label top left
    cv2.putText(frame, f"TARGET: {label_display}", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 160, 40), 2)

    # Existing sample count top right
    if existing > 0:
        cv2.putText(frame, f"Existing: {existing}", (w - 200, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 1)

    # Tip text (wrapped) at bottom
    words = tip.split()
    line, lines = "", []
    for w_txt in words:
        if len(line + " " + w_txt) > 70:
            lines.append(line)
            line = w_txt
        else:
            line = (line + " " + w_txt).strip()
    if line:
        lines.append(line)
    for i, ln in enumerate(lines[-2:]):
        cv2.putText(frame, ln, (20, h - 120 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 200, 255), 1)

    return frame


def draw_transition_screen(frame, sign_done: str, sign_next: str,
                           collected: int, existing: int):
    """Draw a screen between batch signs telling user to get ready."""
    h, w = frame.shape[:2]
    screen = frame.copy()
    overlay = screen.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
    screen = cv2.addWeighted(overlay, 0.85, screen, 0.15, 0)

    cx = w // 2

    cv2.putText(screen, f"Done: {display_name(sign_done)}", (cx - 160, h // 2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 80), 2)
    cv2.putText(screen, f"{collected} new samples (total: {existing + collected})",
                (cx - 180, h // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(screen, f"Next: {display_name(sign_next)}", (cx - 160, h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 160, 40), 2)

    tip = SIGN_TIPS.get(sign_next, "Hold the sign naturally in front of the camera.")
    cv2.putText(screen, tip[:70], (cx - 220, h // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 200, 255), 1)

    cv2.putText(screen, "Press [S] to start  |  [N] to skip  |  [Q] to quit",
                (cx - 240, h // 2 + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 255), 1)

    return screen


def collect_one_sign(sign: str, target: int, save_path: str,
                     cap, detector, extractor, existing: int = 0,
                     v2_metadata: dict | None = None):
    """
    Collect samples for a single sign.

    Returns:
        int: number of new samples collected, or -1 if user quit (Q).
    """
    tip = SIGN_TIPS.get(sign, "Hold the sign naturally in front of the camera.")

    print(f"\n  ── Collecting: {display_name(sign)} ──")
    print(f"     Existing: {existing}  |  Target new: {target}")

    recording  = False
    count      = 0
    last_saved = 0
    SAVE_INTERVAL = 0.05  # seconds between saved samples (avoid near-duplicate frames)
    last_v2_attempt = 0.0
    v2_session_features = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Camera read failed — retrying…")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        frame, _ = detector.find_hands(frame, draw=True)

        if v2_metadata is not None:
            tip = V2_VARIATION_PROMPTS[(count // 25) % len(V2_VARIATION_PROMPTS)]
        else:
            tip = SIGN_TIPS.get(sign, "Hold the sign naturally in front of the camera.")

        if v2_metadata is None and recording and count < target:
            now = time.time()
            if now - last_saved >= SAVE_INTERVAL:
                lm_list = detector.get_landmarks(frame, hand_no=0)
                if lm_list:
                    features = extractor.extract_features(lm_list)
                    with open(save_path, "a") as f:
                        f.write(f"{sign}," + ",".join(map(str, features)) + "\n")
                    count     += 1
                    last_saved = now
                    if count % 25 == 0:
                        print(f"     {count}/{target} samples collected…")

            if count >= target:
                recording = False
                print(f"\n     ✅  Done! {target} new samples for '{display_name(sign)}'.")
                print(f"     Total for {sign}: {existing + count}")
                # Show completion screen briefly
                h, w = frame.shape[:2]
                msg_frame = frame.copy()
                cv2.rectangle(msg_frame, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.putText(msg_frame, "Collection Complete!", (w // 2 - 160, h // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 80), 2)
                cv2.putText(msg_frame, f"{count} new samples for {display_name(sign)} (total: {existing + count})",
                            (w // 2 - 220, h // 2 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(msg_frame, "Continuing in 2s...",
                            (w // 2 - 100, h // 2 + 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
                cv2.imshow("BridgeSign — Data Collector", msg_frame)
                cv2.waitKey(2000)
                return count

        frame = draw_overlay(
            frame, sign, count, target, recording, tip, existing,
            v2_mode=v2_metadata is not None,
        )
        cv2.imshow("BridgeSign — Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and v2_metadata is not None:
            now = time.monotonic()
            if now - last_v2_attempt < 0.35:
                continue
            last_v2_attempt = now
            problem, features = _v2_capture_problem(detector, frame, extractor)
            if problem:
                print(f"     Sample not saved: {problem}")
                continue
            feature_key = tuple(round(value, 8) for value in features)
            if feature_key in v2_session_features:
                print("     Sample not saved: identical to a sample already collected in this session.")
                continue
            _append_v2_sample(save_path, sign, features, v2_metadata, frame.shape)
            v2_session_features.add(feature_key)
            count += 1
            if count % 25 == 0 or count == target:
                print(f"     {count}/{target} samples collected; total for {sign}: {existing + count}")
            if count >= target:
                print(f"\n     Done! {count} new V2 samples for '{display_name(sign)}'.")
                return count
        elif key == ord("s") and not recording:
            recording = True
            count = 0
            print(f"     ▶  Recording started — sign '{display_name(sign)}' naturally…")
        elif key == ord("n"):
            # Skip this sign (batch mode)
            print(f"     ⏭  Skipped {display_name(sign)} ({count} collected this round)")
            return count
        elif key == ord("q"):
            print(f"     ⏹  Quit requested ({count} collected for {display_name(sign)})")
            return -1

    return count


def main():
    args      = parse_args()
    save_path = os.path.join(args.folder, "dataset_v2.csv" if args.v2 else "dataset.csv")

    # ── Build list of signs to collect ────────────────────────────────────
    if args.batch:
        signs = [s.upper() for s in args.batch]
    else:
        signs = [args.sign.upper()]

    if args.v2:
        motion_letters = [sign for sign in signs if sign in V2_MOTION_LETTERS]
        if motion_letters:
            raise SystemExit(
                "V2 static collection excludes motion letters "
                f"{', '.join(motion_letters)}; collect J and Z with the gesture sequence workflow."
            )
        non_letters = [sign for sign in signs if len(sign) != 1 or sign not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        if non_letters:
            raise SystemExit(f"V2 collection accepts static A-Z letters except J and Z; invalid: {', '.join(non_letters)}")
        session_id = args.session_id.strip() if args.session_id else uuid.uuid4().hex
        v2_metadata = {
            "signer_id": args.signer_id.strip(),
            "session_id": session_id,
            "source": "webcam",
            "device": args.device.strip() or f"camera-{config.CAMERA_INDEX}",
        }
        print(f"  V2 signer  : {v2_metadata['signer_id']}")
        print(f"  V2 session : {v2_metadata['session_id']}")
    else:
        v2_metadata = None

    batch_mode = len(signs) > 1

    # ── Count existing samples ────────────────────────────────────────────
    existing_counts = count_v2_samples(save_path) if args.v2 else count_existing_samples(save_path)
    current_counts = existing_counts.copy()
    current_session_counts = (
        count_v2_samples(save_path, v2_metadata["session_id"]) if args.v2 else {}
    )

    def target_for(sign):
        if args.v2:
            base_target = args.samples if args.samples is not None else 300
            return base_target + (args.priority_extra if sign in V2_PRIORITY_LETTERS else 0)
        letter = is_letter_sign(sign)
        default_target = 100 if letter else 200
        return args.samples if args.samples is not None else (
            default_target if not batch_mode else max(default_target, 400)
        )

    def session_target_for(sign):
        if args.session_samples is None:
            base_target = args.samples if args.samples is not None else 300
            base_session_target = min(base_target, 60)
            priority_session_target = (
                min(args.priority_extra, 40)
                if sign in V2_PRIORITY_LETTERS else 0
            )
            return base_session_target + priority_session_target
        return args.session_samples + (
            args.session_priority_extra if sign in V2_PRIORITY_LETTERS else 0
        )

    print("\n" + "=" * 62)
    print(f"  BridgeSign Data Collector {'(Batch Mode)' if batch_mode else ''}")
    print(f"  Signs    : {', '.join(display_name(s) for s in signs)}")
    print(f"  Save to  : {save_path}")
    print("=" * 62)

    if batch_mode:
        print(f"\n  📊 Current sample counts:")
        for s in signs:
            cur = existing_counts.get(s, 0)
            target = target_for(s)
            if args.v2:
                session_count = current_session_counts.get(s, 0)
                session_target = session_target_for(s)
                needed = min(max(0, target - cur), max(0, session_target - session_count))
                status = "✅ done" if needed == 0 else f"need {needed} in this session"
                print(f"     {display_name(s):12s}  total {cur:4d}/{target}, "
                      f"session {session_count:4d}/{session_target} ({status})")
            else:
                needed = max(0, target - cur)
                status = "✅ done" if needed == 0 else f"need {needed} more"
                print(f"     {display_name(s):12s}  {cur:4d} / {target}  ({status})")
        print()

    print("  Controls: [S] save one sample  [N] skip sign  [Q] quit\n" if args.v2
          else "  Controls: [S] start recording  [N] skip sign  [Q] quit\n")

    # ── Init camera + detector ────────────────────────────────────────────
    detector  = HandDetector(max_hands=2 if args.v2 else 1)
    extractor = FeatureExtractor()
    cap       = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.FPS)

    total_collected = {}  # {sign: new_count}

    try:
        for idx, sign in enumerate(signs):
            overall_target = target_for(sign)
            existing = current_counts.get(sign, 0)
            if args.v2:
                session_count = current_session_counts.get(sign, 0)
                needed = min(
                    max(0, overall_target - existing),
                    max(0, session_target_for(sign) - session_count),
                )
            else:
                needed = max(0, overall_target - existing)

            if needed == 0:
                if args.v2:
                    print(
                        f"\n  ⏭  {display_name(sign)} has total {existing}/{overall_target}; "
                        f"current session has {session_count}/{session_target_for(sign)} — skipping"
                    )
                else:
                    print(f"\n  ⏭  {display_name(sign)} already has {existing} samples (target: {overall_target}) — skipping")
                total_collected[sign] = 0
                continue

            # Show transition screen in batch mode (except for the first sign)
            if batch_mode and idx > 0:
                print(f"\n  ── Get ready for: {display_name(sign)} ──")
                waiting = True
                while waiting:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue
                    frame = cv2.flip(frame, 1)
                    prev_sign = signs[idx - 1]
                    prev_collected = total_collected.get(prev_sign, 0)
                    prev_existing = current_counts.get(prev_sign, 0) - prev_collected
                    screen = draw_transition_screen(frame, prev_sign, sign,
                                                   prev_collected, prev_existing)
                    cv2.imshow("BridgeSign — Data Collector", screen)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        waiting = False
                    elif key == ord("n"):
                        print(f"  ⏭  Skipped {display_name(sign)}")
                        total_collected[sign] = 0
                        break
                    elif key == ord("q"):
                        print("  ⏹  Quit requested")
                        raise KeyboardInterrupt
                else:
                    # Normal flow: proceed to collection
                    pass

                if total_collected.get(sign) == 0:
                    continue  # was skipped

            result = collect_one_sign(sign, needed, save_path,
                                      cap, detector, extractor, existing,
                                      v2_metadata=v2_metadata)
            if result == -1:
                # User pressed Q — stop everything
                total_collected[sign] = 0
                break
            total_collected[sign] = result
            current_counts[sign] = current_counts.get(sign, 0) + result
            if args.v2:
                current_session_counts[sign] = current_session_counts.get(sign, 0) + result

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ── Summary ───────────────────────────────────────────────────────
        print("\n" + "=" * 62)
        print("  Session Summary")
        print("=" * 62)
        grand_total = 0
        for s in signs:
            new = total_collected.get(s, 0)
            old = existing_counts.get(s, 0)
            grand_total += new
            if new > 0:
                print(f"    {display_name(s):12s}  +{new:4d}  (total: {old + new})")
            else:
                print(f"    {display_name(s):12s}  skip  (total: {old})")
        print(f"\n  Total new samples: {grand_total}")
        if grand_total > 0:
            print(f"  Saved to: {save_path}")
            if args.v2:
                print("\n  Next steps: python static_model_v2.py quality")
                print("              python static_model_v2.py train")
            else:
                print(f"\n  Next step: python model_trainer.py")
        print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
