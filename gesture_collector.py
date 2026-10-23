"""
BridgeSign – Gesture Data Collector
====================================
Collects training data for MOTION-BASED signs (J, Z, word signs).

Unlike the static data_collector.py which saves one frame per sample,
this captures a SEQUENCE of 15 frames per sample — recording the
full movement of the gesture.

Usage:
  python gesture_collector.py --sign J --samples 200
  python gesture_collector.py --batch J Z --samples 200

Controls:
  [S] = Perform the gesture NOW (records next 15 frames)
  [N] = Skip this sign
  [Q] = Quit

Legacy data is saved to: data/gesture_dataset.csv
Use --v2 with --signer-id to write provenance-aware data to
data/gesture_dataset_v2.csv without changing the legacy dataset.
Format: label, f1_1, f1_2, ..., f1_42, f2_1, ..., f15_42
        (1 label + 15 frames × 42 features = 631 values per row)
"""

import cv2
import os
import time
import argparse
import csv
import numpy as np
import uuid
from datetime import datetime, timezone
import config
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor
from gesture_feature_extractor import GESTURE_WINDOW, STATIC_FEATURES
from gesture_data_utils import GESTURE_V2_FIELDS

try:
    from word_signs import SIGN_TIPS, display_name
except ImportError:
    SIGN_TIPS = {}
    display_name = lambda s: s.replace("_", " ")

# Extra tips for gesture letters
GESTURE_TIPS = {
    "J": "Start with pinky up (I shape), then trace a J hook downward and curl",
    "Z": "Start with index finger pointing, trace a Z zigzag in the air",
    "STATIC": "Keep hand still or make small random repositioning movements",
}
GESTURE_TIPS.update(SIGN_TIPS)

SAVE_FILE = os.path.join(config.DATA_DIR, "gesture_dataset.csv")
SAVE_FILE_V2 = os.path.join(config.DATA_DIR, "gesture_dataset_v2.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BridgeSign Gesture Collector — collect motion-based sign data.",
        epilog="""
Examples:
  python gesture_collector.py --sign J --samples 200
  python gesture_collector.py --batch J Z --samples 200
  python gesture_collector.py --batch HELP STOP --samples 150
        """
    )
    parser.add_argument("--sign",    type=str, default=None, help="Single gesture to collect")
    parser.add_argument("--batch",   type=str, nargs="+", default=None, help="Multiple gestures")
    parser.add_argument("--samples", type=int, default=200, help="Samples per gesture (default: 200)")
    parser.add_argument("--v2", action="store_true",
                        help="Write a new provenance-aware dataset without modifying the legacy CSV.")
    parser.add_argument("--signer-id", type=str, default=None,
                        help="Signer identifier required with --v2.")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Collection session identifier for --v2; generated if omitted.")
    parser.add_argument("--device", type=str, default="camera-0",
                        help="Camera/source identifier recorded in V2 mode.")
    args = parser.parse_args()
    if args.sign is None and args.batch is None:
        parser.error("Specify --sign or --batch")
    if args.sign is not None and args.batch is not None:
        parser.error("Use --sign or --batch, not both.")
    if args.samples < 1:
        parser.error("--samples must be positive.")
    if args.v2 and not (args.signer_id and args.signer_id.strip()):
        parser.error("--signer-id is required with --v2.")
    if not args.v2 and (args.signer_id or args.session_id):
        parser.error("--signer-id and --session-id require --v2.")
    return args


def count_existing(save_path, versioned=False):
    counts = {}
    if os.path.exists(save_path):
        if versioned:
            with open(save_path, "r", newline="", encoding="utf-8-sig") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != GESTURE_V2_FIELDS:
                    raise ValueError(f"V2 gesture dataset header changed: {save_path}")
                for row in reader:
                    label = (row.get("label") or "").strip()
                    if label:
                        counts[label] = counts.get(label, 0) + 1
        else:
            with open(save_path, "r", encoding="utf-8-sig") as source:
                for line in source:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        counts[parts[0]] = counts.get(parts[0], 0) + 1
    return counts


def _append_v2_sample(
    save_path,
    sign,
    features,
    metadata,
    frame_shape,
    captured_at_utc,
    capture_times,
):
    sequence = np.asarray(features, dtype=np.float32)
    if sequence.size != GESTURE_WINDOW * STATIC_FEATURES or not np.isfinite(sequence).all():
        raise ValueError("Gesture sample must contain 15 valid frames of 42 features.")

    file_exists = os.path.exists(save_path)
    if file_exists:
        with open(save_path, "r", newline="", encoding="utf-8-sig") as source:
            if next(csv.reader(source), None) != GESTURE_V2_FIELDS:
                raise ValueError(f"V2 gesture dataset header changed: {save_path}")

    elapsed = capture_times[-1] - capture_times[0] if len(capture_times) > 1 else 0.0
    capture_fps = (len(capture_times) - 1) / elapsed if elapsed > 0 else 0.0
    height, width = frame_shape[:2]
    row = [
        uuid.uuid4().hex,
        sign,
        metadata["signer_id"],
        metadata["session_id"],
        captured_at_utc,
        metadata["device"],
        width,
        height,
        round(capture_fps, 3),
        round(elapsed * 1000, 3),
        *sequence.reshape(-1).tolist(),
    ]
    with open(save_path, "a", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        if not file_exists:
            writer.writerow(GESTURE_V2_FIELDS)
        writer.writerow(row)


def draw_hud(frame, sign, count, target, state_text, tip, existing=0):
    """Draw collection HUD onto frame."""
    h, w = frame.shape[:2]

    # Dark bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 140), (w, h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # Sign label top left
    cv2.putText(frame, f"GESTURE: {display_name(sign)}", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 50), 2)

    # State indicator
    if state_text == "RECORDING":
        color = (0, 0, 255)
        if int(time.time() * 3) % 2 == 0:
            cv2.circle(frame, (w - 28, 24), 10, color, -1)
        cv2.putText(frame, "CAPTURING GESTURE...", (w - 280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    elif state_text == "READY":
        cv2.putText(frame, "Press [S] to perform gesture", (16, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    # Progress
    total = existing + count
    cv2.putText(frame, f"Collected: {count}/{target}  (total: {total})",
                (16, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Progress bar
    if target > 0:
        bar_w = int((count / target) * (w - 40))
        cv2.rectangle(frame, (20, h - 28), (w - 20, h - 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 28), (20 + bar_w, h - 10), (0, 200, 80), -1)

    # Tip
    cv2.putText(frame, tip[:80], (16, h - 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 200, 255), 1)

    # Existing count
    if existing > 0:
        cv2.putText(frame, f"Existing: {existing}", (w - 180, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 200, 255), 1)

    return frame


def collect_one_gesture(
    sign,
    target,
    save_path,
    cap,
    detector,
    extractor,
    existing=0,
    metadata=None,
):
    """
    Collect gesture samples for one sign.

    Each sample = user presses [S], then we capture GESTURE_WINDOW frames
    of landmarks as one sequence.

    Returns: number of new samples collected, or -1 if quit.
    """
    tip = GESTURE_TIPS.get(sign, "Perform the gesture naturally in front of the camera.")
    count = 0
    capturing = False
    capture_buffer = []
    capture_times = []
    captured_at_utc = ""

    print(f"\n  ── Collecting gesture: {display_name(sign)} ──")
    print(f"     Existing: {existing}  |  Target new: {target}")
    print(f"     Press [S] to perform the gesture, we'll capture {GESTURE_WINDOW} frames")

    while count < target:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        frame, _ = detector.find_hands(frame, draw=True)

        if capturing:
            # Capture frames for one gesture sample
            lm_list = detector.get_landmarks(frame, hand_no=0)
            if lm_list:
                features = extractor.extract_features(lm_list)
                if (
                    features is not None
                    and len(features) == STATIC_FEATURES
                    and np.isfinite(features).all()
                ):
                    capture_buffer.append(features)
                    capture_times.append(time.perf_counter())

            # Check if we have enough frames
            if len(capture_buffer) >= GESTURE_WINDOW:
                # Save this sequence as one sample
                flat = np.concatenate(capture_buffer[:GESTURE_WINDOW])
                if metadata is None:
                    with open(save_path, "a", encoding="utf-8") as destination:
                        destination.write(f"{sign}," + ",".join(map(str, flat)) + "\n")
                else:
                    _append_v2_sample(
                        save_path,
                        sign,
                        flat,
                        metadata,
                        frame.shape,
                        captured_at_utc,
                        capture_times[:GESTURE_WINDOW],
                    )
                count += 1
                capturing = False
                capture_buffer = []
                capture_times = []
                if count % 10 == 0:
                    print(f"     {count}/{target} gesture samples collected...")

            frame = draw_hud(frame, sign, count, target, "RECORDING", tip, existing)
        else:
            frame = draw_hud(frame, sign, count, target, "READY", tip, existing)

        cv2.imshow("BridgeSign - Gesture Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and not capturing:
            capturing = True
            capture_buffer = []
            capture_times = []
            captured_at_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        elif key == ord("n"):
            print(f"     Skipped {display_name(sign)} ({count} collected)")
            return count
        elif key == ord("q"):
            print(f"     Quit ({count} collected for {display_name(sign)})")
            return -1

    print(f"\n     ✅  Done! {count} gesture samples for '{display_name(sign)}'")
    return count


def main():
    args = parse_args()

    if args.batch:
        signs = [s.upper() for s in args.batch]
    else:
        signs = [args.sign.upper()]

    save_file = SAVE_FILE_V2 if args.v2 else SAVE_FILE
    existing_counts = count_existing(save_file, versioned=args.v2)
    metadata = None
    if args.v2:
        metadata = {
            "signer_id": args.signer_id.strip(),
            "session_id": (args.session_id or "").strip() or uuid.uuid4().hex,
            "device": args.device.strip() or "camera-0",
        }

    print("\n" + "=" * 62)
    print(f"  BridgeSign Gesture Collector")
    print(f"  Signs    : {', '.join(display_name(s) for s in signs)}")
    print(f"  Samples  : {args.samples} per sign")
    print(f"  Window   : {GESTURE_WINDOW} frames per gesture")
    print(f"  Save to  : {save_file}")
    if metadata:
        print(f"  Signer   : {metadata['signer_id']}")
        print(f"  Session  : {metadata['session_id']}")
    print("=" * 62)

    for s in signs:
        cur = existing_counts.get(s, 0)
        print(f"    {display_name(s):12s}  {cur:4d} existing")

    print(f"\n  Controls: [S] perform gesture  [N] skip  [Q] quit\n")

    detector = HandDetector(max_hands=1)
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    total_collected = {}

    try:
        for sign in signs:
            existing = existing_counts.get(sign, 0)
            needed = max(0, args.samples - existing)
            if needed == 0:
                print(f"\n  ⏭  {display_name(sign)} already has {existing} samples — skipping")
                total_collected[sign] = 0
                continue

            result = collect_one_gesture(
                sign,
                needed,
                save_file,
                cap,
                detector,
                extractor,
                existing,
                metadata=metadata,
            )
            if result == -1:
                total_collected[sign] = 0
                break
            total_collected[sign] = result

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 62)
        print("  Gesture Collection Summary")
        print("=" * 62)
        grand_total = 0
        for s in signs:
            new = total_collected.get(s, 0)
            old = existing_counts.get(s, 0)
            grand_total += new
            print(f"    {display_name(s):12s}  +{new:4d}  (total: {old + new})")
        print(f"\n  Total new samples: {grand_total}")
        if grand_total > 0:
            print(f"  Saved to: {save_file}")
            print(f"\n  Next step: python gesture_trainer.py --dataset \"{save_file}\"")
        print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
