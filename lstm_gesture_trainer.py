"""
BridgeSign -- LSTM Gesture Trainer
====================================
Trains the GestureLSTM model on gesture data collected by gesture_collector.py.

Usage:
  python lstm_gesture_trainer.py

Expects gesture data in: data/gesture_dataset.csv
Each row: label, f1, f2, ..., f42, f1, f2, ..., f42, ...  (15 frames x 42 features = 630 values)

Output:
  models/gesture_lstm_candidate.pt       -- candidate model weights
  models/gesture_lstm_candidate_meta.json -- class labels + validation metrics
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config
from lstm_gesture_model import GestureLSTM, create_model
from gesture_data_utils import load_gesture_sequences, grouped_train_validation_split
from gesture_feature_extractor import (
    FEATURE_CLIP,
    GESTURE_WINDOW,
    mirror_sequence,
    resample_sequence,
    sanitize_sequence,
)

DATA_PATH   = os.path.join(config.DATA_DIR, "gesture_dataset.csv")
MODEL_PATH  = os.path.join(config.MODELS_DIR, "gesture_lstm_candidate.pt")

# Training hyperparameters
EPOCHS        = 80
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15   # early stopping


class GestureSequenceDataset(Dataset):
    """PyTorch dataset that loads gesture sequences from the CSV."""

    def __init__(self, sequences, labels):
        """
        Parameters
        ----------
        sequences : np.ndarray of shape (N, 15, 42)
        labels    : np.ndarray of shape (N,)  -- integer class IDs
        """
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_gesture_data(path):
    """
    Load gesture sequences from CSV.
    
    The gesture_collector saves each sample as:
      label, f1, f2, ..., f630  (15 frames * 42 features = 630 values)
    
    We reshape each sample back into (15, 42) for the LSTM.
    """
    sequences, labels, _, _ = load_gesture_sequences(path)
    return sequences, labels


# Speed factors cover the gap between the ~30 fps desktop collector and the
# ~10-14 fps browser loop (a 15-frame live window spans 2-3x more time), plus
# natural variation in how fast people sign.
SPEED_RANGE = (0.6, 3.0)
LIVE_EVAL_SPEEDS = (1.0, 2.0, 3.0)


def _rotate_scale(X, angles, scales):
    """Rotate/scale every (x, y) landmark pair about the wrist origin."""
    cos = (np.cos(angles) * scales)[:, None, None]
    sin = (np.sin(angles) * scales)[:, None, None]
    xs, ys = X[..., 0::2], X[..., 1::2]
    out = np.empty_like(X)
    out[..., 0::2] = xs * cos - ys * sin
    out[..., 1::2] = xs * sin + ys * cos
    return out


def augment_sequences(X, y, copies=6, noise_std=0.02, seed=42):
    """
    Build augmented copies with speed (frame-rate) changes, left/right mirroring,
    small rotations/scales, and Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    aug_X, aug_y = [X], [y]
    n = len(X)

    for _ in range(copies):
        speeds = rng.uniform(*SPEED_RANGE, size=n)
        centers = rng.uniform(0.35, 0.65, size=n) * (X.shape[1] - 1)
        warped = np.stack([
            resample_sequence(X[i], length=X.shape[1], speed=speeds[i], center=centers[i])
            for i in range(n)
        ])
        mirror = rng.random(n) < 0.5
        warped[mirror] = mirror_sequence(warped[mirror])
        warped = _rotate_scale(
            warped,
            rng.normal(0.0, np.deg2rad(8.0), size=n),
            rng.uniform(0.9, 1.1, size=n),
        )
        warped += rng.normal(0, noise_std, warped.shape).astype(np.float32)
        aug_X.append(sanitize_sequence(warped))
        aug_y.append(y)

    return np.vstack(aug_X), np.concatenate(aug_y)


def load_held_pose_sequences(path, per_label=40, holdout_fraction=0.25, seed=7):
    """
    Turn static-letter rows (label + 42 features) into 15-frame "held still"
    sequences for the STATIC class. The recorded STATIC clips cover only a few
    hand poses, so without these the model reads a held I as J and a held
    L/P/X as Z. Returns (train_sequences, holdout_sequences).
    """
    rng = np.random.default_rng(seed)
    by_label = {}
    with open(path, "r", encoding="utf-8-sig") as source:
        for line in source:
            parts = line.strip().split(",")
            if len(parts) < 43:
                continue
            try:
                features = np.array(parts[1:43], dtype=np.float32)
            except ValueError:
                continue
            if np.isfinite(features).all() and np.abs(features).max() <= FEATURE_CLIP:
                by_label.setdefault(parts[0], []).append(features)
    train, holdout = [], []
    for rows in by_label.values():
        picks = rng.permutation(len(rows))[:per_label]
        n_hold = int(round(len(picks) * holdout_fraction))
        for k, index in enumerate(picks):
            sigma = rng.uniform(0.002, 0.012)
            drift = rng.normal(0, sigma * 0.5, (1, 42)) * np.linspace(0, 1, GESTURE_WINDOW)[:, None]
            sequence = rows[index][None, :] + drift + rng.normal(0, sigma, (GESTURE_WINDOW, 42))
            (holdout if k < n_hold else train).append(sequence.astype(np.float32))
    return np.array(train, dtype=np.float32), np.array(holdout, dtype=np.float32)


def drop_outlier_sequences(X, labels, limit=FEATURE_CLIP):
    """Drop samples with degenerate (exploded) coordinates from bad detections."""
    keep = np.isfinite(X).all(axis=(1, 2)) & (np.abs(X).max(axis=(1, 2)) <= limit)
    return keep


def evaluate_accuracy(model, X, y, device, batch_size=256):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            preds = model(batch).argmax(1).cpu().numpy()
            correct += int((preds == y[start:start + batch_size]).sum())
    return correct / max(len(X), 1)


def live_like_accuracy(model, X, y, device):
    """
    Accuracy when validation clips are replayed the way the browser sees them:
    sped up (lower frame rate) and possibly mirrored.
    """
    results = {}
    for speed in LIVE_EVAL_SPEEDS:
        warped = np.stack([resample_sequence(seq, length=X.shape[1], speed=speed) for seq in X])
        results[f"speed_{speed:g}"] = evaluate_accuracy(model, warped, y, device)
        results[f"speed_{speed:g}_mirrored"] = evaluate_accuracy(
            model, mirror_sequence(warped), y, device
        )
    return results


def export_onnx(model, classes, out_path):
    """Export the checkpoint the way GestureClassifier loads it (classes embedded)."""
    import onnx

    model = model.to("cpu").eval()
    dummy = torch.randn(1, GESTURE_WINDOW, 42)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        dynamo=False,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        },
    )
    onnx_model = onnx.load(out_path)
    prop = onnx_model.metadata_props.add()
    prop.key = "bridgesign.classes"
    prop.value = json.dumps(classes)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, out_path)


def main():
    parser = argparse.ArgumentParser(description="Train BridgeSign's temporal gesture LSTM.")
    parser.add_argument("--dataset", default=DATA_PATH, help="Legacy or provenance-aware gesture CSV.")
    parser.add_argument("--output", default=MODEL_PATH, help="Candidate checkpoint output path.")
    parser.add_argument(
        "--export-onnx",
        default=None,
        help="Also export the best checkpoint to this .onnx path (refuses to overwrite).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--static-poses",
        default=None,
        help="Static-letter CSV (label + 42 features) whose held poses are added to STATIC.",
    )
    args = parser.parse_args()
    output_path = args.output
    output_root, output_extension = os.path.splitext(output_path)
    if output_extension.lower() != ".pt":
        parser.error("--output must have a .pt extension.")
    metadata_path = output_root + "_meta.json"
    outputs = [output_path, metadata_path] + ([args.export_onnx] if args.export_onnx else [])
    for output in outputs:
        if os.path.exists(output):
            raise FileExistsError(
                f"Refusing to overwrite existing training artifact: {output}"
            )

    print("\n-- BridgeSign LSTM Gesture Trainer --")
    print(f"  Dataset : {args.dataset}")
    print(f"  Candidate output: {output_path}\n")

    # 1. Load data
    X, labels_raw, signer_ids, session_ids = load_gesture_sequences(args.dataset)
    keep = drop_outlier_sequences(X, labels_raw)
    if not keep.all():
        print(f"  Dropped {int((~keep).sum())} samples with exploded coordinates (|v| > {FEATURE_CLIP}).")
    X = X[keep]
    labels_raw = [label for label, kept in zip(labels_raw, keep) if kept]
    signer_ids = [v for v, kept in zip(signer_ids, keep) if kept] if signer_ids is not None else None
    session_ids = [v for v, kept in zip(session_ids, keep) if kept] if session_ids is not None else None
    classes = sorted(set(labels_raw))
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_id[l] for l in labels_raw], dtype=np.int64)

    print(f"  Samples     : {len(X)}")
    print(f"  Seq shape   : {X.shape[1:]}  (frames x features)")
    print(f"  Classes ({len(classes)}): {classes}")

    # Class distribution
    for c in classes:
        count = sum(1 for l in labels_raw if l == c)
        print(f"    {c:12s}: {count} samples")

    # 2. Keep duplicates and available capture groups in one partition.
    train_indices, val_indices, split_info = grouped_train_validation_split(
        X,
        labels_raw,
        signer_ids=signer_ids,
        session_ids=session_ids,
        test_size=0.2,
        random_state=42,
    )
    print(
        f"  Validation split: {split_info['split_basis']} groups; "
        f"{split_info['duplicate_extra_rows']} duplicate rows kept together."
    )
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    print(f"\n  Train (raw) : {len(X_train)}")
    print(f"  Val (clean) : {len(X_val)}")

    held_holdout = None
    if args.static_poses:
        if "STATIC" not in label_to_id:
            parser.error("--static-poses needs a STATIC class in the gesture dataset.")
        held_train, held_holdout = load_held_pose_sequences(args.static_poses)
        X_train = np.concatenate([X_train, held_train])
        y_train = np.concatenate([y_train, np.full(len(held_train), label_to_id["STATIC"])])
        print(f"  Held static poses: {len(held_train)} train, {len(held_holdout)} held out")

    # 3. Augment training set
    X_train, y_train = augment_sequences(X_train, y_train)
    print(f"  Train (aug) : {len(X_train)}\n")

    # 4. Create dataloaders
    train_ds = GestureSequenceDataset(X_train, y_train)
    val_ds   = GestureSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(num_classes=len(classes), device=device)

    # 6. Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= class_weights.sum()
    class_weights *= len(classes)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    # 7. Training loop
    best_val_acc = -1.0
    patience_counter = 0

    print("  Training...")
    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # -- Validate --
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={train_loss/train_total:.4f}  "
                  f"train={train_acc:.2%}  val={val_acc:.2%}  lr={lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            output_directory = os.path.dirname(output_path)
            if output_directory:
                os.makedirs(output_directory, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "input_size": 42,
                "hidden_size": 64,
                "num_layers": 2,
                "num_classes": len(classes),
            }, output_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # 8. Final evaluation
    print(f"\n  Best validation accuracy: {best_val_acc:.2%}")

    # Load best model for final report
    if not os.path.isfile(output_path):
        raise RuntimeError("Training finished without producing a candidate checkpoint.")
    checkpoint = torch.load(output_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Per-class accuracy
    class_correct = {c: 0 for c in classes}
    class_total   = {c: 0 for c in classes}
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(1).cpu()
            for pred, true in zip(preds, y_batch):
                true_class = classes[true.item()]
                class_total[true_class] += 1
                if pred.item() == true.item():
                    class_correct[true_class] += 1

    print("\n  Per-class accuracy:")
    report = {}
    for c in classes:
        acc = class_correct[c] / max(class_total[c], 1)
        print(f"    {c:12s}: {acc:.0%} ({class_correct[c]}/{class_total[c]})")
        report[c] = {"accuracy": acc, "correct": class_correct[c], "total": class_total[c]}

    live_metrics = live_like_accuracy(model, X_val, y_val, device)
    if held_holdout is not None and len(held_holdout):
        held_y = np.full(len(held_holdout), label_to_id["STATIC"])
        live_metrics["held_letter_poses_read_as_static"] = evaluate_accuracy(
            model, held_holdout, held_y, device
        )
    print("\n  Live-like validation (sped up = lower browser frame rate):")
    for name, value in live_metrics.items():
        print(f"    {name:22s}: {value:.2%}")

    # Save metadata
    meta = {
        "live_like_val_accuracy": live_metrics,
        "augmentation": {
            "speed_range": list(SPEED_RANGE),
            "mirror": True,
            "rotation_deg_std": 8.0,
            "scale_range": [0.9, 1.1],
        },
        "dropped_outlier_samples": int((~keep).sum()),
        "classes": classes,
        "best_val_accuracy": best_val_acc,
        "per_class": report,
        "epochs_trained": epoch,
        "validation_split_basis": split_info["split_basis"],
        "duplicate_extra_rows": split_info["duplicate_extra_rows"],
        "validation_is_signer_independent": split_info["split_basis"] == "signer",
        "model_params": GestureLSTM.count_parameters(model),
    }

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Model saved  -> {output_path}")
    print(f"  Meta saved   -> {metadata_path}")
    if args.export_onnx:
        export_onnx(model, classes, args.export_onnx)
        print(f"  ONNX saved   -> {args.export_onnx}")
    print("\n-- Training complete! --\n")


if __name__ == "__main__":
    main()
