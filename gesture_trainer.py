"""
BridgeSign – Gesture Model Trainer
====================================
Trains a RandomForest classifier on gesture (motion) data collected
by gesture_collector.py.

Each sample in gesture_dataset.csv is a sequence of 15 frames of
hand landmarks, capturing the full movement of a gesture.

Usage:
  python gesture_trainer.py

Output:
  models/gesture_model_candidate.pkl          <- candidate pipeline
  models/gesture_model_candidate_classes.json <- class labels
  models/gesture_model_candidate_metrics.json <- validation metrics
"""

import os
import json
import pickle
import argparse
import numpy as np
import config
from gesture_feature_extractor import extract_from_sequence
from gesture_data_utils import load_gesture_sequences, grouped_train_validation_split

DATA_PATH    = os.path.join(config.DATA_DIR, "gesture_dataset.csv")
MODEL_PATH   = os.path.join(config.MODELS_DIR, "gesture_model_candidate.pkl")


def load_gesture_dataset(path):
    """
    Load gesture_dataset.csv.
    Each row: label, f1_1, f1_2, ..., f15_42
    (1 label + 15×42 = 630 raw values)
    """
    sequences, labels, _, _ = load_gesture_sequences(path)
    features = [
        extract_from_sequence(list(sequence))
        for sequence in sequences
    ]
    return np.asarray(features, dtype=np.float32), labels


def augment_gestures(X, y, n_copies=3, noise_std=0.01, rng=None):
    """
    Augment gesture data with noise and time-shift.
    Safe transforms only — no flipping (directional gestures).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    aug_X, aug_y = [X], [y]

    for i in range(n_copies):
        # Noise copy
        intensity = noise_std * (0.7 + 0.6 * i / max(n_copies - 1, 1))
        noisy = X + rng.normal(0, intensity, X.shape).astype(np.float32)
        aug_X.append(noisy)
        aug_y.append(y)

        # Scale jitter
        scale = rng.uniform(0.95, 1.05, size=(X.shape[0], 1)).astype(np.float32)
        scaled = X * scale
        scaled += rng.normal(0, noise_std * 0.3, scaled.shape).astype(np.float32)
        aug_X.append(scaled)
        aug_y.append(y)

    return np.vstack(aug_X), np.concatenate(aug_y)


def main():
    parser = argparse.ArgumentParser(description="Train the legacy temporal-feature gesture classifier.")
    parser.add_argument("--dataset", default=DATA_PATH, help="Legacy or provenance-aware gesture CSV.")
    parser.add_argument("--output", default=MODEL_PATH, help="Candidate model output path.")
    args = parser.parse_args()
    output_path = args.output
    output_root, output_extension = os.path.splitext(output_path)
    if output_extension.lower() != ".pkl":
        parser.error("--output must have a .pkl extension.")
    classes_path = output_root + "_classes.json"
    metrics_path = output_root + "_metrics.json"
    for output in (output_path, classes_path, metrics_path):
        if os.path.exists(output):
            raise FileExistsError(
                f"Refusing to overwrite existing training artifact: {output}"
            )

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n-- BridgeSign Gesture Trainer --------")
    print(f"  Dataset : {args.dataset}")
    print(f"  Candidate output: {output_path}\n")

    # 1. Load
    sequences, labels_raw, signer_ids, session_ids = load_gesture_sequences(args.dataset)
    X = np.asarray(
        [extract_from_sequence(list(sequence)) for sequence in sequences],
        dtype=np.float32,
    )
    classes = sorted(set(labels_raw))
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_id[l] for l in labels_raw], dtype=np.int32)

    print(f"  Samples  : {len(X)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Gestures : {classes}")

    # 2. Keep duplicates and available capture groups in one partition.
    train_indices, val_indices, split_info = grouped_train_validation_split(
        sequences,
        labels_raw,
        signer_ids=signer_ids,
        session_ids=session_ids,
        test_size=0.18,
        random_state=42,
    )
    print(
        f"  Validation split: {split_info['split_basis']} groups; "
        f"{split_info['duplicate_extra_rows']} duplicate rows kept together."
    )
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    print(f"\n  Train : {len(X_train)}")
    print(f"  Val   : {len(X_val)}\n")

    # 3. Augment
    print("  Augmenting (3 copies: noise + scale jitter)...")
    X_train, y_train = augment_gestures(X_train, y_train)
    print(f"  Train (augmented): {len(X_train)}\n")

    # 4. Train
    print("  Training RandomForest (300 trees)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        ))
    ])
    pipeline.fit(X_train, y_train)

    # 5. Validate
    val_acc = pipeline.score(X_val, y_val)
    y_pred = pipeline.predict(X_val)
    print(f"  Validation accuracy: {val_acc:.2%}")
    print(classification_report(y_val, y_pred, target_names=classes, zero_division=0))

    # 6. Save
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "classes": classes}, f)
    print(f"  Model saved -> {output_path}")

    with open(classes_path, "w") as f:
        json.dump(classes, f, indent=2)

    metrics = {
        "validation_accuracy": float(val_acc),
        "classes": classes,
        "samples_per_class": {c: int(np.sum(y == label_to_id[c])) for c in classes},
        "validation_split_basis": split_info["split_basis"],
        "duplicate_extra_rows": split_info["duplicate_extra_rows"],
        "validation_is_signer_independent": split_info["split_basis"] == "signer",
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n-- Gesture training complete! --------")
    print(f"  Next: python app.py\n")


if __name__ == "__main__":
    main()
