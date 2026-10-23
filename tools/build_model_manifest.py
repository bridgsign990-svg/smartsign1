"""
Write models/model_manifest.json: one provenance record per runtime artifact.

The app checks loaded models against this file (labels, feature count,
SHA-256) and reports differences in /ready. Re-run after retraining or
replacing any model, then review the diff before committing:

    python tools/build_model_manifest.py
"""

import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_artifacts import MANIFEST_FILENAME, is_git_lfs_pointer, sha256_file  # noqa: E402

# Role of each known artifact. "production" is what the app selects first;
# "fallback" is only used when production is missing; "candidate" is kept for
# evaluation and is not promoted by this tool.
ARTIFACTS = {
    "sign_model_light.pkl": {"family": "static_letters", "status": "production"},
    "sign_model_v2.pkl": {
        "family": "static_letters",
        "status": "fallback",
        "note": "Interim one-signer/session candidate; loaded only when the production model is missing.",
    },
    "gesture_lstm_v2.onnx": {
        "family": "motion_gestures",
        "status": "production",
        "metadata": "gesture_lstm_v2_meta.json",
    },
    "gesture_lstm.onnx": {
        "family": "motion_gestures",
        "status": "fallback",
        "metadata": "gesture_lstm_meta.json",
    },
    "hand_landmarker.task": {"family": "mediapipe_hand", "status": "production"},
    "pose_landmarker_lite.task": {"family": "mediapipe_pose", "status": "production"},
    "sentence_brain/encoder.onnx": {
        "family": "sentence_brain",
        "status": "production",
        "note": "T5-small encoder, int8; see models/sentence_brain/meta.json and docs/SENTENCE_MODE.md.",
    },
    "sentence_brain/decoder.onnx": {"family": "sentence_brain", "status": "production"},
    "holistic_landmarker.task": {
        "family": "mediapipe_holistic",
        "status": "production",
        "note": "Used by the MSL dataset pipeline (tools/msl_dataset), not the live camera path.",
    },
}


def _describe_pickle(path):
    with path.open("rb") as artifact:
        payload = pickle.load(artifact)
    pipeline = payload["pipeline"]
    return {
        "labels": list(payload["classes"]),
        "n_features": int(getattr(pipeline, "n_features_in_", 0)) or None,
    }


def _describe_onnx(path, metadata_name):
    import onnx

    model = onnx.load(str(path))
    shape = [
        dim.dim_value or dim.dim_param or None
        for dim in model.graph.input[0].type.tensor_type.shape.dim
    ]
    details = {"input_shape": shape}
    if shape and isinstance(shape[-1], int):
        details["n_features"] = shape[-1]
    meta_path = path.parent / metadata_name if metadata_name else None
    if meta_path and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        details["labels"] = list(meta.get("classes", []))
        details["evaluation"] = {
            key: meta.get(key)
            for key in ("best_val_accuracy", "validation_split_basis", "validation_is_signer_independent")
            if key in meta
        }
    return details


def build(models_dir):
    artifacts = {}
    for name, role in ARTIFACTS.items():
        path = models_dir / name
        if not path.is_file():
            print(f"skip {name}: missing")
            continue
        if is_git_lfs_pointer(path):
            print(f"skip {name}: unresolved Git LFS pointer (run `git lfs pull`)")
            continue
        entry = {key: value for key, value in role.items() if key != "metadata"}
        entry["sha256"] = sha256_file(path)
        entry["size_bytes"] = path.stat().st_size
        if path.suffix == ".pkl":
            entry.update(_describe_pickle(path))
        elif path.suffix == ".onnx":
            entry.update(_describe_onnx(path, role.get("metadata")))
        artifacts[name] = entry
        print(f"recorded {name}")
    return {"generated": date.today().isoformat(), "artifacts": artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    manifest = build(args.models_dir)
    out = args.models_dir / MANIFEST_FILENAME
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
