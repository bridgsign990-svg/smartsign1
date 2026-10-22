from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tools.experimental_dynamic import wlasl_experiment
from tools.experimental_dynamic import wlasl_live_demo


def test_candidate_selection_preserves_official_gloss_and_signer_metadata():
    metadata = [
        {
            "gloss": "water",
            "instances": [
                {
                    "video_id": "001",
                    "signer_id": 17,
                    "source": "official-source",
                    "split": "train",
                    "url": "https://example.org/water.mp4",
                    "fps": 25,
                    "frame_start": 3,
                    "frame_end": 30,
                },
                {
                    "video_id": "002",
                    "signer_id": 17,
                    "url": "https://example.org/another.mp4",
                },
                {
                    "video_id": "003",
                    "signer_id": 18,
                    "url": "https://www.youtube.com/watch?v=official-id",
                },
                {
                    "video_id": "004",
                    "signer_id": 19,
                    "url": "https://example.org/unavailable.swf",
                },
            ],
        }
    ]

    rows = wlasl_experiment.select_candidates(metadata, ("water",), candidates_per_class=4)

    assert len(rows) == 1
    assert rows[0]["gloss"] == "water"
    assert rows[0]["video_id"] == "001"
    assert rows[0]["signer_id"] == "17"
    assert rows[0]["url"] == "https://example.org/water.mp4"
    assert rows[0]["frame_start"] == "3"
    assert rows[0]["frame_end"] == "30"
    assert rows[0]["download_status"] == "PENDING"


def test_candidate_selection_fails_when_official_gloss_is_missing():
    with pytest.raises(ValueError, match="absent from WLASL metadata"):
        wlasl_experiment.select_candidates([], ("invented-gloss",))


def test_wlasl_demo_rejects_non_wlasl_or_non_asl_checkpoints():
    checkpoint = {
        "dataset": "MS-ASL",
        "language": "ASL",
        "experiment_tag": wlasl_live_demo.EXPERIMENT_TAG,
    }
    with patch.object(
        wlasl_live_demo,
        "load_checkpoint",
        return_value=(object(), checkpoint, "cpu"),
    ):
        with pytest.raises(ValueError, match="only a checkpoint trained on WLASL"):
            wlasl_live_demo.load_wlasl_demo_model(Path("model.pt"))


def test_live_predict_checks_feature_dimensions_before_inference():
    class Model:
        def eval(self):
            return self

        def __call__(self, values, lengths):
            import torch

            return torch.tensor([[2.0, 1.0]])

    checkpoint = {
        "input_size": 3,
        "label_to_index": {"water": 0, "book": 1},
    }
    sequence = np.zeros((30, 3), dtype=np.float32)

    label, confidence = wlasl_live_demo._predict(Model(), checkpoint, "cpu", sequence)

    assert label == "water"
    assert 0.0 < confidence < 1.0
    with pytest.raises(ValueError, match="Expected live input shape"):
        wlasl_live_demo._predict(Model(), checkpoint, "cpu", np.zeros((15, 3), dtype=np.float32))
