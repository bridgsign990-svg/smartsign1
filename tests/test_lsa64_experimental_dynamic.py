from pathlib import Path

import pytest

from tools.experimental_dynamic.baseline import _experiment_tag
from tools.experimental_dynamic.lsa64 import (
    parse_lsa64_filename,
    scan_dataset,
    select_subset,
)


def test_lsa64_identity_parsing_keeps_source_ids_and_non_msl_label():
    identity = parse_lsa64_filename(Path("064_010_005.mp4"))

    assert identity == {
        "source_record_id": "064_010_005",
        "video_id": "064_010_005",
        "class_id": "064",
        "sign_label": "LSA64_CLASS_064",
        "signer_id": "LSA64_SIGNER_010",
        "repetition_id": "REP_005",
    }


@pytest.mark.parametrize("filename", ["064_011_001.mp4", "000_001_001.mp4", "bad.mp4"])
def test_lsa64_identity_parser_rejects_invalid_ids(filename):
    with pytest.raises(ValueError):
        parse_lsa64_filename(filename)


def test_subset_selection_is_reproducible_and_balanced():
    records = []
    for class_number in range(1, 9):
        for signer_number in range(1, 11):
            for repetition_number in range(1, 6):
                records.append({
                    "class_id": f"{class_number:03d}",
                    "signer_id": f"LSA64_SIGNER_{signer_number:03d}",
                    "repetition_id": f"REP_{repetition_number:03d}",
                })

    classes_a, selected_a = select_subset(records, 5, 42)
    classes_b, selected_b = select_subset(list(reversed(records)), 5, 42)

    assert classes_a == classes_b
    assert [row["class_id"] for row in selected_a] == [row["class_id"] for row in selected_b]
    assert len(selected_a) == 250
    assert {row["class_id"] for row in selected_a} == set(classes_a)
    assert len({row["signer_id"] for row in selected_a}) == 10
    assert len({row["repetition_id"] for row in selected_a}) == 5


def test_scan_rejects_incomplete_source_structure(tmp_path):
    (tmp_path / "001_001_001.mp4").write_bytes(b"not a video; only identity scanning is exercised")

    with pytest.raises(ValueError, match="structure validation failed"):
        scan_dataset(tmp_path)


def test_lsa64_receives_distinct_experiment_tag():
    assert _experiment_tag("LSA64") == "EXPERIMENTAL_LSA_ARCHITECTURE_VALIDATION"
