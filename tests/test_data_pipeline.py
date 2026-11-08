import json

import pytest

from aya_edge.data import build_dataset


def record(record_id, category, text, source="fixture", license_name="fixture-only"):
    return {
        "id": record_id,
        "category": category,
        "text": text,
        "source": source,
        "license": license_name,
    }


def test_pipeline_requires_provenance_unless_explicit_defaults(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"id": "1", "category": "general", "text": "hello"}) + "\n")
    with pytest.raises(ValueError):
        build_dataset([source], tmp_path / "strict")
    manifest = build_dataset(
        [source],
        tmp_path / "annotated",
        default_source="development-fixture",
        default_license="fixture-only",
    )
    assert manifest["records"]["accepted"] == 1


def test_pipeline_deduplicates_and_stratifies(tmp_path):
    source = tmp_path / "input.jsonl"
    rows = [
        record("a", "general", "same text"),
        record("b", "general", "same   text"),
        record("c", "agent", "call weather"),
        record("d", "agent", "recover after timeout"),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "out"
    manifest = build_dataset([source], out, validation_fraction=0.25, test_fraction=0.25)
    assert manifest["records"]["accepted"] == 3
    assert manifest["records"]["rejected"] == 1
    assert manifest["processing"]["split"]["stratified_by"] == "category"
    assert (out / "manifest.json").exists()
