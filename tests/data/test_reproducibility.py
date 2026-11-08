import json

import pipeline


def test_normalize_deduplicate_and_manifest_outputs_are_repeatable(configure_pipeline, fixture_source):
    source = fixture_source("docs")
    _, data, _ = configure_pipeline([source])
    raw = data / "raw/docs/pilot.jsonl"
    raw.parent.mkdir(parents=True)
    rows = [{"id": str(i), "text": f"This is a distinct reproducible fixture record number {i}."} for i in range(12)]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    pipeline.normalize_datasets("pilot")
    pipeline.deduplicate("pilot")
    pipeline.build_splits("pilot", seed=42)
    first = (data / "manifests/splits.jsonl").read_bytes()
    pipeline.normalize_datasets("pilot")
    pipeline.deduplicate("pilot")
    pipeline.build_splits("pilot", seed=42)
    second = (data / "manifests/splits.jsonl").read_bytes()
    assert first == second
