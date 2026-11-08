import pipeline


def test_normalized_duplicate_removed_and_distinct_text_retained(configure_pipeline, fixture_source):
    source = fixture_source("docs")
    _, data, _ = configure_pipeline([source])
    rows = [
        {"sample_id": "a", "source_id": "docs", "category": "general", "text": "same text", "dedup_group": "a"},
        {"sample_id": "b", "source_id": "docs", "category": "general", "text": " SAME   TEXT ", "dedup_group": "b"},
        {"sample_id": "c", "source_id": "docs", "category": "general", "text": "different content", "dedup_group": "c"},
    ]
    pipeline.write_jsonl(data / "processed/pilot/normalized.jsonl", rows)
    assert pipeline.deduplicate("pilot") == 0
    kept = list(pipeline.read_jsonl(data / "processed/pilot/deduplicated.jsonl"))
    duplicates = list(pipeline.read_jsonl(data / "reports/duplicates.jsonl"))
    assert len(kept) == 2
    assert len(duplicates) == 1
    assert duplicates[0]["duplicate_type"] == "normalized_exact"
