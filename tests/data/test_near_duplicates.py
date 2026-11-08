import pipeline


def test_near_duplicate_is_flagged_retained_and_grouped_together(configure_pipeline, fixture_source):
    _, data, _ = configure_pipeline([fixture_source("docs")])
    first = "The agent should plan carefully use tools inspect results recover errors and explain its final answer while preserving every useful source and decision boundary."
    second = first[:-1] + "!"
    pipeline.write_jsonl(data / "processed/pilot/normalized.jsonl", [
        {"sample_id": "a", "source_id": "docs", "category": "general", "text": first},
        {"sample_id": "b", "source_id": "docs", "category": "general", "text": second},
    ])
    pipeline.deduplicate("pilot")
    kept = list(pipeline.read_jsonl(data / "processed/pilot/deduplicated.jsonl"))
    duplicates = list(pipeline.read_jsonl(data / "reports/duplicates.jsonl"))
    report = pipeline.load_json(data / "reports/deduplication.json")
    assert len(kept) == 2
    assert report["near_duplicates_flagged"] == 1
    assert [row["duplicate_type"] for row in duplicates] == ["near_duplicate_flag"]
    assert kept[0]["dedup_group"] == kept[1]["dedup_group"]
