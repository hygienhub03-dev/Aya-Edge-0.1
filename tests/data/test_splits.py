import pipeline


def test_splits_repeat_identically_and_keep_groups_together(configure_pipeline, fixture_source):
    _, data, _ = configure_pipeline([fixture_source("docs")])
    rows = [{"sample_id": f"s{i}", "source_id": "docs", "category": "general", "text": f"document {i}", "dedup_group": f"g{i}"} for i in range(30)]
    rows += [{"sample_id": "s0-copy", "source_id": "docs", "category": "general", "text": "document 0", "dedup_group": "g0"}]
    pipeline.write_jsonl(data / "processed/pilot/deduplicated.jsonl", rows)
    pipeline.build_splits("pilot", seed=13, validation_fraction=0.2, test_fraction=0.2)
    first = (data / "manifests/splits.jsonl").read_bytes()
    pipeline.build_splits("pilot", seed=13, validation_fraction=0.2, test_fraction=0.2)
    second = (data / "manifests/splits.jsonl").read_bytes()
    assert first == second
    locations = {}
    for split in ("train", "validation", "test"):
        for row in pipeline.read_jsonl(data / f"processed/pilot/{split}.jsonl"):
            locations.setdefault(row["dedup_group"], set()).add(split)
    assert all(len(splits) == 1 for splits in locations.values())
