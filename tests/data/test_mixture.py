import yaml

import pipeline


def test_mixture_uses_token_counts_and_reports_deviation(configure_pipeline, fixture_source, tmp_path):
    _, data, _ = configure_pipeline([fixture_source("records")])
    rows = []
    for i in range(4):
        category = "general" if i < 2 else "instruction"
        rows.append({"sample_id": f"s{i}", "source_id": "records", "category": category, "text": "abcdefgh", "dedup_group": f"g{i}"})
    pipeline.write_jsonl(data / "processed/pilot/deduplicated.jsonl", rows)
    pipeline.write_jsonl(data / "processed/pilot/train.jsonl", rows)
    pipeline.count_tokens("pilot", "aya-edge-dev-byte-v1")
    cfg = tmp_path / "mixture.yaml"
    cfg.write_text(yaml.safe_dump({
        "seed": 4,
        "tokenizer": {"id": "aya-edge-dev-byte-v1", "production_required": False},
        "category_targets": {"general": 0.5, "agent": 0.5},
        "category_members": {"general": ["general"], "agent": ["instruction", "agent", "tool_use"]},
    }), encoding="utf-8")
    pipeline.build_mixture(cfg, "pilot")
    report = pipeline.load_json(data / "reports/mixture.json")
    assert report["actual_total_tokens"] == 32
    assert report["actual"]["general"] == 16
    assert report["actual"]["agent"] == 16
    assert report["sources"]["records"]["tokens"] == 32
    assert report["production_tokenizer"] is False
