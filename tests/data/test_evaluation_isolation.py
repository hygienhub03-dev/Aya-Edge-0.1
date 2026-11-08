import pipeline


def test_protected_evaluation_source_in_training_fails_validation(configure_pipeline, fixture_source):
    protected = fixture_source("bfcl-test", use_class="held_out", approval_status="unverified", license_verified=False)
    _, data, _ = configure_pipeline([protected])
    row = {"sample_id": "benchmark-leak", "source_id": "bfcl-test", "source_revision": "a" * 40, "license_id": "unknown", "category": "general", "title": "", "text": "This protected benchmark record must not be training data.", "provenance": {"original_record_id": "1", "source_split": "test", "synthetic": False, "transformation_version": "1.0.0"}, "dedup_group": "g"}
    pipeline.write_jsonl(data / "processed/pilot/deduplicated.jsonl", [row])
    pipeline.write_jsonl(data / "processed/pilot/train.jsonl", [row])
    pipeline.write_jsonl(data / "processed/pilot/validation.jsonl", [])
    pipeline.write_jsonl(data / "processed/pilot/test.jsonl", [])
    assert pipeline.validate_dataset("pilot") == 1
    report = pipeline.load_json(data / "reports/validation.json")
    assert any("protected evaluation source" in error.get("reason", "") for error in report["errors"])
