import pipeline


def document(sample_id, source_id="fixture", revision=None, license_id=None, provenance=None):
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "source_revision": revision or "a" * 40,
        "license_id": license_id or "Fixture-only synthetic terms",
        "category": "general",
        "title": "A test document",
        "text": "A sufficiently long test document for schema validation.",
        "provenance": provenance or {
            "original_record_id": sample_id,
            "source_split": "train",
            "synthetic": True,
            "transformation_version": pipeline.TRANSFORM_VERSION,
            "source_file": "raw/fixture/pilot.jsonl",
            "source_checksum": "b" * 64,
        },
        "raw_record": {"id": sample_id},
    }


def test_invalid_schema_provenance_and_license_rows_are_not_counted_as_valid(
    configure_pipeline, fixture_source
):
    approved = fixture_source("fixture")
    blocked = fixture_source("blocked", approval_status="blocked_pending_review")
    _, data, _ = configure_pipeline([approved, blocked])

    valid = document("valid-1")
    wrong_revision = document("wrong-revision", revision="c" * 40)
    wrong_license = document("wrong-license", license_id="Not the reviewed license")
    invalid_schema = document("invalid-schema")
    invalid_schema["category"] = "unsupported-category"
    invalid_provenance = document("invalid-provenance")
    invalid_provenance["provenance"]["source_split"] = "validation"
    unapproved = document("unapproved", source_id="blocked")
    unknown_source = document("unknown-source", source_id="not-in-registry")

    records = [valid, wrong_revision, wrong_license, invalid_schema, invalid_provenance, unapproved, unknown_source]
    pipeline.write_jsonl(data / "processed/pilot/deduplicated.jsonl", records)
    pipeline.write_jsonl(data / "processed/pilot/train.jsonl", records)
    pipeline.write_jsonl(data / "processed/pilot/validation.jsonl", [])
    pipeline.write_jsonl(data / "processed/pilot/test.jsonl", [])

    assert pipeline.validate_dataset("pilot") == 1  # Missing real counts/mixture also fails closed.
    report = pipeline.load_json(data / "reports/validation.json")
    assert report["counts"]["input"] == len(records)
    assert report["counts"]["valid"] == 1
    reasons = [entry["reason"] for entry in report["errors"]]
    assert any("pinned registry revision" in reason for reason in reasons)
    assert any("reviewed source license" in reason for reason in reasons)
    assert any("was expected" in reason for reason in reasons)
    assert any("source_split" in reason for reason in reasons)
    assert any("blocked from training" in reason for reason in reasons)
    assert any("unknown source" in reason for reason in reasons)


def test_all_invalid_records_produce_zero_valid_count(configure_pipeline, fixture_source):
    source = fixture_source("fixture")
    _, data, _ = configure_pipeline([source])
    invalid = document("bad")
    invalid.pop("provenance")
    invalid["sample_id"] = ["not-a-scalar-id"]
    invalid_source = document("bad-source-id")
    invalid_source["source_id"] = {"unexpected": "object"}
    pipeline.write_jsonl(data / "processed/pilot/deduplicated.jsonl", [invalid, invalid_source])
    for split in ("train", "validation", "test"):
        pipeline.write_jsonl(data / f"processed/pilot/{split}.jsonl", [])

    assert pipeline.validate_dataset("pilot") == 1
    report = pipeline.load_json(data / "reports/validation.json")
    assert report["counts"]["input"] == 2
    assert report["counts"].get("valid", 0) == 0
    assert any("scalar string" in item["reason"] for item in report["errors"])
    assert any("unknown source" in item["reason"] for item in report["errors"])
