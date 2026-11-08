import json

import pipeline


def test_high_confidence_sensitive_patterns_are_labeled_not_echoed():
    assert "email" in pipeline.sensitive_data_indicators("Contact alice@example.com for the file.")
    assert "us_ssn" in pipeline.sensitive_data_indicators("The identifier is 123-45-6789.")
    synthetic_prefix = "gh" + "p_"
    synthetic_suffix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "1234567890"
    assert "credential_or_api_secret" in pipeline.sensitive_data_indicators("token=" + synthetic_prefix + synthetic_suffix)


def test_normalizer_rejects_email_and_keeps_only_reason_and_digest(configure_pipeline, fixture_source):
    source = fixture_source("docs")
    _, data, _ = configure_pipeline([source])
    raw_path = data / "raw/docs/pilot.jsonl"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({"id": "p-1", "title": "Contact", "text": "Please contact alice@example.com about this private information."}) + "\n", encoding="utf-8")
    pipeline.normalize_datasets("pilot")
    rejected = list(pipeline.read_jsonl(data / "processed/pilot/rejected.jsonl"))
    assert len(rejected) == 1
    assert "email" in rejected[0]["reason"]
    assert len(rejected[0]["raw_record_sha256"]) == 64
    assert "raw_record" not in rejected[0]
    assert "alice@example.com" not in json.dumps(rejected[0])
