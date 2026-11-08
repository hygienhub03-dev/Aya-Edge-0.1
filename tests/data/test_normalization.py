import hashlib
import json
from pathlib import Path

import pipeline


def test_document_normalization_preserves_raw_and_provenance_and_hashes_file_once(configure_pipeline, fixture_source, monkeypatch):
    source = fixture_source("docs", normalizer={"kind": "document", "text_field": "body", "title_field": "title"})
    _, data, _ = configure_pipeline([source])
    raw = data / "raw/docs/pilot.jsonl"
    raw.parent.mkdir(parents=True)
    raw_bytes = (
        (json.dumps({"id": "p-1", "title": "A title", "body": "A sufficiently long text record.", "lang": "en"}) + "\n")
        + (json.dumps({"id": "p-2", "title": "Another title", "body": "Another sufficiently long text record.", "lang": "en"}) + "\n")
    ).encode("utf-8")
    raw.write_bytes(raw_bytes)
    expected_checksum = hashlib.sha256(raw_bytes).hexdigest()
    checksum_calls = []
    original_sha256_file = pipeline.sha256_file

    def tracked_sha256_file(path):
        checksum_calls.append(Path(path))
        return original_sha256_file(path)

    monkeypatch.setattr(pipeline, "sha256_file", tracked_sha256_file)
    assert pipeline.normalize_datasets("pilot") == 0
    rows = list(pipeline.read_jsonl(data / "processed/pilot/normalized.jsonl"))
    assert len(rows) == 2
    row = rows[0]
    assert row["text"] == "A sufficiently long text record."
    assert row["raw_record"]["lang"] == "en"
    assert row["provenance"]["original_record_id"] == "p-1"
    assert row["source_revision"] == "a" * 40
    assert checksum_calls == [raw]
    assert {row["provenance"]["source_checksum"] for row in rows} == {expected_checksum}
    assert {row["provenance"]["source_file"] for row in rows} == {"raw/docs/pilot.jsonl"}


def test_malformed_or_missing_text_is_rejected_with_reason(configure_pipeline, fixture_source):
    source = fixture_source("docs")
    _, data, _ = configure_pipeline([source])
    raw = data / "raw/docs/pilot.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"id": "bad", "text": ""}) + "\n", encoding="utf-8")
    pipeline.normalize_datasets("pilot")
    rejected = list(pipeline.read_jsonl(data / "processed/pilot/rejected.jsonl"))
    assert len(rejected) == 1
    assert "empty text" in rejected[0]["reason"]
