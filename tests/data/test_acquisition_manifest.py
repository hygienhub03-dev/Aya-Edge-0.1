import yaml

import pipeline


def test_no_active_jobs_write_explicit_zero_download_manifest(configure_pipeline, fixture_source):
    source = fixture_source("fixture", approval_status="blocked_pending_review")
    _, data, registry_path = configure_pipeline([source])
    plan_path = data.parent / "acquisition.yaml"
    plan_path.write_text(yaml.safe_dump({"source_registry": str(registry_path), "sources": [{"source_id": "fixture", "enabled": False, "stage": "stage1"}]}), encoding="utf-8")
    assert pipeline.download_datasets(plan_path, "pilot") == 0
    [manifest] = list(pipeline.read_jsonl(data / "manifests/acquisition_run.jsonl"))
    assert manifest["status"] == "no_sources_approved_or_enabled"
    assert manifest["external_payloads_downloaded"] == 0
    assert manifest["source_jobs_considered"] == 1
