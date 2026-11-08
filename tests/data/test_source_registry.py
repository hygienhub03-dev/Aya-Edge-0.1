from pathlib import Path

import pytest
import yaml

import pipeline


def test_registry_requires_unique_source_ids(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"sources": [{"source_id": "x"}, {"source_id": "x"}]}), encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="Duplicate source_id"):
        pipeline.registry_sources(path)


def test_revision_must_be_full_immutable_sha():
    assert pipeline.parse_revision({"revision": "1" * 40}) == "1" * 40
    with pytest.raises(pipeline.PipelineError, match="immutable 40-character"):
        pipeline.parse_revision({"revision": "main"})
