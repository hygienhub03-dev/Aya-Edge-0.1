from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/data"))
import pipeline  # noqa: E402


def approved_source(source_id="fixture", **overrides):
    source = {
        "source_id": source_id,
        "dataset_name": "Synthetic test fixture",
        "repository": "local/test-fixture",
        "revision": "a" * 40,
        "configuration": "fixture-v1",
        "split": "train",
        "source_url": "https://example.invalid/fixture",
        "license_name": "Fixture-only synthetic terms",
        "license_url": "https://example.invalid/fixture-license",
        "license_verified": True,
        "commercial_use": "not_applicable",
        "redistribution": "not_applicable",
        "attribution_required": False,
        "share_alike": False,
        "additional_restrictions": [],
        "upstream_sources": [],
        "synthetic_or_human": "synthetic",
        "known_limitations": [],
        "approval_status": "approved",
        "review_date": "2026-11-07",
        "review_evidence": "Synthetic fixture authored for tests; no third-party content.",
        "use_class": "stage1",
        "normalizer": {"kind": "document", "text_field": "text", "title_field": "title"},
    }
    source.update(overrides)
    return source


@pytest.fixture
def configure_pipeline(tmp_path, monkeypatch):
    def configure(sources):
        root = tmp_path / "repo"
        registry_path = root / "configs/data/source_registry.yaml"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(yaml.safe_dump({"schema_version": 1, "sources": sources}, sort_keys=False), encoding="utf-8")
        holdout_path = root / "configs/data/evaluation_holdout.yaml"
        holdout_path.write_text(yaml.safe_dump({"protected_source_ids": ["bfcl-test"]}), encoding="utf-8")
        shutil.copytree(ROOT / "data/schemas", root / "data/schemas")
        monkeypatch.setattr(pipeline, "ROOT", root)
        monkeypatch.setattr(pipeline, "DATA_ROOT", tmp_path / "data")
        monkeypatch.setattr(pipeline, "REGISTRY_DEFAULT", registry_path)
        return root, tmp_path / "data", registry_path
    return configure


@pytest.fixture
def fixture_source():
    return approved_source
