import pytest

from aya_edge.data.language_corpus import assert_training_license


def test_fineweb_edu_license_gate_accepts_odc_by():
    assert assert_training_license("odc-by") == "ODC-By-1.0"
    assert assert_training_license(["ODC-By-1.0"]) == "ODC-By-1.0"


def test_fineweb_edu_license_gate_rejects_other_licences():
    with pytest.raises(ValueError, match="ODC-By"):
        assert_training_license("mit")
    with pytest.raises(ValueError, match="ODC-By"):
        assert_training_license(None)


def test_language_config_uses_the_0_2_model():
    import yaml
    from pathlib import Path

    root = Path(__file__).parents[1]
    base = yaml.safe_load((root / "configs" / "aya_edge_0_2.yaml").read_text(encoding="utf-8"))
    language = yaml.safe_load((root / "configs" / "aya_edge_0_2_language.yaml").read_text(encoding="utf-8"))
    assert language["model"] == base["model"]



def test_conditional_fineweb_approval_requires_both_acknowledgements():
    from aya_edge.data.language_corpus import approval_record

    kwargs = {
        "revision": "a" * 40,
        "license_audit_sha256": "b" * 64,
        "corpus_manifest_path": "manifest.json",
        "document_count": 10,
        "character_count": 10000,
    }
    with pytest.raises(ValueError, match="acknowledgements"):
        approval_record(**kwargs, acknowledgements=["commoncrawl_terms"])
    record = approval_record(
        **kwargs,
        acknowledgements=["commoncrawl_terms", "residual_page_rights_risk"],
    )
    assert record["approved"] is True
    assert record["approval_status"] == "approved_with_conditions"
    assert record["scope"] == "first_language_slice_only"
