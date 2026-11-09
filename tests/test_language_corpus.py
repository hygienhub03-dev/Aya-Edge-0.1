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
