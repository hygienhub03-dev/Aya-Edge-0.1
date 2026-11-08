import pipeline


def test_stable_id_changes_with_source_revision_and_is_repeatable():
    raw = {"id": "17", "text": "document"}
    first = pipeline.stable_id("fixture", "a" * 40, "train", "17", raw)
    assert first == pipeline.stable_id("fixture", "a" * 40, "train", "17", raw)
    assert first != pipeline.stable_id("fixture", "b" * 40, "train", "17", raw)


def test_record_identifier_supports_conversation_and_trajectory():
    assert pipeline.record_id({"sample_id": "conversation-1"}) == "conversation-1"
    assert pipeline.record_id({"trajectory_id": "trajectory-1"}) == "trajectory-1"
