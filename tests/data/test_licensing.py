import pytest

import pipeline


def test_unapproved_source_fails_closed(fixture_source):
    source = fixture_source("blocked", approval_status="blocked_pending_review", license_verified=False)
    ok, _ = pipeline.is_source_approved(source)
    assert not ok
    with pytest.raises(pipeline.PipelineError, match="blocked from training"):
        pipeline.ensure_trainable(source, "blocked")


def test_conditional_approval_requires_conditions_to_be_satisfied(fixture_source):
    source = fixture_source("conditional", approval_status="approved_with_conditions", conditions_satisfied=False)
    assert pipeline.is_source_approved(source)[0] is False
    source["conditions_satisfied"] = True
    assert pipeline.is_source_approved(source) == (True, "approved")
