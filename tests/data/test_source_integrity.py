import json

import pytest

import pipeline


def test_jsonl_resume_count_and_corruption_detection(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text(json.dumps({"x": 1}) + "\n" + json.dumps({"x": 2}) + "\n", encoding="utf-8")
    assert pipeline.existing_jsonl_count(path) == 2
    path.write_text('{"x":1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="Invalid JSON"):
        pipeline.existing_jsonl_count(path)
