import pipeline


def test_all_canonical_record_schemas_validate_with_provenance():
    schemas = pipeline.load_schemas()
    provenance = {"original_record_id": "1", "source_split": "train", "synthetic": False, "transformation_version": "1.0.0"}
    records = [
        {"sample_id": "doc", "source_id": "fixture", "source_revision": "20261001", "license_id": "CC BY-SA 4.0", "category": "general", "title": "Title", "text": "A valid documented article sample.", "provenance": provenance, "raw_record": {"page_id": "1"}},
        {"sample_id": "conv", "source_id": "fixture", "source_revision": "a" * 40, "license_id": "CC BY 4.0", "category": "instruction", "messages": [{"role": "user", "content": "Please explain this clearly."}, {"role": "assistant", "content": "Here is a clear explanation."}], "provenance": provenance},
        {"trajectory_id": "traj", "source_id": "fixture", "source_revision": "b" * 40, "license_id": "CC BY 4.0", "category": "tool_use", "events": [{"sequence_index": 0, "event_type": "user_goal", "content": "Find a value."}, {"sequence_index": 1, "event_type": "tool_call", "tool_name": "lookup", "arguments": {"key": "x"}}, {"sequence_index": 2, "event_type": "tool_result", "content": "A value."}, {"sequence_index": 3, "event_type": "final_answer", "content": "The value is A."}], "provenance": provenance},
        {"call_id": "call-1", "source_id": "fixture", "source_revision": "c" * 40, "license_id": "CC BY 4.0", "tool_name": "lookup", "arguments": {"key": "x"}, "result": "A value.", "status": "success", "sequence_index": 1, "provenance": provenance, "raw_call": {"function": "lookup"}, "raw_result": "A value."},
    ]
    assert [pipeline.validate_record(record, schemas) for record in records] == [[], [], [], []]
