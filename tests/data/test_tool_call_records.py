import pipeline


def test_canonical_tool_call_export_pairs_actual_results_and_marks_derived_ids():
    provenance = {"original_record_id": "1", "source_split": "train", "synthetic": False, "transformation_version": "1.0.0"}
    events = pipeline.trajectory_from_messages([
        {"role": "user", "content": "Use the lookup tool and report what happened."},
        {"role": "assistant", "tool_calls": [{"id": "call-source", "function": {"name": "lookup", "arguments": "{\"key\":\"x\"}"}}]},
        {"role": "tool", "tool_call_id": "call-source", "content": {"error": "not found"}},
        {"role": "assistant", "content": "The lookup failed because the value was not found."},
    ])
    record = {"trajectory_id": "trajectory-1", "source_id": "fixture", "source_revision": "a" * 40, "license_id": "fixture license", "provenance": provenance, "events": events}
    [call] = pipeline.canonical_tool_calls_from_trajectory(record)
    assert call["source_call_id"] == "call-source"
    assert call["call_id_origin"] == "source"
    assert call["result"] == {"error": "not found"}
    assert call["status"] == "error"

    events_without_source_id = pipeline.trajectory_from_messages([
        {"role": "user", "content": "Use the lookup tool and report what happened."},
        {"role": "assistant", "function_call": {"name": "lookup", "arguments": "{\"key\":\"x\"}"}},
        {"role": "tool", "content": "Found value."},
        {"role": "assistant", "content": "I found the value."},
    ])
    record["events"] = events_without_source_id
    [derived] = pipeline.canonical_tool_calls_from_trajectory(record)
    assert derived["call_id_origin"] == "derived_from_trajectory_order"
    assert "source_call_id" not in derived
    assert derived["result"] == "Found value."
