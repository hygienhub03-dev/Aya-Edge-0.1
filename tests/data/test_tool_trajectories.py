import json

import pipeline


def call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_single_tool_call_preserves_goal_call_result_and_final_answer():
    events = pipeline.trajectory_from_messages([
        {"role": "user", "content": "Find the current status."},
        {"role": "assistant", "content": "I will search.", "tool_calls": [call("c1", "search", {"query": "status"})]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "Status: ready."},
        {"role": "assistant", "content": "The status is ready."},
    ])
    assert [e["event_type"] for e in events] == ["user_goal", "assistant_decision", "tool_call", "tool_result", "final_answer"]
    assert events[2]["arguments"] == {"query": "status"}
    assert events[3]["content"] == "Status: ready."


def test_multiple_sequential_tool_calls_and_second_decision_are_ordered():
    events = pipeline.trajectory_from_messages([
        {"role": "user", "content": "Look up and then summarize."},
        {"role": "assistant", "tool_calls": [call("c1", "search", {"query": "topic"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "Found one page."},
        {"role": "assistant", "content": "I need a second source.", "tool_calls": [call("c2", "open", {"url": "https://example.test"})]},
        {"role": "tool", "tool_call_id": "c2", "content": "Second source confirms it."},
        {"role": "assistant", "content": "Here is the summary."},
    ])
    assert [e["sequence_index"] for e in events] == list(range(len(events)))
    assert [e["event_type"] for e in events].count("tool_call") == 2
    assert [e["event_type"] for e in events].count("tool_result") == 2
    assert events[3]["content"] == "I need a second source."
    assert events[-1]["content"] == "Here is the summary."


def test_tool_failure_is_retained_and_not_rewritten_as_success():
    events = pipeline.trajectory_from_messages([
        {"role": "user", "content": "Try to retrieve the record."},
        {"role": "assistant", "tool_calls": [call("c1", "lookup", {"id": "x"})]},
        {"role": "tool", "tool_call_id": "c1", "content": {"error": "timeout"}},
        {"role": "assistant", "content": "I could not complete the lookup because the tool timed out."},
    ])
    result = next(e for e in events if e["event_type"] == "tool_result")
    assert result["content"] == {"error": "timeout"}
    assert events[-1]["event_type"] == "final_answer"
    assert "could not complete" in events[-1]["content"]


def test_missing_tool_result_is_rejected_not_fabricated():
    messages = [
        {"role": "user", "content": "Do something."},
        {"role": "assistant", "tool_calls": [call("c1", "tool", {"x": 1})]},
    ]
    try:
        pipeline.trajectory_from_messages(messages)
    except ValueError as exc:
        assert "no tool result" in str(exc)
    else:
        raise AssertionError("missing result should be rejected")
