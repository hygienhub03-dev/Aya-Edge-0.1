import pytest

from aya_edge.training.agent_sft import SFTDataError, encode_trajectory, validate_trajectory


class FakeTokenizer:
    def encode(self, text):
        # Stable one-token-per-character IDs are sufficient to test boundaries.
        return [1000 + ord(char) for char in text]


def tool_schema():
    return [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }]


def valid_row():
    return {
        "uuid": "row-1",
        "tools": tool_schema(),
        "messages": [
            {"role": "user", "content": "Find x."},
            {"role": "assistant", "content": "I will look.", "tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": {"key": "x"}}}]},
            {"role": "tool", "name": "lookup", "tool_call_id": "c1", "content": "value=x"},
            {"role": "assistant", "content": "The value is x."},
        ],
    }


def test_tool_calls_are_validated_and_normalized():
    messages = validate_trajectory(valid_row())
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"key": "x"}


def test_unknown_tool_is_rejected():
    row = valid_row()
    row["messages"][1]["tool_calls"][0]["function"]["name"] = "missing"
    with pytest.raises(SFTDataError, match="undeclared"):
        validate_trajectory(row)


def test_unmatched_tool_result_is_rejected():
    row = valid_row()
    row["messages"][2]["tool_call_id"] = "wrong"
    with pytest.raises(SFTDataError, match="does not match"):
        validate_trajectory(row)


def test_assistant_only_mask_has_supervised_targets_only_in_assistant_turns():
    encoded = encode_trajectory(valid_row(), FakeTokenizer(), max_length=10000)
    assert len(encoded.input_ids) == len(encoded.labels)
    assert encoded.supervised_tokens > 0
    # User and tool-result text must not be supervised.
    user_start = encoded.input_ids.index(1000 + ord("<"))
    assert all(label == -100 for label in encoded.labels[:user_start])
    # The final assistant content is supervised.
    final_text = "The value is x."
    final_ids = [1000 + ord(char) for char in final_text]
    assert any(encoded.input_ids[index:index + len(final_ids)] == tuple(final_ids) for index in range(len(encoded.input_ids)))


def test_oversized_trajectory_is_rejected():
    with pytest.raises(SFTDataError, match="max_length"):
        encode_trajectory(valid_row(), FakeTokenizer(), max_length=8)
