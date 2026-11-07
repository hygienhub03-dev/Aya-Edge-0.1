from aya_edge.agents import AgentFinalEnvelope, ToolCallEnvelope
from aya_edge.evaluation import (
    AgentCompletionScore,
    SchemaValidationError,
    SimulatedRuntime,
    ToolSpec,
    score_trace,
    validate_arguments,
)


def schema(name):
    return {"type": "object", "required": [name], "properties": {name: {"type": "string", "minLength": 1}}, "additionalProperties": False}


def test_tool_call_round_trip_and_final_no_tool_envelope():
    call = ToolCallEnvelope("set_theme", {"theme": "dark"}, call_id="c1", step=2)
    assert ToolCallEnvelope.from_dict(call.to_dict()) == call
    assert AgentFinalEnvelope("4").to_dict()["type"] == "final"


def test_schema_validation_reports_required_and_extra_fields():
    try:
        validate_arguments({"extra": "x"}, schema("theme"))
    except SchemaValidationError as exc:
        messages = " ".join(issue.message for issue in exc.issues)
        assert "required" in messages and "additional" in messages
    else:
        raise AssertionError("invalid arguments should be rejected")


def test_no_tool_case_scores_complete_without_calls():
    trace = SimulatedRuntime([]).run([])
    score = score_trace(trace, expected_tools=[], expected_state={})
    assert score.final_completion == 1.0
    assert score.selection == 1.0
    assert score.geometric_mean() == 1.0


def test_multi_step_state_and_argument_scoring():
    runtime = SimulatedRuntime([
        ToolSpec("set_name", schema("name"), lambda args, state: state.update(name=args["name"])),
        ToolSpec("set_language", schema("language"), lambda args, state: state.update(language=args["language"])),
    ])
    trace = runtime.run([
        ToolCallEnvelope("set_name", {"name": "Aya"}, step=0),
        ToolCallEnvelope("set_language", {"language": "Japanese"}, step=1),
    ])
    score = score_trace(trace, expected_tools=["set_name", "set_language"], expected_arguments=[{"name": "Aya"}, {"language": "Japanese"}], expected_state={"name": "Aya", "language": "Japanese"})
    assert trace.state == {"name": "Aya", "language": "Japanese"}
    assert score.selection == score.arguments == score.execution == score.final_completion == 1.0


def test_injected_failure_can_be_recovered():
    runtime = SimulatedRuntime([ToolSpec("set_theme", schema("theme"), lambda args, state: state.update(theme=args["theme"]))], fail_on_calls={"set_theme": 1})
    trace = runtime.run([ToolCallEnvelope("set_theme", {"theme": "light"}), ToolCallEnvelope("set_theme", {"theme": "light"}, step=1)])
    assert [result.ok for result in trace.results] == [False, True]
    score = score_trace(trace, expected_tools=["set_theme", "set_theme"], expected_state={"theme": "light"}, recovered_after_failure=True)
    assert score.recovery == 1.0 and score.final_completion == 1.0


def test_score_vector_is_bounded_and_complete():
    score = AgentCompletionScore(*(0.5 for _ in range(8)))
    assert set(score.as_dict()) == {"intent", "plan", "selection", "arguments", "execution", "observation", "recovery", "final_completion"}
