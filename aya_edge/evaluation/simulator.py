"""A deterministic, dependency-light tool runtime for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from aya_edge.agents import ToolCallEnvelope, ToolResultEnvelope
from .validation import SchemaValidationError, validate_arguments


ToolHandler = Callable[[dict[str, Any], dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    parameters: Mapping[str, Any]
    handler: ToolHandler


@dataclass
class ExecutionTrace:
    initial_state: dict[str, Any]
    state: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def results(self) -> list[ToolResultEnvelope]:
        return [event["result"] for event in self.events if event["kind"] == "tool_result"]

    @property
    def successful_calls(self) -> list[ToolCallEnvelope]:
        return [event["call"] for event in self.events if event["kind"] == "tool_result" and event["result"].ok]


class SimulatedRuntime:
    def __init__(self, tools: list[ToolSpec], initial_state: Mapping[str, Any] | None = None,
                 fail_on_calls: Mapping[str, int] | None = None, max_steps: int = 8):
        self.tools = {tool.name: tool for tool in tools}
        self.initial_state = dict(initial_state or {})
        self.fail_on_calls = dict(fail_on_calls or {})
        self.max_steps = max_steps
        self._call_counts: dict[str, int] = {}

    def run(self, calls: list[ToolCallEnvelope]) -> ExecutionTrace:
        if len(calls) > self.max_steps:
            raise ValueError(f"maximum steps exceeded: {self.max_steps}")
        state = dict(self.initial_state)
        trace = ExecutionTrace(dict(self.initial_state), state)
        self._call_counts = {}
        for call in calls:
            trace.events.append(self.execute(call, state))
        return trace

    def execute(self, call: ToolCallEnvelope, state: dict[str, Any]) -> dict[str, Any]:
        count = self._call_counts.get(call.tool_name, 0) + 1
        self._call_counts[call.tool_name] = count
        tool = self.tools.get(call.tool_name)
        if tool is None:
            result = ToolResultEnvelope(call.call_id, call.tool_name, False, error="unknown tool", step=call.step)
        else:
            try:
                validate_arguments(call.arguments, tool.parameters)
                failure_count = self.fail_on_calls.get(call.tool_name, 0)
                if count <= failure_count:
                    raise RuntimeError("injected tool failure")
                output = tool.handler(dict(call.arguments), state)
                result = ToolResultEnvelope(call.call_id, call.tool_name, True, output=output, step=call.step)
            except (SchemaValidationError, RuntimeError, ValueError, KeyError) as exc:
                result = ToolResultEnvelope(call.call_id, call.tool_name, False, error=str(exc), step=call.step)
        return {"kind": "tool_result", "call": call, "result": result, "state": dict(state)}


def set_value(key: str, argument: str = "value") -> ToolHandler:
    """Build a simple fixture handler that writes one argument into state."""
    def handler(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        state[key] = arguments[argument]
        return {key: state[key]}
    return handler
