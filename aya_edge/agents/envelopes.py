"""Typed envelopes exchanged between an agent and the evaluation runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ToolCallEnvelope:
    """A single requested tool invocation.

    ``arguments`` is deliberately kept as JSON-compatible data. Runtime schema
    validation belongs to :mod:`aya_edge.evaluation.validation` so envelopes
    remain useful at model and transport boundaries.
    """

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = "call-1"
    step: int = 0

    def __post_init__(self) -> None:
        if not self.tool_name or not isinstance(self.tool_name, str):
            raise ValueError("tool_name must be a non-empty string")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolCallEnvelope":
        if value.get("type", "tool_call") != "tool_call":
            raise ValueError("not a tool-call envelope")
        return cls(
            tool_name=value["tool_name"],
            arguments=value.get("arguments", {}),
            call_id=value.get("call_id", "call-1"),
            step=value.get("step", 0),
        )


@dataclass(frozen=True)
class ToolResultEnvelope:
    """The runtime's observable result for one tool call."""

    call_id: str
    tool_name: str
    ok: bool
    output: Any = None
    error: str | None = None
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "tool_result",
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "output": self.output,
            "step": self.step,
        }
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class AgentFinalEnvelope:
    """A final response that intentionally requests no tool call."""

    content: str
    call_id: str = "final"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "final", "call_id": self.call_id, "content": self.content}


@dataclass(frozen=True)
class AgentClarificationEnvelope:
    """A safe response when the task lacks information needed to act."""

    question: str
    call_id: str = "clarify"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "clarify", "call_id": self.call_id, "question": self.question}


Envelope = ToolCallEnvelope | AgentFinalEnvelope | AgentClarificationEnvelope
