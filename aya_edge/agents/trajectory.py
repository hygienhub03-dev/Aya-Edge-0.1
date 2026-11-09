"""Observable agent-training trajectories; deliberately excludes private reasoning."""
from __future__ import annotations

from typing import Any, Mapping

from aya_edge.evaluation.validation import SchemaValidationError, ValidationIssue, validate_json

TRAJECTORY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AYA Edge Observable Agent Trajectory",
    "type": "object",
    "required": ["task", "context", "steps", "final_response", "success"],
    "properties": {
        "task": {"type": "string", "minLength": 1},
        "context": {"type": "object"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "required": ["rationale", "tool", "arguments", "result"],
            "properties": {
                "rationale": {"type": "string", "minLength": 1},
                "tool": {"type": "string", "minLength": 1},
                "arguments": {"type": "object"},
                "result": {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}, "output": {}, "error": {"type": "string"}}},
                "recovery": {"type": "string", "minLength": 1},
                "verification": {"type": "object", "required": ["passed", "evidence"], "properties": {"passed": {"type": "boolean"}, "evidence": {"type": "string", "minLength": 1}}},
            },
            "additionalProperties": False,
        }},
        "final_response": {"type": "string", "minLength": 1},
        "success": {"type": "boolean"},
    },
    "additionalProperties": False,
}


def validate_trajectory(value: Mapping[str, Any]) -> None:
    """Raise a useful validation error for malformed observable trajectories."""
    issues = validate_json(value, TRAJECTORY_SCHEMA)
    if not issues:
        for index, step in enumerate(value.get("steps", [])):
            result = step.get("result", {})
            if result.get("ok") is False and "recovery" not in step:
                issues.append(ValidationIssue(f"$.steps[{index}].recovery", "is required after a failed result"))
    if issues:
        raise SchemaValidationError(issues)
