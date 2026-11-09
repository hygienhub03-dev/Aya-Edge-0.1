import pytest

from aya_edge.agents import validate_trajectory
from aya_edge.evaluation import SchemaValidationError


def test_observable_trajectory_accepts_recovery_and_verification():
    validate_trajectory({
        "task": "Calculate 2 + 2", "context": {},
        "steps": [{"rationale": "Use the calculator.", "tool": "calculator", "arguments": {"expression": "2+2"}, "result": {"ok": True, "output": "4"}, "verification": {"passed": True, "evidence": "calculator returned 4"}}],
        "final_response": "4", "success": True,
    })


def test_failed_step_requires_observable_recovery():
    with pytest.raises(SchemaValidationError, match="recovery"):
        validate_trajectory({
            "task": "Find note", "context": {},
            "steps": [{"rationale": "Look up the note.", "tool": "notes.get", "arguments": {}, "result": {"ok": False, "error": "missing"}}],
            "final_response": "Not found", "success": False,
        })
