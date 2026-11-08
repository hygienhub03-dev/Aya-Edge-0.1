"""Diagnostic score vector for compact agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import prod
from typing import Any, Mapping

from .simulator import ExecutionTrace


_DIMENSIONS = ("intent", "plan", "selection", "arguments", "execution", "observation", "recovery", "final_completion")


@dataclass(frozen=True)
class AgentCompletionScore:
    intent: float = 0.0
    plan: float = 0.0
    selection: float = 0.0
    arguments: float = 0.0
    execution: float = 0.0
    observation: float = 0.0
    recovery: float = 0.0
    final_completion: float = 0.0

    def __post_init__(self) -> None:
        for name in _DIMENSIONS:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def geometric_mean(self) -> float:
        return prod(self.as_dict().values()) ** (1 / len(_DIMENSIONS))

    @property
    def completion(self) -> float:
        return self.final_completion


def score_trace(
    trace: ExecutionTrace,
    *,
    expected_tools: list[str] | None = None,
    expected_state: Mapping[str, Any] | None = None,
    expected_arguments: list[Mapping[str, Any]] | None = None,
    recovered_after_failure: bool | None = None,
    observed_results: bool = True,
    intent_correct: bool | None = None,
    plan_correct: bool | None = None,
) -> AgentCompletionScore:
    """Score a runtime trace with transparent, bounded heuristics.

    Callers may supply model-judged intent/plan and expected call arguments;
    runtime-derived dimensions remain deterministic and reproducible.
    """
    calls = [event["call"] for event in trace.events]
    results = trace.results
    expected_tools = expected_tools or []
    selected = [call.tool_name for call in calls]
    selection = 1.0 if selected == expected_tools else (1.0 if not expected_tools and not calls else 0.0)
    if expected_arguments is None:
        arguments = sum(result.ok for result in results) / len(results) if results else (1.0 if not expected_tools else 0.0)
    else:
        arguments = sum(call.arguments == dict(expected) for call, expected in zip(calls, expected_arguments)) / max(len(expected_arguments), 1)
    execution = sum(result.ok for result in results) / max(len(results), 1) if results else (1.0 if not expected_tools else 0.0)
    final = 1.0 if expected_state is not None and trace.state == dict(expected_state) else (1.0 if expected_state is None and not expected_tools else 0.0)
    failures = sum(not result.ok for result in results)
    if recovered_after_failure is None:
        recovery = 1.0 if failures == 0 else 0.0
    else:
        recovery = float(recovered_after_failure) if failures else 1.0
    return AgentCompletionScore(
        intent=float(intent_correct) if intent_correct is not None else 0.0,
        plan=float(plan_correct) if plan_correct is not None else 0.0,
        selection=selection,
        arguments=max(0.0, min(1.0, arguments)), execution=execution,
        observation=float(observed_results), recovery=recovery, final_completion=final,
    )
