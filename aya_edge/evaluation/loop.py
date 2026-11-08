"""Iterative model/policy-in-the-loop evaluation against the runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from aya_edge.agents import AgentClarificationEnvelope, AgentFinalEnvelope, Envelope, ToolCallEnvelope
from .simulator import ExecutionTrace, SimulatedRuntime


@dataclass(frozen=True)
class PolicyObservation:
    prompt: str
    results: tuple[object, ...]
    step: int


Policy = Callable[[PolicyObservation], Envelope]


@dataclass(frozen=True)
class LoopResult:
    trace: ExecutionTrace
    terminal: AgentFinalEnvelope | AgentClarificationEnvelope | None
    policy_steps: int


def run_model_in_loop(
    policy: Policy,
    runtime: SimulatedRuntime,
    prompt: str,
    *,
    max_steps: int | None = None,
) -> LoopResult:
    """Let a policy choose each call after observing prior tool results.

    The policy is deliberately injected so tests can use a scripted policy and
    future runs can wrap a real model decoder. The runtime remains authoritative
    for schemas, execution, state, injected failures, and step limits.
    """
    limit = max_steps if max_steps is not None else runtime.max_steps
    if limit <= 0:
        raise ValueError("max_steps must be positive")
    trace = runtime.begin()
    terminal: AgentFinalEnvelope | AgentClarificationEnvelope | None = None
    for step in range(limit):
        observation = PolicyObservation(prompt, tuple(trace.results), step)
        decision = policy(observation)
        if isinstance(decision, (AgentFinalEnvelope, AgentClarificationEnvelope)):
            terminal = decision
            break
        if not isinstance(decision, ToolCallEnvelope):
            raise TypeError("policy must return a tool call, final, or clarification envelope")
        call = ToolCallEnvelope(decision.tool_name, decision.arguments, decision.call_id, step)
        trace.events.append(runtime.execute(call, trace.state))
    return LoopResult(trace=trace, terminal=terminal, policy_steps=len(trace.events))
