# AYA Edge — Stage 4 RL Objectives

## Governing link

All objectives below are subordinate to the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). They refine, rather than replace, the Stage 3 supervised baseline.

## Primary objective

Maximise verified task utility under explicit safety, permission, user-constraint, and resource limits. For task \(x\), the policy should choose an action sequence \(a_{1:T}\) that reaches a verifier-confirmed goal state while avoiding prohibited transitions and unnecessary work.

A useful optimization framing is:

```text
maximize  verified_outcome + useful_process_quality
subject to permission, safety, schema, budget, and honesty gates
```

The objective is deliberately constrained: a fluent answer, a valid JSON call, or a model assertion of success is insufficient without runtime evidence.

## Competencies to improve

1. **Intent and tool-need decisions:** distinguish answer, tool, clarification, abstention, and refusal cases.
2. **Plan and next-action quality:** preserve dependencies, constraints, and a bounded state representation.
3. **Tool selection:** choose the least-privileged tool that can satisfy the goal; avoid distractors.
4. **Argument correctness:** produce schema-valid and semantically grounded arguments from available evidence.
5. **Observation interpretation:** use status, provenance, affected state, and retryability rather than assumptions.
6. **Error recovery:** correct arguments, retry only when retryable, switch tools when justified, ask for clarification, or terminate safely.
7. **Verified completion:** stop when the verifier confirms the end state; otherwise report partial/non-completion honestly.
8. **Efficiency:** minimise turns, calls, tokens, and wall-clock time only after correctness and safety hold.

## Curriculum

- **A — reasoning optimisation:** offline preference/replay learning over correct next decisions, constraint tracking, uncertainty, and concise plans.
- **B — general task optimisation:** deterministic sandbox tasks with no-tool, clarify, structured-output, and multi-step outcome labels.
- **C — agentic optimisation:** typed calls, state changes, injected failures, alternate tools, and completion verification.
- **D — restricted robustness:** unseen templates, adversarial distractors, repeated trials, and read-only or approval-gated tasks.

Keep a no-RL Stage 3 policy in every comparison. Change one major variable at a time where practical.

## Hard constraints

The runtime must veto, not merely penalise:

- missing or invalid permissions;
- policy-prohibited or out-of-scope actions;
- attempts to bypass approval or sandbox boundaries;
- irreversible actions without explicit approval;
- malformed schemas or unknown tools;
- calls exceeding turn, tool, token, time, or monetary budgets;
- data exfiltration, hidden side effects, or unlogged execution.

A vetoed action may receive a diagnostic label, but cannot be made optimal by reward shaping.

## Success target

Promotion requires improvement over Stage 3 on held-out verified completion and recovery, with no hard-gate failure and no unacceptable regression in instruction following, structured output, safety, context, or general language checks.

## Non-claims

These objectives do not claim that learned reasoning traces are faithful, that reward maximisation finds the globally best plan, or that sandbox policies transfer automatically to live tools.

## References

- [Official model capability specification](../../MODEL_SPECIFICATION.md)
- [Agentic models research synthesis](../research/AGENTIC_MODELS_RESEARCH.md)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)
