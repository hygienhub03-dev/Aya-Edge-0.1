# AYA Edge — Stage 4: Reinforcement Learning

## Governing specification

This plan is subordinate to the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Stage 4 starts from the verified Stage 3 supervised checkpoint and preserves AYA Edge's identity as a compact, agent-first assistant: **Understand → Plan → Act → Observe → Adapt → Complete**.

## Purpose

Stage 4 improves decision quality and verified task execution through feedback. The target is not persuasive agent prose or a higher reward number; it is a policy that selects useful actions, uses typed tools correctly, recovers from errors, respects constraints, and only claims completion when an external verifier confirms the requested outcome.

## Scope and operating contract

The model proposes the next action. A sandbox/runtime validates schemas, enforces permissions, executes tools, records state, and returns an observation envelope. A task verifier determines whether the requested end state was achieved. The model's completion claim is never evidence by itself.

Every trajectory should make the following state explicit:

1. user goal and constraints;
2. current state and available tools;
3. one bounded next decision;
4. validated tool call or a clarification/abstention;
5. runtime observation, status, provenance, and retryability;
6. updated state and a completion decision.

Structured ReAct is the default interaction pattern. Branching search or reflection is experimental and must remain bounded by risk, latency, and action budgets.

## Training progression

| Phase | Environment | Main question | Promotion evidence |
|---|---|---|---|
| A. Offline replay | Frozen, logged Stage 3 trajectories and counterfactual labels | Does the objective prefer valid, useful decisions? | No regression in held-out supervised and safety suites; replay metrics improve |
| B. Deterministic sandbox | Synthetic tools, seeded state, injected failures | Does the policy reach verified end states? | Held-out completion and recovery gains with zero hard-gate violations |
| C. Shadow evaluation | Read-only or no-side-effect tasks, unseen task templates | Does it generalize without exploiting the evaluator? | Independent verifier agreement, repeatability, no false-completion increase |
| D. Restricted trial | Least-privilege tools, approvals for irreversible effects | Is limited interaction safe and reproducible? | Pre-registered gates pass; immutable audit trail and rollback checkpoint |

Broad online RL, unrestricted external access, and unsupervised autonomous operation are out of scope for this stage.

## Required deliverables

- Versioned offline trajectory and task manifests.
- Typed tool/runtime protocol and deterministic sandbox.
- Reward and hard-gate specification.
- Reproducible training configurations and seeds.
- Stage 3 comparison report, evaluation harness, and failure taxonomy.
- Best Stage 4 checkpoint plus preserved Stage 3 rollback checkpoint.
- Experiment log with rejected runs and reasons.

## Explicit non-claims

Stage 4 does not establish general autonomy, consciousness, perfect planning, factual omniscience, safe operation in arbitrary environments, or production readiness. Sandbox success does not prove success with live services. Public benchmark results do not replace task-level verification. A reward improvement does not imply capability improvement unless independent held-out outcomes agree.

## References

- [Official model capability specification](../../MODEL_SPECIFICATION.md)
- [Agentic models research synthesis](../research/AGENTIC_MODELS_RESEARCH.md)
