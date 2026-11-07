# AYA Edge — Stage 4 Training Plan

## Governing link and starting point

This plan is subordinate to the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Start from the verified Stage 3 checkpoint, its tokenizer/configuration, tool protocol, evaluation harness, and regression report. Freeze a named Stage 3 rollback checkpoint before every RL run.

## Data and environment contract

Maintain separate versioned manifests for task definitions, tool schemas, initial states, trajectories, verifier code, reward configuration, and train/validation/test splits. Keep test tasks and adversarial canaries inaccessible to training. Synthetic tasks must cover no-tool answers, clarification, abstention, safe refusal, single calls, multi-step state changes, partial information, contradictory results, timeouts, permission failures, malformed arguments, unavailable tools, and alternate-tool recovery.

The sandbox must provide deterministic seeds, virtual time, fake credentials, least-privilege capabilities, resettable state, immutable event logs, and independent end-state verification. No secrets or uncontrolled network access are permitted.

## Phased progression

### Phase 0 — Instrumentation and replay

- Validate trajectory serialization and typed call/result envelopes.
- Replay Stage 3 on fixed tasks; establish component baselines.
- Verify reward calculations against hand-scored examples.
- Run evaluator mutation tests and deliberately corrupted traces.

Exit only when replay is deterministic and every score has a traceable evidence source.

### Phase A — Offline objective optimisation

Use logged or generated trajectories with verified labels. Begin with conservative policy/value or preference objectives that improve next-action quality without allowing the policy to invent outcomes. Hold out task templates, wording, states, and failure combinations. Compare every run with Stage 3 using identical decoding and budgets.

### Phase B — Deterministic sandbox RL

Train against resettable tools and bounded episodes. Introduce failures gradually: first syntax and transient errors, then permission denials, stale state, contradictory observations, and distractor tools. Cap episode turns and retries. Sample successful, failed, vetoed, and environment-error episodes for review.

### Phase C — Shadow and adversarial evaluation

Run unseen tasks in read-only/no-side-effect mode. Keep the policy's proposed calls and predicted outcomes, but do not grant mutation authority. Test verifier disagreement, reward perturbation, prompt injection in tool results, and repeated trials from varied initial states.

### Phase D — Restricted trial

Only after exit gates pass, allow narrowly scoped tools with least privilege. Require explicit user/runtime approval for irreversible effects. Use immutable audit logs, kill switches, rate/cost limits, automatic rollback where possible, and a named operator review. This is a controlled experiment, not deployment.

## Reproducibility checklist

Record Git commit, checkpoint hash, dataset/environment/verifier versions, reward weights, model/config/tokenizer, optimizer and schedule, seeds for model/data/environment, sampler/temperature, max turns and tool calls, hardware/software versions, wall-clock, failures, and exact command. Save RNG states, manifests, logs, checkpoints, and evaluation outputs. A second run with the same seed must reproduce task ordering, actions, reward components, and metrics within declared numerical tolerance.

## Monitoring and stop conditions

Track loss or objective terms, verified completion, pass@1 and pass@3, recovery, false completion, hard-gate violations, wrong-tool and bad-argument rates, unnecessary calls, turns/calls to success, verifier disagreement, safety incidents, throughput, latency, memory, and cost. Stop or roll back on non-finite training, permission bypass, unexplained reward spikes, repetition loops, rising false completion, safety regression, or verifier disagreement that cannot be explained.

## Non-claims

Offline gains do not establish live-tool competence; deterministic sandbox transfer is an empirical question. A reproducible run can reproduce a flawed objective. Restricted trials do not constitute production authorization.

## References

- [Official model capability specification](../../MODEL_SPECIFICATION.md)
- [Agentic models research synthesis](../research/AGENTIC_MODELS_RESEARCH.md)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [τ-bench](https://arxiv.org/html/2406.12045)
