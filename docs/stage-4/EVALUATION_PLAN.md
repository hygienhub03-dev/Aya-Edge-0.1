# AYA Edge — Stage 4 Evaluation Plan

## Governing link

Evaluation follows the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md): task completion is first-class, and a proposed tool call is distinct from actual execution.

## Evaluation tiers

1. **Unit/protocol:** JSON/schema parsing, permission checks, budget enforcement, state transitions, verifier correctness, reset determinism.
2. **Offline replay:** fixed Stage 3 trajectories and counterfactual next-action labels.
3. **Sandbox tasks:** seeded, stateful workflows with independent final-state verification.
4. **Robustness:** unseen templates, distractors, injected failures, contradictory observations, insufficient information, prompt injection, and repeated trials.
5. **Regression:** Stage 3 versus candidate on general language, instruction following, context retrieval, structured output, uncertainty, refusal, and safety.
6. **Restricted trial:** only after all gates; read-only or approval-gated operations with audit review.

## Metric vector

Report counts, denominators, confidence intervals where appropriate, and failure reasons. Never collapse the vector into one number without publishing the components.

| Metric | Definition |
|---|---|
| Tool-need F1 | Correct tool/no-tool/clarify/abstain classification |
| Selection accuracy | Exact appropriate tool; also wrong-tool rate |
| Schema validity | Runtime-accepted calls / attempted calls |
| Argument correctness | Semantically correct required and optional values |
| Runtime acceptance | Calls accepted after permissions and validation |
| Verified completion | Independent verifier pass at episode end |
| `pass@1`, `pass@3` | Success on first attempt and at least one of three repeats |
| Recovery success | Verified success after injected failure without unsafe/repeated calls |
| Constraint adherence | All user/task constraints satisfied |
| False completion | Claim of success without verified success |
| Unnecessary action rate | Calls not needed for the verified solution |
| Efficiency | Turns, calls, tokens, wall-clock, and cost per verified success |
| Hard-gate violations | Permission, safety, approval, budget, or audit failures |
| Verifier disagreement | Difference between policy report and independent outcome |

A diagnostic Agent Completion Score may report the vector across intent, plan, selection, arguments, execution, interpretation, and completion, but full credit requires verified completion. A geometric/gated aggregate is preferable to an additive score that lets fluent planning mask execution failure.

## Required task matrix

Include single- and multi-step tasks, no-tool questions, ambiguous requests, required clarification, stateful dependencies, partial success, unavailable tools, transient and permanent failures, permission denials, approval-required actions, malformed schemas, stale data, contradictory results, and tasks where the correct result is safe non-completion.

## Statistical and procedural controls

Pin task IDs, revisions, prompts, tool schemas, initial states, decoding, budgets, runtime, verifier, and seed. Keep test tasks and canaries isolated. Use paired comparisons with Stage 3, bootstrap or exact intervals for proportions, and report all attempted runs including failures. Review all hard-gate incidents and a stratified sample of successes. Public suites such as BFCL, ToolSandbox, τ-bench, or AgentBench are supplementary diagnostics, not exit evidence by themselves.

## Report format

Every report includes checkpoint/hash, Git commit, data/environment/verifier versions, configuration, seeds, metric definitions, denominators, confidence intervals, baseline comparison, regression table, failure taxonomy, representative traces, safety review, and promotion decision.

## Non-claims

A benchmark score does not prove general autonomy, real-world safety, or transfer to tools absent from the test environment. `pass@3` is repeatability on a defined task set, not reliability in general. No evaluation score overrides a hard safety or permission failure.

## References

- [Official model capability specification](../../MODEL_SPECIFICATION.md)
- [Agentic models research synthesis](../research/AGENTIC_MODELS_RESEARCH.md)
- [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [ToolSandbox](https://arxiv.org/html/2408.04682v1)
- [τ-bench](https://arxiv.org/html/2406.12045)
- [AgentBench](https://arxiv.org/html/2308.03688v3)
