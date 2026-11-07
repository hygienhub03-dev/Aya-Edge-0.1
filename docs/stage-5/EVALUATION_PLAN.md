# AYA Edge — Stage 5 Evaluation Plan

## Governing specification

This plan is subordinate to the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Evaluation must measure completed outcomes, not merely plausible plans or teacher similarity.

## 1. Evaluation populations

Maintain fixed, hidden, and versioned splits for: no-tool and clarify cases; single-step tools; multi-step stateful workflows; injected failures; safety and permission boundaries; structured output; general instruction and conversation; short/long context; and adversarial or distribution-shifted tasks. Prevent source, prompt-template, and near-duplicate leakage from teacher data.

Evaluate Stage 4, Stage 5-SFT, and Stage 5-preference checkpoints with identical prompts, tools, runtime, budgets, seeds/decoding policy, and evaluator versions. Repeat stochastic workflows at least three times where feasible and report `pass^1` and `pass^3`.

## 2. Execution-grounded Agent Completion Score

Report the full vector:

1. Intent and constraint recognition.
2. Plan validity and dependency order.
3. Tool-needed decision and tool selection.
4. Schema and semantic argument correctness.
5. Runtime acceptance and intermediate-state reachability.
6. Recovery success after injected failure.
7. Final-state verification and honest completion.
8. Turns, calls, retries, latency, and unnecessary-action rate.

A final answer receives full completion credit only when an independent verifier confirms the requested end state. A model statement such as “done” is untrusted evidence. Record partial success and a normalized failure reason.

## 3. Core metrics

- Tool-needed precision/recall and clarify/abstain quality.
- Exact tool-selection accuracy and wrong-tool rate.
- Schema validity, required-field accuracy, and semantic argument accuracy.
- Runtime acceptance, state reachability, final-state success, repeatability, and recovery success.
- False-completion rate, fabricated-result rate, permission/policy violations, repeated-failure rate, and unnecessary-call rate.
- Constraint adherence, response usefulness, structured parse/schema success, conversation consistency, and uncertainty calibration.
- General language/context performance, safety performance, latency, throughput, and memory.

## 4. Teacher and preference validation

Measure accepted-data precision on a stratified audit sample, validator agreement, reviewer agreement, rejection rate by reason, pair tie rate, pair winner agreement, outcome-verification coverage, and train/evaluation contamination. Teacher win rate is diagnostic only; it is never a promotion criterion by itself.

## 5. Mandatory regression gates

Compared with Stage 4 on fixed held-out suites, the candidate must not show an unacceptable regression in general language, context retrieval, structured outputs, safety/refusal behaviour, tool-call syntax, or conversation consistency. The project must pre-register numeric tolerances before inspecting results. Any breach requires investigation, replay/repair, or rejection; do not average away a critical safety or false-completion regression.

## 6. Reporting

Each report includes checkpoint and parent hashes, code commit, dataset/manifest and teacher-audit IDs, exact revisions and terms status, prompts/task IDs, runtime/tool schemas, decoding, budgets, evaluator version, complete metric vector, confidence intervals or repeat counts, failure examples, regressions, limitations, and promotion decision.

## 7. Minimum test cases

Include missing information, contradictory observations, malformed arguments, timeouts, denied permissions, unavailable tools, alternate-tool recovery, irreversible-action approval, empty results, partial state changes, user correction, no-tool answers, and tasks where honest non-completion is correct.

## References

- [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md)
- [Agentic Models Research Report](../research/AGENTIC_MODELS_RESEARCH.md)
- [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [ToolSandbox](https://arxiv.org/html/2408.04682v1)
- [τ-bench](https://arxiv.org/html/2406.12045)
- [AgentBench](https://arxiv.org/html/2308.03688v3)
