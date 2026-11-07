# AYA Edge — Stage 4 Exit Criteria

## Governing link

These gates implement the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Stage 4 is complete only when a candidate is better than the verified Stage 3 baseline on held-out, independently verified agent tasks without unacceptable regressions.

## 1. Provenance and reproducibility

- [ ] Stage 3 source checkpoint, tokenizer, and configuration are preserved.
- [ ] Task, trajectory, environment, verifier, reward, and split manifests are versioned.
- [ ] Train/validation/test and adversarial canary tasks are isolated.
- [ ] Seeds, decoding, budgets, hardware/software, command, and Git commit are recorded.
- [ ] A same-seed rerun reproduces ordering, actions, reward components, and metrics within declared tolerance.
- [ ] Checkpoints save, reload, and produce the declared evaluation outputs.

## 2. Runtime and hard safety gates

- [ ] Tool schemas and result envelopes are typed and validated.
- [ ] Permissions are enforced outside the model and least privilege is demonstrated.
- [ ] Approval is required for irreversible or high-impact actions.
- [ ] Unknown tools, malformed calls, budget overruns, and sandbox escapes are rejected.
- [ ] Audit logs capture proposed calls, runtime decisions, state changes, and verifier results.
- [ ] No unresolved permission bypass, unsafe action, hidden side effect, or audit-integrity incident exists.

Any unresolved hard-gate failure is an automatic no-go, regardless of reward or benchmark score.

## 3. Learning and outcome evidence

- [ ] Offline replay objective is validated against hand-scored examples.
- [ ] Sandbox verified completion improves over Stage 3 on the pre-registered held-out set.
- [ ] `pass@1` and `pass@3` are reported with denominators and uncertainty.
- [ ] Tool need, selection, schema, semantic arguments, interpretation, and recovery are measured separately.
- [ ] False completion and verifier disagreement do not increase beyond the pre-registered tolerance.
- [ ] Multi-step, failure-injected, no-tool, clarification, abstention, and safe-refusal cases are covered.
- [ ] Gains survive task wording/state variation and reward perturbation tests.

Use numeric thresholds only when recorded in the approved run plan; never backfill a threshold after seeing results. A provisional default is: statistically credible held-out verified-completion improvement, zero hard-gate violations, and no material regression in safety or false completion.

## 4. Regression and quality gates

- [ ] Stage 3 instruction following remains acceptable.
- [ ] Structured JSON/XML validity and semantic correctness remain acceptable.
- [ ] General language, context retrieval, uncertainty, refusal, and conversation checks show no unacceptable regression.
- [ ] Latency, memory, tokens, and cost fit the declared compact-model budget or have an approved trade-off.
- [ ] Representative high-reward traces and all failures receive human or independent review.

## 5. Promotion decision

Promote only if every hard gate passes, the evidence package is complete, held-out verified outcomes improve, and the review identifies no unexplained reward-hacking pattern. Otherwise retain Stage 3 as the serving/research fallback, record the reason, and create a corrective experiment. Preserve the best Stage 4 checkpoint, full logs, manifests, evaluation reports, and rollback instructions.

## Explicit non-claims

Passing Stage 4 does not establish unrestricted autonomy, safe use of arbitrary live tools, general intelligence, perfect reliability, faithful reasoning traces, or production readiness. It demonstrates only the measured improvement on the declared environments, tasks, verifiers, budgets, and safety controls.

## References

- [Official model capability specification](../../MODEL_SPECIFICATION.md)
- [Agentic models research synthesis](../research/AGENTIC_MODELS_RESEARCH.md)
- [Specification gaming](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
- [ToolSandbox](https://arxiv.org/html/2408.04682v1)
- [τ-bench](https://arxiv.org/html/2406.12045)
