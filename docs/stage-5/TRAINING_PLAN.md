# AYA Edge — Stage 5 Training Plan

## Governing specification

This plan is subordinate to the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). The best Stage 4 checkpoint remains the fallback and must never be overwritten.

## 1. Starting state and controls

- Freeze the Stage 4 checkpoint, tokenizer, runtime, tool schemas, evaluation revisions, and baseline report.
- Pin dataset manifests, teacher audit, code commit, dependency lockfile, hardware, precision, seeds, and decoding settings.
- Use offline replay or deterministic simulators first. Online collection must be least-privilege, budgeted, approval-gated for irreversible actions, and audit logged.
- Never update the teacher or data mixture mid-run without a new audit and experiment ID.

## 2. Training sequence

### Phase A — Pipeline and loss smoke test

Load a small manifest, validate message serialization, assistant/tool masking, preference labels, tokenizer compatibility, and schema handling. Run forward/backward checks, verify finite gradients and optimizer updates, and save/reload a test checkpoint.

### Phase B — Verifier-backed SFT

Train first on accepted trajectories and verified corrections. Mix successful execution, recovery, clarification, safe refusal, and honest non-completion. Keep source and task-family weights explicit in tokens and episodes. Evaluate checkpoints against Stage 4 before proceeding.

### Phase C — Preference-pair calibration

Build high-confidence pairs from independently validated alternatives. Measure label agreement, tie rate, margin by dimension, and outcome-verification coverage. Remove ambiguous pairs before optimization.

### Phase D — Preference optimization

Run a small controlled DPO or equivalent experiment from the best SFT candidate. Keep the reference policy fixed per run, pin beta/temperature and sequence limits, and compare against both Stage 4 and Stage 5-SFT. Do not use preference loss as evidence of real-world task improvement.

### Phase E — Replay and regression repair

If targeted gains cause regressions, add measured replay from general language, context, structured output, safety, and Stage 4 agent baselines. Change one major variable at a time; record the reason and expected benefit.

## 3. Configuration record

Every run records: experiment ID; parent checkpoint hash; model/tokenizer revisions; dataset and manifest hashes; teacher audit ID; SFT or preference objective; learning rate, scheduler, warm-up, weight decay, batch and accumulation; sequence length; epochs/steps; preference beta/reference; precision; gradient clipping; seed; hardware/software; wall time; throughput; peak memory; and checkpoint hashes.

## 4. Monitoring and stop conditions

Track training/validation loss, preference loss and margins, gradient norms, learning rate, tokens, examples, throughput, memory, checkpoint integrity, and evaluation metrics by task family. Stop or quarantine runs with non-finite values, masking errors, data-hash mismatch, provenance loss, reward/preference collapse, rising false completion, unsafe action rate, repeated-call rate, or any mandatory regression breach.

## 5. Promotion order

Promote only the checkpoint that passes the complete evaluation plan. The order is: pipeline integrity → SFT targeted evaluation → preference-pair audit → preference candidate evaluation → Stage 4 regression gates → reproducibility rerun → final exit review. A lower loss or higher teacher agreement cannot override a failed execution gate.

## References

- [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md)
- [Agentic Models Research Report](../research/AGENTIC_MODELS_RESEARCH.md)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [τ-bench](https://arxiv.org/html/2406.12045)
