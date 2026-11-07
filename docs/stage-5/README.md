# AYA Edge — Stage 5: Online and Preference Distillation

## Governing specification

This stage plan is subordinate to the repository-level [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Stage 5 must preserve AYA Edge’s identity as a compact, agent-first assistant whose objective is verified task completion: **Understand → Plan → Act → Observe → Adapt → Complete**.

## Purpose

Stage 5 refines the verified Stage 4 checkpoint with higher-quality demonstrations, execution-grounded feedback, and preference information. The intended teacher reference is **MiniCPM5-1B**, but no teacher checkpoint, output, or hosted service may be used until its exact revision, license, notices, usage terms, and distribution rights have been verified and recorded.

Stage 5 is distillation, not a transfer of authority. Teacher outputs are candidates. A runtime, validator, evaluator, or qualified reviewer must establish correctness before an example can enter training or preference data.

## Primary objectives

- Improve instruction following, planning, tool selection, argument correctness, result interpretation, recovery, and honest completion reporting.
- Preserve broad language, context, structured-output, safety, and Stage 4 capabilities.
- Convert validated trajectories into supervised examples and high-confidence preference pairs.
- Measure outcomes in an execution-capable harness rather than rewarding persuasive agent prose.
- Keep provenance, licensing, privacy, filtering, and rejection reasons auditable.

## Non-negotiable principles

1. Freeze and preserve the best Stage 4 checkpoint before data generation or training.
2. Verify the exact MiniCPM5-1B revision, immutable artifact digest, license text, model-card terms, notices, and any restrictions before use.
3. Record source, prompt, tool schemas, runtime version, execution trace, evaluator version, decoding settings, and transformation history for every retained example.
4. Separate model-proposed actions from runtime-executed actions and observed outcomes.
5. Reject or quarantine unsupported, unsafe, privacy-sensitive, duplicated, contaminated, contradictory, malformed, or unverifiable examples.
6. Prefer verified SFT before preference optimization; run preference training only when pair quality and regression gates pass.
7. Never treat teacher confidence, fluency, self-critique, or a claim of completion as evidence of completion.
8. Promote a checkpoint only on held-out, execution-grounded gains without unacceptable regressions.

## Stage 5 workflow

1. Audit teacher identity, revision, license, terms, artifacts, and access method.
2. Freeze Stage 4 and establish baseline metrics and immutable evaluation splits.
3. Generate targeted candidate trajectories using typed tools and bounded budgets.
4. Execute candidates in deterministic simulators or least-privilege sandboxes where possible.
5. Validate syntax, semantics, constraints, safety, intermediate state, and final state.
6. Filter, deduplicate, redact, split, and manifest accepted and rejected records.
7. Train a verifier-backed SFT candidate; evaluate it before preference optimization.
8. Build preference pairs from independently validated alternatives, not teacher fluency alone.
9. Train a controlled preference candidate and compare with Stage 4 and the SFT candidate.
10. Apply all regression gates and preserve the best checkpoint and complete evidence package.

## Expected deliverables

- Teacher audit and terms snapshot.
- Versioned candidate, accepted, rejected, and quarantined data manifests.
- Execution traces and validator reports.
- SFT and preference-training configurations.
- Stage 4, Stage 5-SFT, and Stage 5-preference evaluation reports.
- Experiment records and promotion decision.

Stage 5 completion does not establish autonomous intelligence. It establishes a better-supported compact agent policy only if the exit criteria are satisfied.

## References

- [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md)
- [Agentic Models Research Report](../research/AGENTIC_MODELS_RESEARCH.md)
- [MiniCPM repository](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM5-1B model card](https://huggingface.co/openbmb/MiniCPM5-1B)
- [DPO paper](https://arxiv.org/abs/2305.18290)
