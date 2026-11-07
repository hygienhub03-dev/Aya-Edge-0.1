# AYA Edge — Stage 4 Experiment Log

## Governing link

Use this log with the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Every run must be attributable, reproducible, and comparable with the frozen Stage 3 baseline.

## Entry template

Copy this block for each run; do not delete failed or rejected experiments.

```yaml
run_id: STAGE4-YYYYMMDD-NNN
status: planned | running | completed | rejected | rolled_back
purpose: "One falsifiable question"
parent_run: null
stage3_checkpoint: "hash/path"
candidate_checkpoint: "hash/path"
git_commit: "hash"
phase: replay | offline | sandbox | shadow | restricted
model_config: "version/path"
tokenizer: "version/hash"
task_manifest: "version/hash"
environment_manifest: "version/hash"
verifier_version: "version/hash"
reward_version: "version/hash"
reward_weights: {}
train_split: "id/version"
validation_split: "id/version"
test_split: "id/version"
seeds:
  model: 0
  data: 0
  environment: 0
decoding: {temperature: 0.0, top_p: 1.0, max_new_tokens: 0}
budgets: {max_turns: 0, max_tool_calls: 0, max_seconds: 0, max_cost: 0}
hardware_software: "image, accelerator, driver, framework"
command: "exact command"
start_utc: ""
end_utc: ""
results: {}
regressions: []
hard_gate_incidents: []
verifier_disagreements: []
artifacts: []
decision: promote | hold | reject | rollback
reason: "Evidence-based decision"
```

## Minimum results to record

Record verified completion, `pass@1`, `pass@3`, recovery success, tool-need and selection accuracy, schema and semantic argument correctness, runtime acceptance, false completion, constraint adherence, unnecessary calls, turns/calls/tokens/time/cost per success, hard-gate violations, verifier disagreement, and all failure labels. Include denominators and confidence intervals where applicable.

## Trace retention

For every sampled or failed episode retain the task version, initial state hash, user request, available tools, model outputs, validated calls, permission decisions, runtime observations, state diff, verifier output, final response, reward components, and termination reason. Redact secrets and personal data while preserving auditability.

## Review rules

- One person or automated job cannot both change the verifier and approve a run without independent review.
- Any hard-gate incident is a hold/rollback until root cause and a regression test exist.
- A reward increase without verified-outcome improvement is a rejected result.
- Environment failures are separated from policy failures.
- Promotion requires a linked evaluation report and preserved Stage 3 checkpoint.

## Non-claims

This log records experiments; it is not evidence that a run is safe outside its declared environment or that an accepted checkpoint is production-ready.

## References

- [Official model capability specification](../../MODEL_SPECIFICATION.md)
- [Agentic models research synthesis](../research/AGENTIC_MODELS_RESEARCH.md)
