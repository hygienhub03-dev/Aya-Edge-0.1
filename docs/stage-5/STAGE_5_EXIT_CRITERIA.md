# AYA Edge — Stage 5 Exit Criteria

## Governing specification

These criteria are subordinate to the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Stage 5 may not be declared complete from loss reduction, teacher agreement, or fluent demonstrations alone.

## 1. Checkpoint and reproducibility

- [ ] Best Stage 4 checkpoint is preserved and hash recorded.
- [ ] Stage 5 parent and candidate checkpoints load and round-trip correctly.
- [ ] Code, dependencies, tokenizer, runtime, tool schemas, seeds, hardware, and configurations are pinned.
- [ ] Training and evaluation rerun is reproducible within documented tolerance.
- [ ] Complete experiment logs and checkpoint hashes are archived.

## 2. Teacher, terms, and provenance

- [ ] Exact MiniCPM5-1B revision/commit and immutable artifact digest are recorded.
- [ ] License text, notices, model-card terms, access terms, and restrictions are verified from authoritative sources and snapshotted.
- [ ] Rights and terms for datasets, APIs, benchmark tasks, prompts, and generated outputs are reviewed.
- [ ] No unapproved hosted-teacher extraction or competing-model restriction remains unresolved.
- [ ] Every retained record has provenance, transformation, privacy, split, and validator metadata.
- [ ] Accepted, rejected, and quarantined counts and reasons reconcile to the raw manifest.

## 3. Data and preference quality

- [ ] Syntax, schema, semantic, safety, constraint, and final-state filters have run.
- [ ] Teacher outputs were independently validated; no automatic trust rule remains.
- [ ] Execution traces distinguish proposed calls, runtime results, and completion evidence.
- [ ] Train/validation/test and benchmark-contamination checks pass.
- [ ] Preference pairs share the same task/environment and have independently justified winners.
- [ ] Ambiguous, tied, unverifiable, duplicated, unsafe, and both-wrong pairs are excluded or quarantined.
- [ ] Data mixture, token counts, and changed proportions are documented with rationale.

## 4. Training integrity

- [ ] Serialization, masking, tokenizer, forward/backward, finite-gradient, optimizer, and checkpoint tests pass.
- [ ] Verifier-backed SFT was evaluated before preference optimization.
- [ ] Preference objective, reference policy, pair manifest, beta/temperature, and stopping conditions are recorded.
- [ ] No non-finite loss, preference collapse, provenance loss, or uncontrolled online action occurred.
- [ ] Stage 4 remains an available rollback checkpoint.

## 5. Agent-first outcome gates

- [ ] Held-out intent, planning, tool-need, selection, argument, interpretation, recovery, and completion metrics are reported.
- [ ] Final-state verification is independent of the model’s completion claim.
- [ ] `pass^1` and repeated success (`pass^3` or documented equivalent) are measured.
- [ ] False-completion, fabricated-result, unsafe-action, permission-violation, repeated-failure, and unnecessary-call rates are reported.
- [ ] Failure injection, clarification, honest non-completion, and alternate-tool recovery cases are included.
- [ ] Targeted agent performance improves over Stage 4 by the pre-registered threshold, or the result is explicitly rejected/inconclusive.

## 6. Regression gates

- [ ] General language and knowledge performance remains within the pre-registered tolerance.
- [ ] Short/long-context retrieval and conversation consistency remain within tolerance.
- [ ] JSON/XML parse, schema, and semantic correctness remain within tolerance.
- [ ] Safety, refusal, uncertainty, and privacy behaviour have no unacceptable regression.
- [ ] Latency, memory, and throughput changes are documented.
- [ ] No critical safety, false-completion, or execution-reliability regression is averaged away by an aggregate score.

## 7. Promotion decision

Stage 5 is complete only when every mandatory checkbox is satisfied, the evidence package is reviewed, and the promotion owner signs the report. Otherwise preserve the candidate for analysis, retain Stage 4 as the production fallback, and record the blocker and next experiment. The promoted checkpoint must be accompanied by its manifests, audit snapshots, evaluation report, and rollback reference.

## 8. Stage completion statement

The approved statement must identify the exact promoted checkpoint and say: AYA Edge demonstrated measured, execution-grounded improvement on the declared held-out tasks without unacceptable regression under the declared gates. Do not claim general autonomy or capabilities outside the evaluated scope.

## References

- [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md)
- [Agentic Models Research Report](../research/AGENTIC_MODELS_RESEARCH.md)
- [MiniCPM repository](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM5-1B model card](https://huggingface.co/openbmb/MiniCPM5-1B)
- [ToolSandbox](https://arxiv.org/html/2408.04682v1)
- [τ-bench](https://arxiv.org/html/2406.12045)
