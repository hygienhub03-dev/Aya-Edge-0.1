# AYA Edge — Stage 5 Experiment Log

This document records online collection, distillation, SFT, preference, and regression experiments. Do not enter estimated results as measured results. Use `Not recorded` when evidence is unavailable.

## Experiment template

### Experiment ID
`STAGE5-EXP-001`

### Hypothesis
State the targeted agent behaviour, expected benefit, and measurable acceptance test.

### Parent and identity
- Stage 4 checkpoint/hash:
- Candidate checkpoint/hash:
- Git commit:
- Model architecture/parameter count:
- Tokenizer revision:
- Runtime/tool-schema revision:
- Teacher: MiniCPM5-1B exact revision/commit:
- Teacher artifact digest:
- Teacher audit ID and verification date:
- License/terms status and snapshot:

### Data and provenance
- Dataset/manifest version:
- Source and transformation versions:
- Accepted/rejected/quarantined records:
- Training/validation/test records and tokens:
- Preference-pair count, tie rate, and label agreement:
- Split and contamination checks:
- Redaction/privacy review:
- Execution-trace and validator versions:

### Configuration
- Objective: SFT / DPO / equivalent:
- Learning rate / scheduler / warm-up:
- Weight decay:
- Batch size / gradient accumulation:
- Sequence length / steps / epochs:
- Preference beta/reference policy:
- Precision / gradient clipping:
- Random seed / decoding settings:
- Tool budget / timeout / permission profile:

### Hardware and software
- GPU and memory:
- CUDA / PyTorch / dependency lock:
- Runtime environment or simulator image:
- Training duration / throughput / peak memory:

### Training results

| Metric | Result |
|---|---|
| Training steps/tokens | Not recorded |
| Training/validation loss | Not recorded |
| Preference loss/margin | Not recorded |
| Finite-gradient check | Not recorded |
| Checkpoint save/reload | Not recorded |

### Execution-grounded evaluation

| Metric | Stage 4 | Candidate | Result/notes |
|---|---:|---:|---|
| Verified final-state success (`pass^1`) | Not measured | Not measured | |
| Repeatable success (`pass^3`) | Not measured | Not measured | |
| Tool selection accuracy | Not measured | Not measured | |
| Semantic argument accuracy | Not measured | Not measured | |
| Recovery success | Not measured | Not measured | |
| False-completion rate | Not measured | Not measured | |
| Unnecessary-call rate | Not measured | Not measured | |

### Regression evaluation

Record general language, context, structured output, safety, refusal, conversation, latency, and memory comparisons against Stage 4. List every failed gate and the adjudication.

### Observations and failures

Record representative traces, validator evidence, rejected examples, anomalies, unsafe outputs, evaluator disagreements, and unresolved limitations. Do not replace evidence with narrative confidence.

### Conclusion
State whether the hypothesis is supported, unsupported, or inconclusive, with metric and provenance references.

### Decision
- Continue.
- Repeat with repaired data.
- Modify objective/configuration.
- Quarantine candidate.
- Reject candidate.
- Promote only after Stage 5 exit review.

---

## Experiment history

No Stage 5 results are established by this document yet.

## References

- [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md)
- [Agentic Models Research Report](../research/AGENTIC_MODELS_RESEARCH.md)
- [DPO paper](https://arxiv.org/abs/2305.18290)
