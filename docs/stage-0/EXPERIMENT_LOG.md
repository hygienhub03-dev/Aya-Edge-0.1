# AYA Edge — Experiment Log

This document records the currently available Stage 0 diagnostic results.

Measurements should be reproduced against the exact Git commit before being treated as official benchmarks.

## Experiment 1 — GPU Forward and Backward

Reported hardware:

- GPU: NVIDIA Tesla T4
- Device: CUDA

Reported model information:

- Parameters: 85,099,520
- Logits shape: (2, 128, 32000)

Reported measurements:

| Metric | Result |
|---|---:|
| Language loss | 473.1667 |
| Auxiliary loss | 1.0023 |
| Gradient-bearing tensors | 203 |
| Peak GPU memory | 0.803 GB |
| Forward + backward time | 2.13 seconds |

### Interpretation

The model successfully executed a forward and backward pass in the reported diagnostic.

The initial language loss is a diagnostic observation, not a model-quality measurement.

---

## Experiment 2 — Optimiser Update

Reported measurements:

| Metric | Result |
|---|---:|
| Initial parameter snapshots | 203 |
| Language loss | 10.4968 |
| Auxiliary loss | 1.0966 |
| Total loss | 10.5078 |
| Gradient-bearing tensors | 203 |
| All gradients finite | True |
| Optimiser | AdamW |
| Changed parameter tensors | 203 |
| Changed scalar weights | 850,995 |

### Interpretation

The diagnostic indicates that finite gradients were produced and the optimiser updated the tracked parameter tensors.

It does not establish that the training configuration is optimal.

---

## Experiment 3 — Small Overfit Run

Available observations:

| Step | Train LM Loss | Validation LM Loss |
|---:|---:|---:|
| 50 | 5.0470 | 3.3071 |
| 100 | 2.8576 | 2.9287 |
| 150 | 2.3320 | 2.7830 |

Only the available observations are included.

### Interpretation

Training loss decreased over the reported interval.

Validation loss also decreased over this interval.

A small overfit experiment checks whether the model can learn a small sample.

It does not demonstrate:

- Generalisation.
- General reasoning.
- Instruction-following capability.
- Reliable tool use.
- Expert specialisation.
- Production readiness.

---

## Future Experiment Template

Every new experiment should record:

- Experiment ID.
- Hypothesis.
- Git commit.
- Configuration.
- Dataset version.
- Random seed.
- Hardware.
- Software versions.
- Training steps.
- Tokens processed.
- Training loss.
- Validation loss.
- GPU memory.
- Training throughput.
- Expert utilisation.
- Checkpoint.
- Evaluation results.
- Observations.
- Conclusion.
