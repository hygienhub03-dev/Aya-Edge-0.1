# AYA Edge 0.1 — Stage 0 Baseline

## 1. Objective

Stage 0 establishes the engineering baseline for AYA Edge.

The objective is to verify that the model can:

- Initialise correctly.
- Execute a forward pass.
- Execute a backward pass.
- Produce finite gradients.
- Perform an optimiser update.
- Save checkpoints.
- Reload checkpoints.
- Learn from a small training sample.
- Produce reproducible measurements.

Stage 0 is not a capability benchmark.

## 2. Reported Baseline Configuration

| Component | Value |
|---|---:|
| Vocabulary | 32,000 |
| Hidden dimension | 512 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Latent KV width | 128 |
| Routed experts per layer | 4 |
| Routing top-k | 2 |
| Expert hidden dimension | 1,024 |
| Shared expert | Enabled |
| Maximum sequence length | 512 |
| Reported parameters | 85,099,520 |
| Teacher reference | openbmb/MiniCPM5-1B |

These values must be verified against the exact repository configuration and model implementation.

## 3. Reported Hardware Diagnostic

GPU:

NVIDIA Tesla T4

Reported observations:

- Logits shape: (2, 128, 32000)
- Gradient-bearing tensors: 203
- Peak GPU memory: 0.803 GB
- Forward/backward time: 2.13 seconds
- Finite gradients: True
- AdamW update: Executed
- Changed parameter tensors: 203

## 4. Verification Checklist

### Model Integrity

- [ ] Verify model initialisation.
- [ ] Verify parameter count.
- [ ] Verify trainable parameter count.
- [ ] Verify embedding/output weight tying.
- [ ] Verify attention dimensions.
- [ ] Verify expert dimensions.
- [ ] Verify router dimensions.

### Forward and Backward

- [ ] Run forward-pass shape tests.
- [ ] Verify causal attention masking.
- [ ] Verify loss calculation.
- [ ] Run backward pass.
- [ ] Check all gradients for NaN and Inf.
- [ ] Verify expected parameters receive gradients.

### Optimisation

- [ ] Execute one optimiser update.
- [ ] Verify intended parameters change.
- [ ] Check optimiser configuration.
- [ ] Record learning rate and weight decay.

### Checkpoint Integrity

- [ ] Save a checkpoint.
- [ ] Load it into a fresh model instance.
- [ ] Compare model outputs before and after loading.
- [ ] Verify optimiser-state restoration if applicable.

### Training Baseline

- [ ] Run the small overfit experiment.
- [ ] Record training loss.
- [ ] Record validation loss.
- [ ] Record tokens processed.
- [ ] Record training throughput.

### MoE Diagnostics

- [ ] Record expert routing counts per layer.
- [ ] Calculate expert utilisation percentages.
- [ ] Identify dead experts.
- [ ] Monitor router imbalance.
- [ ] Record auxiliary loss.

### Reproducibility

- [ ] Record Git commit hash.
- [ ] Record configuration snapshot.
- [ ] Record random seed.
- [ ] Record Python and PyTorch versions.
- [ ] Record CUDA version.
- [ ] Record GPU model.
- [ ] Store experiment logs.

## 5. Acceptance Criteria

Stage 0 should only be considered complete when the verification checklist has been executed against a known commit.

A successful forward and backward pass alone is insufficient.

The checkpoint round-trip, causal-mask correctness, parameter accounting, and reproducible training diagnostics are essential.

## 6. Known Limitations

1. The reported 85.1M parameter count requires independent verification.
2. MLA-inspired attention requires an implementation audit.
3. Sparse expert-dispatch efficiency requires measurement.
4. Expert specialisation has not been established.
5. The small overfit experiment does not establish generalisation.
6. Long-context behaviour has not been verified.
7. Instruction following remains a future training objective.
8. Tool-use capability remains a future training objective.
9. Production readiness has not been established.

## 7. Stage 1 Readiness

Before beginning large-scale pretraining:

- Complete Stage 0 verification.
- Establish fixed validation and test datasets.
- Create dataset manifests.
- Verify checkpoint integrity.
- Confirm reproducible training.
- Establish baseline evaluation metrics.

Only then should AYA Edge progress to Stage 1.
