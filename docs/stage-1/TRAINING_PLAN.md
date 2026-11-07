# AYA Edge — Stage 1 Training Plan

## 1. Objective

Establish stable and reproducible language-model pretraining.

## 2. Starting configuration

Use the verified Stage 0 configuration as the initial reference.

Reported starting values:

| Parameter | Value |
|---|---:|
| Vocabulary | 32,000 |
| Hidden dimension | 512 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Maximum sequence length | 512 |
| Learning rate | 0.0003 |
| Weight decay | 0.1 |
| Batch size | 2 |
| Maximum steps | 1,000 |
| Auxiliary loss weight | 0.01 |
| Random seed | 42 |

These are reported scaffold settings, not mandatory final training settings.

The training configuration must be reviewed before a full run.

## 3. Training sequence

### Phase A — Smoke test

- Load the dataset.
- Initialise the model.
- Execute a small number of training steps.
- Check loss values.
- Check gradients.
- Verify optimiser updates.
- Save and reload a checkpoint.

### Phase B — Short training run

- Train on a small representative dataset.
- Monitor training loss.
- Monitor validation loss.
- Inspect expert routing.
- Measure memory consumption.

### Phase C — Controlled pretraining

Increase training volume only after the short run is stable.

Track tokens processed rather than relying exclusively on training steps.

### Phase D — Checkpoint evaluation

Evaluate checkpoints at fixed intervals.

Do not automatically promote the latest checkpoint.

## 4. Learning-rate strategy

Use a controlled learning-rate schedule.

The intended broader training direction is:

1. Stable initial learning.
2. Controlled decay.
3. Longer training decay.

Warm-up and decay parameters must be selected using measured stability and validation behaviour.

## 5. Monitoring

Record:

- Training loss.
- Validation loss.
- Learning rate.
- Tokens processed.
- Tokens per second.
- GPU utilisation.
- Peak GPU memory.
- Gradient norm.
- Gradient anomalies.
- Expert routing counts.
- Auxiliary loss.

## 6. Failure conditions

Stop or investigate training if:

- Loss becomes NaN or Inf.
- Gradients become non-finite.
- Validation loss deteriorates unexpectedly.
- Expert routing collapses.
- Checkpoint saving fails.
- Training data becomes corrupted.
- GPU memory usage becomes unstable.

## 7. Checkpoint policy

Save checkpoints with:

- Model weights.
- Optimiser state.
- Scheduler state.
- Training step.
- Tokens processed.
- Configuration.
- Random-number-generator state.
- Dataset version.

Maintain a separate best-validation checkpoint.

## 8. Important rule

Do not change architecture, dataset mixture, learning rate, and batch configuration simultaneously.

Change one major variable at a time whenever possible.
