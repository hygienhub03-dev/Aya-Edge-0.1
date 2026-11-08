# AYA Edge Training Pipeline

**Purpose:** staged, resumable T4 experiments for the 0.2 research candidate. This document specifies a protocol; it does not claim that a full 100,000-step run has been completed.

## Stages

### Stage 0 — integrity smoke test

Instantiate the exact YAML, verify total/tied/structural-active counts, run causal forward/backward with finite gradients, check routing telemetry, save/reload, and compare outputs and optimizer state. Use the mini dataset only here or for bounded overfit checks.

### Stage 1 — short T4 baseline

On a Tesla T4, begin with batch size 1, gradient accumulation 16, sequence length 1,024, seed 42, AdamW, learning rate `2e-4`, weight decay `0.1`, warmup 1,000 steps, gradient clipping 1.0, and the configured auxiliary-loss weight 0.01. Treat these as supplied experiment settings, not validated optima. Measure loss, validation loss, tokens/s, peak memory, GPU utilization, gradient norm, routing, and checkpoint latency.

### Stage 2 — controlled ablations

Change one major variable at a time: sequence length, accumulation, precision, router top-k/shared path, or dispatch implementation. Keep data manifest, seed policy, optimizer, and evaluation fixed where possible. Compare quality at matched tokens and report memory/throughput, not only loss.

### Stage 3 — longer resumable run

Use a licensed token-counted corpus only after the dataset gate passes. Continue from the best verified checkpoint, maintain periodic and best-validation checkpoints, and evaluate held-out language and agent diagnostics. Stage 1 language improvement does not establish agent competence.

## Checkpoint contract

`aya_edge.training.save_checkpoint` stores model, optimizer, optional scheduler/scaler, step, tokens processed, RNG state, configuration, and dataset version. A production experiment must additionally record Git revision, YAML hash, dataset manifest/hash, tokenizer revision, Python/PyTorch/CUDA versions, GPU model, dtype, sequence length, micro-batch/accumulation, and logging schema. On resume, restore all available state, verify config/data identity, replay one deterministic batch where feasible, and compare step/loss behavior.

## T4 operating procedure

Preflight CUDA visibility, free memory, driver/PyTorch versions, deterministic setting, and checkpoint destination. Run a short warm-up before throughput measurement. Use mixed precision only when numerically checked; record actual dtype and scaler behavior. Save before expected interruption, at configured 500-step intervals, and after validation. Never overwrite the best-validation artifact with a later checkpoint.

## Stop and promotion criteria

Stop on NaN/Inf loss or gradients, unstable memory, checkpoint failure, data corruption, unexpected validation deterioration, or routing collapse. Promote only with reproducible held-out improvement, finite training, successful resume, acceptable memory/throughput, and no structured-output, tool-use, or short-context regression. Do not use the 120-example corpus for production claims.

## Future work

Add a true data loader/packing path, explicit attention masks, grouped dispatch, KV-cache benchmarking, and verifier-backed agent evaluations before large-scale training or RL. Online RL and teacher distillation remain later stages gated by provenance and held-out evidence.

## References

- [`aya_edge/training.py`](../../aya_edge/training.py)
- [`configs/aya_edge_0_2.yaml`](../../configs/aya_edge_0_2.yaml)
- [`docs/stage-1/TRAINING_PLAN.md`](../stage-1/TRAINING_PLAN.md)
- [`docs/stage-1/DATA_PIPELINE.md`](../stage-1/DATA_PIPELINE.md)
- [`MODEL_SPECIFICATION.md`](../../MODEL_SPECIFICATION.md)
