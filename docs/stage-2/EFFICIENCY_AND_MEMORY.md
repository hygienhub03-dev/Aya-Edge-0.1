# AYA Edge — Stage 2 Efficiency and Memory

## 1. Objective

Measure the computational cost of increasing AYA Edge's context length.

## 2. Measurements

For every sequence-length experiment, record:

- Peak allocated GPU memory.
- Peak reserved GPU memory.
- Training tokens per second.
- Inference tokens per second.
- Forward-pass duration.
- Backward-pass duration.
- Batch size.
- Sequence length.
- Precision.
- GPU utilisation.

## 3. Memory considerations

Longer sequences can significantly increase attention and activation memory.

Measure actual consumption rather than relying on theoretical estimates.

## 4. Optimisation experiments

Potential techniques:

- Mixed precision.
- Activation checkpointing.
- Gradient accumulation.
- Efficient attention kernels.
- Reduced padding.
- Sequence packing.
- Memory-efficient batching.

Each optimisation requires correctness tests.

## 5. Attention implementation audit

Determine whether the current attention implementation supports the intended sequence lengths efficiently.

Investigate:

- Attention complexity.
- Intermediate tensor sizes.
- KV representation.
- Position handling.
- Mask construction.
- Kernel availability.

## 6. Mixture-of-Experts overhead

Measure:

- Router execution time.
- Expert dispatch overhead.
- Expert computation time.
- Memory per expert.
- Routing distribution.

Do not attribute all training overhead to attention.

## 7. Benchmark table

| Sequence length | Peak memory | Tokens/sec | Validation loss |
|---|---|---|---|
| 512 | Not measured | Not measured | Not measured |
| 1,024 | Not measured | Not measured | Not measured |
| 2,048 | Not measured | Not measured | Not measured |
| 4,096 | Not measured | Not measured | Not measured |
| 8,192 | Not measured | Not measured | Not measured |

## 8. Success definition

Stage 2 must establish a documented performance profile for supported sequence lengths.

Do not claim an efficiency improvement without measurements against a consistent baseline.
