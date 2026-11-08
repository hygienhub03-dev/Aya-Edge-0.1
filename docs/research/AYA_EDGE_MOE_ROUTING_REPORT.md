# AYA Edge MoE Routing Report

## Current implementation

The 0.2 candidate has ten bias-free routed experts per layer, top-2 selection, renormalized selected probabilities, an always-computed shared SwiGLU, Switch-style assignment/importance balance loss, and router z-loss. Dispatch loops over expert IDs, gathers matching tokens, runs each expert, and combines with `index_add_`.

## What is verified

The model records per-layer assignment fractions, probability fractions, selected expert IDs, entropy, balance loss, and z-loss. These diagnostics can expose skew, low entropy, dead experts, and disagreement between selected assignments and router mass. The Python loop is a numerical/correctness reference and is explicitly configured as `python_reference` in the 0.2 YAML.

## What is not established

Diagnostics are not semantic specialization. A balanced router only shows distributional behavior; it does not show that one expert learned planning, another language, or another tool use. Semantic specialization requires controlled probes, token/task-conditioned analysis, causal expert ablations or swaps, persistence across seeds/checkpoints, and quality changes on held-out tasks. Current code has no capacity factor, overflow/drop policy, persistent dead-expert tracker, device-level balance, grouped GEMM, or expert parallelism.

## Measurement plan

For every layer and run, log requested and accepted routes, per-expert token counts, probability mass, entropy, z-loss, auxiliary loss, overflow/drops, residual bypass, dead-expert duration, and local/global counts. Time router, index/permutation, expert kernels, combine, synchronization, and peak activation memory separately. Compare top-1 and top-2 only at matched active compute and identical data/optimization budgets.

## Dispatch recommendation

Keep the Python implementation as a golden numerical oracle and unit-test grouped dispatch against it. Before scaling, benchmark vectorized/grouped token packing, offsets, batched expert computation, inverse permutation, and combine. Consider dropless/block-sparse or expert-parallel paths only after measurements justify their complexity. Do not claim sparse efficiency from parameter count alone.

## Routing hypotheses

Top-2 and a shared expert may improve retention or reduce brittle routing, but they add selected work and dispatch cost. Fine-grained routing may increase combinations but can amplify overhead. These are hypotheses for matched experiments, not conclusions from frontier-scale papers.

## References

- [`aya_edge/model.py`](../../aya_edge/model.py)
- [`configs/aya_edge_0_2.yaml`](../../configs/aya_edge_0_2.yaml)
- [`BASE_MODEL_AUDIT.md`](BASE_MODEL_AUDIT.md)
- Switch Transformers: https://arxiv.org/abs/2101.03961
- ST-MoE: https://arxiv.org/abs/2202.08906
- MegaBlocks: https://arxiv.org/abs/2211.15841
- [`ARCHITECTURE_DECISION_REPORT.md`](ARCHITECTURE_DECISION_REPORT.md)
