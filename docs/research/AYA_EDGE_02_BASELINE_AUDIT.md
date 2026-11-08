# AYA Edge 0.2 Baseline Audit

**Status:** Documentation audit for `feature/aya-edge-02-production`
**Audit date:** 2026-11-07
**Scope:** supplied 0.2 YAML, authoritative `aya_edge/` implementation, existing research and Stage 0/1 documents.

## Executive finding

The supplied 0.2 configuration instantiates the current experimental decoder-only scaffold at **310,115,904 trainable parameters**. This is a verified count produced by instantiating `AyaEdgeForCausalLM` with `configs/aya_edge_0_2.yaml` and summing registered trainable tensors. It is not evidence of production readiness, model quality, or successful agent behavior.

The repository implementation remains an experimental control: MLA-inspired latent-KV attention, a top-2 sparse MoE with an optional shared expert, and Python-loop expert dispatch. It has useful telemetry and checkpoint helpers, but no measured grouped dispatch, KV cache, capacity/overflow policy, mask-aware attention, or demonstrated semantic expert specialization.

## Verified facts

- The authoritative package is `aya_edge/`; the duplicated `base model/aya_edge/` tree is a historical/reference copy and is not the root package path.
- The 0.2 YAML sets `vocab_size=32,768`, `d_model=672`, `n_layers=12`, `n_heads=8`, `latent_dim=192`, `ffn_dim=1,024`, `n_experts=10`, `top_k=2`, shared expert enabled, and maximum sequence length 1,024.
- Embedding and LM-head weights are tied and counted once by PyTorch parameter registration.
- The attention implementation uses full-width query/key/value projections after a latent bottleneck and causal PyTorch SDPA. The repository correctly calls this MLA-inspired, not faithful DeepSeek MLA.
- Each block has two LayerNorms, a bias-free router, ten routed SwiGLU experts, and one shared SwiGLU expert. Routing uses top-k probabilities renormalized over selected experts.
- Dispatch is a Python loop over experts with `index_add_`; this is a correctness/reference path, not a production throughput claim.
- Routing telemetry records assignment fraction, probability/importance fraction, selected IDs, entropy, balance loss, and z-loss. These are diagnostics only.
- `count_active_parameters()` is a structural estimate that counts configured selected experts and the shared expert per layer. It does not measure FLOPs, token imbalance, dispatch overhead, memory, optimizer state, activation memory, or decode-time expert variation.
- The 120-example mini dataset is explicitly synthetic, has simulated tools, and is documented for overfitting/smoke tests rather than production training.

## Parameter verification

For this implementation, with `V=vocab_size`, `D=d_model`, `L=n_layers`, `C=latent_dim`, `F=ffn_dim`, `E=n_experts`, and `S=1` when the shared expert is enabled:

```text
Embedding/tied LM head       = V × D
One attention block          = 2D² + 3DC
Two LayerNorms               = 4D
Router                       = D × E
All expert SwiGLUs           = 3DF × (E + S)
Final LayerNorm              = 2D

Total = V D + L[2D² + 3DC + 4D + DE + 3DF(E + S)] + 2D
```

Substitution:

```text
= 32,768×672
  + 12[(2×672²) + (3×672×192) + (4×672) + (672×10)
       + (3×672×1,024×(10+1))]
  + 2×672
= 310,115,904
```

This agrees with the instantiated model count. The corresponding repository structural active estimate is **111,935,040 parameters**. It should be reported as “structural active estimate,” not active FLOPs or a measured number of parameters touched by a particular batch.

## Hypotheses and unverified claims

The 0.2 dimensions are a research candidate, not a demonstrated optimum. The latent bottleneck may or may not reduce useful cache/memory cost because the implementation reconstructs full-width K/V and has no KV cache. Top-2 plus a shared path may improve retention or agent reliability, but no matched-compute result establishes that. Router telemetry may reveal imbalance, but balanced assignments do not prove semantic specialization.

## Required next work

1. Pin a Git revision and independently rerun total/unique/tied counts, active proxy, forward/backward, and checkpoint round-trip.
2. Add or benchmark grouped/vectorized dispatch against the Python oracle, separating routing, permutation, expert, combine, and synchronization time.
3. Define capacity/overflow semantics and persistent dead-expert reporting before routing comparisons.
4. Measure prefill/decode memory and latency with a real KV-cache implementation before making attention-efficiency claims.
5. Use a licensed, provenance-complete corpus and token-counted manifests before Stage 1 production-scale experiments.
6. Gate promotion on held-out language loss, structured-output validity, verified tool completion, recovery, memory, throughput, and regression checks.

## References

- [`MODEL_SPECIFICATION.md`](../../MODEL_SPECIFICATION.md)
- [`ARCHITECTURE_DECISION_REPORT.md`](ARCHITECTURE_DECISION_REPORT.md)
- [`BASE_MODEL_AUDIT.md`](BASE_MODEL_AUDIT.md)
- [`configs/aya_edge_0_2.yaml`](../../configs/aya_edge_0_2.yaml)
- [`aya_edge/model.py`](../../aya_edge/model.py)
- [`aya_edge/config.py`](../../aya_edge/config.py)
- [`aya_edge/training.py`](../../aya_edge/training.py)
