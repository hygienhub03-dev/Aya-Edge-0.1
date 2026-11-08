# AYA Edge 0.2 Architecture Specification

**Status:** Research candidate; not production-ready
**Normative identity:** Agent-first compact assistant governed by [`MODEL_SPECIFICATION.md`](../../MODEL_SPECIFICATION.md).

## 1. System boundary

AYA follows **Understand → Plan → Act → Observe → Adapt → Complete**. The model interprets intent, plans, selects tools, emits typed calls, interprets observations, and chooses the next action. An external runtime owns permissions, argument validation, execution, state, and error handling. A generated tool call is not proof that a tool ran.

## 2. Candidate configuration

| Item | Value |
|---|---:|
| Vocabulary | 32,768 |
| Hidden width | 672 |
| Layers | 12 |
| Attention heads | 8 (84-dimensional head width) |
| Latent width | 192 |
| Expert hidden width | 1,024 |
| Routed experts/layer | 10 |
| Routed experts/token | 2 |
| Shared expert | Enabled |
| Maximum sequence length | 1,024 |
| Dispatch | Python reference loop |
| Attention | MLA-inspired latent-KV path |
| Verified trainable total | 310,115,904 |

The YAML target of 310,000,000 is a rounded budget; the instantiated count is the authoritative figure for this implementation.

## 3. Block design

Each pre-normalized decoder block applies causal latent attention followed by a sparse MoE residual path. Attention projects Q to hidden width, compresses X to 192 dimensions, reconstructs full-width K and V, applies RoPE, and calls causal SDPA. The MoE applies a bias-free router, top-2 selection with probability renormalization, ten routed SwiGLUs, and an optional always-computed shared SwiGLU. The language head is tied to token embeddings.

“MLA-inspired” is deliberate: this is not a faithful DeepSeek MLA implementation. It lacks a decoupled positional latent path, decode KV cache, paged allocation, and explicit cache-byte accounting.

## 4. Required invariants

- Causal masking must prevent future-token access.
- `d_model / n_heads` must be even for RoPE.
- The configured top-k must be within the expert count.
- Tied embedding/head storage must be counted once.
- Tool execution must remain outside the model.
- Every promoted checkpoint must include exact config, revision, data manifest, and reproducibility metadata.

## 5. Control and ablation plan

The first control is the readable current implementation: MHA-equivalent full-width projections with causal SDPA, top-1 routing at matched active compute, and Python dispatch as a numerical oracle. Candidate ablations are top-2, shared expert on/off, GQA ratios, latent-KV variants with correct positional accounting, and grouped dispatch. Change one major variable at a time and compare language loss, active proxy, throughput, peak memory, structured output, and verified task completion.

## 6. Promotion gates

Promotion requires independent parameter accounting, finite forward/backward, causal-mask tests, checkpoint output equivalence, measured dispatch breakdown, stable routing telemetry, and held-out capability tests. No architecture is retained merely because it is sophisticated. No benchmark or paper result is transferred as an AYA capability claim without a pinned reproduction.

## References

- [`MODEL_SPECIFICATION.md`](../../MODEL_SPECIFICATION.md)
- [`ARCHITECTURE_DECISION_REPORT.md`](ARCHITECTURE_DECISION_REPORT.md)
- [`AYA_EDGE_02_BASELINE_AUDIT.md`](AYA_EDGE_02_BASELINE_AUDIT.md)
- [`configs/aya_edge_0_2.yaml`](../../configs/aya_edge_0_2.yaml)
- [`aya_edge/model.py`](../../aya_edge/model.py)
- DeepSeek-V2 MLA background: https://arxiv.org/abs/2405.04434 (paper context, not an implementation claim)
