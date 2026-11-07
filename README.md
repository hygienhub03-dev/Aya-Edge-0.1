# AYA Edge 0.1 — Architecture Scaffold

Experimental PyTorch scaffold for a compact, agent-first sparse Mixture-of-Experts language model.

## Status

This is a **production-oriented prototype**, not a trained 310M-parameter model or a production-ready assistant. It uses an MLA-inspired latent key/value bottleneck with rotary positional embeddings (RoPE) and a top-k routed MoE feed-forward layer with a shared expert. It is not a faithful implementation of DeepSeek MLA or MiniCPM5. The approved identity is defined in [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md): agent-first, balanced reasoning, and hybrid knowledge.

## Initial architecture

- Token embeddings and language-model head (weight-tied)
- Pre-normalised Transformer blocks
- Latent-compressed key/value attention with RoPE
- SwiGLU-style routed experts
- Top-k expert routing and a shared expert
- Load-balancing auxiliary loss
- Per-layer routing diagnostics and structural active-parameter reporting
- Reproducible checkpoint save/load helpers
- Simple `generate()` helper (temperature + top-k)

## Quick start

```bash
python -m pip install -e .
python scripts/inspect_model.py --config configs/aya_edge_0_1.yaml
pytest
```

The default configuration is intentionally small enough for architecture inspection. Adjust dimensions only after validating memory and compute requirements on the target GPU.

## Teacher

Initial distillation teacher: `openbmb/MiniCPM5-1B`. The teacher is dense; this project does not assume it contains experts that can simply be pruned. Dense-to-MoE conversion and distillation require explicit experiments.

## Documentation

- [Official model capability specification](MODEL_SPECIFICATION.md)
- [Base model audit](docs/research/BASE_MODEL_AUDIT.md)
- [Tiny-MoE research](docs/research/TINY_MOE_RESEARCH.md)
- [Agentic models research](docs/research/AGENTIC_MODELS_RESEARCH.md)
- [Architecture decision report](docs/research/ARCHITECTURE_DECISION_REPORT.md)
- [Stage 0](docs/stage-0/README.md) · [Stage 1](docs/stage-1/README.md) · [Stage 2](docs/stage-2/README.md) · [Stage 3](docs/stage-3/README.md)
- [Stage 4 — Reinforcement learning](docs/stage-4/README.md)
- [Stage 5 — Distillation](docs/stage-5/README.md)
- [Kaggle Tesla T4 training notebook](notebooks/AYA_Edge_T4_Training.ipynb)

## Limitations / next steps

- Attention still omits KV caching and long-context optimisations.
- Router dispatch is a simple prototype implementation, not an optimised sparse kernel.
- The Kaggle notebook is a reproducible smoke-test/prototype harness; it has not been claimed as successfully executed on Kaggle until run there.
- No production-scale training or distillation pipeline is included yet (see the mini-dataset under `aya_edge_mini_dataset_v1/` for overfitting experiments).
- Unit tests are provided but must be run in the target environment.
