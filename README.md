# AYA Edge 0.1 — Architecture Scaffold

Experimental PyTorch scaffold for a compact, agent-specialised sparse Mixture-of-Experts language model.

## Status

This is an architecture prototype, **not** a trained 310M-parameter model. It uses an MLA-inspired latent key/value bottleneck and a top-k routed MoE feed-forward layer with a shared expert. It is not a faithful implementation of DeepSeek MLA or MiniCPM5.

## Initial architecture

- Token embeddings and language-model head
- Pre-normalised Transformer blocks
- Latent-compressed key/value attention (prototype)
- SwiGLU-style routed experts
- Top-k expert routing and a shared expert
- Load-balancing auxiliary loss

## Quick start

```bash
python -m pip install -e .
python scripts/inspect_model.py --config configs/aya_edge_0_1.yaml
pytest
```

The default configuration is intentionally small enough for architecture inspection. Adjust dimensions only after validating memory and compute requirements on the target GPU.

## Teacher

Initial distillation teacher: `openbmb/MiniCPM5-1B`. The teacher is dense; this project does not assume it contains experts that can simply be pruned. Dense-to-MoE conversion and distillation require explicit experiments.

## Limitations / next steps

- Attention omits production-grade RoPE, KV caching, and long-context optimisations.
- Router dispatch is a simple prototype implementation, not an optimised sparse kernel.
- No tokenizer, dataset, training loop, checkpointing, or distillation pipeline is included yet.
- Unit tests are provided but must be run in the target environment.
