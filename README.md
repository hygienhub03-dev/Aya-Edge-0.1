# AYA Edge — Agent-First MoE Research Repository

Experimental PyTorch scaffold for a compact, agent-first sparse Mixture-of-Experts language model.

## Status

The merged 0.1 work is a production-oriented prototype. The 0.2 branch adds a reproducible approximately 310.1M-parameter architecture candidate, tokenizer/data contracts, agent evaluation, and resumable training scaffolding. It is **not** a trained production model or a claim of agent competence. The approved identity is defined in [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md): agent-first, balanced reasoning, and hybrid knowledge.

## Initial architecture

- Token embeddings and language-model head (weight-tied)
- Pre-normalised Transformer blocks
- Latent-compressed key/value attention with RoPE
- SwiGLU-style routed experts
- Top-k expert routing and a shared expert
- Load-balancing auxiliary loss
- Per-layer routing diagnostics and structural active-parameter reporting
- Reproducible checkpoint save/load helpers
- Optional router z-loss diagnostics
- Simple `generate()` helper (temperature + top-k)

## Quick start

```bash
python -m pip install -e .
python scripts/inspect_model.py --config configs/aya_edge_0_1.yaml
python scripts/inspect_model.py --config configs/aya_edge_0_2.yaml
python scripts/train.py --config configs/aya_edge_0_2.yaml --dry-run
pytest
```

The 0.1 configuration is intentionally small for inspection. The 0.2 configuration targets 310,115,904 total trainable parameters with tied embeddings; validate memory and compute on the target GPU before any training.

## Teacher

Initial distillation teacher: `openbmb/MiniCPM5-1B`. The teacher is dense; this project does not assume it contains experts that can simply be pruned. Dense-to-MoE conversion and distillation require explicit experiments.

## Documentation

- [Official model capability specification](MODEL_SPECIFICATION.md)
- [Base model audit](docs/research/BASE_MODEL_AUDIT.md)
- [Tiny-MoE research](docs/research/TINY_MOE_RESEARCH.md)
- [Agentic models research](docs/research/AGENTIC_MODELS_RESEARCH.md)
- [Architecture decision report](docs/research/ARCHITECTURE_DECISION_REPORT.md)
- [AYA Edge 0.2 baseline audit](docs/research/AYA_EDGE_02_BASELINE_AUDIT.md)
- [AYA Edge 0.2 architecture](docs/research/AYA_EDGE_02_ARCHITECTURE_SPEC.md)
- [AYA Edge 0.2 parameter budget](docs/research/AYA_EDGE_02_PARAMETER_BUDGET.md)
- [Tokenizer specification](docs/research/AYA_EDGE_TOKENIZER_SPEC.md)
- [Production 32K BPE tokenizer specification](docs/research/AYA_EDGE_PRODUCTION_TOKENIZER_SPEC.md)
- [Dataset strategy](docs/research/AYA_EDGE_DATASET_STRATEGY.md)
- [Dataset acquisition and licensing](docs/data/DATA_ACQUISITION.md)
- [Dataset/tokenizer audit](docs/data/TOKENIZER_AUDIT.md)
- [MoE routing report](docs/research/AYA_EDGE_MOE_ROUTING_REPORT.md)
- [Agent evaluation specification](docs/research/AYA_EDGE_AGENT_EVALUATION_SPEC.md)
- [Training pipeline](docs/research/AYA_EDGE_TRAINING_PIPELINE.md)
- [0.2 implementation report](docs/research/AYA_EDGE_02_IMPLEMENTATION_REPORT.md)
- [Dataset preparation instructions](datasets/README.md)
- [Stage 0](docs/stage-0/README.md) · [Stage 1](docs/stage-1/README.md) · [Stage 2](docs/stage-2/README.md) · [Stage 3](docs/stage-3/README.md)
- [Stage 4 — Reinforcement learning](docs/stage-4/README.md)
- [Stage 5 — Distillation](docs/stage-5/README.md)
- [Kaggle Tesla T4 training notebook](notebooks/AYA_Edge_T4_Training.ipynb)
- [AYA Edge 0.2 Kaggle Tesla T4 notebook](notebooks/AYA_Edge_02_T4_Training.ipynb)

## Limitations / next steps

- Attention still omits KV caching and long-context optimisations.
- Router dispatch is a simple prototype implementation, not an optimised sparse kernel.
- The Kaggle notebook is a reproducible smoke-test/prototype harness; it has not been claimed as successfully executed on Kaggle until run there.
- The 0.2 training script is a resumable pipeline scaffold; the byte tokenizer and small fixtures are development adapters, not the approved production corpus/tokenizer.
- The production BPE trainer is approval-gated; no production tokenizer artifact or checksum is included until the corpus is cleared.
- Unit tests are provided but must be run in the target environment.
