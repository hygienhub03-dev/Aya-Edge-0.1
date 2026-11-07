# AYA Edge 0.1 — Model Overview

## 1. Project Vision

AYA Edge is an experimental compact language-model project focused on developing an efficient assistant architecture.

The long-term objective is to develop a model capable of:

- General language understanding.
- Instruction following.
- Structured JSON and XML generation.
- Multi-turn conversational interaction.
- Reasoning and problem solving.
- Tool calling and agentic workflows.
- Efficient inference on constrained hardware.

These are development objectives, not capabilities that have already been demonstrated.

## 2. Current Model Snapshot

The current scaffold has a reported trainable parameter count of:

85,099,520 parameters.

Reported configuration:

| Component | Configuration |
|---|---:|
| Vocabulary size | 32,000 |
| Hidden dimension | 512 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Latent KV width | 128 |
| Routed experts per layer | 4 |
| Routing top-k | 2 |
| Expert hidden dimension | 1,024 |
| Shared expert | Enabled |
| Maximum sequence length | 512 |

The parameter count must be independently verified against the instantiated model.

## 3. Architectural Direction

The longer-term AYA Edge architecture aims to combine:

- Latent attention inspired by MLA.
- Fine-grained sparse Mixture-of-Experts.
- Shared expert capacity.
- Efficient routing.
- Compact model dimensions.

The intended architecture target is approximately 310M parameters.

The current 85.1M scaffold must not be described as the completed 310M target.

## 4. Development Strategy

The project follows a staged approach:

1. Stage 0 — Reproducibility and baseline validation.
2. Stage 1 — Language-model pretraining.
3. Stage 2 — Mid-training and context development.
4. Stage 3 — Supervised fine-tuning.
5. Stage 4 — Reinforcement learning.
6. Stage 5 — Online/preference distillation.

## 5. Stage 0 Scope

Stage 0 focuses on engineering correctness.

It does not attempt to establish:

- Competitive general knowledge.
- Reliable multi-step reasoning.
- Robust tool use.
- Expert specialisation.
- Long-context performance.
- Production readiness.

These capabilities require dedicated training and evaluation.
