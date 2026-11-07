# AYA Edge 0.1 — Architecture Documentation

## 1. Overview

AYA Edge 0.1 is a decoder-style language-model scaffold incorporating transformer blocks and Mixture-of-Experts components.

The architecture is intended to provide a foundation for future compact sparse-model experimentation.

## 2. Configuration

| Component | Value |
|---|---:|
| Vocabulary | 32,000 |
| Hidden dimension | 512 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Latent KV dimension | 128 |
| Routed experts | 4 |
| Experts activated per token | 2 |
| Expert hidden dimension | 1,024 |
| Shared expert | Enabled |
| Maximum sequence length | 512 |

The repository configuration remains the authoritative source of truth.

## 3. Attention Architecture

The current design uses a latent-KV approach inspired by Multi-head Latent Attention.

The term MLA-inspired is intentional.

The implementation should not be described as a faithful reproduction of DeepSeek MLA without a complete architectural audit.

Stage 0 should verify:

- Projection dimensions.
- Causal attention masking.
- Numerical stability.
- Output dimensions.
- Correct loss alignment.
- Gradient propagation.

## 4. Mixture-of-Experts Architecture

The reported configuration contains four routed experts per transformer layer.

Top-2 routing selects two routed experts for each token.

A shared expert is also enabled.

Important considerations:

- Routing decisions must be measurable.
- Expert utilisation must be monitored.
- Dead experts must be identified.
- Load imbalance must be investigated.
- Shared expert computation must be included in performance measurements.

Expert utilisation alone does not prove that experts have developed meaningful specialisations.

## 5. Implementation Considerations

The current implementation should be audited for sparse-dispatch efficiency.

If Python-loop dispatch is used, its overhead must be measured before scaling training.

Future optimisation should investigate:

- Efficient token grouping.
- Batched expert computation.
- Reduced Python overhead.
- Memory-efficient routing.
- GPU utilisation.
- Tokens processed per second.

## 6. Parameter Accounting

The reported parameter count is:

85,099,520.

This figure must be reproduced directly from the instantiated model.

Record:

- Total parameters.
- Trainable parameters.
- Unique parameters when weights are tied.
- Embedding parameters.
- Output-head parameters.
- Attention parameters.
- Expert parameters.
- Router parameters.

Parameter counts must never be estimated solely from configuration dimensions.

## 7. Future Architecture Work

Potential improvements include:

- More efficient latent attention.
- Improved sparse dispatch.
- Router load balancing.
- Better expert utilisation.
- Optimised memory usage.
- Context-length scaling.
