# AYA Edge 0.2 Implementation Report

## Scope and status

This report records the documentation audit and implementation state of the production-oriented branch. The branch now includes tokenizer, dataset, agent-evaluation, training, model, configuration, test, and notebook scaffolding. It remains a **research candidate, not production-ready**.

## Verified implementation state

The root `aya_edge/` package is authoritative. The model is a pre-normalized decoder scaffold with tied embedding/head weights, latent-KV MLA-inspired attention, causal SDPA, top-2 routed SwiGLU experts, an optional shared expert, explicit auxiliary/router losses, sampling generation without KV cache, and checkpoint helpers that preserve optimizer/scheduler/scaler/RNG metadata. The supplied 0.2 YAML instantiates **310,115,904** trainable parameters; the code’s structural active estimate is **111,935,040**.

The MoE path currently dispatches with a Python expert loop and `index_add_`. Telemetry is present, but semantic specialization, production throughput, capacity behavior, grouped dispatch, expert parallelism, and device-level load balance are not established. Attention has no mask-aware padded batching or decode KV cache, so latent width must not be presented as proven cache savings.

## Cross-report decisions

- **Baseline:** retain the readable implementation as a correctness control.
- **Architecture:** call the candidate MLA-inspired and sparse-MoE; avoid faithful-implementation claims.
- **Budget:** use instantiated/tied count as authoritative; label active count a structural proxy.
- **Data:** reject the 120-example corpus for production; require licensing, provenance, dedupe, source-aware splits, manifests, and token counts.
- **Routing:** treat router statistics as diagnostics; test specialization causally and on held-out tasks.
- **Training:** use staged, resumable T4 experiments with exact checkpoint metadata and one-major-variable-at-a-time ablations.

## Acceptance gates

Before any production claim, verify a pinned revision, deterministic checkpoint round-trip, causal correctness, parameter accounting, real training/resume behavior, grouped-dispatch comparison, memory and throughput at target workloads, and held-out language plus agent completion metrics. Tool-call text is not execution; completion requires external verification. No unsupported benchmark claims are made here.

## Limitations and future work

No report in this set establishes generalization, agent competence, long-context quality, semantic experts, production latency, or legal clearance. Future work must scale the data pipeline with approved sources, add mask-aware attention and KV-cache measurements, benchmark grouped dispatch, implement capacity/overflow telemetry, run the verifier-backed evaluation on held-out data, and complete licensing review for all external and teacher-derived artifacts.

## Report index

- [`AYA_EDGE_02_BASELINE_AUDIT.md`](AYA_EDGE_02_BASELINE_AUDIT.md)
- [`AYA_EDGE_02_ARCHITECTURE_SPEC.md`](AYA_EDGE_02_ARCHITECTURE_SPEC.md)
- [`AYA_EDGE_02_PARAMETER_BUDGET.md`](AYA_EDGE_02_PARAMETER_BUDGET.md)
- [`AYA_EDGE_DATASET_STRATEGY.md`](AYA_EDGE_DATASET_STRATEGY.md)
- [`AYA_EDGE_MOE_ROUTING_REPORT.md`](AYA_EDGE_MOE_ROUTING_REPORT.md)
- [`AYA_EDGE_TRAINING_PIPELINE.md`](AYA_EDGE_TRAINING_PIPELINE.md)
- [`AYA_EDGE_TOKENIZER_SPEC.md`](AYA_EDGE_TOKENIZER_SPEC.md)
- [`AYA_EDGE_AGENT_EVALUATION_SPEC.md`](AYA_EDGE_AGENT_EVALUATION_SPEC.md)

## Primary sources

- [`MODEL_SPECIFICATION.md`](../../MODEL_SPECIFICATION.md)
- [`ARCHITECTURE_DECISION_REPORT.md`](ARCHITECTURE_DECISION_REPORT.md)
- [`BASE_MODEL_AUDIT.md`](BASE_MODEL_AUDIT.md)
- [`configs/aya_edge_0_2.yaml`](../../configs/aya_edge_0_2.yaml)
