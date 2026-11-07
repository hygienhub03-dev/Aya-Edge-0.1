# AYA Edge Architecture Decision Report

**Decision scope:** compact sparse-MoE model and agent-first training architecture
**Status:** Experimental decision record; not a production-readiness assessment
**Governing specification:** [`MODEL_SPECIFICATION.md`](../../MODEL_SPECIFICATION.md)

## Decision summary

**Decision:** retain the current AYA direction as a measurement-driven architecture program, with a simple control condition and narrowly scoped ablations:

1. **MoE control:** a MiniMind-style top-1 router is the first compact baseline.
2. **AYA hypothesis tests:** add a shared expert and compare top-2/fine-grained routing only at matched active compute.
3. **Attention control:** establish ordinary causal MHA with SDPA/FlashAttention-compatible execution; compare GQA before an MLA-inspired latent-KV variant.
4. **Dispatch:** keep the Python-loop implementation only as a correctness oracle; benchmark vectorized/grouped dispatch before scaling.
5. **Agent policy:** train typed tool trajectories and use structured ReAct as the default controller; gate deliberative search and RL.
6. **Distillation:** keep MiniCPM5-1B as the documented teacher reference, but verify exact revision, terms, filtering, and regression gates before using outputs.

This is a research plan, not a claim that any option is production-ready or that AYA’s target quality has been demonstrated.

## Non-negotiable project facts

AYA is an agent-first compact assistant. Its official operating loop is **Understand → Plan → Act → Observe → Adapt → Complete**. The model is responsible for intent, plans, tool selection, structured calls, observation interpretation, and next actions; the external runtime is responsible for permissions, argument validation, execution, state, and errors. Task completion—not plausible prose—is the central agent measure. [1]

The current scaffold is reported at **85,099,520 parameters (approximately 85.1M)** with vocabulary 32,000, hidden size 512, 8 layers, 8 attention heads, latent KV width 128, 4 routed experts, top-2 routing, expert hidden size 1,024, a shared expert, and maximum sequence length 512. The count requires independent verification. The longer-term architecture target is approximately **310M parameters**. The 85.1M prototype and 310M target are distinct and must never be conflated. [2,3]

The official specification does not prescribe AYA’s expert count, top-k, capacity factor, router coefficients, latent width, context limit, or parameter allocation. These are experimental variables, not facts to infer from external models. [1]

## Evidence-based options

### Option A: compact top-1 MoE control

MiniMind-3-MoE documents 198M total/64M active parameters, 8 layers, 768 hidden size, 6,400 vocabulary, four experts, and top-1 routing. Its native-PyTorch project includes pretraining, SFT, tool use, agentic-RL, and distillation paths. This makes it the most practical control for separating compact training-system effects from routing complexity. It is Apache-2.0 according to the project sources. [4–6]

**Decision:** adopt as control, but reproduce parameter counts and runtime measurements in AYA’s environment. Do not treat project demos as capability proof.

### Option B: shared expert and top-2 routing

Tiny-MoE documents approximately 150M–200M total and 70M–90M active parameters, 8 routed plus 1 shared expert, top-2 routing, capacity factor 1.25, router diagnostics, grouped GEMM, and memory-conscious training. DeepSeekMoE provides the underlying shared-expert and fine-grained specialization rationale, but its 2B/16B/145B-scale configurations do not establish compact AYA requirements. [7,8]

**Decision:** test as a matched-compute ablation. Measure whether the shared path improves general language and tool reliability enough to justify its fixed compute and memory. Do not assume top-2 or fine-grained routing is better.

### Option C: fine-grained routing and balance mechanisms

Fine-grained expert segmentation can increase routing combinations while preserving approximate total expert capacity if selected count scales with split factor. It also increases selected-expert work and can raise dispatch and communication overhead. Balance losses address different signals: realized token assignment, router probability mass, device/communication load, and router-logit stability. [8–10]

**Decision:** implement only after a simple router is instrumented. Log requested/accepted routes, overflow/drops, per-expert fractions, entropy, dead experts, router logits, and local/global counts. Coefficients from Switch or ST-MoE are starting-point evidence, not AYA defaults. [9]

### Option D: latent attention, GQA, and exact kernels

FlashAttention/SDPA reduce memory traffic and prefill intermediates but do not by themselves reduce persistent decode KV-cache bytes. GQA reduces cache heads with a simpler implementation; MLA compresses K/V into a latent and uses a decoupled RoPE path whose positional state must also be cached. DeepSeek-V2’s cache and throughput figures are paper-specific and cannot be transferred to AYA. PagedAttention improves allocation and sharing but is complementary to cache compression. [11]

**Decision:** establish MHA + causal SDPA baseline; test GQA ratios next; then test the current MLA-inspired design with explicit cache-byte accounting, decoupled positional handling, numerical checks, and agent-specific quality tests. The repository itself says the current implementation is not a faithful DeepSeek MLA implementation. [3,12]

### Option E: dispatch implementation

A per-expert Python loop is an appropriate numerical reference but can create many small launches. Grouped/batched dispatch requires expert counts/offsets, token permutation, expert computation, inverse permutation, and—under expert parallelism—all-to-all communication. MegaBlocks and related implementations show that permutation, padding, synchronization, and communication can dominate small expert computations. [13]

**Decision:** keep loop dispatch for unit tests; build a vectorized/grouped path before large-scale training. Report router, permutation, synchronization, expert kernel, combine, peak activation memory, and dropped-token time separately. Consider dropless/block-sparse dispatch only when measurements justify it.

### Option F: agent training and policy

Typed tool-call trajectories are the correct supervised unit. Structured ReAct matches AYA’s loop and should be the default routine controller. ToT/LATS-like search may be useful for bounded, high-branching tasks; Reflexion-style summaries may help after verified failure, but neither proves correctness. [14–18]

**Decision:** adopt typed SFT and structured ReAct; test search/reflection behind explicit cost, depth, expiry, and verifier gates. Reward externally verified outcomes, not self-reported completion. Defer broad online RL until offline and sandboxed tests show held-out improvement over Stage 3 without safety or general-language regressions. [18,19]

## Chosen experimental architecture

The immediate architecture should be described as:

> An experimental decoder-only compact Transformer with a latent-KV attention path inspired by MLA, a sparse routed MoE feed-forward block with an optional shared expert, explicit routing diagnostics, and an external typed-tool runtime.

The word **inspired** is required: the existing implementation is not a faithful DeepSeek MLA or MiniCPM5 implementation. [3,12]

### Baseline/control matrix

| Dimension | Control | Ablation 1 | Ablation 2 | Promotion evidence |
|---|---|---|---|---|
| Router | Top-1, simple auxiliary balance | Top-2 at matched active compute | Fine-grained/shared path | Loss, active FLOPs, expert load, overflow, tool completion |
| Shared expert | Off | One shared path | Alternative shared width/count | General language + agent reliability per added cost |
| Attention | MHA + causal SDPA | GQA ratios | MLA-inspired latent KV | Cache bytes, prefill/decode, retrieval, JSON/tool quality |
| Dispatch | Loop reference | Vectorized/grouped | Dropless/block-sparse | End-to-end throughput and memory at target batch/sequence |
| Agent policy | Typed one-step ReAct | Bounded ToT | LATS-like search | Verified completion, recovery, cost, false completion |
| Optimization | SFT | DPO/preferences | Sandboxed RL | Held-out gain over prior stage and no regressions |

## Measurement and promotion gates

### Architecture and systems gates

For every checkpoint/configuration record total parameters, trainable parameters, tied-weight accounting, active parameters/token, exact config, dtype, sequence length, batch/concurrency, peak memory, prefill tokens/s, decode tokens/s, latency percentiles, and selected attention backend. For MoE, add requested/accepted routes, capacity utilization, overflow/dropped tokens, residual bypass, per-expert distribution, entropy, dead-expert persistence, router z-loss, and dispatch breakdown. [9,11,13]

### Capability gates

The local agent harness should test no-tool decisions, tool selection, schema and semantic arguments, execution, observation interpretation, recovery, clarification, structured output, long-context retrieval, and final-state completion. Public BFCL, ToolSandbox, maintained τ-bench/τ³-bench, and AgentBench runs should be pinned by revision and treated as complementary diagnostics rather than a universal score. [20–23]

### Stage gates

- **Stage 0:** verify parameter accounting, checkpoint round-trip, causal masking, gradients, dispatch correctness, and reproducible diagnostics. A forward/backward pass is insufficient. [3]
- **Stage 1:** require stable language modeling, held-out improvement, reproducible checkpoints, and monitored expert utilization; do not claim agent competence. [1]
- **Stage 2:** require measured retrieval and memory/throughput improvements at supported lengths without unacceptable short-context regression; do not claim durable memory or perfect recall. [1]
- **Stage 3:** require valid structured outputs, correct calls, planning, observation interpretation, uncertainty, and supervised agent baseline metrics. [1]
- **Stage 4:** require held-out improvement over Stage 3, verified outcomes, constraint adherence, recovery, and no reward-hacking/safety regression. [1,18]
- **Stage 5:** require filtered/provenanced teacher data, preference quality, improvement over Stage 4, and no unacceptable general-language, safety, context, or structured-output regression. [1,24]

## Licensing and provenance decision

Use only artifacts whose exact code, model, data, benchmark, and output terms have been checked. MiniMind and Tiny-MoE are documented as Apache-2.0; Phi-tiny-MoE is documented as MIT while its SlimMoE paper has separate terms; DeepSeek V3 code and weights have separate licensing; Megatron, MegaBlocks, Tutel, PyTorch examples, and benchmark repositories each require artifact-specific review. [4–10,13,20–23]

For agent data, ToolBench’s Apache-2.0 repository does not automatically license RapidAPI-derived content; API-Bank’s cited sources do not establish redistribution rights; ToolSandbox has a custom permissive license and third-party notices. Hosted teacher terms may restrict automated output collection or training competing models. Record revision hashes, license/terms snapshots, prompts, tool/data provenance, filtering code, and privacy decisions. [14,24]

## Final adopt/reject/test decisions

- **Adopt now:** measured top-1 MoE control; typed tool protocol; structured ReAct; causal SDPA/MHA control; explicit diagnostics; verifier-backed local evaluation.
- **Test next:** shared expert; top-2 at matched active compute; GQA; MLA-inspired latent KV with correct positional accounting; grouped dispatch; bounded reflection/search.
- **Defer:** fine-grained/device-aware routing, dropless kernels, broad online RL, and SlimMoE-like compression until baseline evidence justifies their complexity.
- **Reject as unsupported claims:** production readiness, agent competence from architecture alone, 310M status for the 85.1M scaffold, quality transfer from frontier-scale papers, or capability claims based only on repository demonstrations.

## Open questions

1. Will a shared expert improve AYA’s general-language retention and verified tool completion enough to offset its compute?
2. At AYA’s batch and sequence distributions, does top-2 improve completion/recovery enough to offset dispatch cost?
3. What latent KV width and positional state preserve exact tool arguments and long-document retrieval?
4. Does GQA deliver most of the cache benefit with less implementation risk than MLA-inspired attention?
5. What capacity/drop policy is acceptable at target micro-batches without hiding quality loss in residual bypass?
6. Which agent rewards correlate with verified outcomes without becoming exploitable proxies?
7. Can MiniCPM5-1B outputs be collected and trained on under the exact current terms and checkpoint revision?
8. What parameter configuration, if any, bridges the verified 85.1M prototype toward the approximately 310M target without assuming undocumented arithmetic?

## References

[1]: ../../MODEL_SPECIFICATION.md "AYA Edge official model capability specification"
[2]: ../stage-0/MODEL_OVERVIEW.md "AYA Edge model overview"
[3]: ../stage-0/ARCHITECTURE.md "AYA Edge architecture"; ../stage-0/STAGE_0_BASELINE.md "AYA Edge Stage 0 baseline"
[4]: https://github.com/jingyaogong/minimind "MiniMind repository"
[5]: https://jingyaogong.github.io/minimind/ "MiniMind project site"
[6]: https://github.com/AbdelrhmanEbied/Tiny-MoE "Tiny-MoE repository"
[7]: https://huggingface.co/microsoft/Phi-tiny-MoE-instruct "Phi-tiny-MoE model card"
[8]: https://arxiv.org/abs/2401.06066 "DeepSeekMoE"; https://arxiv.org/abs/2405.04434 "DeepSeek-V2"
[9]: https://arxiv.org/abs/2101.03961 "Switch Transformers"; https://arxiv.org/abs/2202.08906 "ST-MoE"; https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md "Megatron Core MoE"
[10]: https://github.com/deepseek-ai/DeepSeek-V3 "DeepSeek-V3 repository"
[11]: https://arxiv.org/abs/2305.13245 "GQA"; https://arxiv.org/abs/2205.14135 "FlashAttention"; https://arxiv.org/abs/2307.08691 "FlashAttention-2"; https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html "PyTorch SDPA"
[12]: https://arxiv.org/abs/2405.04434 "DeepSeek-V2 MLA"; https://arxiv.org/abs/2305.13245 "GQA"
[13]: https://arxiv.org/abs/2211.15841 "MegaBlocks"; https://github.com/databricks/megablocks/blob/main/megablocks/layers/moe.py "MegaBlocks implementation"; https://github.com/microsoft/tutel "Tutel repository"
[14]: https://arxiv.org/abs/2302.04761 "Toolformer"; https://developers.openai.com/api/docs/guides/function-calling "Function calling guide"
[15]: https://arxiv.org/abs/2210.03629 "ReAct"; https://react-lm.github.io/ "ReAct project"
[16]: https://arxiv.org/abs/2305.10601 "Tree of Thoughts"; https://arxiv.org/abs/2310.04406 "LATS"
[17]: https://arxiv.org/abs/2303.11366 "Reflexion"
[18]: https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/ "Specification gaming"; https://arxiv.org/abs/1606.06565 "Concrete Problems in AI Safety"
[19]: https://arxiv.org/abs/2305.18290 "DPO"; https://arxiv.org/abs/2406.12045 "τ-bench"
[20]: https://gorilla.cs.berkeley.edu/leaderboard.html "BFCL"; https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard "BFCL repository"
[21]: https://arxiv.org/html/2408.04682v1 "ToolSandbox"; https://github.com/apple-aiml-research/ToolSandbox "ToolSandbox repository"
[22]: https://arxiv.org/html/2406.12045 "τ-bench"; https://github.com/sierra-research/tau-bench "τ-bench repository"
[23]: https://arxiv.org/html/2308.03688v3 "AgentBench"; https://github.com/THUDM/AgentBench "AgentBench repository"
[24]: https://github.com/OpenBMB/MiniCPM "MiniCPM repository"; https://huggingface.co/openbmb/MiniCPM5-1B "MiniCPM5-1B model card"; https://openai.com/policies/terms-of-use/ "OpenAI Terms"; https://www.anthropic.com/legal/commercial-terms "Anthropic Commercial Terms"
