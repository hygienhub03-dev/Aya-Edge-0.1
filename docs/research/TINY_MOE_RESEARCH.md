# Tiny-MoE Research Report

**Project:** AYA Edge 0.1
**Status:** Research synthesis; no production-readiness claim
**Evidence basis:** the project research notes, official model specification, and cited upstream sources

## Executive conclusion

The most useful compact-MoE comparison is not a single “best” model. **MiniMind-3-MoE** is the strongest practical training baseline because its official project documents a 198M-total/64M-active release and an end-to-end native-PyTorch training, tool-use, and agentic-RL path. **Tiny-MoE** is the best design reference for shared experts, top-2 routing, router diagnostics, grouped expert execution, and memory-conscious training, but its own repository describes it as a learning/research project and reports approximate parameter ranges. **Phi-tiny-MoE** is a useful quality/latency and later compression reference, not a genuinely tiny from-scratch target: it was distilled and slimmed from a much larger model using 400B tokens and 64 A100 GPUs. [1–3]

The recommendation is to **adopt MiniMind-style top-1 routing as a control condition, retain Tiny-MoE-style shared-expert/top-2 routing as a matched-compute ablation, and defer SlimMoE-style compression** until AYA has a capable teacher and verified training/evaluation pipeline. None of these sources establishes AYA agent competence or production readiness.

## AYA constraints and current baseline

AYA is an agent-first compact assistant whose operating loop is **Understand → Plan → Act → Observe → Adapt → Complete**. Tool execution, permission enforcement, argument validation, and operational state belong to the external runtime; the model must plan, select tools, emit structured calls, interpret observations, recover, and verify completion. Task completion is the central capability measure. [4]

The current repository scaffold is reported at **85,099,520 trainable parameters**, with a 32,000 vocabulary, 512 hidden dimension, 8 layers, 8 attention heads, latent KV width 128, 4 routed experts, top-2 routing, 1,024 expert hidden dimension, a shared expert, and a 512-token maximum sequence length. The count requires independent verification against the instantiated model. The **85.1M scaffold is not the longer-term approximately 310M target** and must not be described as that completed target. [5]

## Implementations and original sources

### Tiny-MoE (AbdelrhmanEbied)

**Original source and date.** The official GitHub repository is `AbdelrhmanEbied/Tiny-MoE`, created according to the research note on 19 May 2026. [1]

**Documented architecture and parameters.** The repository describes a native-PyTorch decoder-only Transformer trained on two NVIDIA T4 GPUs on Kaggle. It reports approximately **150M–200M total parameters** and **70M–90M active parameters per token** as model specifications, not as a precise checkpoint count. Its documented design has 14 layers, hidden size 512, 8 routed experts, 1 shared expert, top-2 routing, 32,000 vocabulary, 1,024 MoE intermediate size, and 512-token base context. It also documents optional YaRN extension to 2,048 tokens and MLA-style attention with KV compression, RoPE, SDPA, and inference weight absorption. [1]

**Training and engineering.** The documented data stream mixes FineWeb-Edu, Cosmopedia v2, and Open-Web-Math at 60%/25%/15%; packs contiguous 512-token windows; and uses 8-bit AdamW, FP16, gradient accumulation, activation checkpointing, Torch Compile, and DeepSpeed ZeRO-2/expert parallelism. The project reports 30,000 training steps. It reports top-2 routing over 8 experts, capacity factor 1.25, auxiliary router loss, router z-loss, load-balancing metrics, a shared expert, and grouped GEMM. [1]

**Strengths.** It is unusually transparent for architecture study; exposes routing diagnostics; gives a concrete shared-expert/top-2 design; and documents practical packing and memory-saving techniques. **Weaknesses.** Parameter reporting is approximate; the repository describes limited conversational ability and a learning/research objective; two-T4 feasibility is not evidence of AYA-quality results; and the reviewed material does not establish robust structured tool calling or multi-step completion. [1]

**AYA recommendation.** **Adopt as a design reference and ablation**, not as the AYA agent baseline. Test one shared path plus routed experts, router z-loss, capacity behavior, and grouped dispatch at matched active compute. Preserve explicit measurements for requested routes, accepted routes, overflow/drops, expert utilization, latency, and tool-task outcomes.

### MiniMind-3-MoE

**Original source and date.** The official Apache-2.0 repository is `jingyaogong/minimind`; the project site and changelog document MiniMind-3-MoE as released 1 April 2026. [2,3]

**Documented architecture and parameters.** The release table documents **198M total / 64M active parameters**, 8 layers, hidden size 768, vocabulary 6,400, 4 experts, and top-1 routing. Earlier variants are separately documented, including MiniMind2-MoE at 145M and MiniMind-v1-MoE at 4×26M; no extra arithmetic should be inferred from model names. The current implementation exposes `num_experts` and `num_experts_per_tok`, computes a softmax gate, selects top-k experts, can normalize top-k probabilities, combines expert outputs, and applies an auxiliary router loss. The 1 April 2026 changelog says the shared-expert design was removed while aligning the structure with the Qwen3/Qwen3-MoE ecosystem. [2–4]

**Training and engineering.** MiniMind’s primary value is system completeness: the repository documents pretraining, SFT, LoRA, DPO, PPO/GRPO/CISPO, tool calling, agentic RL, adaptive thinking, and distillation. The changelog documents tool-call data in SFT, a native multi-turn agentic-RL script, and OpenAI-compatible serving. The project site reports roughly 0.5 GB memory for dense MiniMind-3 and roughly 1.0 GB for MiniMind-3-MoE; these deployment figures require validation under AYA’s runtime. [2,3]

**Strengths.** It has the smallest documented active budget among the reviewed current releases; exact release/configuration figures; a simple top-1 router; and a comparatively complete path to tool use and agentic RL. **Weaknesses.** Top-1 supplies less expert diversity than top-2; the current branch removes shared experts, so it does not directly test one of AYA’s architecture priorities; and repository demonstrations are not independent evidence of capability. [2–4]

**AYA recommendation.** **Adopt as the initial compact-MoE control condition**, preferably at a configuration whose total and active counts are recorded from the instantiated model. Measure language loss, throughput, router entropy, per-expert requested/accepted load, structured-output validity, tool selection, argument correctness, recovery, and verified task completion.

### Phi-tiny-MoE and SlimMoE

**Original source and dates.** Microsoft’s official `Phi-tiny-MoE-instruct` model card documents the released model; the SlimMoE paper is arXiv v1 dated 23 June 2025. [5,6]

**Documented architecture and parameters.** The model card documents a decoder-only MoE with **3.8B total / 1.1B active parameters** and 4k context. The SlimMoE paper documents 32 layers, width 4,096, 16 experts, top-2 routing, expert width 448, and the same 3.8B/1.1B counts. The method retains experts while pruning redundant neurons, uses staged compression rather than one-shot pruning, and applies knowledge distillation. [5,6]

**Training.** The model card reports compression/distillation from the Phi-3.5-MoE/GRIN-MoE family, supervised fine-tuning and DPO, 400B tokens including Phi-3 synthetic data and filtered public documents, training between September 2024 and March 2025, and 64 A100-80G GPUs for 11 days. [5]

**Strengths.** It is a useful compact-MoE performance and latency reference and supplies evidence that structured expert slimming plus distillation can outperform naive one-shot pruning. **Weaknesses.** Its training budget is not a tiny from-scratch recipe; its architecture and compression pipeline are much more complex than AYA’s baseline; and model-card caveats and English-focused intended use make it a reference rather than an immediate multilingual/agentic foundation. [5,6]

**AYA recommendation.** **Reject as the initial architecture target; test later as a quality/latency reference or teacher candidate.** Do not transfer its parameter counts, quality, or training feasibility to AYA.

## Cross-cutting architecture findings

Fine-grained MoE research supports splitting experts into smaller units, increasing selected routed experts proportionally, and isolating shared experts for common knowledge. The DeepSeekMoE 2B example documents 1 shared plus 63 routed experts with 7 routed active, but this is a reference configuration, not an AYA requirement. Fine-grained routing increases combinations but can also raise dispatch, memory, and communication overhead. [7]

Load balance is not one scalar. AYA should monitor router probability mass and realized assignments separately, plus entropy, dead experts, capacity utilization, overflow/drops, residual bypasses, logit stability, and local/global counts. Switch’s reported coefficient and ST-MoE’s reported z-loss coefficient are paper-specific starting points, not AYA defaults. Router math should remain in FP32 or higher where justified. [8]

Python per-expert loops are suitable as a correctness reference, not a serving design. A production-oriented experiment should flatten tokens, retain expert IDs and offsets, use vectorized gather/scatter and grouped/batched expert computation, and measure permutation, synchronization, dispatch, MLP, and combine time separately. Dropless or block-sparse paths should be introduced only if capacity drops or padding dominate. [9]

## Licensing caveats

- Tiny-MoE and MiniMind repositories are documented as Apache-2.0. [1–3]
- Phi-tiny-MoE’s model card lists MIT; the SlimMoE paper is separately identified as CC BY 4.0 at arXiv. A paper license is not automatically a model-weight or code license. [5,6]
- DeepSeek papers are research sources distributed under arXiv terms; the DeepSeek-V3 repository separates MIT-licensed code from a separate model license for weights. Reimplement mechanisms or audit exact code/weight terms before reuse. [7]
- Any training data, generated text, benchmark items, and third-party API material require their own provenance and license review. Apache/MIT on a repository does not grant rights to unrelated data or outputs.

## Decision matrix

| Candidate | AYA use | Decision | Required test |
|---|---|---|---|
| MiniMind-3-MoE | Compact top-1 control | **Adopt for baseline** | Matched language, routing, tool-use, recovery, completion, memory, and latency suite |
| Tiny-MoE | Shared/top-2 and diagnostics reference | **Adopt as ablation** | Matched active compute; overflow, dispatch, z-loss, shared-path and agent metrics |
| Phi-tiny-MoE/SlimMoE | External compact quality/compression reference | **Defer/reject initial target** | Later teacher/compression study with licensing and teacher-quality checks |
| DeepSeekMoE ideas | Shared/fine-grained routing hypotheses | **Test selectively** | One-device, fixed-compute experiments; no copied scale assumptions |

## References

[1]: https://github.com/AbdelrhmanEbied/Tiny-MoE "AbdelrhmanEbied/Tiny-MoE official repository"
[2]: https://github.com/jingyaogong/minimind "jingyaogong/minimind official repository"
[3]: https://jingyaogong.github.io/minimind/ "MiniMind official project site"
[4]: https://raw.githubusercontent.com/jingyaogong/minimind/master/model/model_minimind.py "MiniMind model source"
[5]: https://huggingface.co/microsoft/Phi-tiny-MoE-instruct "Microsoft Phi-tiny-MoE-instruct model card"
[6]: https://arxiv.org/html/2506.18349 "SlimMoE: Structured Compression of Large MoE Models via Expert Slimming and Distillation"
[7]: https://arxiv.org/abs/2401.06066 "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
[8]: https://arxiv.org/abs/1701.06538 "Sparsely-Gated MoE"; https://arxiv.org/abs/2006.16668 "GShard"; https://arxiv.org/abs/2101.03961 "Switch Transformers"; https://arxiv.org/abs/2202.08906 "ST-MoE"; https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md "Megatron Core MoE documentation"
[9]: https://arxiv.org/abs/2211.15841 "MegaBlocks"; https://github.com/databricks/megablocks/blob/main/megablocks/layers/moe.py "MegaBlocks implementation"; https://docs.pytorch.org/ao/stable/eager_tutorials/mxfp8_expert_parallel_training.html "PyTorch expert-parallel tutorial"
