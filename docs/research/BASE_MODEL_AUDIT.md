# AYA Edge Base Model Audit

**Audit status:** Repository audit for the production-oriented prototype branch

**Governing specification:** [`MODEL_SPECIFICATION.md`](../../MODEL_SPECIFICATION.md)

**Scope:** `aya_edge/`, `base model/`, `configs/`, `scripts/`, `tests/`, the mini dataset, and Stage 0 documentation.

## Authority and repository state

`aya_edge/` is the authoritative Python implementation. `base model/aya_edge/` is byte-for-byte equivalent for the current `config.py` and `model.py`, but it duplicates the package and is not included by the root `pyproject.toml`. It should be treated as a historical/reference copy until it is removed or explicitly assigned a maintenance role. The root package, root configuration, root tests, and root scripts are the supported development path.

The repository contains an approximately 85.1M-parameter prototype, not the longer-term approximately 310M target. The implementation is an MLA-inspired latent-KV path rather than a faithful DeepSeek MLA or MiniCPM5 implementation. These distinctions remain part of the acceptance criteria.

## Existing implementation findings

### Model and objective

The model has tied token embedding and language-model head weights, pre-normalised decoder blocks, latent key/value projections, RoPE, PyTorch causal scaled-dot-product attention, a SwiGLU feed-forward implementation, a top-k routed MoE, an optional shared expert, and next-token cross-entropy with `ignore_index=-100`.

The causal language-model shift is explicit: logits at positions `[:, :-1]` are compared with labels at `[:, 1:]`. Padding or other ignored labels are supported through the cross-entropy ignore index. The model does not currently accept an attention mask, so padded batches must be handled by packing or by a future mask-aware interface.

### Attention

The current `LatentAttention` projects queries to full hidden width and compresses inputs to `latent_dim` before projecting separate full-width keys and values. RoPE is applied to the reconstructed full-width key and query tensors. `scaled_dot_product_attention(..., is_causal=True)` supplies the causal mask for the unpadded sequence.

This is a readable correctness baseline. It does not expose a KV cache, decode-time cache accounting, GQA, paged allocation, explicit padding masks, or a decoupled positional latent path. The current latent bottleneck therefore must not be described as proven memory-saving MLA.

### MoE

Each block has a bias-free router, four routed SwiGLU experts by default, top-2 selection by default, renormalised selected probabilities, optional shared SwiGLU computation, and a Switch-style auxiliary balance term. Routing is currently performed with a Python loop over experts and `index_add_`, which is suitable as a numerical reference but may create many small GPU launches and synchronization points.

The implementation now records detached per-layer assignment fractions, mean router probability fractions, selected expert IDs, and router entropy in `routing_stats`. These are diagnostics, not evidence of meaningful expert specialisation. The current code has no capacity factor, overflow policy, dead-expert persistence tracker, z-loss, device-level balancing, grouped GEMM, or expert-parallel dispatch.

### Generation

`generate()` performs autoregressive sampling with temperature and optional top-k filtering. It now validates sampling arguments and uses a sliding context window so a prompt at the maximum configured length can still receive new tokens. It remains a reference implementation: it recomputes the full context at every step and has no KV cache.

## Engineering changes on this branch

- Configuration validation now rejects non-positive dimensions, invalid RoPE head widths, latent widths larger than the model width, invalid expert counts/top-k, and non-boolean shared-expert flags.
- Configuration values can be serialized with `to_dict()` and reconstructed with `from_dict()` with unknown-key rejection.
- Routing diagnostics are exposed from the model forward result without retaining graph references.
- Generation validates temperature, top-k, and token-count arguments and supports sliding-window continuation.
- `aya_edge.training` provides explicit seed handling plus model/optimizer/scheduler/scaler/RNG checkpoint round trips and metadata for step, token count, configuration, and dataset version.
- Tests cover routing telemetry, configuration errors, sliding-window generation, active-parameter estimates, finite model integrity, and checkpoint restoration.

These changes improve observability and reproducibility. They do not make the model production-ready or replace a measured grouped-dispatch implementation.

## Parameter accounting

The authoritative count must be produced by instantiating the model and summing trainable tensors. Tied embedding/output weights are counted once by PyTorch parameter registration. The branch also exposes a structural `count_active_parameters()` estimate that includes the configured top-k routed experts and optional shared expert per layer while counting shared attention, router, norm, and embedding parameters once.

The active estimate is not a FLOP count. It does not account for token imbalance, padding, dispatch overhead, optimizer state, activation memory, or the fact that generation may touch different experts over time. Any architecture promotion must report both total and active estimates alongside actual throughput and peak memory measurements.

## Reproducibility and checkpoint integrity

A valid run should record the exact Git revision, YAML configuration, model configuration, dataset manifest/version, seed, Python and PyTorch versions, CUDA version, GPU model, dtype, sequence length, batch/micro-batch settings, optimizer and scheduler settings, tokens processed, losses, throughput, peak memory, and routing diagnostics.

The checkpoint helper stores model weights, optimizer state, optional scheduler and scaler state, Python/PyTorch/CUDA RNG state, training step, tokens processed, configuration, and dataset version. The test suite checks a model and optimizer round trip. A real training run must additionally compare outputs before and after loading and verify resumed loss/step behaviour.

## Dataset audit

`aya_edge_mini_dataset_v1/` contains 120 JSONL examples split into 80 train, 20 validation, and 20 test records. The validator reports no exact split leakage, valid required fields, balanced general/agentic categories, valid simulated tool calls, and an estimated average of approximately 62 tokens per example. The data is suitable for smoke tests and overfitting experiments only.

Tool results in the dataset are simulated. The dataset does not demonstrate live retrieval, external execution, generalisation, or agent competence. The Stage 1 50/50 general-language/instruction-agent direction remains an experimental allocation and requires a larger, licensed, token-counted dataset before pretraining claims.

## Remaining risks and next measurements

1. Install and pin the development dependencies, then run the full test suite on CPU and a CUDA smoke test.
2. Measure loop dispatch against a vectorized/grouped reference at several batch and sequence lengths before scaling training.
3. Add explicit capacity/overflow metrics and persistent dead-expert reports before comparing routing policies.
4. Add mask-aware attention tests before using padded batches or packed examples.
5. Benchmark prefill and decode memory/latency with and without a real KV cache; do not infer cache savings from the latent width alone.
6. Verify the exact 85.1M count, active estimate, tied-weight accounting, and checkpoint outputs against a pinned commit.
7. Audit the duplicate `base model/` directory and either remove it in a separate change or document a deliberate synchronization policy.
8. Build the local verifier-backed agent harness described by the official capability specification before Stage 4.

## Audit conclusion

The repository is a useful experimental sparse-MoE scaffold and is suitable for a production-oriented prototype branch. It is not a production-ready model, complete training system, optimized inference engine, or demonstrated agent. The safe path is to preserve the readable implementation as a control, add diagnostics and checkpoint discipline, and promote architectural changes only when matched-quality, memory, throughput, and task-completion evidence supports them.
