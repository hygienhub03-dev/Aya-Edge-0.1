# AYA Edge 0.2 dual-T4 DDP smoke test

This milestone validates infrastructure only: exactly two visible Tesla T4 GPUs, one process per GPU, one distributed forward/backward/optimizer step, and a rank-0 checkpoint. It does not establish language learning, agent competence, production throughput, or corpus quality.

## Safety gates

The Kaggle notebook [`AYA_Edge_02_Dual_T4_DDP_Smoke.ipynb`](../../notebooks/AYA_Edge_02_Dual_T4_DDP_Smoke.ipynb) clones the dedicated branch, checks out an immutable commit, runs the repository test suite, and refuses to proceed unless exactly two devices are visible and both identify as Tesla T4. It does not silently fall back to single-GPU execution.

The worker [`scripts/dual_t4_ddp_smoke.py`](../../scripts/dual_t4_ddp_smoke.py) uses PyTorch DDP with one complete model replica per GPU. The two 16-GiB devices do not become one pooled 32-GiB model. `find_unused_parameters=True` is enabled because sparse MoE routing can leave expert parameters unused in a synthetic microbatch.

Synthetic token IDs are sampled from the 268-token development adapter while the model embedding vocabulary is 32,768. This deliberately exercises valid embedding lookup, distributed gradients, optimizer state, checkpoint writing, and metric collection while keeping `production_training_allowed: false`. It must not be interpreted as a tokenizer or pretraining test.

## Expected artifacts

Rank 0 writes:

- `outputs/dual_t4_smoke/metrics.json`
- `outputs/dual_t4_smoke/aya_edge_02_ddp_smoke.pt`

The metrics record the world size, GPU names, total and structural-active parameter counts, loss, auxiliary loss, gradient norm, whether embedding weights changed, elapsed time, and explicit production-training status. The checkpoint records model/optimizer state and the vocabulary mismatch for auditability. It is an infrastructure artifact only and must not initialize production training.

## Production gate

Real pretraining remains blocked until the approved 32,768-entry BPE tokenizer, its artifact and corpus-manifest hashes, and a licensed/accepted training corpus are available and validated together. The production tokenizer path is documented in [`AYA_EDGE_PRODUCTION_TOKENIZER_SPEC.md`](AYA_EDGE_PRODUCTION_TOKENIZER_SPEC.md).
