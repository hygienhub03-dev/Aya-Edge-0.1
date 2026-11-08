# Tokenizer and 0.2 vocabulary audit

**Status: blocking issue for 0.2 production data accounting/training.** This tokenizer milestone is based on merged AYA Edge 0.2 `main`. It preserves the byte adapter, adds the production BPE contract and approval-gated trainer, separates development byte counts from production counts, and makes the distinction visible in mixture reports.

| Repository state | Declared model vocabulary | Tokenizer identity/effective IDs | Consequence |
|---|---:|---:|---|
| Merged `main` after PR #2 | `configs/aya_edge_0_2.yaml`: 32,768 | `aya_edge/tokenizer/byte_level.py` uses a fixed 256-byte vocabulary plus special tokens, and the tokenizer specification declares 268 IDs total. | The model embedding/output vocabulary and development tokenizer ID space do not match: **32,768 vs 268**. |
| Tokenizer milestone | `configs/aya_edge_0_2.yaml`: 32,768 | `aya_edge/tokenizer/bpe.py` defines a fixed 32,768 byte-level BPE contract with reserved IDs 0–15; no artifact is generated without corpus approval. | This is the intended production direction, not evidence that a production tokenizer already exists. |
| Data pipeline development proxy | Not a model vocabulary | `aya-edge-dev-byte-v1`: one token ID per UTF-8 byte; deterministic only. | Useful for pipeline tests, never label as production model-token count. |

The merged 0.2 config and development tokenizer cannot be used together as production inputs: the adapter emits only 268 IDs while the model allocates for 32,768. The new production BPE trainer addresses the contract but does not bypass corpus approval. A final tokenizer must be pinned by exact artifact/checksum, its vocabulary size and special-token IDs must agree with the model config, and the same artifact must be used by counting, training, and inference. No 32,000-vocabulary setting may be silently substituted for 0.2.

The pipeline sets `production_required: false` in the pilot-only mixture config so tests can demonstrate byte-token accounting, and reports `production_tokenizer: false`. Before Stage 1 training, pin the chosen production tokenizer JSON and vocabulary checksum, set its exact `id`/path in `mixture.yaml`, and change `production_required` to `true`. `build_mixture.py` then refuses the development proxy. The acceptance report must list production-token counts and actual versus target category proportions; byte counts from this branch are not acceptable substitutes.

Evidence reviewed in the repository: merged `main` `configs/aya_edge_0_2.yaml`, `aya_edge/tokenizer/byte_level.py`, `aya_edge/tokenizer/bpe.py`, the dataset acquisition manifests, and `docs/research/AYA_EDGE_PRODUCTION_TOKENIZER_SPEC.md`. This audit describes repository configuration, not a claim that a production tokenizer or checkpoint exists.
