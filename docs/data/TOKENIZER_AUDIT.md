# Tokenizer and 0.2 vocabulary audit

**Status: blocking issue for 0.2 production data accounting/training.** This data-pipeline branch is based on the audited stable `main` branch and does not change the model or tokenizer. It introduces an explicit tokenizer ID, separates development byte counts from production counts, and makes the distinction visible in mixture reports.

| Repository state | Declared model vocabulary | Tokenizer identity/effective IDs | Consequence |
|---|---:|---:|---|
| Stable `main` scaffold | `aya_edge/config.py`: 32,000 | Existing repo was inspected before acquisition work; no verified production tokenizer contract is present in this branch base. | This is not the 0.2 config and is not a production count basis for 0.2. |
| Open 0.2 production PR #2 | `configs/aya_edge_0_2.yaml`: 32,768 | `aya_edge/tokenizer/byte_level.py` uses a fixed 256-byte vocabulary plus special tokens, and the tokenizer specification declares 268 IDs total. | The model embedding/output vocabulary and tokenizer ID space do not match: **32,768 vs 268**. |
| Data pipeline development proxy | Not a model vocabulary | `aya-edge-dev-byte-v1`: one token ID per UTF-8 byte; deterministic only. | Useful for pipeline tests, never label as production model-token count. |

The open 0.2 PR’s config and tokenizer therefore cannot be used together as-is: inputs can only produce the 268 tokenizer IDs, while the model config allocates for 32,768. A tokenizer trained or selected for the intended model must be pinned by exact artifact/checksum, its vocabulary size and special-token IDs must agree with the model config, and the same artifact must be used by counting, training, and inference. Alternatively, the 0.2 model config could be revised to match the fixed byte tokenizer, but that decision belongs to the model/architecture PR and is not made in this data branch. The stable-main 32,000 setting is a separate identity and must not be silently substituted for 0.2.

The pipeline sets `production_required: false` in the pilot-only mixture config so tests can demonstrate byte-token accounting, and reports `production_tokenizer: false`. Before Stage 1 training, pin the chosen production tokenizer JSON and vocabulary checksum, set its exact `id`/path in `mixture.yaml`, and change `production_required` to `true`. `build_mixture.py` then refuses the development proxy. The acceptance report must list production-token counts and actual versus target category proportions; byte counts from this branch are not acceptable substitutes.

Evidence reviewed in the repository: stable `main` `aya_edge/config.py`; open branch `feature/aya-edge-02-production` `configs/aya_edge_0_2.yaml`, `aya_edge/tokenizer/byte_level.py`, and `docs/research/AYA_EDGE_02_TOKENIZER_SPEC.md`. This audit describes repository configuration, not a claim that a production tokenizer or checkpoint exists.
