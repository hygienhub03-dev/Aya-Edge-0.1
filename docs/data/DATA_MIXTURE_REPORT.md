# Stage 1 data-mixture report

**Run status: blocked before acquisition.** No external data was acquired, so the observed mixture contains **0 records and 0 tokens**. This report records the configured objective and why no training-ready mixture can be reported.

## Intended objective

The pilot config sets a deterministic **50% general text/knowledge and 50% agentic/instruction/tool-use mixture by token count**, not by record count. The general group contains the `general` category, intended for a later-approved FineWeb or Wikimedia text subset. The agent group contains `instruction`, `agent`, and `tool_use`, intended for later-approved instruction-following and tool-interaction material. The pipeline samples category groups with a stable seed, stops at token-count targets, and reports target versus actual counts and deviations. Because records vary in length, row counts are not used as a proxy for token balance.

| Mixture group | Target token share | Candidate categories | Approved sources in current run | Actual tokens |
|---|---:|---|---:|---:|
| General | 50% | `general` | 0 | 0 |
| Agentic/instruction/tool-use | 50% | `instruction`, `agent`, `tool_use` | 0 | 0 |
| **Total** | **100%** |  | **0** | **0** |

The RL pivot datasets are later-stage candidates only and are not members of the Stage 1 mixture. BFCL V4 is held out for evaluation and never included in training. Every external Stage 1 job is currently disabled because its registry status is `blocked_pending_review`; changing the acquisition flag alone cannot bypass the code's license gate.

## Token accounting limit

The pilot-only development proxy `aya-edge-dev-byte-v1` counts UTF-8 bytes as IDs. It is useful for deterministic pipeline tests but is **not** a production tokenizer and cannot support the model's token-budget or embedding vocabulary claims. The open 0.2 PR defines `vocab_size: 32768` while its fixed byte tokenizer defines 268 total IDs. Stable `main` separately reports 32,000. This is a blocking mismatch; see `TOKENIZER_AUDIT.md`.

Before any production mixture is accepted, the model/tokenizer owners must resolve the vocabulary mismatch, pin a tokenizer artifact and checksum, use that exact artifact in training/inference/counting, and set `production_required: true`. Then acquire only source-approved subsets, count training tokens with the production tokenizer, and regenerate the target/actual mixture report, including per-source token shares and deviations. No Stage 1 training is authorized by this configuration.
