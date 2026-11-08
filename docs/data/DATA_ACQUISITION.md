# Data acquisition plan and current status

## Decision

The pipeline is implemented and the requested datasets are pinned for review, but **no external dataset payload was acquired**. All Stage 1 candidate jobs are disabled because source-specific license or rights questions remain open. `verify_licenses.py --strict` therefore fails closed; `download_datasets.py` confirms there are no approved/enabled Stage 1 jobs and downloads nothing. This is intentional, not a failed download.

`configs/data/acquisition.yaml` is a reproducible *proposed* pilot. Once legal/source review resolves the blockers recorded in `DATASET_LICENSING.md`, a reviewer may update a source approval and enable only that subset. The downloader verifies the approved source, exact registry/acquisition revision match, positive record cap, and checksummed completed-file manifest before accepting payloads. It streams and can resume from verified JSONL partials bound to the same revision/config/split. Raw payloads and Hugging Face caches are excluded from Git.

## Source plan

| Intended role | Source and pinned identity | Pilot cap if later approved | Current status |
|---|---|---:|---|
| General pretraining | FineWeb `sample-10BT`, `train`, commit `9bb295ddab0e05d785b879661af7260fed5140fc` | 100 records | Disabled; webpage rights/terms unresolved |
| Agentic SFT | Nemotron-Agentic-v1 `interactive_agent`, commit `650d590978ca35c8f1ecea2faf136e5fac421b62` | 50 records | Disabled; Glaive component and schema unresolved |
| Tool-use SFT | Nemotron-Agentic-v1 `tool_calling`, same commit | 100 records | Disabled; component/tool-set lineage unresolved |
| Instruction SFT | Nemotron-Instruction-Following-Chat-v1 `chat_if`, commit `83dcd3aded0d289b0bbc018d3f9af4c5dd4005df` | 100 records | Disabled; subset/upstream terms unresolved |
| Structured-output SFT | Same NVIDIA repo, `structured_outputs`, same commit | 100 records | Disabled; upstream/model terms unresolved |
| General knowledge | Wikimedia English `pages-articles`, dated snapshot `20261001` | 100 pages from one checksum-pinned local shard | Disabled; item-level rights and namespace review unresolved |
| Later-stage only | Nemotron conversational pivot, commit `bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e` | 100 records, after later review | Disabled; not Stage 1; provenance/count mismatch |
| Later-stage only | Nemotron function pivot, commit `0ac34ec81bb6976d3746407df306a57ce48f0bd8` | 100 records, after later review | Disabled; not Stage 1; scope/provenance/count mismatch |
| Evaluation only | BFCL V4 evaluator commit `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000` | None in training pipeline | Held out; never acquire into training |

The maximum Stage 1 record caps total 550 across candidates if every one were later approved, but those are ceilings, not a target corpus size. The FineWeb `sample-10BT` configuration itself is a 10-billion-token pool; the streaming cap, not the source configuration, bounds the proposed pilot. Wikimedia has no automatic bulk downloader in this branch: the local XML converter requires the exact snapshot, official shard SHA-256, and registry approval before it reads the shard.

## Mixture and tokenizer

The intended training mixture is 50% general text/knowledge and 50% agentic/instruction/tool-use **by tokenizer token count**, not by row count. `configs/data/mixture.yaml` groups `general` separately from `instruction`, `agent`, and `tool_use`, uses a deterministic seed, and reports both target and actual token shares and deviations. The 50/50 report cannot be populated until an approved dataset is acquired.

The current pilot tokenizer ID `aya-edge-dev-byte-v1` counts UTF-8 bytes as token IDs. This provides deterministic development accounting only. It is not the production tokenizer. PR #2’s 0.2 model config sets `vocab_size: 32768`, while its byte-level tokenizer has 268 total token IDs (256 byte IDs plus special tokens); the audited stable-main scaffold reports 32,000. This mismatch blocks any 0.2 production token-count report or training run until the model/tokenizer contracts are reconciled. See `TOKENIZER_AUDIT.md`.

## Kaggle T4 reproducibility

The project does not train the model or download data as part of setup. On a Kaggle T4 notebook, use session-scoped paths and install only data/test tooling after confirming PyTorch is already available:

```bash
python -m pip install --upgrade datasets huggingface_hub pyarrow pandas pyyaml jsonschema tokenizers pytest
export AYA_DATA_ROOT=/kaggle/working/aya-edge-data
export HF_HOME=/kaggle/temp/hf-cache
python scripts/data/inspect_sources.py --config configs/data/source_registry.yaml
python scripts/data/verify_licenses.py --config configs/data/source_registry.yaml
python scripts/data/verify_licenses.py --config configs/data/source_registry.yaml --strict
```

With the current registry, the strict command returns a non-zero status for blocked acquisition candidates. That is the required stop. Do not toggle any `enabled` flag to bypass it. If an approved license decision is recorded, first inspect the source card and official revision again, then enable only the authorized source and run:

```bash
python scripts/data/download_datasets.py --config configs/data/acquisition.yaml --stage pilot
python scripts/data/inspect_samples.py --limit 100
python scripts/data/normalize_datasets.py --stage pilot
python scripts/data/deduplicate.py --stage pilot
python scripts/data/build_splits.py --stage pilot
python scripts/data/count_tokens.py --stage pilot --tokenizer-id <pinned-production-tokenizer-id> --tokenizer-path <local-tokenizer.json>
python scripts/data/build_mixture.py --config configs/data/mixture.yaml --stage pilot
python scripts/data/validate_dataset.py --stage pilot
python scripts/data/run_acceptance_tests.py --stage pilot
python scripts/data/generate_report.py --stage pilot
```

For a licensed Wikimedia shard, obtain the shard and official SHA-256 separately, then run `convert_wikimedia_dump.py` with `--snapshot`, `--expected-sha256`, `--namespaces 0`, and a small `--limit`; the current registry blocks this command before it reads the file. Confirm the selected namespace from the snapshot docs/metadata, since the documented `pages-articles` namespace coverage is ambiguous.

Kaggle working and cache directories are ephemeral. Preserve approved pilot outputs only in a private, authorized storage destination, with a license/provenance manifest and checksums. Never upload source payloads to public storage, and do not commit raw records, cookies, or access tokens.
