# AYA Edge 0.2 datasets

The repository contains a **pipeline and manifest**, not a production training corpus. No public dataset is automatically assumed to be legally reusable. Before ingestion, record the exact source revision, license and notices, collection date, permitted use, language distribution, filtering decisions, and contamination risk.

## Record contract

Each input JSONL record must contain non-empty `id`, `category`, `text`, `source`, and `license` fields. Optional provenance fields are preserved. The pipeline normalizes whitespace, removes NUL characters, computes a normalized-text SHA-256, rejects duplicate IDs and normalized text, and creates deterministic category-stratified splits.

## Build command

```bash
python scripts/build_dataset.py \
  path/to/licensed_records.jsonl \
  --output-dir datasets/aya_edge_v0_2_prepared \
  --seed 42
```

The output includes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`. The manifest records counts, categories, rejected rows, split policy, and source metadata. Review the generated manifest before training.

Stage 1 begins with an experimental 50/50 mixture of general language/knowledge and instruction/agent patterns. This is a sampling target, not permission to lower quality. The first language milestone uses educational web text only. Agent instruction data starts in the agent-supervision stage. The existing 120-example synthetic dataset remains a development fixture only and must not be reported as the 0.2 production corpus.

## First language slice

`HuggingFaceFW/fineweb-edu`, config `sample-10BT`, is the selected source. Its dataset card declares ODC-By 1.0. Common Crawl terms also apply. The prepare command checks the live licence tag, records the dataset revision, and writes an approval file only for the documents it collects:

```bash
python -m pip install -e ".[data]"
python scripts/prepare_language_corpus.py --char-budget 25000000 \
  --accept-commoncrawl-terms \
  --accept-residual-page-rights-risk
python scripts/train_bpe_tokenizer.py \
  --input datasets/language_v1/train.jsonl \
  --input datasets/language_v1/validation.jsonl \
  --output artifacts/tokenizer/aya_edge_bpe_32768.json \
  --manifest-output artifacts/tokenizer/aya_edge_bpe_32768.manifest.json \
  --corpus-manifest datasets/language_v1/manifest.json \
  --approval-record datasets/language_v1/tokenizer_approval.json \
  --implementation-commit "$(git rev-parse HEAD)"
python scripts/train_language_model.py \
  --config configs/aya_edge_0_2_language.yaml \
  --input-jsonl datasets/language_v1/train.jsonl \
  --validation-jsonl datasets/language_v1/validation.jsonl
```

The 1,000-step configuration is the first saved language run of the 310,115,904-parameter model. It is not a finished pretraining run. Resume the checkpoint to continue toward the 100,000-step plan.


## Bounded agent-SFT pilot

UltraData-SFT-Agent-2609 is the selected agent-supervision candidate, separate from language pretraining. The first pilot is capped at 25,000 trajectories: 12,000 Tool-Use, 5,000 English Search-Agent, and 8,000 General-Agent. The preparation script streams rows and writes structured JSONL splits without materializing the full Hub dataset.

Review `docs/data/ULTRADATA_AGENT_SFT.md` before running:

```bash
python scripts/prepare_ultradata_agent.py \
  --tool-use 12000 \
  --search-agent 5000 \
  --general-agent 8000 \
  --accept-upstream-terms \
  --accept-residual-upstream-rights-risk
```

The dataset card declares Apache-2.0 for the project but also requires compliance with upstream licenses. This pilot is local/personal-research scope only; it is not blanket clearance for commercial deployment or redistribution. The script preserves tool definitions and message boundaries but does not make the frozen virtual tools executable.
