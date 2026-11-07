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

Stage 1 begins with an experimental 50/50 mixture of general language/knowledge and instruction/agent patterns. This is a sampling target, not permission to lower quality. The existing 120-example synthetic dataset remains a development fixture only and must not be reported as the 0.2 production corpus.
