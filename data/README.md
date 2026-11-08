# AYA Edge data pipeline

This directory holds code-adjacent schemas, small provenance manifests, and locally generated pilot outputs. Raw dataset payloads are written to `data/raw/` by default and are excluded from Git. Set `AYA_DATA_ROOT` to an explicit path to move raw data, caches, and generated reports elsewhere. Never place Hugging Face tokens in configuration or committed files; use `HF_TOKEN` or the Hugging Face CLI login.

The acquisition pipeline fails closed. A source can enter a Stage 1 training corpus only when the source registry contains an immutable 40-character repository commit (or an official dated Wikimedia snapshot), a reviewed license record, `license_verified: true`, and an allowed approval status. Held-out BFCL and later-RL candidates are not Stage 1 sources. No source is approved merely because it is public or hosted on Hugging Face.

## Environment setup

For Kaggle T4, install the data tools without downloading corpus payloads first:

```bash
python -m pip install --upgrade datasets huggingface_hub pyarrow pandas pyyaml jsonschema tokenizers pytest
export AYA_DATA_ROOT=/kaggle/working/aya-edge-data
export HF_HOME=/kaggle/temp/hf-cache
```

Kaggle working and cache paths are session-scoped. Save any approved pilot corpus outside the ephemeral session, for example as a private Kaggle Dataset or another authorized object store, and record its checksums. Data preparation and model training are separate; these commands do not start training.

## Commands

Run from the repository root. The source inspection resolves revisions for review; it does not rewrite the pinned registry. License review must pass before download.

```bash
python scripts/data/inspect_sources.py
python scripts/data/verify_licenses.py --config configs/data/source_registry.yaml
python scripts/data/verify_licenses.py --config configs/data/source_registry.yaml --strict
python scripts/data/download_datasets.py --config configs/data/acquisition.yaml --stage pilot
python scripts/data/inspect_samples.py --limit 100
python scripts/data/normalize_datasets.py --stage pilot
python scripts/data/deduplicate.py --stage pilot
python scripts/data/build_splits.py --stage pilot
python scripts/data/count_tokens.py --stage pilot
python scripts/data/build_mixture.py --config configs/data/mixture.yaml --stage pilot
python scripts/data/validate_dataset.py --stage pilot
python scripts/data/run_acceptance_tests.py --stage pilot
python scripts/data/generate_report.py --stage pilot
```

The repository's development byte tokenizer is explicitly labeled `aya-edge-dev-byte-v1`. Its UTF-8 byte-ID counts are valid only for that development tokenizer and are not production-token counts. The pilot mixture report retains that warning. Before Stage 1 training, configure a pinned local tokenizer JSON that matches the chosen 0.2 model and require `production_required: true` in `configs/data/mixture.yaml`.

`data/manifests/` records source revisions, checksums, split assignments, and acquisition metadata. `data/processed/` and run-specific report outputs are generated locally. Small reviewed source and summary manifests may be committed; raw data, caches, credentials, and large processed payloads must not be.

Wikimedia XML is not fetched by the Hugging Face downloader. After the legal gate is cleared, stage only an approved shard from the pinned snapshot, verify its SHA-256 against Wikimedia's official `dumpstatus.json`, then use the streaming parser with a small row cap:

```bash
python scripts/data/convert_wikimedia_dump.py \
  --source-id wikimedia_enwiki \
  --dump /path/to/enwiki-20261001-pages-articles-multistream1.xml-p1p41242.bz2 \
  --snapshot 20261001 \
  --expected-sha256 <official-sha256> \
  --namespaces 0 \
  --limit 100
```

The command checks the registry approval and dated snapshot before parsing; the current registry blocks it, so it will refuse to process content. The shard name above is illustrative: select an actual file and its checksum from the pinned official dump status only after the rights and namespace review. The parser retains page/revision IDs, revision timestamps, attribution metadata, text, source URL, snapshot, and shard checksum.
