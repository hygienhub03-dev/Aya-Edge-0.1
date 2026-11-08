# Repository audit before data-pipeline implementation

**Audit date:** 2026-11-07. **Repository:** `hygienhub03-dev/Aya-Edge-0.1`. This branch is intentionally based on stable `main`; it does not include or modify the open 0.2 PR branch.

## State and branch choice

- Default branch: `main`, based on commit `6c5577bcfbfd0be8693a1018268fa76e2afd30e4` at the start of work.
- Pull request #1 had been merged into `main`; the current base is the merge commit for that earlier research/prototype work.
- Pull request #2, “AYA Edge 0.2: production-oriented 310M agent-first candidate”, was open against `main` during the audit and remains a separate line of work. The latest observed PR head is `90b763f6bc8963f3c403622fdd8bc84a18c58488` on `feature/aya-edge-02-production`.
- This work uses `feature/aya-edge-dataset-acquisition` branched from `main` at the audited base, not from PR #2. It does not modify PR #2 files or rewrite its history, and it must not be merged automatically.

## Existing infrastructure reviewed

The repository already includes the locked AYA Edge identity/capability specification, a small model scaffold, training/research documentation, and a development tokenizer. The 0.2 PR contains an architecture-oriented model config and a fixed byte-level tokenizer. Existing Python package metadata and `.gitignore` rules were reused and narrowly extended. The branch was checked for pre-existing dataset scripts, data schemas, acquisition configs, notebooks, tests, and manifest machinery before additions; there was no end-to-end source-licensed acquisition, normalization, contamination, token-mixture, and acceptance pipeline to reuse.

## New infrastructure supplied by this branch

This branch adds source-specific registry/acquisition/filter/mixture/holdout configs; revision and license checks; a streaming/resumable Hugging Face pilot downloader; bounded sample inspection; canonical document/conversation/trajectory/tool-call schemas; normalization and rejection logs; exact and conservative near-deduplication; group-aware deterministic splits; explicit tokenizer counts; token-weighted category sampling; validation, reporting, and test wrappers; local Wikimedia XML shard parsing behind the same license and SHA gates; synthetic tests; research, licensing, quality, mixture, contamination, tokenizer, acquisition and acceptance reports; and Kaggle-oriented usage guidance. External payloads remain disabled and raw/processed data are ignored.

## Reusable files and deliberate changes

- Reused the repository's `pyproject.toml`, `.gitignore`, stable `main` model identity, and documented Kaggle environment expectation.
- Added optional data dependencies rather than changing core model dependencies.
- Preserved model architecture and training objective unchanged.
- Kept BFCL held out and the two RL pivot sources outside Stage 1.

## Dependency and conflict with 0.2

The data processing code can be developed and tested independently against synthetic fixtures, but production token accounting and training data cannot be accepted until the 0.2 model/tokenizer vocabulary is reconciled. PR #2 declares `vocab_size: 32768`; its fixed byte tokenizer declares 268 total IDs. Stable `main` has a separate 32,000 vocabulary setting. The data branch does not choose a correction; it refuses to represent development byte counts as production tokens.
