# AYA Edge data pipeline acceptance report

**Run/report date:** 2026-11-08. **Scope:** pipeline and synthetic acceptance fixtures only. This is not a dataset-quality pass and does not authorize model training.

## Environment

- Python 3.12.3 on Ubuntu Linux sandbox.
- pytest 9.2.0; PyYAML 6.0.3; jsonschema 4.26.0; pandas 3.1.0.
- PyTorch is not installed in the local audit environment, so the existing model test suite cannot collect locally. The successful GitHub Actions repository-test job installed CPU PyTorch and ran the full suite.
- `datasets`, `huggingface-hub`, `pyarrow`, and `tokenizers` are not installed in the audit environment; no external payload was acquired. Those are declared as optional `.[data]` dependencies for an approved Kaggle T4 acquisition environment. Data acceptance tests do not require downloading datasets.
- All experiment payload paths are environment-configurable (`AYA_DATA_ROOT`, `HF_HOME`); raw/cached data are excluded from Git.

## Commands executed and observed result

| Command | Result |
|---|---|
| `python3 scripts/data/inspect_sources.py --config configs/data/source_registry.yaml` | Passed for the seven unique Hugging Face source repositories; printed immutable current commit hashes. The dated Wikimedia snapshot and pinned BFCL GitHub evaluator were explicitly skipped by the HF resolver. Registry pins were not rewritten. |
| `python3 scripts/data/verify_licenses.py --config configs/data/source_registry.yaml` | Returned per-source gate status. BFCL V4 passed only as conditional evaluation-only; eight acquisition candidates remained blocked. |
| `python3 scripts/data/download_datasets.py --config configs/data/acquisition.yaml --stage pilot` | Exit 0; reported no Stage 1 sources approved/enabled and downloaded no payload. |
| `python3 scripts/data/run_acceptance_tests.py --stage pilot` | **Passed: 30 tests in 0.17 seconds**, including invalid-record accounting and a once-per-file checksum regression. |
| `python3 scripts/data/check_ci.py --base-ref origin/main` | **Passed**; scanned 67 changed files for YAML, JSON Schema, notebook syntax, disabled acquisition sources, dataset payloads and credential material. |
| GitHub Actions run `42484521742` at `85870c1546e98b864fcc7714b2a441f0d5459072` | **Passed**: data acceptance/hygiene job and existing repository-test job. The full `python -m pytest -q` run passed **38 tests** in 3.11 seconds (one harmless PyTorch warning about missing NumPy); GitGuardian also passed. |
| `python3 -m pytest -q` (local sandbox) | Local attempt stopped at collection because `torch` is not installed (`ModuleNotFoundError`); the dependency-equipped GitHub Actions result above is the repository-wide test result. |
| `python3 scripts/data/verify_licenses.py --config configs/data/source_registry.yaml --strict` | Exit 2, expected: eight blocked acquisition sources prevent a strict acquisition approval. |
| `python3 scripts/data/validate_dataset.py --stage pilot` | Exit 1, expected: empty pilot, absent token-count manifest, and absent token-count-aware mixture are hard failures. No empty-corpus pass is claimed. |
| `python3 scripts/data/generate_report.py --stage pilot` | Executed in an isolated temporary `AYA_DATA_ROOT`; wrote a valid reproducibility summary with zero acquired sources. This is not a data-quality pass. |

Other executed checks: YAML config parsing; revision pin inspection; tokenizer/model config audit; positive canonical JSON Schema validation; streaming MediaWiki XML parser fixture; license-gate/no-download behavior; isolated no-payload CLI smoke run through sample inspection, normalization, deduplication, split generation, development-byte count, empty mixture and final report; syntax checks for scripts/configs/schemas/notebook/pyproject; data ignore/payload scan. The empty-corpus validator returned its expected failure. No actual dataset examples or payload bytes were inspected.

## Counts and acquisition result

- External datasets acquired: **0**.
- External records inspected/normalized/rejected: **0 / 0 / 0**.
- External production tokens: **0**.
- Training records and token mixture: **0 records / 0 tokens**.
- Stage 1 sources approved and included: **0**.
- Source identities researched/pinned: eight dataset families/subsets plus BFCL held-out evaluator; nine registry entries.
- Synthetic fixture tests: 30 passed (code behavior only).

The explicit acquisition, checksums, source, and split manifests record this no-download state. The licensed-source report retains blockers rather than treating public visibility or a repository license tag as permission to download/train/redistribute.

## Acceptance gate summary

- **Provenance and canonical schemas:** implementation tests pass against synthetic documents, conversations, tool trajectories, and tool-call rows; no external record passed.
- **Tool trajectory integrity:** tests cover single/multiple calls, sequential results, failure, completion/inability messaging, missing results, and distinct canonical tool-call export.
- **Deduplication and splits:** synthetic deterministic exact/normalized/near duplicate, group isolation, and reproducibility tests pass. Real source overlap was not tested.
- **License gate:** expected pass for held-out BFCL only; all eight candidate acquisitions blocked. No training source cleared.
- **Token/mixture:** explicit tokenizer identity and 50/50 token-target mechanism are implemented, but no actual mixture was built. The development UTF-8 byte tokenizer is not a production count.
- **Evaluation isolation:** BFCL source-ID exclusion is exercised with a fixture. There was no external corpus to compare to BFCL, so zero real-world contamination is not claimed.
- **Quality/PII checks:** synthetic tests pass. Heuristics flag common email, phone, US SSN, Luhn-valid card, and secret-like patterns but are incomplete; manual review remains mandatory.
- **Kaggle T4:** documented configurable paths, optional dependency installation, streaming, staged outputs and ephemeral-storage warnings. No Kaggle session or T4 acquisition run was executed.

## Blockers and decision

1. Eight sources remain `blocked_pending_review` for unresolved content rights, component licenses, subset terms, upstream provenance, or rights/namespace obligations. See `DATASET_LICENSING.md`.
2. No raw rows were acquired, so sample-level format checks, corruption/quality rates, rejection examples and source-specific cleaning have not been observed.
3. PR #2 has an unresolved `32768` model vocabulary vs `268` tokenizer-ID mismatch; stable `main`'s 32,000 setting is separate. Pin/reconcile a production tokenizer before token counts or training.
4. BFCL and the RL evaluation partitions remain excluded; no actual corpus-to-test contamination comparison occurred.
5. Wikipedia snapshot namespace ambiguity, per-page exceptions, attribution/ShareAlike plan, and XML sunset need review before a dump shard can be processed.

**Final status:** pipeline synthetic acceptance tests pass locally; PR CI has dedicated data/hygiene and repository-test jobs. External dataset pilot acceptance: **NOT RUN / NOT PASSED**. Training corpus: **NOT APPROVED for Stage 1**. No training started.
