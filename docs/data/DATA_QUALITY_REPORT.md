# Pilot data-quality and sample-inspection report

**Status date: 2026-11-07. No external data payload was downloaded or inspected.** Source cards, repository metadata, rendered previews, schema summaries, and license documents were reviewed; those metadata previews are not a substitute for sampling raw records. Therefore this report contains no observed corpus-quality percentages and makes no claim that any requested external dataset passed acceptance.

## Actual data counts

| Metric | Observed |
|---|---:|
| External raw records acquired | 0 |
| External raw records inspected | 0 |
| Canonical normalized external records | 0 |
| Rejected external records | 0 |
| Token-counted external records | 0 |
| Production model tokens | 0 |
| Mixture records/tokens | 0 |
| Stage 1 sources approved and included | 0 |
| Synthetic/unit-test cases executed | 27 tests passed |

The `verify_licenses` manifest reports BFCL V4 conditionally approved for **evaluation-only** and eight acquisition candidates blocked pending further review. Since there is no source approval for Stage 1, no external corpus can be reported as accepted. The validator intentionally fails an empty pilot corpus, missing token counts, or missing mixture report.

## Implemented quality checks

Normalization retains source ID/revision, original record ID, source split, acquisition-relative source file, raw-file checksum, transformation version, declared license ID, and original raw record for accepted data. The canonical schemas distinguish documents, conversations, ordered tool-use trajectories, and complete tool-call/result rows. Textual documents are never relabeled as conversations. Tool-call arguments/results are preserved in trajectory order and also exported as canonical tool-call records; a source call ID is marked as source-provided, while a deterministic correlation ID is marked as derived. Missing tool results, broken call/result matching, and truncated tool trajectories are rejected, not fabricated. Tool errors remain explicit `status: error` results.

Configurable filters reject empty/extremely short, low-information, overlong, unsupported image/audio/video, and malformed source-constrained structured-output records. Structured JSON is validated against a source-provided schema when available. The default limits are 12 characters minimum, 3 whitespace-delimited words minimum, and 200,000 characters maximum. Exact normalized duplicates are removed across sources. A deterministic five-word-shingle detector flags near duplicates at Jaccard similarity 0.98, retains them for review, and groups them together to prevent cross-split leakage. Repeated numeric-template groups at or above the configured threshold are reported. Splits are deterministic, hash-group aware, and include stable group/sample assignments in a manifest.

A high-confidence sensitive-data filter flags common emails, phone numbers, US SSNs, Luhn-valid payment-card-like strings, and common exposed API-secret patterns. Rejected records store a reason and SHA-256 digest rather than the raw payload by default. This is a privacy aid, **not** a complete PII detector: names, postal addresses, private facts, contextual sensitivity, copyrighted quotations, and false positives require manual review. Accepted rows still contain their raw source record; keep raw/processed outputs in access-controlled storage and do not commit them.

## Synthetic acceptance fixtures

The 27 repository tests exercise source/revision uniqueness, conditional-license gates, stable IDs, document normalization and rejection, sequential tool calls, tool errors, missing results, canonical schema validation, exact and near-duplicate handling, deterministic grouped splits, explicit tokenizer accounting, token mixture reporting, protected BFCL holdout rejection, corrupt JSONL detection, end-to-end reproducibility, privacy-minimized rejection logs, and a synthetic bzip2 MediaWiki XML fixture. These tests establish pipeline behavior against constructed fixtures, not quality characteristics of the unacquired source datasets.

## Outstanding checks before any corpus can pass

1. Resolve the source-specific blockers in `DATASET_LICENSING.md`; no download may start before that gate.
2. Inspect at least 100 rows per approved subset, using a documented deterministic sample and a bounded output. Verify schemas and field retention, raw-source/license metadata, language/quality distributions, PII and sensitive content, broken/truncated dialogues, malformed structured JSON, tool/result order, failure/recovery/correction examples, low-information items, repeated templates, exact/near duplicates, and source-specific contamination.
3. Run the canonicalizer and reviewer on those inspected samples; record accepted/rejected counts and rejection reasons by source and category. No dataset card assertion should be substituted for a raw-row check.
4. Reconcile the 0.2 model vocabulary against its tokenizer; pin a production tokenizer artifact and checksum before token totals or the Stage 1 mixture can be accepted.
5. Build train/validation/test groups after global deduplication; confirm no group leakage and keep the full BFCL V4 evaluation categories, prompts, schemas, expected answers, and derived material out of training.
6. Report actual target/observed token shares, per-source contributions, and token-count deviations. The 50/50 objective is by tokens, not samples.

**Current outcome:** code-level acceptance fixtures pass; external pilot acceptance is **NOT RUN / NOT PASSED** because no source payload was approved or acquired. No training readiness is implied.
