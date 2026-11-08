# AYA Edge Dataset Strategy

**Decision:** Reject `aya_edge_mini_dataset_v1` (120 examples) as a production corpus. Retain it only for smoke tests, checkpoint wiring, routing telemetry, and bounded overfit experiments.

## Evidence and decision

The mini dataset has 80 train, 20 validation, and 20 test records; 60 general-language and 60 agentic records; short examples; and fictional tools with simulated results. Its QC report finds valid JSON, unique IDs, no exact split duplicates, and no reported near-duplicates. Those are useful integrity checks, but 120 examples cannot support generalization, broad knowledge, tool grounding, or production pretraining. Its “production-ready” wording is limited to the intended overfitting experiment and must not be broadened.

## Required source register

Every source must have a stable identifier, URL or archive location, revision/commit, retrieval date, copyright/licence and terms snapshot, allowed uses, attribution requirements, transformation record, language/domain metadata, and removal/contact procedure. Repository code, model weights, benchmark data, hosted API content, and teacher outputs must be audited separately; a permissive code licence does not automatically license data returned by an API.

MiniCPM5-1B remains the documented teacher reference, but exact checkpoint revision, output-collection terms, privacy, and competing-model training restrictions must be verified before use. Teacher outputs are candidates requiring filtering, not ground truth.

## Processing contract

1. Inventory and license-check raw sources before downloading at scale.
2. Preserve immutable raw manifests and cryptographic hashes.
3. Normalize encoding and schema; remove corruption, empty records, unsupported languages, and sensitive personal data.
4. Deduplicate exact normalized text and then detect near duplicates using document/semantic fingerprints.
5. Inspect quality, toxicity, privacy, contamination, and tool-call validity.
6. Split by document/source/conversation group, not random rows, after dedupe and before tokenization.
7. Tokenize with the exact production tokenizer revision and record special-token behavior.
8. Pack sequences without crossing evaluation boundaries; emit token-counted manifests.

## Mixture and splits

The official Stage 1 starting hypothesis is 50% general language/knowledge and 50% instruction/agent patterns **by tokens**, not records. Produce separate, immutable train/validation/test manifests with source and document group IDs. Keep a contamination/benchmark holdout and a hidden agent-task set. Validation and test must never be used for training, teacher filtering, prompt tuning, or route policy selection.

Recommended acceptance metadata includes raw/accepted/rejected counts, exact and near-duplicate counts, token counts, source distribution, language distribution, hash, tokenizer revision, split seed, and processing code revision.

## Licensing caveats

Do not infer redistribution or model-training rights from a source URL, repository licence, or benchmark citation alone. Review third-party notices, terms of service, personal-data obligations, and jurisdictional restrictions. For ToolBench/RapidAPI-like material, repository terms do not necessarily cover API-derived responses. For public benchmarks, use them as evaluation data only unless their specific terms permit training.

## Acceptance gate

No corpus is production-eligible until legal/provenance review, dedupe and contamination reports, source-aware splits, token-counted manifests, tokenizer alignment, and reproducible regeneration all pass. The 120-example mini dataset fails the scale and provenance requirements even though its local QC checks pass.

## References

- [`aya_edge_mini_dataset_v1/README.md`](../../aya_edge_mini_dataset_v1/README.md)
- [`aya_edge_mini_dataset_v1/dataset_manifest.json`](../../aya_edge_mini_dataset_v1/dataset_manifest.json)
- [`aya_edge_mini_dataset_v1/QC_REPORT.md`](../../aya_edge_mini_dataset_v1/QC_REPORT.md)
- [`docs/stage-1/DATA_PIPELINE.md`](../stage-1/DATA_PIPELINE.md)
- [`docs/stage-0/DATA_STRATEGY.md`](../stage-0/DATA_STRATEGY.md)
- [`MODEL_SPECIFICATION.md`](../../MODEL_SPECIFICATION.md)
