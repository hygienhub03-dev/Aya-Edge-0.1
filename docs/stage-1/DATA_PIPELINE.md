# AYA Edge — Stage 1 Data Pipeline

## 1. Objective

Create a reproducible pipeline that transforms raw datasets into validated training-ready token sequences.

## 2. Proposed directory structure

data/
    raw/
        general/
        agent/

    processed/
        general.jsonl
        agent.jsonl

    tokenized/
        general/
        agent/

    manifests/
        sources.yaml
        train_manifest.json
        validation_manifest.json
        test_manifest.json

## 3. General dataset

File: general.jsonl

Purpose:

- General language modelling.
- Knowledge-oriented text.
- Reading comprehension.
- Explanations.
- Natural language structures.

## 4. Agent dataset

File: agent.jsonl

Purpose:

- Instructions.
- Conversations.
- Task descriptions.
- Planning patterns.
- Structured responses.
- Tool-use examples.
- JSON and XML formatting.

## 5. Processing pipeline

Each dataset should pass through:

1. Source collection.
2. Provenance and licence verification.
3. Text extraction.
4. Encoding normalisation.
5. Empty-record filtering.
6. Corruption filtering.
7. Language filtering.
8. Exact deduplication.
9. Near-duplicate detection.
10. Quality inspection.
11. Dataset splitting.
12. Tokenisation.
13. Sequence packing.
14. Manifest generation.

## 6. Token accounting

The 50/50 mixture must be calculated using actual token counts.

Do not use record counts to determine the mixture.

For each source, record:

- Raw records.
- Accepted records.
- Rejected records.
- Token count.
- Language distribution.
- Source identifier.
- Dataset version.

## 7. Data leakage prevention

Validation and test examples must not be included in training.

Where possible, split by document or source rather than individual rows.

Deduplication should occur before final split assignment.

## 8. Tokenizer consistency

The tokenizer used for dataset preparation must match the tokenizer used by the model.

Record:

- Tokenizer revision.
- Vocabulary size.
- Special token IDs.
- Padding behaviour.
- End-of-sequence behaviour.
- Truncation behaviour.

The current reported vocabulary size is 32,000, but the actual tokenizer must be verified.

## 9. Reproducibility

Every processed dataset must have:

- A version identifier.
- A source manifest.
- Processing configuration.
- Dataset hashes.
- Token counts.
- Creation date.
