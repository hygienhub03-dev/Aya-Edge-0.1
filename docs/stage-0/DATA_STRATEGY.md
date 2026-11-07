# AYA Edge — Data Strategy

## 1. Initial Data Mixture

The initial Stage 1 data strategy proposes:

| Category | Token allocation |
|---|---:|
| General language and knowledge | 50% |
| Instruction and agent patterns | 50% |
| Coding | 0% |
| Dedicated mathematics | 0% |

This is an initial hypothesis that must be evaluated.

The proportions should be calculated using token counts rather than the number of JSONL records.

## 2. Dataset Structure

Suggested structure:

data/
    manifests/
        sources.yaml
        train_manifest.json
        validation_manifest.json
        test_manifest.json

    processed/
        general.jsonl
        agent.jsonl

## 3. General Dataset

general.jsonl

Intended purpose:

- General language modelling.
- Language understanding.
- General knowledge.
- Reading comprehension.
- Natural language patterns.
- Explanatory text.

## 4. Agent Dataset

agent.jsonl

Intended purpose:

- Instruction following.
- Structured responses.
- Tool-use patterns.
- Multi-step task examples.
- Planning demonstrations.
- JSON/XML formatting.
- Conversational interaction.

## 5. Data Quality

Every dataset should undergo:

- Exact deduplication.
- Near-duplicate detection.
- Empty-record removal.
- Corruption checks.
- Language filtering.
- Quality inspection.
- Provenance tracking.
- Licence review.

Sensitive personal information should be filtered appropriately.

## 6. Evaluation Separation

Training, validation, and test datasets must remain separate.

Avoid data leakage between splits.

Where possible, use document-aware or source-aware splitting rather than blindly splitting individual records.

## 7. Teacher-Generated Data

If MiniCPM5-1B is used as a teacher, record:

- Exact model identifier.
- Model revision.
- Prompt template.
- Generation settings.
- Date of generation.
- Filtering procedure.
- Data provenance.

Teacher outputs require quality control.

## 8. Data Documentation

Every dataset version should include:

- Dataset name.
- Source.
- Number of records.
- Token count.
- Language distribution.
- Processing procedure.
- Licence information.
- Split assignment.
- Version identifier.
