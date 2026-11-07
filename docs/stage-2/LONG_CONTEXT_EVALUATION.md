# AYA Edge — Long-Context Evaluation

## 1. Objective

Measure whether AYA can use information distributed across longer sequences.

## 2. Evaluation lengths

Evaluate at multiple sequence lengths.

| Evaluation length | Purpose |
|---|---|
| 128 | Short-context baseline |
| 256 | Short-context retention |
| 512 | Stage 0 reference |
| 1,024 | Initial expansion |
| 2,048 | Moderate context |
| 4,096 | Extended context |
| 8,192 | Experimental long context |

Only test lengths actually supported by the implementation.

## 3. Evaluation categories

### A. Position retrieval

Place relevant information at different positions.

Test whether the model can retrieve it.

### B. Multi-section understanding

Provide documents containing several related sections.

Test whether AYA can connect information across sections.

### C. Long-document questions

Ask questions requiring information from different parts of a document.

### D. Context retention

Evaluate whether relevant information remains available later in a sequence.

### E. Distractor resistance

Include irrelevant information between relevant facts.

Evaluate whether the model can identify the correct information.

### F. Contextual summarisation

Evaluate whether the model can produce summaries that preserve important information.

## 4. Metrics

Record:

- Accuracy.
- Exact-match score where applicable.
- Retrieval accuracy.
- Position-dependent accuracy.
- Validation loss.
- Perplexity.
- Memory consumption.
- Processing speed.

## 5. Position analysis

Report performance separately for:

- Beginning of sequence.
- Early-middle.
- Middle.
- Late-middle.
- End of sequence.

Do not rely on one aggregate score.

## 6. Short-context regression

Compare against Stage 1 using identical short-context evaluation examples.

## 7. Reporting

Every evaluation must identify:

- Model checkpoint.
- Sequence length.
- Dataset version.
- Evaluation method.
- Results.
- Known limitations.

## 8. Success definition

The model must demonstrate meaningful use of information across longer sequences.

Merely accepting a longer input is insufficient.
