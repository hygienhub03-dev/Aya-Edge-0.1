# AYA Edge — Stage 1 Evaluation Plan

## 1. Purpose

Determine whether pretraining improves AYA Edge's language-modelling ability.

## 2. Core metrics

### Language modelling

- Training loss.
- Validation loss.
- Perplexity.
- Loss by dataset category.

Perplexity should be calculated consistently from the relevant token-level loss.

### Training efficiency

- Tokens per second.
- Total tokens processed.
- GPU memory consumption.
- Training time.
- Checkpoint size.

### MoE behaviour

For every layer, record:

- Expert routing counts.
- Expert utilisation percentages.
- Dead experts.
- Routing imbalance.
- Auxiliary loss.

## 3. Evaluation datasets

Maintain separate held-out datasets for:

- General language.
- Knowledge-oriented text.
- Instruction-pattern text.

Evaluation data must not be included in training.

## 4. Generalisation checks

Evaluate whether improvements transfer to unseen examples.

Inspect:

- Unseen writing styles.
- Unseen instructions.
- New document topics.
- Different sequence lengths.
- Previously unseen combinations of concepts.

## 5. Memorisation checks

Investigate whether the model is memorising training examples.

Use:

- Training/validation loss comparisons.
- Duplicate detection.
- Exact-match checks on selected examples.
- Unseen document evaluations.

## 6. Expert evaluation

Expert utilisation should be compared across:

- Layers.
- Training checkpoints.
- Dataset categories.
- Different random seeds, where practical.

Balanced routing does not automatically imply expert specialisation.

## 7. Reporting

Every evaluation report must include:

- Model checkpoint.
- Git commit.
- Dataset version.
- Evaluation configuration.
- Metrics.
- Comparison against previous checkpoints.
- Known limitations.

## 8. Stage 1 outcome

The outcome must be based on held-out performance and reproducibility.

Training loss reduction alone is not sufficient evidence of successful pretraining.
