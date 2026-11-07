# AYA Edge — Stage 2 Mid-Training Plan

## 1. Objective

Continue training the Stage 1 model to improve contextual processing and sequence-length capabilities.

## 2. Starting checkpoint

Required:

- Verified Stage 1 checkpoint.
- Original model configuration.
- Tokenizer version.
- Optimiser and scheduler history, where available.
- Dataset manifests.
- Evaluation baseline.

## 3. Training data

Potential data categories:

- Long-form educational material.
- Books and properly licensed documents.
- Technical explanations.
- Multi-section documents.
- Long conversational exchanges.
- Document summarisation examples.
- Cross-document information examples.

Maintain source provenance and licensing records.

## 4. Training curriculum

### Phase A — Baseline continuation

Continue from the Stage 1 checkpoint using the verified baseline sequence length.

### Phase B — Moderate context expansion

Introduce longer sequences while monitoring training stability.

### Phase C — Long-sequence training

Increase sequence length only after the previous phase passes evaluation.

### Phase D — Mixed-context training

Combine short and long examples to preserve performance across different input sizes.

## 5. Optimisation considerations

Longer sequences increase memory and computational requirements.

Investigate:

- Gradient accumulation.
- Micro-batch size.
- Activation checkpointing.
- Mixed precision.
- Efficient attention implementations.
- Sequence packing.
- Padding efficiency.

These optimisations must preserve numerical correctness.

## 6. Training monitoring

Track:

- Training loss.
- Validation loss.
- Loss by sequence length.
- Tokens processed.
- Tokens per second.
- Peak GPU memory.
- Gradient norm.
- Attention stability.
- Expert utilisation.

## 7. Checkpoint policy

Save checkpoints after meaningful context-length milestones.

Never overwrite the best verified Stage 1 checkpoint.

Maintain separate records for each context-length experiment.

## 8. Failure conditions

Investigate or stop training if:

- Loss becomes non-finite.
- Long-sequence loss diverges.
- Short-context performance deteriorates significantly.
- Memory usage becomes unstable.
- Position encoding behaves incorrectly.
- Checkpoint compatibility fails.
