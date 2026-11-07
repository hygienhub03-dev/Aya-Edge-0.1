# AYA Edge — Stage 3 SFT Training Plan

## 1. Starting checkpoint

Use the best verified Stage 2 checkpoint.

Preserve the original checkpoint before fine-tuning.

## 2. Training objective

Train AYA using supervised examples of the desired assistant behaviour.

The exact loss implementation must respect message boundaries and assistant-target masking.

## 3. Training phases

### Phase A — Dataset smoke test

- Load a small SFT dataset.
- Validate conversation serialization.
- Verify tokenisation.
- Verify assistant-target masking.
- Run forward and backward passes.
- Check gradients.
- Save a test checkpoint.

### Phase B — General instruction tuning

Focus on:

- Following instructions.
- Answering questions.
- Respecting constraints.
- Producing clear explanations.
- Handling simple conversations.

### Phase C — Conversational tuning

Introduce:

- Multi-turn conversations.
- Context-dependent responses.
- Follow-up instructions.
- Corrections.
- Summarisation.
- Conversation consistency.

### Phase D — Structured-output tuning

Train on:

- JSON.
- XML.
- Schema compliance.
- Nested structures.
- Output validation.

### Phase E — Agent and tool-use tuning

Train on:

- Tool selection.
- Tool-call formatting.
- Argument generation.
- Tool-result interpretation.
- Error handling.
- Multi-step tool workflows.

### Phase F — Safety and uncertainty tuning

Introduce:

- Honest uncertainty.
- Appropriate refusals.
- Ambiguity handling.
- Unsupported-request handling.
- Avoidance of fabricated tool results.

## 4. Loss masking

Ensure that the training objective is correctly aligned with the conversation format.

Important considerations:

- User messages should not accidentally become assistant targets.
- System instructions should be handled consistently.
- Tool responses should use the intended training mask.
- Assistant responses should receive the intended loss.
- Padding tokens must not contribute incorrectly to the loss.
- End-of-turn and tool-call tokens must be handled correctly.

## 5. Hyperparameter experiments

Do not assume Stage 1 hyperparameters remain optimal for SFT.

Test controlled variations of:

- Learning rate.
- Batch size.
- Gradient accumulation.
- Weight decay.
- Warm-up.
- Scheduler.
- Epoch count.
- Sequence length.

Record every experiment.

## 6. Monitoring

Track:

- SFT training loss.
- SFT validation loss.
- Instruction-following accuracy.
- Structured-output validity.
- Tool-call accuracy.
- Gradient norms.
- Learning rate.
- Tokens processed.
- GPU memory.
- Training throughput.

## 7. Preventing catastrophic forgetting

Evaluate the SFT checkpoint against Stage 2 language-modelling benchmarks.

Investigate any significant regression.

Possible approaches include:

- Lower learning rates.
- Mixed general-language replay data.
- Controlled SFT mixture adjustments.
- Checkpoint selection.
- Carefully designed multi-task training.

## 8. Checkpoint policy

Save:

- Model weights.
- Optimiser state.
- Scheduler state.
- Training step.
- Configuration.
- Dataset version.
- Random state.

Keep the best validation checkpoint.

## 9. Important rule

A lower SFT loss does not necessarily mean better assistant behaviour.

Always evaluate the actual tasks the model is expected to perform.
