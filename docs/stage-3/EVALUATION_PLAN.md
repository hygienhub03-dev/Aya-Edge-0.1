# AYA Edge — Stage 3 Evaluation Plan

## 1. Objective

Determine whether supervised fine-tuning improves AYA Edge's assistant behaviour.

## 2. Instruction-following evaluation

Test:

- Direct instructions.
- Multiple constraints.
- Formatting requirements.
- Short answers.
- Detailed explanations.
- Ambiguous requests.
- Follow-up instructions.

Measure task completion and constraint adherence.

## 3. Conversational evaluation

Test:

- Multi-turn conversations.
- Context retention.
- Follow-up questions.
- Corrections.
- Conversation consistency.
- Summarisation.

## 4. Reasoning evaluation

Use held-out tasks involving:

- Logical reasoning.
- Comparisons.
- Planning.
- Multi-step problem solving.
- Error detection.
- Explanations.

Evaluate answer correctness rather than response length.

## 5. Structured-output evaluation

### JSON

Measure:

- Parse success rate.
- Required-field accuracy.
- Schema compliance.
- Value correctness.
- Nested structure correctness.

### XML

Measure:

- Parse success rate.
- Element correctness.
- Attribute correctness.
- Schema compliance.
- Content correctness.

## 6. Tool-use evaluation

Evaluate:

- Tool selection.
- Tool-call syntax.
- Argument correctness.
- Tool execution success.
- Tool-result interpretation.
- Recovery from tool failures.
- Multi-step tool workflows.

Separate model-generated tool calls from runtime execution results.

## 7. Safety evaluation

Test:

- Uncertainty handling.
- Unsupported requests.
- Ambiguous instructions.
- Fabricated claims.
- Unsafe requests.
- Appropriate refusals.

## 8. Regression evaluation

Compare Stage 3 against Stage 2.

Monitor:

- General language-modelling loss.
- Context retention.
- Response quality.
- Structured output.
- Long-context performance.

## 9. Evaluation report

Every report must include:

- Checkpoint.
- Git commit.
- Dataset version.
- Evaluation prompts.
- Scoring methodology.
- Results.
- Failure examples.
- Comparison with previous checkpoints.

## 10. Success definition

Stage 3 succeeds when AYA demonstrates measurable improvement in assistant-oriented tasks without unacceptable regression in its pretrained language and context capabilities.
