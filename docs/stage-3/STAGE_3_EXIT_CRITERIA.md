# AYA Edge — Stage 3 Exit Criteria

## Objective

Verify that AYA Edge has developed measurable instruction-following and assistant-oriented capabilities.

## 1. Dataset readiness

- [ ] SFT datasets prepared.
- [ ] Dataset provenance documented.
- [ ] Dataset mixture verified.
- [ ] Conversation serialization verified.
- [ ] Tokenizer compatibility verified.
- [ ] Training and evaluation splits separated.
- [ ] Data quality checks completed.

## 2. Training integrity

- [ ] Stage 2 checkpoint loads correctly.
- [ ] Assistant-target masking verified.
- [ ] Forward pass verified.
- [ ] Backward pass verified.
- [ ] Gradients remain finite.
- [ ] Optimiser updates verified.
- [ ] Checkpoint saving verified.
- [ ] Checkpoint loading verified.

## 3. Instruction following

- [ ] Direct instruction evaluation completed.
- [ ] Multi-constraint evaluation completed.
- [ ] Follow-up instruction evaluation completed.
- [ ] Ambiguity handling evaluated.
- [ ] Response consistency evaluated.

## 4. Structured outputs

- [ ] JSON parsing tested.
- [ ] JSON schema compliance tested.
- [ ] XML parsing tested.
- [ ] XML structure tested.
- [ ] Semantic correctness evaluated.

## 5. Agent behaviour

- [ ] Tool selection evaluated.
- [ ] Tool-call formatting evaluated.
- [ ] Tool argument correctness evaluated.
- [ ] Tool-result interpretation evaluated.
- [ ] Tool failure handling evaluated.
- [ ] Multi-step tool workflows evaluated.

## 6. Reasoning and safety

- [ ] Reasoning demonstrations evaluated.
- [ ] Multi-step problem solving evaluated.
- [ ] Uncertainty handling evaluated.
- [ ] Unsupported claims evaluated.
- [ ] Appropriate refusal behaviour evaluated.

## 7. Regression testing

- [ ] General language performance compared with Stage 2.
- [ ] Context performance compared with Stage 2.
- [ ] Long-context performance compared with Stage 2.
- [ ] Training regressions investigated.

## 8. Acceptance criteria

Stage 3 can be considered complete when:

1. SFT training is reproducible.
2. Held-out instruction-following performance improves.
3. Structured outputs demonstrate measurable validity.
4. Tool-call generation is evaluated independently of tool execution.
5. Conversational behaviour is evaluated.
6. Safety and uncertainty behaviour are tested.
7. Pretrained language and context performance remain acceptable.
8. The best checkpoint and complete experiment records are preserved.

## 9. Stage 4 readiness

Preserve:

- Best Stage 3 checkpoint.
- SFT dataset manifests.
- Training configuration.
- Evaluation results.
- Tool-call evaluation reports.
- Regression reports.
- Experiment logs.

Stage 4 reinforcement learning must begin from a verified supervised baseline.

Do not introduce reinforcement learning before establishing reliable supervised performance.
