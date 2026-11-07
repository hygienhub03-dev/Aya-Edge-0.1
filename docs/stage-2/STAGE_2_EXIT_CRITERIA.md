# AYA Edge — Stage 2 Exit Criteria

## Objective

Verify that AYA Edge has improved its ability to process longer sequences.

## 1. Checkpoint integrity

- [ ] Stage 1 checkpoint preserved.
- [ ] Stage 2 starting checkpoint documented.
- [ ] Model loads successfully.
- [ ] Tokenizer compatibility verified.
- [ ] Checkpoint round-trip tested.

## 2. Context support

- [ ] Positional encoding audited.
- [ ] Attention implementation verified.
- [ ] Supported context lengths documented.
- [ ] Longer sequences tested.
- [ ] Position-dependent evaluation completed.

## 3. Training stability

- [ ] Loss remains finite.
- [ ] Gradients remain finite.
- [ ] Optimiser updates verified.
- [ ] Long-sequence training stable.
- [ ] Short-context regression measured.

## 4. Evaluation

- [ ] Position retrieval evaluated.
- [ ] Long-document understanding evaluated.
- [ ] Context retention evaluated.
- [ ] Distractor resistance evaluated.
- [ ] Summarisation evaluated.
- [ ] Results compared with Stage 1.

## 5. Efficiency

- [ ] Peak memory recorded.
- [ ] Throughput recorded.
- [ ] Training time recorded.
- [ ] Attention overhead investigated.
- [ ] MoE overhead investigated.

## 6. Acceptance criteria

Stage 2 can be considered complete when:

1. Longer context lengths are supported by the actual implementation.
2. Long-context evaluations demonstrate measurable improvement.
3. Short-context performance remains acceptable.
4. Memory and throughput are documented.
5. Checkpoint integrity is verified.
6. Training is reproducible.

## 7. Stage 3 readiness

Preserve:

- Best Stage 2 checkpoint.
- Tokenizer.
- Training configuration.
- Context evaluation results.
- Memory benchmarks.
- Experiment logs.

Stage 3 must begin from a verified Stage 2 checkpoint.
