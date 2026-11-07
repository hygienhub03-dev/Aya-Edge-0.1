# AYA Edge — Stage 1 Exit Criteria

## Objective

Determine whether AYA Edge has completed a reproducible pretraining phase.

## 1. Data readiness

- [ ] General dataset prepared.
- [ ] Agent dataset prepared.
- [ ] Token counts verified.
- [ ] Initial mixture measured.
- [ ] Dataset provenance documented.
- [ ] Deduplication completed.
- [ ] Validation data isolated.
- [ ] Test data isolated.
- [ ] Dataset manifests generated.

## 2. Training integrity

- [ ] Training is reproducible.
- [ ] Loss remains finite.
- [ ] Gradients remain finite.
- [ ] Optimiser updates are verified.
- [ ] Checkpoints save successfully.
- [ ] Checkpoints reload successfully.
- [ ] Training configuration is versioned.

## 3. Model evaluation

- [ ] Validation loss measured.
- [ ] Perplexity measured.
- [ ] General language evaluation completed.
- [ ] Instruction-pattern evaluation completed.
- [ ] Memorisation checks completed.
- [ ] Expert utilisation evaluated.

## 4. Performance

- [ ] Tokens per second recorded.
- [ ] Peak GPU memory recorded.
- [ ] Training duration recorded.
- [ ] Checkpoint sizes recorded.

## 5. Acceptance criteria

Stage 1 can be considered complete when:

1. The training pipeline is reproducible.
2. Held-out language-modelling performance demonstrates measurable improvement over the established baseline.
3. No unexplained training instability remains.
4. Data provenance and dataset splits are documented.
5. Checkpoint integrity is verified.
6. Expert routing behaviour has been investigated.

## 6. Stage 2 readiness

Before moving to Stage 2, preserve:

- The best Stage 1 checkpoint.
- The complete training configuration.
- Dataset manifests.
- Evaluation results.
- Expert utilisation reports.
- Experiment logs.

Stage 2 must begin from a verified checkpoint.
