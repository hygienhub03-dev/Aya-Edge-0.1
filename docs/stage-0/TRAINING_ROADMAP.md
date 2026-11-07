# AYA Edge — Training Roadmap

## Stage 0 — Baseline and Reproducibility

Objective:

Establish that the architecture can execute a reliable training workflow.

Requirements:

- Deterministic model initialisation.
- Successful forward pass.
- Successful backward pass.
- Finite gradients.
- Successful optimiser update.
- Checkpoint saving.
- Checkpoint loading.
- Small overfit experiment.
- Parameter-count verification.
- GPU memory measurement.

Exit criteria:

All baseline tests must pass and be reproducible against a known Git commit.

---

## Stage 1 — Language-Model Pretraining

Initial proposed data mixture:

| Dataset category | Token allocation |
|---|---:|
| General language and knowledge | 50% |
| Instruction and agent patterns | 50% |
| Coding | 0% |
| Dedicated mathematics | 0% |

The percentages are an initial experiment, not a permanent decision.

Priorities:

- Dataset quality.
- Tokenisation consistency.
- Data deduplication.
- Stable training.
- Validation-loss monitoring.
- Expert utilisation monitoring.
- Reproducible checkpoints.

---

## Stage 2 — Mid-Training and Context Development

Objectives:

- Gradually increase context length.
- Evaluate longer sequences.
- Monitor memory consumption.
- Measure training throughput.
- Test short-context retention.
- Identify long-context regressions.

A configured maximum sequence length does not prove long-context competence.

---

## Stage 3 — Supervised Fine-Tuning

Objectives:

- Instruction following.
- Conversational behaviour.
- Structured JSON generation.
- XML generation.
- Tool-call formatting.
- Argument correctness.
- Multi-turn interaction.
- Appropriate uncertainty handling.

Evaluation must measure both formatting validity and actual task correctness.

---

## Stage 4 — Reinforcement Learning

Reinforcement learning should only begin after a reliable supervised baseline exists.

Initial priorities:

- Narrow task definitions.
- Reliable reward signals.
- Reward-hacking detection.
- Regression monitoring.
- Reproducible experiments.

---

## Stage 5 — Online and Preference Distillation

Explore teacher-generated demonstrations and preference signals.

The teacher reference currently specified in the configuration is:

openbmb/MiniCPM5-1B

Record the exact teacher revision, prompts, decoding settings, and data filtering procedures.

Teacher-generated responses must not automatically be treated as correct ground truth.

---

## Experiment Reporting

Every experiment should record:

- Git commit hash.
- Configuration.
- Dataset manifest.
- Random seed.
- Hardware.
- Software versions.
- Training steps.
- Tokens processed.
- Training loss.
- Validation loss.
- GPU memory.
- Throughput.
- Expert utilisation.
- Checkpoint identifier.
- Evaluation results.
