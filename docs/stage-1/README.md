# AYA Edge — Stage 1: Language-Model Pretraining

## Governing specification

This stage plan is subordinate to the repository-level [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Any change to objectives, data, training, architecture, or evaluation must preserve the approved agent-first identity or document its reason, expected benefit, and evaluation method.

## Purpose

Stage 1 establishes the language foundation of AYA Edge.

The objective is to train the model on a carefully curated mixture of general language, knowledge, instructions, and agent-oriented patterns.

Stage 1 builds on the engineering baseline established in Stage 0.

## Primary objectives

- Develop general language-modelling ability.
- Improve next-token prediction.
- Learn common linguistic structures.
- Develop basic knowledge representation through training data.
- Expose the model to instruction patterns.
- Introduce structured response patterns.
- Establish a reproducible pretraining pipeline.
- Monitor Mixture-of-Experts routing behaviour.

## Initial data mixture

| Dataset | Token allocation |
|---|---:|
| General language and knowledge | 50% |
| Instructions and agent patterns | 50% |
| Coding | 0% |
| Dedicated mathematics | 0% |

The mixture is an initial hypothesis and must be evaluated.

## Training principles

1. Data quality takes priority over raw dataset size.
2. Dataset proportions are measured by tokens.
3. Validation data must remain isolated.
4. Training must be reproducible.
5. Expert utilisation must be monitored.
6. Checkpoints must be evaluated before promotion.
7. Training loss alone cannot establish model quality.

## Expected deliverables

- Versioned training datasets.
- Data manifests.
- Reproducible training configuration.
- Training and validation logs.
- Expert utilisation reports.
- Checkpoints.
- Evaluation reports.

## Important distinction

Stage 1 is foundation training.

It does not guarantee instruction-following reliability, reasoning, or tool-use competence.

Those capabilities require dedicated evaluation and further training.
