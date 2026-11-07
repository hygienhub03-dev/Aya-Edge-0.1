# AYA Edge — Stage 3: Supervised Fine-Tuning

## Purpose

Stage 3 transforms the pretrained AYA Edge model into an instruction-following assistant through supervised fine-tuning.

The starting checkpoint must come from the verified Stage 2 model.

## Primary objectives

- Instruction following.
- Conversational interaction.
- Multi-turn context handling.
- Structured JSON generation.
- XML generation.
- Reasoning demonstrations.
- Planning and task decomposition.
- Tool-call generation.
- Tool-result interpretation.
- Consistent response formatting.

## Core training philosophy

AYA should learn to understand what the user wants, determine the appropriate response format, and produce useful and accurate outputs.

Training should prioritise quality over dataset size.

## Proposed SFT categories

| Category | Initial allocation |
|---|---:|
| General instruction following | 25% |
| Conversational interaction | 20% |
| Reasoning and explanations | 20% |
| Structured outputs | 15% |
| Tool use and agent patterns | 15% |
| Safety and uncertainty | 5% |

These proportions are proposed experimental allocations, not established optimal values.

Calculate mixture proportions using tokens.

## Expected deliverables

- Curated SFT datasets.
- Dataset manifests.
- Conversation formatting specification.
- Tool-call schema.
- SFT training configuration.
- Fine-tuned checkpoints.
- Instruction-following evaluation.
- Structured-output evaluation.
- Tool-use evaluation.
- Stage 3 experiment reports.

## Important distinction

SFT can teach a model to produce tool-call instructions.

It does not automatically give the model access to external tools or guarantee successful execution.

Actual tool execution requires an external runtime.
