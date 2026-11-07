# AYA Edge — Stage 2 Context Strategy

## 1. Objective

Develop the ability to process longer sequences while maintaining useful language-modelling behaviour.

## 2. Starting point

Use the best validated Stage 1 checkpoint.

The reported Stage 0 maximum sequence length was 512 tokens.

The actual Stage 1 training context length must be verified before Stage 2 begins.

## 3. Gradual context expansion

Use progressive context-length experiments.

Suggested experimental sequence:

| Experiment | Sequence length |
|---|---:|
| Baseline | 512 |
| Context experiment 1 | 1,024 |
| Context experiment 2 | 2,048 |
| Context experiment 3 | 4,096 |
| Context experiment 4 | 8,192 |

These are proposed targets, not guaranteed supported lengths.

Do not automatically progress to the next length.

Each increase must pass memory, stability, and evaluation checks.

## 4. Context training strategy

Introduce longer sequences gradually.

Potential approaches:

- Sequence-length curriculum.
- Variable-length training batches.
- Long-document training examples.
- Packed document sequences.
- Position-aware evaluation.
- Long-range dependency examples.

The selected approach must be compatible with the actual attention implementation.

## 5. Positional encoding audit

Before increasing sequence length, inspect:

- Positional embedding implementation.
- RoPE configuration, if applicable.
- Frequency scaling.
- Maximum supported position.
- Position interpolation or scaling.
- Checkpoint compatibility.

Do not assume that increasing a YAML configuration value extends positional support correctly.

## 6. Context quality objectives

Evaluate whether AYA can:

- Identify information near the beginning of a sequence.
- Use information near the end.
- Connect information from different positions.
- Answer questions about long documents.
- Maintain relevant conversational context.
- Avoid confusing unrelated sections.

## 7. Short-context preservation

Long-context training must not unnecessarily damage short-context behaviour.

Continue evaluating 128-, 256-, and 512-token examples.

## 8. Success definition

Stage 2 succeeds when longer sequences produce measurable improvements on held-out context evaluations without unacceptable degradation in short-context performance or training stability.
