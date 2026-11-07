# AYA Edge — Stage 1 Pretraining Objectives

## 1. Primary objective

Train AYA Edge to predict the next token across a diverse and carefully selected dataset.

The model should develop a useful language representation before more specialised training begins.

## 2. General language and knowledge

Target allocation: 50% of training tokens.

Intended data characteristics:

- Natural language.
- Explanatory writing.
- Educational material.
- General informational text.
- Reading comprehension material.
- High-quality conversational text.
- Diverse writing styles.
- Properly licensed reference material.

The purpose is to develop broad language modelling rather than memorisation of a narrow dataset.

## 3. Instruction and agent patterns

Target allocation: 50% of training tokens.

Intended data characteristics:

- User instructions.
- Question-and-answer examples.
- Multi-turn conversations.
- Structured responses.
- Task descriptions.
- Planning examples.
- Tool-use demonstrations.
- JSON and XML examples.
- Context-dependent instructions.

The purpose is to expose the model to instruction-oriented language patterns.

This does not mean that the model will automatically become a reliable agent.

## 4. Coding and mathematics

Initial allocation:

- Dedicated coding data: 0%.
- Dedicated mathematics data: 0%.

These categories are intentionally excluded from the initial mixture.

They can be introduced in later experiments if evaluation identifies a need.

General language data may naturally contain references to technical or mathematical topics.

## 5. Learning objectives

Monitor:

- Training language-model loss.
- Validation language-model loss.
- Perplexity.
- Token throughput.
- General language evaluation.
- Instruction-pattern evaluation.
- Dataset memorisation.
- Expert utilisation.
- Training stability.

## 6. Success definition

Stage 1 succeeds when AYA demonstrates reproducible improvement on held-out language-modelling evaluations without unacceptable degradation in stability, generality, or expert utilisation.

A decreasing training loss alone is insufficient.
