# AYA Edge — Stage 3 SFT Dataset Design

## 1. Objective

Create high-quality supervised training examples that teach AYA how to behave as an assistant.

## 2. Proposed directory structure

data/
    sft/
        raw/
            instruction/
            conversation/
            reasoning/
            structured/
            tools/
            safety/

        processed/
            sft_train.jsonl
            sft_validation.jsonl
            sft_test.jsonl

        manifests/
            sft_sources.yaml
            sft_train_manifest.json
            sft_validation_manifest.json
            sft_test_manifest.json

## 3. Proposed dataset mixture

| Category | Initial token share |
|---|---:|
| General instruction following | 25% |
| Conversational interaction | 20% |
| Reasoning and explanations | 20% |
| Structured outputs | 15% |
| Tool use and agent patterns | 15% |
| Safety and uncertainty | 5% |

This mixture should be adjusted using evaluation results.

## 4. Conversation format

Use a consistent message representation.

Example:

{
  "messages": [
    {
      "role": "user",
      "content": "Explain what a neural network is."
    },
    {
      "role": "assistant",
      "content": "A neural network is a computational model..."
    }
  ]
}

The actual serialization must match the tokenizer and training implementation.

## 5. Multi-turn examples

Include conversations where the assistant must remember information from earlier messages.

Example:

User: My project uses Python.

Assistant: Got it.

User: What language does my project use?

Assistant: Your project uses Python.

The example is illustrative; training records should be diverse and quality-controlled.

## 6. Structured-output examples

Include:

- JSON generation.
- XML generation.
- Schema-constrained responses.
- Nested structures.
- Error correction.
- Invalid-format detection.

Validate generated targets automatically wherever possible.

## 7. Tool-use examples

Suggested conceptual format:

{
  "messages": [
    {
      "role": "user",
      "content": "Search for the latest information about a topic."
    },
    {
      "role": "assistant",
      "tool_call": {
        "name": "web_search",
        "arguments": {
          "query": "example topic"
        }
      }
    },
    {
      "role": "tool",
      "name": "web_search",
      "content": "Example search results."
    },
    {
      "role": "assistant",
      "content": "Here is the summary."
    }
  ]
}

This is a conceptual schema.

The final implementation must match the actual tool runtime and model serialization.

## 8. Data quality

Every SFT example should be checked for:

- Correctness.
- Relevance.
- Formatting.
- Instruction-response alignment.
- Duplicates.
- Contradictions.
- Malformed messages.
- Incomplete conversations.
- Unsafe or inappropriate content.

## 9. Train/validation/test separation

Keep all evaluation examples outside the training dataset.

Avoid placing near-identical demonstrations in different splits.

## 10. Dataset provenance

Record:

- Source.
- Licence.
- Dataset revision.
- Processing steps.
- Token count.
- Record count.
- Quality filters.
- Split assignment.

## 11. Teacher-generated demonstrations

If MiniCPM5-1B is used to generate demonstrations:

- Record its exact revision.
- Record prompts and decoding settings.
- Validate generated answers.
- Filter incorrect demonstrations.
- Track synthetic-data provenance.

Teacher outputs must not automatically be considered correct.
