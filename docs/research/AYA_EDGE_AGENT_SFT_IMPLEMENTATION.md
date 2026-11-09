# AYA Edge agent SFT implementation

The repository now contains a native PyTorch agent-SFT path for the custom AYA Edge causal language model. It does not use TRL directly because AYA Edge is not a Transformers `PreTrainedModel`; it adopts the same data and masking semantics locally.

## Training contract

Each JSONL record contains `messages` and `tools`. Tool declarations are JSON Schema objects. Assistant tool calls must reference a declared function, contain object-valued arguments, and pass JSON Schema validation. Every tool result must match a pending call by ID or function name, and every trajectory must finish with an assistant response. This follows the Transformers and TRL dataset contract for tool-use SFT, where tool calls live in assistant `tool_calls`, tool outputs use the `tool` role, and the `tools` column contains JSON schemas. [1] [2]

The serializer uses AYA Edge’s locked production control tokens: `<|im_start|>`, `<|im_sep|>`, `<|im_end|>`, `<|tool_call|>`, and `<|tool_result|>`. It emits canonical JSON with sorted keys and compact separators so equivalent calls have stable training text. The preflight rejects over-length examples instead of silently truncating a tool call or its result.

Labels are aligned one-to-one with `input_ids`. System, user, and tool-result tokens receive `-100`; assistant content, assistant tool calls, and the assistant end marker retain their token IDs. AYA Edge’s existing causal loss shifts logits and labels internally and already ignores `-100`, so this produces assistant-only loss without changing the model. This is equivalent in intent to TRL’s `assistant_only_loss=True`, which requires a chat template capable of producing assistant-generation masks. [1]

## Commands

Install the repository with the data and development extras:

```bash
python -m pip install -e ".[data,dev]"
```

Prepare and approve the bounded UltraData pilot using the existing gated command:

```bash
python scripts/prepare_ultradata_agent.py \
  --output-dir datasets/ultradata_agent_v1 \
  --accept-upstream-terms \
  --accept-residual-upstream-rights-risk
```

Prepare the approved language corpus and production tokenizer using the existing license-gated commands. The tokenizer command requires a real implementation commit and approval record; it refuses placeholder metadata:

```bash
python scripts/prepare_language_corpus.py \
  --output-dir datasets/language_v1 \
  --accept-commoncrawl-terms \
  --accept-residual-page-rights-risk
python scripts/train_bpe_tokenizer.py \
  --input datasets/language_v1/train.jsonl \
  --output artifacts/aya_edge_tokenizer.json \
  --manifest-output artifacts/aya_edge_tokenizer.manifest.json \
  --corpus-manifest datasets/language_v1/manifest.json \
  --implementation-commit "$(git rev-parse HEAD)" \
  --approval-record datasets/language_v1/tokenizer_approval.json
```

Run the Kaggle-equivalent preflight before training:

```bash
python scripts/kaggle_preflight.py \
  --config configs/aya_edge_0_2_language.yaml \
  --tokenizer artifacts/aya_edge_tokenizer.json \
  --train-jsonl datasets/ultradata_agent_v1/train.jsonl \
  --validation-jsonl datasets/ultradata_agent_v1/validation.jsonl \
  --manifest datasets/ultradata_agent_v1/manifest.json \
  --output artifacts/agent_sft_preflight.json
```

Start a bounded first experiment only after preflight passes:

```bash
python scripts/train_agent_sft.py \
  --config configs/aya_edge_0_2_language.yaml \
  --tokenizer artifacts/aya_edge_tokenizer.json \
  --input-jsonl datasets/ultradata_agent_v1/train.jsonl \
  --validation-jsonl datasets/ultradata_agent_v1/validation.jsonl \
  --output checkpoints/aya_edge_agent_sft_step_100.pt \
  --max-steps 100
```

## Validation scope

`tests/test_agent_sft.py` covers declared-tool lookup, JSON Schema argument validation, unmatched result rejection, assistant-only masking, and length rejection. The full project test suite remains the required gate in an environment with the project dependencies installed.

## References

[1]: https://huggingface.co/docs/trl/en/sft_trainer "TRL SFTTrainer documentation"
[2]: https://huggingface.co/docs/trl/en/dataset_formats "TRL dataset formats and tool calling"
[3]: https://huggingface.co/docs/transformers/en/chat_templating "Transformers chat templates and training masks"
[4]: https://huggingface.co/docs/transformers/en/chat_extras "Transformers tool-use and JSON schema documentation"
[5]: https://json-schema.org/learn/getting-started-step-by-step "JSON Schema validation guide"
