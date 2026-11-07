# AYA Edge tokenizer specification

## Scope

AYA Edge uses an offline, dependency-light tokenizer for general conversation,
tool calls and results, JSON/XML, and multi-turn context. The implementation is
`aya_edge.tokenizer.ByteLevelTokenizer` and requires only the Python standard
library. It does not download data or learn vocabulary entries.

## Vocabulary and determinism

The vocabulary is fixed at **268 tokens**:

* IDs `0..255` are the literal byte values. Text is encoded as UTF-8 bytes,
  preserving punctuation, whitespace, JSON, XML, non-ASCII text, and arbitrary
  Unicode without an unknown-character path.
* IDs `256..267` are special tokens in this exact order: `pad`, `bos`, `eos`,
  `unk`, `turn_start`, `turn_end`, `system`, `user`, `assistant`, `tool`,
  `tool_call`, and `tool_result`.

Their surface strings are respectively:

```text
<|pad|> <|bos|> <|eos|> <|unk|> <|turn_start|> <|turn_end|>
<|system|> <|user|> <|assistant|> <|tool|> <|tool_call|> <|tool_result|>
```

This layout is part of the model interface. The implementation rejects changed
vocabulary size, changed special-token names/order, duplicate markers, changed
special IDs, or unsupported normalization rather than silently changing IDs.
There is no learned vocabulary and no merge table.

## Encoding and decoding

`encode(text)` returns a list of integer IDs. UTF-8 bytes map directly to IDs
`0..255`. A recognized special marker maps to its one special ID; the longest
marker is preferred. `add_bos` and `add_eos` optionally add boundary tokens.
No normalization is applied, so whitespace and case are retained.

`decode(ids)` reconstructs UTF-8 text and special markers. It raises for invalid
IDs and uses strict UTF-8 decoding by default; callers can pass `errors="replace"`
for inspection of intentionally incomplete byte sequences. Passing
`skip_special_tokens=True` removes special markers from the decoded text.

## Multi-turn conversations

`encode_messages` accepts a sequence of exactly shaped mappings:

```python
[
    {"role": "system", "content": "You are AYA."},
    {"role": "user", "content": "Return JSON."},
    {"role": "assistant", "content": "{\"ok\":true}"},
]
```

Each message is serialized as:

```text
<|turn_start|><|role|>content<|turn_end|>
```

The sequence defaults to a leading `<|bos|>` and trailing `<|eos|>`. Supported
roles include `system`, `user`, `assistant`, `tool`, `tool_call`, and
`tool_result`; JSON/XML content remains ordinary UTF-8 byte content. The decoder
provides a faithful text view and does not infer or rewrite message structure.

## Persistence

`save(path)` writes a readable JSON document containing format/version metadata,
the fixed byte vocabulary size, encoding and normalization policy, special-token
strings, and computed special-token IDs. `ByteLevelTokenizer.load(path)` validates
the complete contract before constructing an instance. A saved file is therefore
portable and auditable, but it is configuration—not a mutable vocabulary cache.

## Preparation script

Run from the repository root:

```bash
python scripts/prepare_tokenizer.py --output artifacts/aya_edge_tokenizer.json
python scripts/prepare_tokenizer.py \
  --input aya_edge_mini_dataset_v1/train.jsonl \
  --output artifacts/aya_edge_tokenizer.json
```

The optional local UTF-8 text/JSONL input is counted for a small preparation
report only. It cannot add tokens or alter IDs. Omitting it is the normal path;
no network access is required.
