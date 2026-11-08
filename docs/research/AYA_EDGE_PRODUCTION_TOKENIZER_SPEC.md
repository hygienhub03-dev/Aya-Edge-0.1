# AYA Edge production tokenizer specification

**Status:** architecture decision and implementation scaffold only. No production tokenizer artifact or checksum is claimed.

## Decision

AYA Edge 0.2 will use a custom byte-level BPE tokenizer with exactly **32,768 vocabulary entries**. The existing 268-token UTF-8 byte tokenizer remains the deterministic development adapter for plumbing tests. It must not be used as evidence of production token efficiency or as the tokenizer for real pretraining.

Byte-level BPE preserves arbitrary Unicode and structured text while learning shorter subword units for natural language, code-like syntax, JSON, XML, tool names, and tool results. The BPE implementation is wrapped through the optional `tokenizers` dependency; the repository does not silently fall back to the 268-token adapter when production mode is requested.

## Locked vocabulary layout (final)

```
32,768 total IDs
├── 0–15:     16 reserved special tokens          (fixed contract)
├── 16–271:   256 byte-alphabet fallback tokens   (ByteLevel initial alphabet)
└── 272–32,767: 32,496 learned BPE entries         (corpus-dependent)
```

This layout is enforced on every load of a production tokenizer artifact.

## Fixed special-token contract

IDs 0–15 are reserved before BPE training and must never be learned or reordered:

| ID | Token | Purpose |
|---:|---|---|
| 0 | `<|pad|>` | Padding |
| 1 | `<|unk|>` | Unknown token |
| 2 | `<|bos|>` | Sequence start |
| 3 | `<|eos|>` | Sequence completion |
| 4 | `<|im_start|>` | Message start |
| 5 | `<|im_sep|>` | Header/content separator |
| 6 | `<|im_end|>` | Message end |
| 7 | `<|tool_call|>` | Tool invocation |
| 8 | `<|tool_result|>` | Tool observation |
| 9 | `<|analysis|>` | Reasoning channel |
| 10 | `<|final|>` | Final response |
| 11 | `<|commentary|>` | Commentary channel |
| 12 | `<|json|>` | Structured JSON output |
| 13 | `<|refusal|>` | Refusal marker |
| 14 | `<|end_of_task|>` | Task completion |
| 15 | `<|meta_sep|>` | Metadata separator |

IDs 16–271 are the fixed 256-byte fallback alphabet supplied by the Hugging Face `ByteLevel` pre-tokenizer. They guarantee that every possible UTF-8 byte sequence can be represented even if the learned merges do not cover it.

IDs 272–32,767 are the learned BPE merge entries produced by training on the approved corpus. The implementation validates the full layout (specials + complete byte-fallback block) on load.

## Training gate

`scripts/train_bpe_tokenizer.py` refuses to train without an explicit approval record containing `approved: true`, a source-registry revision, and a license-audit hash. The corpus must be the exact normalized output of the dataset acquisition pipeline, not the 120-example fixture. Approval must cover redistribution, model training, privacy filtering, and evaluation contamination controls.

Example command after approval:

```bash
python scripts/train_bpe_tokenizer.py \
  --input data/processed/approved/train.jsonl \
  --input data/processed/approved/validation.jsonl \
  --output artifacts/tokenizer.json \
  --manifest-output artifacts/tokenizer.manifest.json \
  --corpus-manifest data/manifests/approved_corpus.json \
  --approval-record data/manifests/tokenizer_approval.json \
  --implementation-commit <actual-implementation-commit>
```

The command must fail if the corpus cannot produce exactly 32,768 entries. This prevents a small or accidental corpus from being mislabeled as the production tokenizer.

## Reproducibility pins

A released artifact must record:

- the tokenizer implementation commit;
- the exact `tokenizers` library version;
- the normalized corpus-manifest SHA-256;
- the final `tokenizer.json` SHA-256;
- the fixed special-token IDs, byte-fallback block, and vocabulary size;
- BPE training parameters and normalization policy.

No guessed, placeholder, or future checksum belongs in the production manifest. The model embedding, tied output head, token-counting tools, training loop, and inference loader must all validate the same vocabulary size and artifact manifest.
