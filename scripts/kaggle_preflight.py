"""Validate all artifacts required before the first AYA Edge agent-SFT run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aya_edge import AyaEdgeConfig
from aya_edge.tokenizer.bpe import ProductionBPETokenizer, sha256_file
from aya_edge.training.agent_sft import read_encoded_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/aya_edge_0_2_language.yaml"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--include-tool-schemas", action="store_true")
    parser.add_argument("--manifest", type=Path, help="optional dataset manifest to record in the report")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = AyaEdgeConfig(**raw["model"])
    tokenizer = ProductionBPETokenizer.load(args.tokenizer)
    tokenizer.assert_model_alignment(cfg)
    max_length = min(cfg.max_seq_len, args.max_length or int(raw["training"].get("sequence_length", cfg.max_seq_len)))
    train = read_encoded_jsonl(
        args.train_jsonl,
        tokenizer,
        max_length=max_length,
        include_tool_schemas=args.include_tool_schemas,
    )
    validation = read_encoded_jsonl(
        args.validation_jsonl,
        tokenizer,
        max_length=max_length,
        include_tool_schemas=args.include_tool_schemas,
    )
    report = {
        "status": "ready",
        "model_vocab_size": cfg.vocab_size,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "max_length": max_length,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_supervised_tokens": sum(example.supervised_tokens for example in train),
        "validation_supervised_tokens": sum(example.supervised_tokens for example in validation),
        "objective": "assistant_only_loss",
        "include_tool_schemas": args.include_tool_schemas,
        "tool_validation": "json-schema arguments plus call/result pairing",
        "dataset_manifest": str(args.manifest) if args.manifest else None,
    }
    if report["train_supervised_tokens"] == 0 or report["validation_supervised_tokens"] == 0:
        raise SystemExit("preflight failed: at least one split has no assistant target tokens")
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
