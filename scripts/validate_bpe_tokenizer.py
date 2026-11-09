"""Validate a pinned AYA Edge production tokenizer artifact before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aya_edge.config import AyaEdgeConfig
from aya_edge.tokenizer.bpe import (
    PRODUCTION_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    ProductionBPETokenizer,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the locked AYA Edge 32k BPE artifact")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/aya_edge_0_2_language.yaml"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("vocab_size") != PRODUCTION_VOCAB_SIZE:
        raise SystemExit("manifest vocabulary is not 32768")
    if manifest.get("special_token_ids") != SPECIAL_TOKEN_IDS:
        raise SystemExit("manifest special-token IDs do not match the locked contract")
    if manifest.get("tokenizer_sha256") != sha256_file(args.tokenizer):
        raise SystemExit("tokenizer checksum does not match its manifest")
    tokenizer = ProductionBPETokenizer.load(args.tokenizer)
    config = AyaEdgeConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8"))["model"])
    tokenizer.assert_model_alignment(config)
    probes = ["ASCII text", "café — 你好", "\x00\x01\xff"]
    for probe in probes:
        if tokenizer.decode(tokenizer.encode(probe)) != probe:
            raise SystemExit("tokenizer round trip failed")
    print(json.dumps({"status": "valid", "vocab_size": tokenizer.vocab_size, "tokenizer": str(args.tokenizer)}, indent=2))


if __name__ == "__main__":
    main()
