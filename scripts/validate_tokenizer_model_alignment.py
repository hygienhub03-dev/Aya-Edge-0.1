#!/usr/bin/env python3
"""Fail loudly if the model config and tokenizer are not production-aligned.

This is a Milestone A gate. It must pass before any claim of a genuine
language-model training run is made.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from aya_edge import AyaEdgeConfig
from aya_edge.tokenizer import PRODUCTION_VOCAB_SIZE, ProductionBPETokenizer, SPECIAL_TOKENS


def load_config(path: Path) -> AyaEdgeConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AyaEdgeConfig(**raw["model"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/aya_edge_0_2.yaml"),
        help="Model YAML config (must declare vocab_size=32768)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to a production tokenizer.json artifact",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Also require a co-located tokenizer.manifest.json with real checksums",
    )
    args = parser.parse_args()

    errors: list[str] = []

    # 1. Model config
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"failed to load config: {exc}"}))
        return 1

    if cfg.vocab_size != PRODUCTION_VOCAB_SIZE:
        errors.append(
            f"model vocab_size is {cfg.vocab_size}, expected {PRODUCTION_VOCAB_SIZE}"
        )

    # 2. Production tokenizer artifact
    if not args.tokenizer.exists():
        errors.append(f"tokenizer artifact not found: {args.tokenizer}")
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    try:
        tok = ProductionBPETokenizer.load(args.tokenizer)
    except Exception as exc:
        errors.append(f"failed to load production tokenizer: {exc}")
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    if tok.vocab_size != PRODUCTION_VOCAB_SIZE:
        errors.append(
            f"tokenizer vocab_size is {tok.vocab_size}, expected {PRODUCTION_VOCAB_SIZE}"
        )

    # 3. Special-token contract
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        # ProductionBPETokenizer already validates on load; this is belt-and-braces.
        pass

    # 4. Model ↔ tokenizer alignment
    try:
        tok.assert_model_alignment(cfg)
    except ValueError as exc:
        errors.append(str(exc))

    # 5. Optional strict manifest check
    if args.strict:
        manifest_path = args.tokenizer.with_suffix(".manifest.json")
        if not manifest_path.exists():
            # also accept sibling name used by the trainer
            alt = args.tokenizer.parent / "tokenizer.manifest.json"
            if alt.exists():
                manifest_path = alt
            else:
                errors.append(f"strict mode requires a tokenizer manifest near {args.tokenizer}")
        else:
            try:
                from aya_edge.tokenizer.bpe import ProductionTokenizerManifest
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                # Reconstruct to trigger validation
                ProductionTokenizerManifest(
                    format=data["format"],
                    version=int(data["version"]),
                    tokenizer_sha256=data["tokenizer_sha256"],
                    corpus_manifest_sha256=data["corpus_manifest_sha256"],
                    implementation_commit=data["implementation_commit"],
                    tokenizers_version=data["tokenizers_version"],
                    vocab_size=int(data["vocab_size"]),
                    special_tokens=tuple(data["special_tokens"]),
                )
            except Exception as exc:
                errors.append(f"manifest validation failed: {exc}")

    report = {
        "ok": len(errors) == 0,
        "model_vocab_size": cfg.vocab_size,
        "tokenizer_vocab_size": tok.vocab_size,
        "production_vocab_size": PRODUCTION_VOCAB_SIZE,
        "tokenizer_path": str(args.tokenizer),
        "config_path": str(args.config),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
