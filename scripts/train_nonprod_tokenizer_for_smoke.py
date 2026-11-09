#!/usr/bin/env python3
"""Train a clearly-labelled NON-PRODUCTION 32 768-vocab BPE tokenizer on the mini fixture.

This exists solely so the dual-T4 smoke test can load a real tokenizer artifact
that satisfies the locked ID layout (0–15 specials / 16–271 byte fallback /
272–32767 learned). It must never be treated as a production tokenizer.

Usage (from repo root, with tokenizers installed):

    python scripts/train_nonprod_tokenizer_for_smoke.py

Outputs:
    artifacts/tokenizer_nonprod.json
    artifacts/tokenizer_nonprod.manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aya_edge.tokenizer.bpe import train_production_bpe, write_manifest, sha256_file

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_PATH = OUT_DIR / "tokenizer_nonprod.json"
MANIFEST_PATH = OUT_DIR / "tokenizer_nonprod.manifest.json"
CORPUS_MANIFEST = REPO / "data" / "manifests" / "approved_corpus_nonprod.json"
APPROVAL = REPO / "data" / "manifests" / "tokenizer_approval_nonprod.json"

INPUTS = [
    REPO / "aya_edge_mini_dataset_v1" / "train.jsonl",
    REPO / "aya_edge_mini_dataset_v1" / "validation.jsonl",
]

def main() -> None:
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if approval.get("approved") is not True or approval.get("label") != "NON-PRODUCTION":
        raise SystemExit("refusing: expected NON-PRODUCTION approval record")

    for p in INPUTS:
        if not p.exists():
            raise SystemExit(f"missing mini-dataset file: {p}")

    print("Training NON-PRODUCTION tokenizer on mini fixture (this may take a minute)...")
    train_production_bpe(INPUTS, TOKENIZER_PATH, min_frequency=1)

    # Pin a deterministic "implementation commit" for the smoke artifact
    impl = "nonprod-smoke-" + sha256_file(TOKENIZER_PATH)[:12]

    manifest = write_manifest(
        TOKENIZER_PATH,
        MANIFEST_PATH,
        corpus_manifest_sha256=sha256_file(CORPUS_MANIFEST),
        implementation_commit=impl,
        min_frequency=1,
    )

    # Extra safety label in the manifest file itself
    data = manifest.to_dict()
    data["label"] = "NON-PRODUCTION"
    data["purpose"] = "dual-T4 smoke / contract validation only"
    MANIFEST_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "label": "NON-PRODUCTION",
        "tokenizer": str(TOKENIZER_PATH),
        "manifest": str(MANIFEST_PATH),
        "tokenizer_sha256": data["tokenizer_sha256"],
        "vocab_size": data["vocab_size"],
        "id_layout": data.get("id_layout"),
    }, indent=2))


if __name__ == "__main__":
    main()
