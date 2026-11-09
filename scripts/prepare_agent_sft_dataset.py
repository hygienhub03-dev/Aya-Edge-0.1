"""Materialize tokenizer-compatible agent-SFT splits without truncating trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from aya_edge.tokenizer.bpe import ProductionBPETokenizer, sha256_file
from aya_edge.training.agent_sft import SFTDataError, encode_trajectory


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--include-tool-schemas", action="store_true")
    args = parser.parse_args()
    if args.max_length < 2:
        raise SystemExit("--max-length must be at least 2")
    tokenizer = ProductionBPETokenizer.load(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    output_files = {}
    for split in ("train", "validation", "test"):
        source = args.input_dir / f"{split}.jsonl"
        destination = args.output_dir / f"{split}.jsonl"
        accepted = 0
        with source.open(encoding="utf-8") as source_handle, destination.open("w", encoding="utf-8") as output_handle:
            for line_number, line in enumerate(source_handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    encode_trajectory(
                        row,
                        tokenizer,
                        max_length=args.max_length,
                        include_tool_schemas=args.include_tool_schemas,
                    )
                except SFTDataError as exc:
                    reason = str(exc).split(": ", 1)[0]
                    counts[f"{split}:{reason}"] += 1
                    continue
                output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                accepted += 1
        output_files[split] = {"path": str(destination), "examples": accepted, "sha256": file_sha256(destination)}
    total = sum(item["examples"] for item in output_files.values())
    if total == 0 or output_files["train"]["examples"] == 0 or output_files["validation"]["examples"] == 0:
        raise SystemExit("tokenizer-aware SFT materialization produced an empty train or validation split")
    manifest = {
        "format": "aya-edge-agent-sft-tokenized-manifest",
        "version": 1,
        "input_dir": str(args.input_dir),
        "input_manifest_sha256": file_sha256(args.input_dir / "manifest.json"),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "max_length": args.max_length,
        "include_tool_schemas": args.include_tool_schemas,
        "objective": "assistant_only_loss",
        "splits": output_files,
        "total_examples": total,
        "rejections": dict(counts),
        "truncation": "none; over-context trajectories rejected",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
