#!/usr/bin/env python3
"""Prepare an Aya Edge byte tokenizer config from local data (no downloads).

The optional input is only inspected for a byte/count report. It cannot alter
vocabulary contents or token IDs; byte-level tokenization needs no training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct execution from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aya_edge.tokenizer import ByteLevelTokenizer  # noqa: E402


def iter_local_text(path: Path):
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
                yield json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        yield path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="output tokenizer JSON config")
    parser.add_argument("--input", type=Path, help="optional local UTF-8 text or JSONL file")
    args = parser.parse_args()

    tokenizer = ByteLevelTokenizer()
    report = {"vocab_size": tokenizer.vocab_size, "input_bytes": 0, "input_records": 0}
    if args.input is not None:
        for text in iter_local_text(args.input):
            report["input_bytes"] += len(text.encode("utf-8"))
            report["input_records"] += 1

    tokenizer.save(args.output)
    print(json.dumps({"output": str(args.output), **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
