"""Build auditable AYA Edge JSONL splits from licensed source records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aya_edge.data import build_dataset


parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs="+", type=Path, help="JSONL source files with provenance fields")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validation-fraction", type=float, default=0.1)
parser.add_argument("--test-fraction", type=float, default=0.1)
parser.add_argument("--default-source", help="Explicit source label for legacy fixture rows")
parser.add_argument("--default-license", help="Explicit license label for legacy fixture rows")
args = parser.parse_args()

manifest = build_dataset(
    args.inputs,
    args.output_dir,
    seed=args.seed,
    validation_fraction=args.validation_fraction,
    test_fraction=args.test_fraction,
    default_source=args.default_source,
    default_license=args.default_license,
    manifest_metadata={
        "purpose": "development pipeline fixture unless source registry is independently approved",
        "source_registry": [str(path) for path in args.inputs],
    },
)
print(json.dumps(manifest, indent=2, sort_keys=True))
