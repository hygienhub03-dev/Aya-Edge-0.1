"""Train and pin the approved AYA Edge production tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aya_edge.tokenizer.bpe import train_production_bpe, write_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description="Train the approved 32,768-vocabulary AYA Edge byte-level BPE tokenizer")
parser.add_argument("--input", action="append", required=True, type=Path, help="approved normalized JSONL/text corpus file; repeatable")
parser.add_argument("--output", type=Path, required=True, help="tokenizer.json output path")
parser.add_argument("--manifest-output", type=Path, required=True)
parser.add_argument("--corpus-manifest", type=Path, required=True, help="approved corpus manifest whose hash is pinned")
parser.add_argument("--implementation-commit", required=True, help="actual Git commit containing tokenizer implementation")
parser.add_argument("--approval-record", type=Path, required=True, help="JSON record with {\"approved\": true} and source-clearance metadata")
parser.add_argument("--min-frequency", type=int, default=2)
args = parser.parse_args()

approval = json.loads(args.approval_record.read_text(encoding="utf-8"))
if approval.get("approved") is not True:
    raise SystemExit("refusing to train: approval record must explicitly contain approved=true")
if not approval.get("source_registry_revision") or not approval.get("license_audit_sha256"):
    raise SystemExit("refusing to train: approval record must pin source_registry_revision and license_audit_sha256")
if len(args.implementation_commit) < 7 or args.implementation_commit.startswith(("TODO", "PENDING")):
    raise SystemExit("refusing to train: provide a real implementation commit")
for path in args.input:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
if args.min_frequency < 1:
    raise SystemExit("--min-frequency must be positive")

train_production_bpe(args.input, args.output, min_frequency=args.min_frequency, show_progress=True)
manifest = write_manifest(
    args.output,
    args.manifest_output,
    corpus_manifest_sha256=sha256_file(args.corpus_manifest),
    implementation_commit=args.implementation_commit,
    min_frequency=args.min_frequency,
)
print(json.dumps(manifest.to_dict(), indent=2))
