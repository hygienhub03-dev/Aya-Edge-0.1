"""Collect the first FineWeb-Edu language slice and write its approval record.

This command is the licence gate. It refuses to write an approval file unless
the Hub dataset card still declares ODC-By and the requested config exists.

Example:
    python scripts/prepare_language_corpus.py --char-budget 25000000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aya_edge.data.language_corpus import (
    FINEWEB_EDU_CONFIG,
    FINEWEB_EDU_DATASET,
    approval_record,
    fetch_fineweb_edu_identity,
    license_audit,
    stream_fineweb_edu_records,
    write_json,
)
from aya_edge.data.pipeline import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the first AYA Edge language corpus slice")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/language_v1"))
    parser.add_argument("--char-budget", type=int, default=25_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=FINEWEB_EDU_CONFIG)
    args = parser.parse_args()

    identity = fetch_fineweb_edu_identity(config=args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = license_audit(
        dataset=str(identity["dataset"]),
        config=str(identity["config"]),
        revision=str(identity["revision"]),
        license_name=str(identity["license"]),
        card_license=identity["card_license"],
        char_budget=args.char_budget,
    )
    audit_sha = write_json(output_dir / "license_audit.json", audit)

    raw_path = output_dir / "raw.jsonl"
    documents = 0
    characters = 0
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in stream_fineweb_edu_records(
            revision=str(identity["revision"]),
            config=str(identity["config"]),
            char_budget=args.char_budget,
        ):
            documents += 1
            characters += len(str(record["text"]))
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if documents == 0:
        raise SystemExit("refusing to approve an empty language slice")

    manifest = build_dataset(
        [raw_path],
        output_dir,
        seed=args.seed,
        validation_fraction=0.05,
        test_fraction=0.05,
        manifest_metadata={
            "purpose": "first language-learning slice",
            "dataset": FINEWEB_EDU_DATASET,
            "config": identity["config"],
            "revision": identity["revision"],
            "license": identity["license"],
            "publisher": "Hugging Face (Lozhkov, Ben Allal, von Werra, Wolf)",
            "char_budget": args.char_budget,
            "documents_before_split": documents,
            "characters_before_split": characters,
            "license_audit_sha256": audit_sha,
            "mixture": {
                "general_language_and_knowledge": 1.0,
                "instruction_and_agent_patterns": 0.0,
            },
        },
    )
    approval = approval_record(
        revision=str(identity["revision"]),
        license_audit_sha256=audit_sha,
        corpus_manifest_path=str(output_dir / "manifest.json"),
        document_count=documents,
        character_count=characters,
    )
    approval_sha = write_json(output_dir / "tokenizer_approval.json", approval)
    print(json.dumps({
        "status": "slice_ready",
        "output_dir": str(output_dir),
        "revision": identity["revision"],
        "license": identity["license"],
        "documents": documents,
        "characters": characters,
        "splits": manifest["records"]["splits"],
        "license_audit_sha256": audit_sha,
        "approval_sha256": approval_sha,
    }, indent=2))


if __name__ == "__main__":
    main()
