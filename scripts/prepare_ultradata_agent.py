"""Prepare a bounded English UltraData agent-SFT pilot without materializing the full Hub dataset.

This exports structured trajectories; it does not execute their frozen tools.
The caller must explicitly acknowledge the dataset card's upstream-rights caveats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATASET = "openbmb/UltraData-SFT-Agent-2609"
REVISION = "f684cc1"  # Dataset-card revision inspected for this pilot; resolve full Hub SHA at runtime.


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def trajectory_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def valid_trajectory(row: dict[str, Any], max_chars: int) -> tuple[bool, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "missing_or_short_messages"
    if not isinstance(row.get("uuid"), str) or not row["uuid"].strip():
        return False, "missing_uuid"
    if not isinstance(row.get("tools", []), list):
        return False, "invalid_tools"
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if len(roles) != len(messages) or not {"user", "assistant"}.issubset(set(roles)):
        return False, "invalid_message_roles"
    text = trajectory_text(messages)
    if not text.strip():
        return False, "empty_trajectory"
    if len(text) > max_chars:
        return False, "trajectory_too_long"
    if has_cjk(text):
        return False, "non_english_or_mixed_language"
    return True, "accepted"


def split_for_id(identifier: str) -> str:
    bucket = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 5:
        return "test"
    if bucket < 10:
        return "validation"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/ultradata_agent_v1"))
    parser.add_argument("--tool-use", type=int, default=12_000)
    parser.add_argument("--search-agent", type=int, default=5_000)
    parser.add_argument("--general-agent", type=int, default=8_000)
    parser.add_argument("--max-characters", type=int, default=80_000)
    parser.add_argument("--max-candidates-per-subset", type=int, default=100_000)
    parser.add_argument("--revision", default=REVISION, help="Hub commit/revision; pin the resolved full SHA in the manifest")
    parser.add_argument("--accept-upstream-terms", action="store_true")
    parser.add_argument("--accept-residual-upstream-rights-risk", action="store_true")
    args = parser.parse_args()

    if not args.accept_upstream_terms or not args.accept_residual_upstream_rights_risk:
        raise SystemExit(
            "This pilot requires explicit acknowledgement of the dataset card's upstream terms "
            "and residual item-level rights caveats. Re-run with --accept-upstream-terms "
            "and --accept-residual-upstream-rights-risk after reviewing docs/data/ULTRADATA_AGENT_SFT.md."
        )
    quotas = {"Tool-Use": args.tool_use, "Search-Agent": args.search_agent, "General-Agent": args.general_agent}
    if any(value < 0 for value in quotas.values()) or sum(quotas.values()) == 0:
        raise SystemExit("subset limits must be non-negative and their total must be positive")
    if args.max_characters < 1 or args.max_candidates_per_subset < 1:
        raise SystemExit("character and candidate limits must be positive")

    try:
        from datasets import load_dataset
        from huggingface_hub import dataset_info
    except ImportError as exc:
        raise SystemExit("install data extras: python -m pip install -e '.[data]'") from exc

    info = dataset_info(DATASET, revision=args.revision)
    resolved_revision = info.sha
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    handles = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("train", "validation", "test")}
    counts = Counter()
    rejects = Counter()
    accepted_by_subset = Counter()
    seen_uuids: set[str] = set()
    complete = False
    try:
        for subset, quota in quotas.items():
            if quota == 0:
                continue
            stream = load_dataset(DATASET, subset, split="train", revision=resolved_revision, streaming=True)
            scanned = 0
            for row in stream:
                scanned += 1
                if scanned > args.max_candidates_per_subset or accepted_by_subset[subset] >= quota:
                    break
                valid, reason = valid_trajectory(row, args.max_characters)
                if not valid:
                    rejects[f"{subset}:{reason}"] += 1
                    continue
                identifier = row["uuid"]
                if identifier in seen_uuids:
                    rejects[f"{subset}:duplicate_uuid"] += 1
                    continue
                domain = str(row.get("domain", ""))
                if subset == "Search-Agent" and domain and domain != "Search_Agent-en":
                    rejects[f"{subset}:non_english_domain"] += 1
                    continue
                record = {
                    "uuid": row["uuid"],
                    "messages": row["messages"],
                    "tools": row.get("tools", []),
                    "source": DATASET,
                    "domain": domain or subset.replace("-", "_"),
                    "subset": subset,
                    "license": "Apache-2.0 dataset declaration; applicable upstream terms unresolved",
                    "declared_dataset_license": "Apache-2.0",
                    "source_revision": resolved_revision,
                    "upstream_rights_review": "not_resolved_per_record",
                    "use_scope": "bounded_personal_research_sft_pilot",
                    "split": split_for_id(row["uuid"]),
                }
                handles[record["split"]].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                seen_uuids.add(identifier)
                counts[record["split"]] += 1
                accepted_by_subset[subset] += 1
            if accepted_by_subset[subset] < quota:
                raise RuntimeError(
                    f"{subset}: requested {quota}, accepted {accepted_by_subset[subset]} "
                    f"after scanning {scanned}; refusing to silently underfill the pilot"
                )
        complete = True
    finally:
        for handle in handles.values():
            handle.close()
        if not complete:
            for split_path in output.glob("*.jsonl"):
                split_path.unlink(missing_ok=True)

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    manifest = {
        "format": "aya-edge-agent-sft-manifest",
        "version": 1,
        "dataset": DATASET,
        "dataset_card_revision": args.revision,
        "resolved_hub_revision": resolved_revision,
        "license": "Apache-2.0 dataset declaration; applicable upstream terms unresolved",
        "declared_dataset_license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "dataset_card_url": f"https://huggingface.co/datasets/{DATASET}",
        "attribution": "MiniCPM Team, UltraData-SFT-Agent-2609 (2026)",
        "upstream_rights_review": "not_resolved_per_record",
        "limitations": [
            "Apache-2.0 dataset declaration does not override upstream dataset/content terms.",
            "The dataset card identifies course-page and web/repository-derived samples; public access alone is not a training license.",
            "This bounded personal research pilot is not legal clearance for commercial use or redistribution.",
            "Frozen tools are demonstrations only; no tool implementations or environments are included.",
            "Diagnostic failure traces are not success labels; review trajectories before assigning supervision masks.",
        ],
        "requested_by_subset": quotas,
        "accepted_by_subset": dict(accepted_by_subset),
        "splits": dict(counts),
        "split_sha256": {name: file_sha256(output / f"{name}.jsonl") for name in ("train", "validation", "test")},
        "unique_uuids": len(seen_uuids),
        "rejections": dict(rejects),
        "split_policy": "stable UUID SHA-256 bucket: 90% train, 5% validation, 5% test",
        "max_characters_per_trajectory": args.max_characters,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
