#!/usr/bin/env python3
"""Convert a locally acquired, checksum-verified Wikimedia XML shard to sampled JSONL.

This is deliberately not a downloader. The source registry must approve the exact
snapshot before this command reads or writes any source content.
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(parent: ET.Element, name: str, default: str = "") -> str:
    for child in parent:
        if local_name(child.tag) == name:
            return child.text or default
    return default


def stream_pages(path: Path, allowed_namespaces: set[str]) -> Iterator[dict[str, Any]]:
    opener = bz2.open if path.name.endswith(".bz2") else open
    with opener(path, "rb") as stream:
        stack: list[ET.Element] = []
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                stack.append(element)
                continue
            if local_name(element.tag) == "page":
                namespace = child_text(element, "ns", "0")
                if namespace in allowed_namespaces:
                    page_id = child_text(element, "id")
                    title = child_text(element, "title")
                    revision = next((item for item in element if local_name(item.tag) == "revision"), None)
                    if page_id and title and revision is not None:
                        text_element = next((item for item in revision if local_name(item.tag) == "text"), None)
                        if text_element is not None and text_element.get("deleted") is None:
                            revision_id = child_text(revision, "id")
                            yield {
                                "page_id": page_id,
                                "title": title,
                                "namespace": namespace,
                                "revision_id": revision_id,
                                "revision_timestamp": child_text(revision, "timestamp"),
                                "parent_revision_id": child_text(revision, "parentid"),
                                "content_model": child_text(revision, "model"),
                                "content_format": child_text(revision, "format"),
                                "sha1": child_text(revision, "sha1"),
                                "text": text_element.text or "",
                                "source_url": f"https://en.wikipedia.org/w/index.php?oldid={revision_id}" if revision_id else f"https://en.wikipedia.org/wiki/Special:Redirect/page/{page_id}",
                                "attribution_url": f"https://en.wikipedia.org/w/index.php?title=Special:History&pageid={page_id}",
                            }
                if len(stack) > 1:
                    stack[-2].remove(element)
                element.clear()
            stack.pop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", default="wikimedia_enwiki")
    parser.add_argument("--dump", type=Path, required=True, help="Locally staged .xml or .xml.bz2 shard")
    parser.add_argument("--snapshot", required=True, help="Dated snapshot, e.g. 20261001")
    parser.add_argument("--expected-sha256", required=True, help="SHA-256 from the official dump-status manifest")
    parser.add_argument("--limit", type=int, required=True, help="Maximum number of accepted namespace-filtered pages")
    parser.add_argument("--namespaces", default="0", help="Comma-separated namespace IDs; default is main namespace only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit <= 0 or not re.fullmatch(r"\d{8}", args.snapshot):
        parser.error("--limit must be positive and --snapshot must be YYYYMMDD")

    registry = pipeline.registry_sources()
    if args.source_id not in registry:
        raise pipeline.PipelineError(f"Unknown source_id: {args.source_id}")
    source = registry[args.source_id]
    pipeline.ensure_trainable(source, args.source_id)
    if pipeline.parse_revision(source) != args.snapshot:
        raise pipeline.PipelineError("Requested Wikimedia snapshot does not match the source registry pin")
    if not args.dump.is_file():
        raise pipeline.PipelineError(f"Dump shard not found: {args.dump}")
    actual_sha = pipeline.sha256_file(args.dump)
    if actual_sha.lower() != args.expected_sha256.lower():
        raise pipeline.PipelineError(f"Dump checksum mismatch: expected {args.expected_sha256}, got {actual_sha}")

    allowed = {value.strip() for value in args.namespaces.split(",") if value.strip()}
    output = args.output or pipeline.DATA_ROOT / "raw" / args.source_id / "pilot.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for page in stream_pages(args.dump, allowed):
        page.update({
            "dump_snapshot": args.snapshot,
            "dump_shard": args.dump.name,
            "dump_shard_sha256": actual_sha,
            "rights_status": "source-level license obligations apply; page-specific rights require separate review",
            "source_revision": args.snapshot,
        })
        rows.append(page)
        if len(rows) >= args.limit:
            break
    if not rows:
        raise pipeline.PipelineError("No pages matched the requested namespace and sample cap")
    pipeline.write_jsonl(output, rows)
    pipeline.append_jsonl(pipeline.DATA_ROOT / "manifests/shards.jsonl", {
        "source_id": args.source_id,
        "snapshot": args.snapshot,
        "shard": args.dump.name,
        "sha256": actual_sha,
        "namespaces": sorted(allowed),
        "records": len(rows),
        "output": str(output.relative_to(pipeline.DATA_ROOT)) if output.is_relative_to(pipeline.DATA_ROOT) else str(output),
        "rights_status": "page-level rights review required",
    })
    print(json.dumps({"source_id": args.source_id, "snapshot": args.snapshot, "records": len(rows), "sha256": actual_sha, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except pipeline.PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
