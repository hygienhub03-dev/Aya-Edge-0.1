"""Dependency-light, provenance-first JSONL dataset preparation."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

_REQUIRED = {"id", "category", "text", "source", "license"}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def normalize_record(record: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(_REQUIRED - set(record))
    if missing:
        raise ValueError(f"record missing required fields: {', '.join(missing)}")
    values = dict(record)
    if not all(isinstance(values[key], str) and values[key].strip() for key in _REQUIRED):
        raise TypeError("id, category, text, source, and license must be non-empty strings")
    text = normalize_text(str(values["text"]))
    if not text:
        raise ValueError("normalized text cannot be empty")
    values["id"] = str(values["id"]).strip()
    values["category"] = str(values["category"]).strip()
    values["text"] = text
    values["source"] = str(values["source"]).strip()
    values["license"] = str(values["license"]).strip()
    values["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return values


def _read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(normalize_record(value))
    return rows


def build_dataset(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    manifest_metadata: Mapping[str, object] | None = None,
    default_source: str | None = None,
    default_license: str | None = None,
) -> dict[str, object]:
    """Build deterministic JSONL splits and return an auditable manifest."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be in [0, 1)")
    raw_rows: list[dict[str, object]] = []
    for path in inputs:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            if "source" not in value and default_source is not None:
                value["source"] = default_source
            if "license" not in value and default_license is not None:
                value["license"] = default_license
            raw_rows.append(value)
    rows = [normalize_record(value) for value in raw_rows]
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    kept: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row["id"])
        text_hash = str(row["text_sha256"])
        if row_id in seen_ids:
            rejected.append({"id": row_id, "reason": "duplicate_id"})
            continue
        if text_hash in seen_hashes:
            rejected.append({"id": row_id, "reason": "duplicate_normalized_text"})
            continue
        seen_ids.add(row_id)
        seen_hashes.add(text_hash)
        kept.append(row)

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in kept:
        by_category[str(row["category"])].append(row)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    for category, category_rows in sorted(by_category.items()):
        rng.shuffle(category_rows)
        n = len(category_rows)
        n_test = int(n * test_fraction)
        n_val = int(n * validation_fraction)
        splits["test"].extend(category_rows[:n_test])
        splits["validation"].extend(category_rows[n_test:n_test + n_val])
        splits["train"].extend(category_rows[n_test + n_val:])
    for values in splits.values():
        values.sort(key=lambda row: str(row["id"]))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split, values in splits.items():
        (destination / f"{split}.jsonl").write_text(
            "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
            encoding="utf-8",
        )
    manifest: dict[str, object] = {
        "format": "aya-edge-jsonl-manifest",
        "version": 1,
        "seed": seed,
        "processing": {
            "normalization": "unicode-preserving whitespace collapse and NUL removal",
            "deduplication": "ID and normalized-text SHA-256",
            "split": {"validation_fraction": validation_fraction, "test_fraction": test_fraction, "stratified_by": "category"},
        },
        "records": {"input": len(rows), "accepted": len(kept), "rejected": len(rejected), "splits": {k: len(v) for k, v in splits.items()}},
        "categories": Counter(str(row["category"]) for row in kept),
        "rejections": rejected,
        "metadata": dict(manifest_metadata or {}),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
