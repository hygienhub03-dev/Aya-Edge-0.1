"""Prepare the first real language-learning slice.

FineWeb-Edu is the selected source for Milestone A. The parent dataset is
published by Hugging Face under ODC-By 1.0 and is also subject to the Common
Crawl terms. This module only accepts that declared licence. It records the
dataset revision and writes an approval file for the bounded slice actually
collected. It does not approve the full 1.3-trillion-token corpus, and it does
not invent instruction or tool-result text.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIG = "sample-10BT"
ACCEPTED_LICENSES = frozenset({"odc-by", "odc-by-1.0"})
PUBLISHER = "Hugging Face (Lozhkov, Ben Allal, von Werra, Wolf)"
ATTRIBUTION = (
    "Lozhkov, Anton; Ben Allal, Loubna; von Werra, Leandro; Wolf, Thomas. "
    "FineWeb-Edu. Hugging Face, 2024. "
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu. "
    "Licensed under ODC-By 1.0."
)


def normalize_license_name(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    return {item.strip().lower().replace("_", "-") for item in items if item and str(item).strip()}


def assert_training_license(value: object) -> str:
    """Accept only the FineWeb-Edu ODC-By declaration."""
    found = normalize_license_name(value)
    if not found & ACCEPTED_LICENSES:
        raise ValueError(
            "refusing corpus: FineWeb-Edu must declare the ODC-By 1.0 licence, "
            f"found {sorted(found) or ['<missing>']}"
        )
    return "ODC-By-1.0"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def license_audit(
    *,
    dataset: str,
    config: str,
    revision: str,
    license_name: str,
    card_license: object,
    char_budget: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "config": config,
        "revision": revision,
        "publisher": PUBLISHER,
        "declared_license": license_name,
        "card_license": sorted(normalize_license_name(card_license)),
        "license_url": "https://opendatacommons.org/licenses/by/1-0/",
        "dataset_url": f"https://huggingface.co/datasets/{dataset}",
        "attribution": ATTRIBUTION,
        "permitted_use_recorded": [
            "local research training of the AYA Edge language model",
            "creation of a derivative database slice with attribution",
        ],
        "obligations": [
            "Public distribution of this database or a derivative database requires ODC-By attribution.",
            "Use of the underlying crawl is also subject to the Common Crawl terms of use.",
        ],
        "not_approved": [
            "The full FineWeb-Edu corpus is not approved by collecting this slice.",
            "Individual source-page copyrights are not cleared document by document.",
            "Upstream email and IP rewriting is recorded; this slice adds no further privacy guarantee.",
        ],
        "char_budget": char_budget,
        "audited_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def approval_record(
    *,
    revision: str,
    license_audit_sha256: str,
    corpus_manifest_path: str,
    document_count: int,
    character_count: int,
) -> dict[str, object]:
    return {
        "approved": True,
        "scope": "first_language_slice_only",
        "label": "PRODUCTION-SLICE",
        "dataset": FINEWEB_EDU_DATASET,
        "config": FINEWEB_EDU_CONFIG,
        "source_registry_revision": revision,
        "license_audit_sha256": license_audit_sha256,
        "corpus_manifest": corpus_manifest_path,
        "license": "ODC-By-1.0",
        "publisher": PUBLISHER,
        "documents": document_count,
        "characters": character_count,
        "mixture": {
            "general_language_and_knowledge": 1.0,
            "instruction_and_agent_patterns": 0.0,
            "reason": (
                "Milestone A teaches ordinary language from educational web text. "
                "The 50/50 instruction allocation starts with agent supervision, "
                "not with synthetic tool results mixed into this slice."
            ),
        },
    }


def fetch_fineweb_edu_identity(dataset: str = FINEWEB_EDU_DATASET, config: str = FINEWEB_EDU_CONFIG) -> dict[str, object]:
    """Read the Hub dataset card and refuse anything other than the pinned source."""
    try:
        from huggingface_hub import dataset_info
    except ImportError as exc:
        raise RuntimeError("install the project data extras before preparing the language corpus") from exc
    info = dataset_info(dataset)
    card = info.card_data
    raw_license = None if card is None else getattr(card, "license", None)
    if raw_license is None and card is not None and isinstance(card, dict):
        raw_license = card.get("license")
    license_name = assert_training_license(raw_license)
    revision = info.sha
    if not revision:
        raise RuntimeError("Hugging Face dataset info did not include a revision SHA")
    try:
        from datasets import get_dataset_config_names
        configs = set(get_dataset_config_names(dataset))
    except Exception as exc:  # noqa: BLE001 - surface the Hub/datasets failure directly
        raise RuntimeError(f"could not list dataset configs for {dataset}") from exc
    if config not in configs:
        raise ValueError(f"dataset config {config} is not in {sorted(configs)}")
    return {"dataset": dataset, "config": config, "revision": revision, "license": license_name, "card_license": raw_license}


def stream_fineweb_edu_records(
    *,
    revision: str,
    config: str = FINEWEB_EDU_CONFIG,
    char_budget: int = 25_000_000,
    min_characters: int = 200,
    max_characters: int = 100_000,
):
    """Stream a bounded, provenance-bearing slice. Stops at the character budget."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the project data extras before preparing the language corpus") from exc
    if char_budget < 1000:
        raise ValueError("char_budget must be at least 1000 characters")
    dataset = load_dataset(
        FINEWEB_EDU_DATASET,
        name=config,
        split="train",
        revision=revision,
        streaming=True,
    )
    characters = 0
    produced = False
    for row in dataset:
        text = row.get("text")
        document_id = row.get("id")
        if not isinstance(text, str) or not isinstance(document_id, str):
            continue
        text = " ".join(text.replace("\x00", " ").split())
        if len(text) < min_characters or len(text) > max_characters:
            continue
        if characters >= char_budget:
            break
        remaining = char_budget - characters
        if len(text) > remaining:
            text = text[:remaining].rsplit(" ", 1)[0].strip()
            if len(text) < min_characters:
                break
        characters += len(text)
        produced = True
        yield {
            "id": f"fineweb-edu-{_sha256_text(document_id)[:16]}",
            "category": "general_language",
            "text": text,
            "source": f"{FINEWEB_EDU_DATASET}:{config}:{revision}",
            "license": "ODC-By-1.0",
            "upstream_id": document_id,
            "dump": row.get("dump"),
            "url": row.get("url"),
        }
    if not produced:
        raise RuntimeError("FineWeb-Edu stream produced no usable documents")
