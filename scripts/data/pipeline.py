#!/usr/bin/env python3
"""Auditable, deterministic AYA Edge data-acquisition and validation helpers.

Raw source payloads live under AYA_DATA_ROOT (default: repository/data) and are
never intended for Git. Downloads are refused unless the source registry marks
the exact source/subset approved and the acquisition config pins a full commit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by CLI environment
    raise SystemExit("PyYAML is required. Install with: pip install -e '.[data]'") from exc

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AYA_DATA_ROOT", ROOT / "data")).expanduser().resolve()
REGISTRY_DEFAULT = ROOT / "configs/data/source_registry.yaml"
ACQUISITION_DEFAULT = ROOT / "configs/data/acquisition.yaml"
MIXTURE_DEFAULT = ROOT / "configs/data/mixture.yaml"
FILTERING_DEFAULT = ROOT / "configs/data/filtering.yaml"
TRANSFORM_VERSION = "1.0.0"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
APPROVED = {"approved", "approved_with_conditions"}


class PipelineError(RuntimeError):
    """Raised when an acquisition or data-integrity gate fails."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise PipelineError(f"Expected a YAML mapping in {path}")
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    tmp.replace(path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise PipelineError(f"Expected a JSON object in {path}:{line_no}")
            yield record


def iter_files(directory: Path, pattern: str = "*.jsonl") -> list[Path]:
    return sorted(directory.glob(pattern)) if directory.exists() else []


def registry_sources(registry_path: Path | None = None) -> dict[str, dict[str, Any]]:
    registry_path = registry_path or REGISTRY_DEFAULT
    registry_doc = load_yaml(registry_path)
    raw = registry_doc.get("sources", [])
    if not isinstance(raw, list):
        raise PipelineError("source_registry.yaml must contain a sources list")
    sources: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not item.get("source_id"):
            raise PipelineError("Every source registry entry needs source_id")
        sid = str(item["source_id"])
        if sid in sources:
            raise PipelineError(f"Duplicate source_id in registry: {sid}")
        item.setdefault("review_date", registry_doc.get("review_date"))
        sources[sid] = item
    return sources


def is_source_approved(source: dict[str, Any]) -> tuple[bool, str]:
    status = source.get("approval_status", "unverified")
    if status not in APPROVED:
        return False, f"approval_status={status!r} is not approved"
    if source.get("license_verified") is not True:
        return False, "license_verified is not true"
    for key in ("license_name", "license_url", "review_date", "review_evidence"):
        if not source.get(key):
            return False, f"missing required licensing field: {key}"
    if status == "approved_with_conditions" and source.get("conditions_satisfied") is not True:
        return False, "approval conditions are not marked satisfied"
    return True, "approved"


def ensure_trainable(source: dict[str, Any], source_id: str) -> None:
    ok, reason = is_source_approved(source)
    if not ok:
        raise PipelineError(f"Source {source_id} is blocked from training: {reason}")
    if source.get("use_class") in {"held_out", "later_rl"}:
        raise PipelineError(f"Source {source_id} is designated {source.get('use_class')}, not Stage 1 training")


def parse_revision(source: dict[str, Any], override: Any = None) -> str:
    revision = str(override or source.get("revision") or "")
    if source.get("revision_kind") == "dated_dump_snapshot" and re.fullmatch(r"\d{8}", revision):
        return revision
    if not FULL_SHA.fullmatch(revision):
        raise PipelineError(f"A full immutable 40-character commit SHA is required; got {revision!r}")
    return revision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_text(record: dict[str, Any]) -> str:
    if isinstance(record.get("text"), str):
        return record["text"]
    if isinstance(record.get("messages"), list):
        parts = []
        for message in record["messages"]:
            if isinstance(message, dict):
                content = message.get("content", "")
                parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, sort_keys=True))
        return "\n".join(parts)
    if isinstance(record.get("events"), list):
        return json.dumps(record["events"], ensure_ascii=False, sort_keys=True)
    return ""


def sensitive_data_indicators(text: str) -> list[str]:
    found = set()
    if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.IGNORECASE):
        found.add("email")
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
        found.add("us_ssn")
    if re.search(r"\b(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._~-]{20,})\b", text):
        found.add("credential_or_api_secret")
    for candidate in re.findall(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)", text):
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 15 and len(set(digits)) > 1:
            found.add("phone_number")
            break
    for candidate in re.findall(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", text):
        digits = re.sub(r"\D", "", candidate)
        total, double = 0, False
        for digit in reversed(digits):
            value = int(digit)
            if double:
                value *= 2
                value = value - 9 if value > 9 else value
            total += value
            double = not double
        if 13 <= len(digits) <= 19 and total % 10 == 0:
            found.add("payment_card_number")
            break
    return sorted(found)


def stable_id(source_id: str, revision: str, split: str, original_id: Any, raw: Any) -> str:
    identity = {"source": source_id, "revision": revision, "split": split, "id": original_id, "record": raw}
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def record_id(record: dict[str, Any]) -> str:
    value = record.get("sample_id") or record.get("trajectory_id")
    if value is None:
        raise PipelineError("Canonical record is missing sample_id or trajectory_id")
    return str(value)


def resolve_hf_revision(repo_id: str, revision: str = "main") -> dict[str, Any]:
    url = f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}"
    request = urllib.request.Request(url, headers={"User-Agent": "AYA-Edge-data-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Could not resolve {repo_id}@{revision}: {exc}") from exc
    return {"repo_id": repo_id, "requested_ref": revision, "commit_sha": payload.get("sha"), "url": url}


def inspect_sources(registry_path: Path = REGISTRY_DEFAULT) -> int:
    sources = registry_sources(registry_path)
    out = DATA_ROOT / "manifests/revision_probes.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    failures = 0
    resolved: dict[str, dict[str, Any]] = {}
    for sid, source in sorted(sources.items()):
        repo = source.get("repository")
        if source.get("revision_kind") != "huggingface_commit" or not repo:
            print(f"SKIP {sid}: pinned as {source.get('revision_kind')} and resolved from its official non-Hugging-Face source")
            continue
        try:
            if str(repo) not in resolved:
                resolved[str(repo)] = resolve_hf_revision(str(repo), "main")
            result = dict(resolved[str(repo)])
            result.update({"source_id": sid, "observed_at": now_utc(), "note": "Probe only; pin commit_sha in config after review."})
            append_jsonl(out, result)
            print(f"{sid}: {result.get('commit_sha')}")
        except PipelineError as exc:
            failures += 1
            print(f"ERROR {sid}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def verify_licenses(registry_path: Path = REGISTRY_DEFAULT, strict: bool = False) -> int:
    sources = registry_sources(registry_path)
    rows = []
    failures = []
    for sid, source in sorted(sources.items()):
        ok, reason = is_source_approved(source)
        row = {"source_id": sid, "license_name": source.get("license_name"), "approval_status": source.get("approval_status", "unverified"), "license_verified": source.get("license_verified", False), "gate": "pass" if ok else "blocked", "reason": reason, "use_class": source.get("use_class")}
        rows.append(row)
        if not ok:
            failures.append(row)
        print(f"{row['gate'].upper():7} {sid}: {row['approval_status']} — {reason}")
    out = DATA_ROOT / "manifests/license_audit.jsonl"
    write_jsonl(out, rows)
    if strict and failures:
        print(f"License gate blocked: {len(failures)} source(s) need verified approval.", file=sys.stderr)
        return 2
    return 0


def get_acquisition_plan(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = load_yaml(config_path)
    registry_path = Path(config.get("source_registry", REGISTRY_DEFAULT))
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    return config, registry_sources(registry_path)


def existing_jsonl_count(path: Path) -> int:
    count = 0
    if not path.exists():
        return count
    for _ in read_jsonl(path):
        count += 1
    return count


def download_datasets(config_path: Path, stage: str) -> int:
    config, registry = get_acquisition_plan(config_path)
    sources_cfg = config.get("sources", [])
    if stage != "pilot":
        raise PipelineError("Only --stage pilot is currently supported; no RL or training stage is implemented")
    report: list[dict[str, Any]] = []
    active_jobs = [job for job in sources_cfg if job.get("enabled", False) and job.get("stage", "stage1") == "stage1"]
    if not active_jobs:
        print("No Stage 1 sources are approved/enabled for acquisition; no payload was downloaded.")
        write_jsonl(DATA_ROOT / "manifests/acquisition_run.jsonl", [{"stage": stage, "run_at": now_utc(), "status": "no_sources_approved_or_enabled", "source_jobs_considered": len(sources_cfg), "external_payloads_downloaded": 0}])
        return 0
    for job in active_jobs:
        sid = str(job.get("source_id", ""))
        if sid not in registry:
            raise PipelineError(f"Unknown source_id in acquisition config: {sid}")
        source = registry[sid]
        ensure_trainable(source, sid)
        revision = parse_revision(source, job.get("revision"))
        if revision != source.get("revision"):
            raise PipelineError(f"Revision mismatch for {sid}: registry={source.get('revision')} config={revision}")
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise PipelineError("Hugging Face datasets is required. Install with: pip install -e '.[data]'") from exc
    for job in sources_cfg:
        if not job.get("enabled", False) or job.get("stage", "stage1") != "stage1":
            print(f"SKIP {job.get('source_id')}: disabled or not Stage 1")
            continue
        sid = str(job["source_id"])
        if sid not in registry:
            raise PipelineError(f"Unknown source_id in acquisition config: {sid}")
        source = registry[sid]
        ensure_trainable(source, sid)
        revision = parse_revision(source, job.get("revision"))
        if revision != source.get("revision"):
            raise PipelineError(f"Revision mismatch for {sid}: registry={source.get('revision')} config={revision}")
        limit = int(job.get("max_records", 0))
        if limit <= 0:
            raise PipelineError(f"{sid}: max_records must be a positive pilot cap")
        config_name = job.get("configuration")
        split = str(job.get("split", "train"))
        dest = DATA_ROOT / "raw" / sid / "pilot.jsonl"
        partial = dest.with_suffix(".jsonl.partial")
        partial_meta = dest.with_suffix(".jsonl.partial.meta.json")
        if dest.exists():
            count = existing_jsonl_count(dest)
            checksum = sha256_file(dest)
            expected = next((row for row in read_jsonl(DATA_ROOT / "manifests/checksums.jsonl") if row.get("path") == str(dest.relative_to(DATA_ROOT))), None)
            provenance = next((row for row in read_jsonl(DATA_ROOT / "manifests/sources.jsonl") if row.get("source_id") == sid), None)
            if not expected or expected.get("sha256") != checksum:
                raise PipelineError(f"Completed file checksum is missing or mismatched for {dest}")
            if not provenance or provenance.get("revision") != revision or provenance.get("configuration") != config_name or provenance.get("split") != split:
                raise PipelineError(f"Completed file provenance does not match the pinned acquisition config for {sid}")
            report.append({"source_id": sid, "revision": revision, "configuration": config_name, "split": split, "records": count, "sha256": checksum, "resumed": False, "status": "already_complete"})
            print(f"EXISTS {sid}: {count} records at {dest}")
            continue
        if partial.exists():
            written = existing_jsonl_count(partial)
            if not partial_meta.exists():
                raise PipelineError(f"Partial file metadata is missing for {partial}; refusing unsafe resume")
            metadata = load_json(partial_meta)
            expected_metadata = {"source_id": sid, "revision": revision, "configuration": config_name, "split": split, "max_records": limit}
            if metadata != expected_metadata:
                raise PipelineError(f"Partial file metadata does not match pinned config for {sid}")
            if written > limit:
                raise PipelineError(f"Partial file has {written} rows, above pilot cap {limit}")
        else:
            written = 0
            partial.parent.mkdir(parents=True, exist_ok=True)
            dump_json(partial_meta, {"source_id": sid, "revision": revision, "configuration": config_name, "split": split, "max_records": limit})
        kwargs: dict[str, Any] = {"path": source["repository"], "split": split, "revision": revision, "streaming": True}
        if config_name:
            kwargs["name"] = config_name
        cache_value = os.path.expandvars(str(config.get("cache_dir", "")))
        if cache_value and "$" not in cache_value:
            kwargs["cache_dir"] = cache_value
        kwargs["download_config"] = DownloadConfig(resume_download=True)
        print(f"ACQUIRE {sid}: {source['repository']} config={config_name!r} split={split} revision={revision} max_records={limit}")
        stream = load_dataset(**kwargs)
        iterator = iter(stream)
        skipped = 0
        while skipped < written:
            try:
                next(iterator)
            except StopIteration:
                raise PipelineError(f"Partial file has {written} rows but source ended after {skipped}; refusing to resume")
            skipped += 1
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("a", encoding="utf-8", newline="\n") as output:
            count = written
            for row in iterator:
                if count >= limit:
                    break
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")) + "\n")
                count += 1
                if count % 1000 == 0:
                    output.flush()
                    os.fsync(output.fileno())
        if count == 0:
            raise PipelineError(f"No records acquired for {sid}")
        partial.replace(dest)
        partial_meta.unlink(missing_ok=True)
        checksum = sha256_file(dest)
        item = {"source_id": sid, "repository": source["repository"], "revision": revision, "configuration": config_name, "split": split, "records": count, "requested_cap": limit, "sha256": checksum, "acquired_at": now_utc(), "datasets_version": importlib.metadata.version("datasets"), "resumed": written > 0, "status": "complete"}
        sources_manifest = DATA_ROOT / "manifests/sources.jsonl"
        prior_sources = [row for row in read_jsonl(sources_manifest) if row.get("source_id") != sid]
        write_jsonl(sources_manifest, [*prior_sources, item])
        append_jsonl(DATA_ROOT / "manifests/checksums.jsonl", {"path": str(dest.relative_to(DATA_ROOT)), "sha256": checksum, "bytes": dest.stat().st_size, "source_id": sid, "revision": revision})
        report.append(item)
        print(f"DONE {sid}: {count} records sha256={checksum}")
    write_jsonl(DATA_ROOT / "manifests/acquisition_run.jsonl", report)
    return 0


def inspect_samples(limit: int) -> int:
    shown = 0
    for path in sorted((DATA_ROOT / "raw").glob("*/pilot.jsonl")):
        print(f"=== {path.relative_to(DATA_ROOT)} ===")
        for row in read_jsonl(path):
            print(json.dumps(row, ensure_ascii=False, indent=2, default=str)[:6000])
            shown += 1
            if shown >= limit:
                return 0
    print(f"Inspected {shown} sample(s); no more raw records found.")
    return 0


def messages_from_record(raw: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    opts = source.get("normalizer", {}) or {}
    field = opts.get("messages_field", "messages")
    values = raw.get(field)
    if values is None and isinstance(raw.get("conversation"), list):
        values = raw["conversation"]
    if not isinstance(values, list):
        raise ValueError(f"missing message list field {field!r}")
    messages = []
    allowed = {"system", "user", "assistant", "tool", "tool_result"}
    role_map = opts.get("role_map", {})
    for i, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"message {i} is not an object")
        role = str(role_map.get(item.get("role"), item.get("role", ""))).lower()
        if role not in allowed:
            raise ValueError(f"invalid role {role!r} at message {i}")
        if "content" not in item and "text" not in item:
            raise ValueError(f"message {i} has no content/text")
        content = item.get("content", item.get("text"))
        def has_unsupported_modality(value: Any) -> bool:
            if isinstance(value, dict):
                kind = str(value.get("type", "")).lower()
                if any(word in kind for word in ("image", "audio", "video", "frame")):
                    return True
                return any(has_unsupported_modality(child) for child in value.values())
            if isinstance(value, list):
                return any(has_unsupported_modality(child) for child in value)
            return False
        if has_unsupported_modality(content):
            raise ValueError(f"unsupported non-text modality in message {i}")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        msg = {"role": role, "content": content}
        for key in ("name", "tool_call_id", "tool_calls", "function_call"):
            if key in item:
                msg[key] = item[key]
        messages.append(msg)
    if not messages:
        raise ValueError("empty conversation")
    return messages


def trajectory_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    index = 0
    for message_index, message in enumerate(messages):
        role, content = message["role"], message.get("content")
        if role == "user":
            events.append({"sequence_index": index, "event_type": "user_goal", "content": content}); index += 1
        elif role == "assistant":
            calls = message.get("tool_calls")
            function_call = message.get("function_call")
            has_calls = bool(calls or function_call)
            future_calls = any(
                item.get("role") == "assistant" and (item.get("tool_calls") or item.get("function_call"))
                for item in messages[message_index + 1 :]
            )
            if content not in (None, ""):
                event_type = "final_answer" if not has_calls and any(e["event_type"] == "tool_call" for e in events) and not future_calls else "assistant_decision"
                events.append({"sequence_index": index, "event_type": event_type, "content": content}); index += 1
            if function_call and not calls:
                calls = [{"id": "", "function": function_call}]
            for call in calls or []:
                fn = call.get("function", call)
                name = fn.get("name")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if not isinstance(args, dict) or not name:
                    raise ValueError("malformed tool call (name and object arguments required)")
                events.append({"sequence_index": index, "event_type": "tool_call", "tool_name": str(name), "arguments": args, "call_id": str(call.get("id", "")), "raw_event": call}); index += 1
        elif role in {"tool", "tool_result"}:
            status = "error" if isinstance(content, dict) and content.get("error") else "success"
            events.append({"sequence_index": index, "event_type": "tool_result", "tool_name": message.get("name"), "call_id": message.get("tool_call_id", ""), "content": content, "status": status}); index += 1
    if any(e["event_type"] == "tool_call" for e in events):
        calls = [e for e in events if e["event_type"] == "tool_call"]
        results = [e for e in events if e["event_type"] == "tool_result"]
        if not results:
            raise ValueError("tool call has no tool result")
        call_ids = [e["call_id"] for e in calls if e.get("call_id")]
        result_ids = [e["call_id"] for e in results if e.get("call_id")]
        if call_ids and not set(call_ids).issubset(set(result_ids)):
            raise ValueError("tool call missing a matching result")
        if events[-1]["event_type"] not in {"final_answer", "failure"}:
            raise ValueError("truncated tool trajectory has no final answer or explicit failure")
    else:
        for event in events:
            if event["event_type"] == "assistant_decision":
                event["event_type"] = "final_answer"
    return events


def canonical_tool_calls_from_trajectory(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = record.get("events", [])
    results = [(i, event) for i, event in enumerate(events) if event.get("event_type") == "tool_result"]
    consumed: set[int] = set()
    output = []
    for event in events:
        if event.get("event_type") != "tool_call":
            continue
        call_id = str(event.get("call_id") or "")
        match = next(((i, result) for i, result in results if i not in consumed and result.get("call_id") == call_id), None) if call_id else None
        if match is None:
            match = next(((i, result) for i, result in results if i not in consumed and result.get("sequence_index", -1) > event.get("sequence_index", -1)), None)
        if match is None:
            raise ValueError(f"tool call at sequence {event.get('sequence_index')} has no result to export")
        result_index, result = match
        consumed.add(result_index)
        output_call_id = call_id or stable_id(record["source_id"], record["source_revision"], "tool_call", event.get("sequence_index"), event.get("raw_event", event))
        row = {
            "call_id": output_call_id,
            "call_id_origin": "source" if call_id else "derived_from_trajectory_order",
            "source_id": record["source_id"],
            "source_revision": record["source_revision"],
            "license_id": record["license_id"],
            "tool_name": event["tool_name"],
            "arguments": event["arguments"],
            "result": result.get("content"),
            "status": result.get("status", "unknown"),
            "sequence_index": event["sequence_index"],
            "provenance": record["provenance"],
            "raw_call": event.get("raw_event", {"tool_name": event["tool_name"], "arguments": event["arguments"]}),
            "raw_result": result.get("content"),
        }
        if call_id:
            row["source_call_id"] = call_id
        output.append(row)
    return output


def normalize_datasets(stage: str) -> int:
    if stage != "pilot":
        raise PipelineError("Only pilot normalization is supported")
    registry = registry_sources()
    normalized_path = DATA_ROOT / "processed/pilot/normalized.jsonl"
    rejected_path = DATA_ROOT / "processed/pilot/rejected.jsonl"
    filtering = load_yaml(FILTERING_DEFAULT) if FILTERING_DEFAULT.exists() else {}
    min_chars = int(filtering.get("minimum_characters", 12))
    max_chars = int(filtering.get("maximum_characters", 200000))
    min_words = int(filtering.get("minimum_words", 3))
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    tool_call_records: list[dict[str, Any]] = []
    for raw_path in sorted((DATA_ROOT / "raw").glob("*/pilot.jsonl")):
        sid = raw_path.parent.name
        if sid not in registry:
            raise PipelineError(f"Raw directory contains unknown source {sid}")
        source = registry[sid]
        ensure_trainable(source, sid)
        revision = parse_revision(source)
        norm = source.get("normalizer", {}) or {}
        kind = norm.get("kind", "document")
        for row_number, raw in enumerate(read_jsonl(raw_path)):
            orig = raw.get(norm.get("id_field", "id"), raw.get("id", row_number))
            provenance = {"original_record_id": str(orig), "source_split": source.get("split", "train"), "synthetic": bool(source.get("synthetic_or_human") == "synthetic"), "transformation_version": TRANSFORM_VERSION, "source_file": str(raw_path.relative_to(DATA_ROOT)), "source_checksum": sha256_file(raw_path)}
            common = {"sample_id": stable_id(sid, revision, provenance["source_split"], orig, raw), "source_id": sid, "source_revision": revision, "license_id": source["license_name"], "provenance": provenance, "raw_record": raw}
            try:
                row_tool_calls: list[dict[str, Any]] = []
                if kind == "document":
                    text_field = norm.get("text_field", "text")
                    text = raw.get(text_field)
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"missing or empty text field {text_field!r}")
                    title = raw.get(norm.get("title_field", "title"), "")
                    record = {**common, "category": "general", "title": title if isinstance(title, str) else str(title), "text": text}
                else:
                    messages = messages_from_record(raw, source)
                    if not any(message["role"] == "user" for message in messages):
                        raise ValueError("conversation has no user turn")
                    if (norm.get("structured_output") or raw.get("json_schema") or raw.get("response_schema")):
                        schema = raw.get("json_schema", raw.get("response_schema"))
                        for message in messages:
                            if message["role"] != "assistant" or not isinstance(message.get("content"), str):
                                continue
                            content = message["content"].strip()
                            if content.startswith("```json"):
                                content = content[len("```json"):].strip()
                                if content.endswith("```"):
                                    content = content[:-3].strip()
                            if content.startswith(("{", "[")):
                                try:
                                    parsed = json.loads(content)
                                except json.JSONDecodeError as exc:
                                    raise ValueError(f"malformed structured JSON output: {exc.msg}") from exc
                                if schema:
                                    try:
                                        import jsonschema
                                        jsonschema.validate(parsed, schema)
                                    except ImportError as exc:
                                        raise PipelineError("jsonschema is required to validate structured outputs") from exc
                                    except Exception as exc:
                                        raise ValueError("structured output does not match source schema") from exc
                    if kind == "trajectory" or any(m.get("tool_calls") or m.get("function_call") or m["role"] in {"tool", "tool_result"} for m in messages):
                        events = trajectory_from_messages(messages)
                        record = {"trajectory_id": common.pop("sample_id"), **common, "category": "tool_use", "events": events}
                        row_tool_calls = canonical_tool_calls_from_trajectory(record)
                    else:
                        category = norm.get("category", "instruction")
                        record = {**common, "category": category, "messages": messages}
                sample_text = record_text(record).strip()
                if len(sample_text) < min_chars:
                    raise ValueError(f"extremely short/low-information sample: {len(sample_text)} characters, minimum is {min_chars}")
                if len(sample_text) > max_chars:
                    raise ValueError(f"excessively long sample: {len(sample_text)} characters, maximum is {max_chars}")
                if len(sample_text.split()) < min_words:
                    raise ValueError(f"low-information sample: fewer than {min_words} whitespace-delimited words")
                if filtering.get("reject_high_confidence_sensitive_identifiers", True):
                    raw_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
                    indicators = sensitive_data_indicators(sample_text + "\n" + raw_text)
                    if indicators:
                        raise ValueError("high-confidence sensitive identifier patterns: " + ", ".join(indicators))
                normalized.append(record)
                tool_call_records.extend(row_tool_calls)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                rejected_row = {"source_id": sid, "source_revision": revision, "original_record_id": str(orig), "reason": str(exc), "raw_record_sha256": hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()}
                if filtering.get("include_rejected_raw", False):
                    rejected_row["raw_record"] = raw
                rejected.append(rejected_row)
    write_jsonl(normalized_path, normalized)
    write_jsonl(rejected_path, rejected)
    write_jsonl(DATA_ROOT / "processed/pilot/tool_calls.jsonl", tool_call_records)
    template_counts: Counter[str] = Counter()
    for record in normalized:
        text = record_text(record)
        template = re.sub(r"\b\d+\b", "<NUM>", " ".join(text.casefold().split()))
        template_counts[hashlib.sha256(template.encode("utf-8")).hexdigest()] += 1
    template_threshold = int(filtering.get("repeated_template_flag_threshold", 5))
    repeated_templates = sum(1 for count in template_counts.values() if count >= template_threshold)
    report = {"stage": stage, "input_records": len(normalized) + len(rejected), "valid_records": len(normalized), "canonical_tool_call_records": len(tool_call_records), "rejected_records": len(rejected), "rejection_reasons": dict(Counter(r["reason"] for r in rejected)), "repeated_template_groups_flagged": repeated_templates, "repeated_template_flag_threshold": template_threshold, "minimum_characters": min_chars, "maximum_characters": max_chars, "minimum_words": min_words, "transformation_version": TRANSFORM_VERSION}
    dump_json(DATA_ROOT / "reports/normalization.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def normalized_signature(record: dict[str, Any]) -> str:
    if isinstance(record.get("text"), str):
        text = " ".join(record["text"].casefold().split())
        basis = {"type": "document", "text": text}
    else:
        basis = {k: record.get(k) for k in ("category", "messages", "events") if k in record}
        basis = json.loads(json.dumps(basis, ensure_ascii=False, sort_keys=True))
        if "messages" in basis:
            for message in basis["messages"]:
                if isinstance(message.get("content"), str):
                    message["content"] = " ".join(message["content"].casefold().split())
    return hashlib.sha256(json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def word_shingles(record: dict[str, Any], width: int = 5) -> set[str]:
    words = re.findall(r"[\w]+", record_text(record).casefold())
    if len(words) < max(12, width + 2):
        return set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}


def shingle_bands(shingles: set[str], bands: int = 8) -> set[str]:
    values = sorted(hashlib.sha256(shingle.encode("utf-8")).hexdigest() for shingle in shingles)
    if len(values) < bands:
        return set()
    band_width = max(1, len(values) // bands)
    return {hashlib.sha256("|".join(values[i * band_width : (i + 1) * band_width]).encode("utf-8")).hexdigest() for i in range(bands)}


def deduplicate(stage: str) -> int:
    if stage != "pilot":
        raise PipelineError("Only pilot deduplication is supported")
    src = DATA_ROOT / "processed/pilot/normalized.jsonl"
    exact_seen: set[str] = set()
    id_seen: set[str] = set()
    kept, duplicates, removed = [], [], []
    filtering = load_yaml(FILTERING_DEFAULT) if FILTERING_DEFAULT.exists() else {}
    similarity_threshold = float(filtering.get("near_duplicate_similarity_threshold", 0.98))
    band_index: dict[str, list[int]] = defaultdict(list)
    shingle_sets: list[set[str]] = []
    input_count = 0
    for record in read_jsonl(src):
        input_count += 1
        rid = record_id(record)
        if rid in id_seen:
            row = {"sample_id": rid, "duplicate_type": "duplicate_sample_id", "kept_sample_id": rid}
            duplicates.append(row); removed.append(row)
            continue
        id_seen.add(rid)
        signature = normalized_signature(record)
        if signature in exact_seen:
            row = {"sample_id": rid, "duplicate_type": "normalized_exact", "signature": signature}
            duplicates.append(row); removed.append(row)
            continue
        exact_seen.add(signature)
        record["dedup_group"] = signature
        shingles = word_shingles(record)
        candidates = sorted({index for band in shingle_bands(shingles) for index in band_index.get(band, [])})
        matched_groups: set[str] = set()
        for candidate_index in candidates:
            candidate_shingles = shingle_sets[candidate_index]
            union = shingles | candidate_shingles
            similarity = len(shingles & candidate_shingles) / len(union) if union else 0.0
            if similarity >= similarity_threshold:
                matched_groups.add(str(kept[candidate_index]["dedup_group"]))
                duplicates.append({"sample_id": rid, "duplicate_type": "near_duplicate_flag", "similarity": round(similarity, 6), "candidate_sample_id": record_id(kept[candidate_index]), "action": "retained_for_review"})
        if matched_groups:
            group_id = min(matched_groups | {signature})
            for previous in kept:
                if previous.get("dedup_group") in matched_groups:
                    previous["dedup_group"] = group_id
            record["dedup_group"] = group_id
        for band in shingle_bands(shingles):
            band_index[band].append(len(kept))
        shingle_sets.append(shingles)
        kept.append(record)
    write_jsonl(DATA_ROOT / "processed/pilot/deduplicated.jsonl", kept)
    write_jsonl(DATA_ROOT / "reports/duplicates.jsonl", duplicates)
    near_flagged = sum(row["duplicate_type"] == "near_duplicate_flag" for row in duplicates)
    dump_json(DATA_ROOT / "reports/deduplication.json", {"input_records": input_count, "exact_duplicates_removed": len(removed), "near_duplicates_flagged": near_flagged, "near_duplicate_similarity_threshold": similarity_threshold, "near_duplicate_policy": "Flag only; retain until human review.", "final_records": len(kept)})
    print(f"deduplicated input={input_count} removed={len(removed)} near_flagged={near_flagged} final={len(kept)}")
    return 0


def build_splits(stage: str, seed: int = 42, validation_fraction: float = 0.05, test_fraction: float = 0.05) -> int:
    if stage != "pilot":
        raise PipelineError("Only pilot splitting is supported")
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise PipelineError("Validation/test fractions must be non-negative and sum to less than 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(DATA_ROOT / "processed/pilot/deduplicated.jsonl"):
        group_id = record.get("dedup_group") or record_id(record)
        groups[str(group_id)].append(record)
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    manifest = []
    for group_id, records in sorted(groups.items()):
        fraction = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) / 0xFFFFFFFFFFFFFFFF
        split = "test" if fraction < test_fraction else "validation" if fraction < test_fraction + validation_fraction else "train"
        buckets[split].extend(records)
        manifest.append({"group_id": group_id, "sample_ids": [r.get("sample_id", r.get("trajectory_id")) for r in records], "split": split, "seed": seed})
    for split, records in buckets.items():
        records.sort(key=lambda r: str(r.get("sample_id", r.get("trajectory_id"))))
        write_jsonl(DATA_ROOT / f"processed/pilot/{split}.jsonl", records)
    write_jsonl(DATA_ROOT / "manifests/splits.jsonl", manifest)
    dump_json(DATA_ROOT / "reports/splits.json", {"seed": seed, "validation_fraction": validation_fraction, "test_fraction": test_fraction, "records": {k: len(v) for k, v in buckets.items()}, "groups": len(groups)})
    return 0


class TokenCounter:
    def __init__(self, tokenizer_id: str, tokenizer_path: str | None = None):
        self.tokenizer_id = tokenizer_id
        self.backend = None
        self.kind = ""
        if tokenizer_id == "aya-edge-dev-byte-v1":
            self.kind = "utf8_byte_ids"
        elif tokenizer_id == "tokenizers-json":
            if not tokenizer_path:
                raise PipelineError("--tokenizer-path is required for tokenizers-json")
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise PipelineError("Install tokenizers with pip install -e '.[data]'") from exc
            self.backend = Tokenizer.from_file(tokenizer_path)
            self.kind = "tokenizers_json"
        else:
            raise PipelineError("Tokenizer must be explicitly selected as aya-edge-dev-byte-v1 or tokenizers-json with a pinned local tokenizer file")

    def count(self, text: str) -> int:
        if self.kind == "utf8_byte_ids":
            return len(text.encode("utf-8"))
        return len(self.backend.encode(text).ids)


def count_tokens(stage: str, tokenizer_id: str, tokenizer_path: str | None = None) -> int:
    if stage != "pilot":
        raise PipelineError("Only pilot token counting is supported")
    counter = TokenCounter(tokenizer_id, tokenizer_path)
    registry = registry_sources()
    rows = []
    for record in read_jsonl(DATA_ROOT / "processed/pilot/deduplicated.jsonl"):
        sid = record["source_id"]
        ensure_trainable(registry[sid], sid)
        text = record_text(record)
        rows.append({"sample_id": record.get("sample_id", record.get("trajectory_id")), "source_id": sid, "category": record.get("category"), "token_count": counter.count(text), "tokenizer_id": tokenizer_id, "tokenizer_kind": counter.kind, "production_tokenizer": tokenizer_id not in {"aya-edge-dev-byte-v1"}})
    write_jsonl(DATA_ROOT / "processed/pilot/token_counts.jsonl", rows)
    by_source: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[row["source_id"]] += row["token_count"]
        by_category[row["category"]] += row["token_count"]
    report = {"tokenizer_id": tokenizer_id, "tokenizer_kind": counter.kind, "production_tokenizer": tokenizer_id != "aya-edge-dev-byte-v1", "token_count_definition": "UTF-8 byte count equals token IDs only for this explicit development byte tokenizer; it is not a production-token estimate." if tokenizer_id == "aya-edge-dev-byte-v1" else "Counted by the explicitly supplied pinned tokenizer JSON.", "records": len(rows), "tokens_by_source": dict(by_source), "tokens_by_category": dict(by_category), "total_tokens": sum(by_source.values())}
    dump_json(DATA_ROOT / "reports/token_counts.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_mixture(config_path: Path, stage: str) -> int:
    if stage != "pilot":
        raise PipelineError("Only pilot mixture building is supported")
    config = load_yaml(config_path)
    tokenizer_cfg = config.get("tokenizer", {})
    expected_tokenizer = str(tokenizer_cfg.get("id", ""))
    count_report = load_json(DATA_ROOT / "reports/token_counts.json")
    if not expected_tokenizer or count_report.get("tokenizer_id") != expected_tokenizer:
        raise PipelineError("Mixture config tokenizer.id must match the explicit token-count run")
    if tokenizer_cfg.get("production_required", False) and not count_report.get("production_tokenizer", False):
        raise PipelineError("A production tokenizer is required; the development byte tokenizer cannot satisfy this gate")
    registry = registry_sources()
    train = list(read_jsonl(DATA_ROOT / "processed/pilot/train.jsonl"))
    counts = {row["sample_id"]: row for row in read_jsonl(DATA_ROOT / "processed/pilot/token_counts.jsonl")}
    groups = {str(k): float(v) for k, v in (config.get("category_targets", {}) or {}).items()}
    if not groups or abs(sum(groups.values()) - 1.0) > 1e-6:
        raise PipelineError("category_targets must be nonempty and sum to 1")
    for record in train:
        ensure_trainable(registry[record["source_id"]], record["source_id"])
    total_tokens = sum(counts[record_id(r)]["token_count"] for r in train)
    target_total = int(config.get("target_tokens_per_window", 0) or total_tokens)
    target_total = min(target_total, total_tokens) if total_tokens else 0
    selected: list[dict[str, Any]] = []
    report = {"tokenizer_id": expected_tokenizer, "production_tokenizer": count_report.get("production_tokenizer", False), "target_total_tokens": target_total, "actual_total_tokens": 0, "targets": {}, "actual": {}, "deviation": {}, "sources": {}}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        by_category[str(row.get("category"))].append(row)
    category_members = config.get("category_members", {}) or {}
    for category, proportion in sorted(groups.items()):
        member_categories = category_members.get(category, [category])
        candidates = [record for member in member_categories for record in by_category.get(member, [])]
        candidates.sort(key=lambda r: hashlib.sha256(f"{config.get('seed',42)}:{record_id(r)}".encode()).hexdigest())
        target = int(round(target_total * proportion))
        actual = 0
        for record in candidates:
            n = counts[record_id(record)]["token_count"]
            if actual >= target and actual > 0:
                break
            selected.append({**record, "token_count": n, "tokenizer_id": expected_tokenizer})
            actual += n
        report["targets"][category] = target
        report["actual"][category] = actual
        report["deviation"][category] = actual - target
        report["actual_total_tokens"] += actual
    by_source: dict[str, int] = defaultdict(int)
    for row in selected:
        by_source[row["source_id"]] += row["token_count"]
    report["sources"] = {
        source_id: {"tokens": token_count, "proportion": token_count / report["actual_total_tokens"] if report["actual_total_tokens"] else 0.0}
        for source_id, token_count in sorted(by_source.items())
    }
    write_jsonl(DATA_ROOT / "processed/pilot/mixture.jsonl", selected)
    dump_json(DATA_ROOT / "reports/mixture.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def load_schemas() -> dict[str, Any]:
    schema_dir = ROOT / "data/schemas"
    return {p.name.removesuffix(".schema.json"): load_json(p) for p in schema_dir.glob("*.schema.json")}


def validate_record(record: dict[str, Any], schemas: dict[str, Any]) -> list[str]:
    try:
        import jsonschema
        from referencing import Registry, Resource
    except ImportError as exc:
        raise PipelineError("jsonschema is required; install with pip install -e '.[dev]'") from exc
    if "tool_name" in record and "arguments" in record:
        key = "canonical_tool_call"
    elif "trajectory_id" in record:
        key = "canonical_trajectory"
    elif "messages" in record:
        key = "canonical_conversation"
    elif "text" in record:
        key = "canonical_document"
    else:
        return ["unknown canonical record type"]
    schema = schemas.get(key)
    if schema is None:
        return [f"missing schema {key}"]
    resources = [(item["$id"], Resource.from_contents(item)) for item in schemas.values() if "$id" in item]
    registry = Registry().with_resources(resources)
    validator = jsonschema.Draft202012Validator(schema, registry=registry)
    return [error.message for error in validator.iter_errors(record)]


def validate_dataset(stage: str) -> int:
    if stage != "pilot":
        raise PipelineError("Only pilot validation is supported")
    registry, schemas = registry_sources(), load_schemas()
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    seen = set()
    for record in read_jsonl(DATA_ROOT / "processed/pilot/deduplicated.jsonl"):
        counts["input"] += 1
        rid = record.get("sample_id", record.get("trajectory_id"))
        if rid in seen:
            errors.append({"sample_id": rid, "reason": "duplicate sample identifier"})
        seen.add(rid)
        sid = record.get("source_id")
        if sid not in registry:
            errors.append({"sample_id": rid, "reason": f"unknown source {sid!r}"})
        else:
            ok, reason = is_source_approved(registry[sid])
            if not ok:
                errors.append({"sample_id": rid, "reason": f"unapproved source {sid}: {reason}"})
        for error in validate_record(record, schemas):
            errors.append({"sample_id": rid, "reason": error})
        counts["valid"] += 1
    tool_call_path = DATA_ROOT / "processed/pilot/tool_calls.jsonl"
    tool_calls = list(read_jsonl(tool_call_path)) if tool_call_path.exists() else []
    counts["tool_call_records"] = len(tool_calls)
    for call in tool_calls:
        sid = call.get("source_id")
        if sid not in registry:
            errors.append({"call_id": call.get("call_id"), "reason": f"unknown tool-call source {sid!r}"})
        else:
            ok, reason = is_source_approved(registry[sid])
            if not ok:
                errors.append({"call_id": call.get("call_id"), "reason": f"unapproved tool-call source {sid}: {reason}"})
        for error in validate_record(call, schemas):
            errors.append({"call_id": call.get("call_id"), "reason": error})
    if counts["input"] == 0:
        errors.append({"reason": "pilot training corpus is empty"})
    rejected = list(read_jsonl(DATA_ROOT / "processed/pilot/rejected.jsonl"))
    counts["rejected"] = len(rejected)
    split_records = {split: list(read_jsonl(DATA_ROOT / f"processed/pilot/{split}.jsonl")) for split in ("train", "validation", "test")}
    id_sets = {split: {r.get("sample_id", r.get("trajectory_id")) for r in records} for split, records in split_records.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = id_sets[a] & id_sets[b]
        if overlap:
            errors.append({"reason": f"split ID leakage {a}/{b}: {len(overlap)}"})
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in split_records.items():
        for record in records:
            group_splits[str(record.get("dedup_group", record.get("sample_id")))].add(split)
    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append({"reason": f"dedup group {group} crosses splits"})
    token_path = DATA_ROOT / "processed/pilot/token_counts.jsonl"
    tokens = list(read_jsonl(token_path)) if token_path.exists() else []
    if not tokens:
        errors.append({"reason": "token counts are missing; no tokenizer-aware acceptance is possible"})
    mixture_path = DATA_ROOT / "reports/mixture.json"
    mixture = load_json(mixture_path) if mixture_path.exists() else None
    if not mixture or not mixture.get("actual_total_tokens"):
        errors.append({"reason": "token-count-aware mixture report is missing or empty"})
    holdout = load_yaml(ROOT / "configs/data/evaluation_holdout.yaml")
    if any(r.get("source_id") in set(holdout.get("protected_source_ids", [])) for r in split_records["train"]):
        errors.append({"reason": "protected evaluation source appears in training"})
    report = {"stage": stage, "counts": dict(counts), "split_counts": {k: len(v) for k, v in split_records.items()}, "token_counts_present": bool(tokens), "errors": errors, "status": "PASS" if not errors else "FAIL", "known_limitation": "Zero contamination can only be claimed for protected sources whose records were technically inspectable."}
    dump_json(DATA_ROOT / "reports/validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


def generate_report(stage: str) -> int:
    report_dir = DATA_ROOT / "reports"
    inputs = {}
    for name in ("normalization", "deduplication", "splits", "token_counts", "mixture", "validation"):
        path = report_dir / f"{name}.json"
        if path.exists():
            inputs[name] = load_json(path)
    out = {"stage": stage, "generated_at": now_utc(), "pipeline_version": TRANSFORM_VERSION, "environment": {"python": sys.version.split()[0], "datasets": _version_or_none("datasets"), "huggingface_hub": _version_or_none("huggingface-hub"), "pyarrow": _version_or_none("pyarrow"), "pandas": _version_or_none("pandas")}, "reports": inputs, "acquired_sources": list(read_jsonl(DATA_ROOT / "manifests/sources.jsonl")) if (DATA_ROOT / "manifests/sources.jsonl").exists() else []}
    dump_json(report_dir / f"{stage}_pipeline_report.json", out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _version_or_none(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_acceptance_tests(stage: str) -> int:
    if stage != "pilot":
        raise PipelineError("Only pilot acceptance tests are supported")
    command = [sys.executable, "-m", "pytest", "tests/data", "-q"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def run_cli(command_name: str) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    if command_name == "inspect_sources":
        parser.add_argument("--config", type=Path, default=REGISTRY_DEFAULT)
    elif command_name == "download_datasets":
        parser.add_argument("--config", type=Path, default=ACQUISITION_DEFAULT)
        parser.add_argument("--stage", choices=["pilot"], default="pilot")
    elif command_name == "verify_licenses":
        parser.add_argument("--config", type=Path, default=REGISTRY_DEFAULT)
        parser.add_argument("--strict", action="store_true")
    elif command_name == "inspect_samples":
        parser.add_argument("--limit", type=int, default=20)
    elif command_name in {"normalize_datasets", "deduplicate", "build_splits", "count_tokens", "build_mixture", "validate_dataset", "generate_report", "run_acceptance_tests"}:
        parser.add_argument("--stage", choices=["pilot"], default="pilot")
        if command_name == "build_splits":
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--validation-fraction", type=float, default=0.05)
            parser.add_argument("--test-fraction", type=float, default=0.05)
        if command_name == "count_tokens":
            parser.add_argument("--tokenizer-id")
            parser.add_argument("--tokenizer-path")
        if command_name == "build_mixture":
            parser.add_argument("--config", type=Path, default=MIXTURE_DEFAULT)
        if command_name == "run_acceptance_tests":
            pass
    args = parser.parse_args()
    try:
        if command_name == "inspect_sources":
            return inspect_sources(args.config)
        if command_name == "download_datasets":
            return download_datasets(args.config, args.stage)
        if command_name == "verify_licenses":
            return verify_licenses(args.config, args.strict)
        if command_name == "inspect_samples":
            if args.limit <= 0:
                raise PipelineError("--limit must be positive")
            return inspect_samples(args.limit)
        if command_name == "normalize_datasets":
            return normalize_datasets(args.stage)
        if command_name == "deduplicate":
            return deduplicate(args.stage)
        if command_name == "build_splits":
            return build_splits(args.stage, args.seed, args.validation_fraction, args.test_fraction)
        if command_name == "count_tokens":
            tokenizer_id = args.tokenizer_id or str(load_yaml(MIXTURE_DEFAULT).get("tokenizer", {}).get("id", ""))
            if not tokenizer_id:
                raise PipelineError("Set tokenizer.id in configs/data/mixture.yaml or pass --tokenizer-id")
            tokenizer_path = args.tokenizer_path or load_yaml(MIXTURE_DEFAULT).get("tokenizer", {}).get("path")
            return count_tokens(args.stage, tokenizer_id, tokenizer_path)
        if command_name == "build_mixture":
            return build_mixture(args.config, args.stage)
        if command_name == "validate_dataset":
            return validate_dataset(args.stage)
        if command_name == "generate_report":
            return generate_report(args.stage)
        if command_name == "run_acceptance_tests":
            return run_acceptance_tests(args.stage)
        raise PipelineError(f"Unknown command {command_name}")
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run_cli(Path(__file__).stem))
