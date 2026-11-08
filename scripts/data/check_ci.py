#!/usr/bin/env python3
"""Validate PR data configs/artifacts and reject changed dataset payloads or secrets."""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DATASET_SUFFIXES = {
    ".arrow", ".bin", ".bz2", ".gz", ".parquet", ".pkl", ".pickle", ".pt",
    ".pth", ".safetensors", ".tar", ".tgz", ".zip", ".zst",
}
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    "OpenAI-style API key": re.compile(rb"\bsk-[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "Slack token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
}
CREDENTIAL_FILENAMES = {".env", "id_rsa", "id_ed25519", "credentials.json", "secrets.json"}


def changed_paths(base_ref: str) -> list[Path]:
    commands = [
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", "-z", f"{base_ref}...HEAD"],
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", "-z"],
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    ]
    paths: set[Path] = set()
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE)
        paths.update(Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item)
    return sorted(paths)


def check_configs_and_schemas() -> list[str]:
    errors: list[str] = []
    try:
        import yaml
        import jsonschema
    except ImportError as exc:
        return [f"CI validation dependency is missing: {exc}"]

    for path in sorted((ROOT / "configs/data").glob("*.yaml")):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                errors.append(f"{path.relative_to(ROOT)} must contain a YAML mapping")
        except Exception as exc:
            errors.append(f"Invalid YAML {path.relative_to(ROOT)}: {exc}")

    for path in sorted((ROOT / "data/schemas").glob("*.schema.json")):
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator.check_schema(schema)
        except Exception as exc:
            errors.append(f"Invalid JSON Schema {path.relative_to(ROOT)}: {exc}")

    notebook_path = ROOT / "notebooks/data/AYA_Edge_Dataset_Inspection.ipynb"
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            errors.append(f"Invalid notebook structure: {notebook_path.relative_to(ROOT)}")
        for number, cell in enumerate(notebook.get("cells", []), 1):
            if cell.get("cell_type") == "code":
                source = cell.get("source", "")
                code = "".join(source) if isinstance(source, list) else str(source)
                try:
                    ast.parse(code, filename=f"{notebook_path.name}:cell-{number}")
                except SyntaxError as exc:
                    errors.append(f"Notebook code syntax error in cell {number}: {exc}")
    except Exception as exc:
        errors.append(f"Invalid notebook JSON {notebook_path.relative_to(ROOT)}: {exc}")

    registry_path = ROOT / "configs/data/source_registry.yaml"
    acquisition_path = ROOT / "configs/data/acquisition.yaml"
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        acquisition = yaml.safe_load(acquisition_path.read_text(encoding="utf-8"))
        statuses = {row.get("source_id"): row.get("approval_status") for row in registry.get("sources", [])}
        enabled = [row.get("source_id") for row in acquisition.get("sources", []) if row.get("enabled") is True]
        if enabled:
            errors.append(f"External acquisition must remain disabled in this PR; enabled sources: {enabled}")
        blocked = [sid for sid, status in statuses.items() if status == "blocked_pending_review"]
        if len(blocked) != 8:
            errors.append(f"Expected all eight reviewed acquisition candidates to remain blocked; found {len(blocked)}")
    except Exception as exc:
        errors.append(f"Could not verify fail-closed licensing configuration: {exc}")
    return errors


def check_changed_files(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        path = ROOT / relative
        parts = {part.lower() for part in relative.parts}
        if "raw" in parts and "data" in parts or "cache" in parts and "data" in parts:
            errors.append(f"Dataset raw/cache path is not allowed in PR: {relative}")
        if relative.as_posix().startswith("data/processed/") and relative.name != ".gitkeep":
            errors.append(f"Processed corpus payload is not allowed in PR: {relative}")
        if path.name.lower() in CREDENTIAL_FILENAMES or path.suffix.lower() in {".pem", ".p12", ".pfx", ".key"}:
            errors.append(f"Credential-like file is not allowed in PR: {relative}")
        if path.exists():
            if path.is_symlink():
                errors.append(f"Changed symlink is not allowed in data-validation PR: {relative}")
                continue
            size = path.stat().st_size
            if size > 10 * 1024 * 1024:
                errors.append(f"Changed file exceeds 10 MiB and may contain a payload: {relative} ({size} bytes)")
            if path.suffix.lower() in DATASET_SUFFIXES:
                errors.append(f"Dataset/binary payload file is not allowed in PR: {relative}")
            if path.suffix.lower() == ".jsonl" and size > 5 * 1024 * 1024:
                errors.append(f"Large JSONL payload is not allowed in PR: {relative} ({size} bytes)")
            try:
                content = path.read_bytes()
            except OSError as exc:
                errors.append(f"Could not scan changed file {relative}: {exc}")
                continue
            for description, pattern in SECRET_PATTERNS.items():
                if pattern.search(content):
                    errors.append(f"Possible {description} found in changed file: {relative}")
        elif relative.as_posix().startswith(("data/raw/", "data/cache/", "data/processed/")):
            # Deleted payloads are safe; only current added/modified contents can be committed.
            continue
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default=os.environ.get("BASE_REF", "origin/main"))
    args = parser.parse_args()
    errors = check_configs_and_schemas()
    try:
        paths = changed_paths(args.base_ref)
    except subprocess.CalledProcessError as exc:
        errors.append(f"Could not determine PR changes against {args.base_ref}: {exc}")
        paths = []
    errors.extend(check_changed_files(paths))
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: YAML, JSON Schemas, notebook code, disabled-source policy and hygiene checks; scanned {len(paths)} changed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
