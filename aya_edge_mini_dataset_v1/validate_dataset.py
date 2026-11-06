#!/usr/bin/env python3
"""
Independent validation script for AYA Edge Mini Dataset v1.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(__file__).parent
REQUIRED_FILES = ["train.jsonl", "validation.jsonl", "test.jsonl", "dataset_manifest.json"]
AVAILABLE_TOOLS = {
    "web_search", "calculator", "get_weather", "get_time",
    "calendar_search", "notes_create", "notes_search"
}
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_RESULT_RE = re.compile(r"<tool_result>\s*(\{.*?\})\s*</tool_result>", re.DOTALL)
REQUIRED_TAGS = ["<system>", "</system>", "<user>", "</user>", "<assistant>", "</assistant>"]


def load_jsonl(path: Path):
    records = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"FAIL: Invalid JSON in {path.name} line {i}: {e}")
                sys.exit(1)
    return records


def validate():
    results = []
    all_records = []
    split_records = {}

    # 1. File presence
    for fname in REQUIRED_FILES:
        p = DATASET_DIR / fname
        if not p.exists():
            results.append(("File presence", fname, "FAIL", "Missing file"))
            print(f"FAIL: Missing {fname}")
            return False
        results.append(("File presence", fname, "PASS", "Present"))

    # 2. Load and count
    for split in ["train", "validation", "test"]:
        recs = load_jsonl(DATASET_DIR / f"{split}.jsonl")
        split_records[split] = recs
        all_records.extend(recs)

    expected = {"train": 80, "validation": 20, "test": 20}
    for split, exp in expected.items():
        actual = len(split_records[split])
        status = "PASS" if actual == exp else "FAIL"
        results.append((f"Split count: {split}", str(actual), status, f"Expected {exp}"))
        if status == "FAIL":
            print(f"FAIL: {split} has {actual} records, expected {exp}")

    total = len(all_records)
    results.append(("Total records", str(total), "PASS" if total == 120 else "FAIL", "Expected 120"))

    # 3. Unique IDs
    ids = [r["id"] for r in all_records]
    unique_ids = set(ids)
    results.append(("Unique IDs", str(len(unique_ids)), "PASS" if len(unique_ids) == 120 else "FAIL", "Expected 120 unique"))

    # 4. Category distribution overall
    cats = Counter(r["category"] for r in all_records)
    results.append(("General language total", str(cats.get("general_language", 0)),
                    "PASS" if cats.get("general_language") == 60 else "FAIL", "Expected 60"))
    results.append(("Agentic total", str(cats.get("agentic", 0)),
                    "PASS" if cats.get("agentic") == 60 else "FAIL", "Expected 60"))

    # 5. Per-split category balance
    for split, recs in split_records.items():
        c = Counter(r["category"] for r in recs)
        exp_g = 40 if split == "train" else 10
        exp_a = 40 if split == "train" else 10
        g_ok = c.get("general_language", 0) == exp_g
        a_ok = c.get("agentic", 0) == exp_a
        results.append((f"{split} general", str(c.get("general_language", 0)),
                        "PASS" if g_ok else "FAIL", f"Expected {exp_g}"))
        results.append((f"{split} agentic", str(c.get("agentic", 0)),
                        "PASS" if a_ok else "FAIL", f"Expected {exp_a}"))

    # 6. Conversation structure
    struct_ok = 0
    for r in all_records:
        text = r.get("text", "")
        if all(tag in text for tag in REQUIRED_TAGS):
            struct_ok += 1
    results.append(("Conversation structure", f"{struct_ok}/120",
                    "PASS" if struct_ok == 120 else "FAIL", "All must have system/user/assistant tags"))

    # 7. Required fields
    field_ok = all("id" in r and "category" in r and "text" in r for r in all_records)
    results.append(("Required fields", "present", "PASS" if field_ok else "FAIL", "id, category, text"))

    # 8. Tool call validation (agentic only)
    tool_issues = 0
    for r in all_records:
        if r["category"] != "agentic":
            continue
        text = r["text"]
        for m in TOOL_CALL_RE.finditer(text):
            try:
                call = json.loads(m.group(1))
                name = call.get("name")
                if name not in AVAILABLE_TOOLS:
                    tool_issues += 1
                    print(f"WARNING: Unknown tool {name} in {r['id']}")
                if "arguments" not in call:
                    tool_issues += 1
            except json.JSONDecodeError:
                tool_issues += 1
                print(f"WARNING: Malformed tool call JSON in {r['id']}")
    results.append(("Tool call validity", str(tool_issues),
                    "PASS" if tool_issues == 0 else "WARNING", "Issues found"))

    # 9. Cross-split duplicate check (exact text)
    train_texts = {r["text"] for r in split_records["train"]}
    val_texts = {r["text"] for r in split_records["validation"]}
    test_texts = {r["text"] for r in split_records["test"]}
    leakage_val = len(train_texts & val_texts)
    leakage_test = len(train_texts & test_texts)
    leakage_vt = len(val_texts & test_texts)
    results.append(("Train-Val leakage", str(leakage_val), "PASS" if leakage_val == 0 else "FAIL", "Exact text overlap"))
    results.append(("Train-Test leakage", str(leakage_test), "PASS" if leakage_test == 0 else "FAIL", "Exact text overlap"))
    results.append(("Val-Test leakage", str(leakage_vt), "PASS" if leakage_vt == 0 else "FAIL", "Exact text overlap"))

    # 10. ID uniqueness across splits
    train_ids = {r["id"] for r in split_records["train"]}
    val_ids = {r["id"] for r in split_records["validation"]}
    test_ids = {r["id"] for r in split_records["test"]}
    id_overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    results.append(("ID split overlap", str(len(id_overlap)), "PASS" if len(id_overlap) == 0 else "FAIL", "No shared IDs"))

    # 11. Length statistics
    lengths = [len(r["text"]) for r in all_records]
    words = [len(r["text"].split()) for r in all_records]
    results.append(("Avg characters", f"{sum(lengths)/len(lengths):.1f}", "PASS", ""))
    results.append(("Avg words", f"{sum(words)/len(words):.1f}", "PASS", ""))
    results.append(("Min / Max chars", f"{min(lengths)} / {max(lengths)}", "PASS", ""))

    # 12. Manifest consistency
    with (DATASET_DIR / "dataset_manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    m_ok = (manifest.get("total_examples") == 120 and
            manifest.get("splits", {}).get("train") == 80)
    results.append(("Manifest consistency", "ok", "PASS" if m_ok else "FAIL", ""))

    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    fails = 0
    warnings = 0
    for check, value, status, note in results:
        print(f"{status:8} | {check:30} | {value:15} | {note}")
        if status == "FAIL":
            fails += 1
        elif status == "WARNING":
            warnings += 1

    print(f"\nTotal checks: {len(results)} | FAIL: {fails} | WARNING: {warnings}")
    if fails > 0:
        print("OVERALL: FAIL")
        return False
    print("OVERALL: PASS")
    return True


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
