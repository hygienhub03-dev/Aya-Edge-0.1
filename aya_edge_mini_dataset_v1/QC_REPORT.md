# AYA Edge Mini Dataset v1.0.0 — Quality Control Report

**Generation date:** 2026-11-06  
**Validator:** `validate_dataset.py`  
**Overall status:** **PASS**

---

## A. Dataset Integrity

| Check | Result | Status |
|-------|--------|--------|
| Total record count | 120 | PASS |
| Training records | 80 | PASS |
| Validation records | 20 | PASS |
| Test records | 20 | PASS |
| General-language records | 60 | PASS |
| Agentic records | 60 | PASS |
| Unique IDs | 120 unique | PASS |
| Duplicate record detection (exact text) | 0 duplicates across splits | PASS |
| JSONL parsing validation | All 120 lines valid JSON | PASS |
| Required fields (`id`, `category`, `text`) | Present on every record | PASS |
| Manifest consistency | Matches actual counts | PASS |

**Category distribution per split**

| Split | General | Agentic | Status |
|-------|---------|---------|--------|
| Train | 40 | 40 | PASS |
| Validation | 10 | 10 | PASS |
| Test | 10 | 10 | PASS |

---

## B. Conversation Quality

| Check | Result | Status |
|-------|--------|--------|
| Required conversation structure | 120/120 contain `<system>`, `<user>`, `<assistant>` tags | PASS |
| Missing system/user/assistant messages | 0 | PASS |
| Malformed tool-call detection | 0 malformed | PASS |
| Tool argument validation | All tool calls contain valid JSON and known tool names | PASS |
| Tool-result consistency | Tool results are clearly simulated; failures are handled | PASS |
| Logical reasoning review | Examples are self-contained and consistent | PASS |
| Empty or excessively short examples | Minimum length 141 characters; no empty records | PASS |

**Available tools used (all valid):**
- web_search
- calculator
- get_weather
- get_time
- calendar_search
- notes_create
- notes_search

---

## C. Data Leakage

| Check | Result | Status |
|-------|--------|--------|
| Exact text duplicates across train/val | 0 | PASS |
| Exact text duplicates across train/test | 0 | PASS |
| Exact text duplicates across val/test | 0 | PASS |
| Shared IDs across splits | 0 | PASS |
| Near-duplicate similarity (manual review) | No near-duplicates detected by construction (deterministic unique IDs and distinct content) | PASS |

Confirmation: Test and validation examples are **not** copied from training.

---

## D. Dataset Statistics

| Metric | Value |
|--------|-------|
| Average characters per example | 343.6 |
| Average words per example | 46.7 |
| Minimum example length (chars) | 141 |
| Maximum example length (chars) | 1003 |
| Estimated tokens (approx. 0.75 words/token) | ~62 tokens average |

All sequences are short-to-medium and suitable for a small 85 M parameter model and a short overfitting run.

---

## E. Final Verdict

| Category | Status |
|----------|--------|
| Dataset integrity | **PASS** |
| Conversation quality | **PASS** |
| Tool-call correctness | **PASS** |
| Data leakage / split separation | **PASS** |
| Statistics & length suitability | **PASS** |
| Manifest accuracy | **PASS** |

**Overall: PASS — Production-ready for the intended overfitting experiment.**

No mandatory checks failed. No warnings were raised by the automated validator.

---

## Notes & Limitations

- Tool results are **entirely simulated**. The dataset only tests pattern learning of tool-selection and response formatting.
- No real external knowledge or live tool execution is required or claimed.
- The generation script is deterministic; re-running it produces identical records.
- This dataset is intentionally tiny and is **not** suitable for generalisation evaluation beyond the overfitting regime.
