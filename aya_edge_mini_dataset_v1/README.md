# AYA Edge Mini Dataset v1.0.0

Synthetic dataset for overfitting experiments with **AYA Edge 0.1**, an experimental 85-million-parameter sparse Mixture-of-Experts causal language model.

## Purpose

This is a **small-dataset overfitting experiment**, not a production training corpus. It tests whether the model can learn:

- Basic language patterns and instruction following
- Simulated agentic tool-selection and tool-use patterns

All tool results are **simulated**. The dataset does not prove real tool execution capability.

## Dataset Statistics

| Split       | General Language | Agentic | Total |
|-------------|------------------|---------|-------|
| Train       | 40               | 40      | 80    |
| Validation  | 10               | 10      | 20    |
| Test        | 10               | 10      | 20    |
| **Total**   | **60**           | **60**  | **120** |

## File Layout

```
aya_edge_mini_dataset_v1/
├── train.jsonl              # 80 training examples
├── validation.jsonl         # 20 validation examples
├── test.jsonl               # 20 test examples
├── dataset_manifest.json    # Metadata and statistics
├── generate_dataset.py      # Reproducible generation script
├── validate_dataset.py      # Independent validation script
├── QC_REPORT.md             # Full quality-control report
└── README.md                # This file
```

## Record Format

Each line in the JSONL files is a single JSON object:

```json
{
  "id": "gen_001",
  "category": "general_language",
  "text": "<system>\nYou are AYA, a helpful language and task assistant.\n</system>\n<user>\n...\n</user>\n<assistant>\n...\n</assistant>"
}
```

Categories are either `"general_language"` or `"agentic"`.

## Simulated Tools

The agentic examples use the following fictional tools:

- `web_search(query)`
- `calculator(expression)`
- `get_weather(location)`
- `get_time(timezone)`
- `calendar_search(date)`
- `notes_create(title, content)`
- `notes_search(query)`

Tool calls appear in the exact format:

```
<tool_call>
{"name":"web_search","arguments":{"query":"example"}}
</tool_call>
```

Tool results appear as:

```
<tool_result>
{"name":"web_search","result":"Simulated result text."}
</tool_result>
```

## Usage

```python
import json

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train = load_jsonl("train.jsonl")
val = load_jsonl("validation.jsonl")
test = load_jsonl("test.jsonl")
```

## Regeneration

```bash
python generate_dataset.py
python validate_dataset.py
```

## Experimental Notes

- Designed for a short (100–300 step) overfitting run on an NVIDIA Tesla T4.
- Sequences are short-to-medium length (average ~344 characters / ~47 words).
- No coding, programming, or advanced mathematics examples.
- No intentional duplicates or split leakage.
- All examples are self-contained; no external knowledge is required beyond the supplied context.

## License / Attribution

Generated for internal AYA Edge 0.1 research. Use at your own discretion for experimental purposes only.
