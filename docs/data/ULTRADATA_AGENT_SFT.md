# UltraData-SFT-Agent-2609: bounded AYA Edge pilot

## Purpose

Use a small, English-focused slice for AYA Edge agent supervised fine-tuning (SFT), separate from language pretraining. The proposed first pilot is 25,000 trajectories:

| Subset | Limit |
|---|---:|
| Tool-Use | 12,000 |
| Search-Agent (English domain) | 5,000 |
| General-Agent | 8,000 |
| **Total** | **25,000** |

The limits are caps, not a promise that every subset will yield that many valid English records. The preparation script fails rather than silently producing an underfilled dataset.

## Declared license and important limits

The dataset card declares Apache License 2.0 for the dataset project. It also states that the collection aggregates upstream datasets, public repositories, web content, and model-generated trajectories; each upstream license continues to apply. It specifically warns that public access to course pages is not permission for training or redistribution, and prohibits unauthorized unchanged reposting/re-hosting of the dataset artifacts.

Therefore:
- Treat Apache-2.0 as the dataset-level declaration, not a blanket license for every embedded source.
- Keep the pilot local and personal-research-only unless upstream rights are reviewed for the intended broader use.
- Do not mirror or redistribute the raw dataset or unchanged artifacts.
- Preserve source, subset, domain, UUID, tools, revision, and license metadata.
- Do not claim row-level rights clearance: the card does not provide a complete per-record rights manifest.
- This engineering note is not legal advice or a legal opinion.

Official card: https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609

## Prepare the pilot

Install the optional data dependencies:

```bash
python -m pip install -e ".[data]"
```

After reviewing the upstream caveats above, run:

```bash
python scripts/prepare_ultradata_agent.py \
  --tool-use 12000 \
  --search-agent 5000 \
  --general-agent 8000 \
  --accept-upstream-terms \
  --accept-residual-upstream-rights-risk
```

The script streams each selected configuration, keeps message/tool structures intact, filters non-English/mixed CJK trajectories and oversized examples, and writes deterministic train/validation/test JSONL splits plus a manifest. It does not execute tools or prove task success. Search-Agent is restricted to the English domain label when present.

The script records the resolved Hugging Face revision. Verify the manifest before using the resulting data. Keep Code-Agent out of the initial AYA scope.

## Training integration requirements

Before SFT:
1. Convert the messages to AYA's exact chat/tool-call template without flattening away tool boundaries.
2. Validate assistant tool-call IDs against tool responses and preserve the frozen tool schemas in context.
3. Map only supported tool definitions to AYA's runtime registry; virtual tools must not be represented as executable capabilities.
4. Review failed, timeout, and recovery trajectories; do not treat all retained traces as successful demonstrations.
5. Apply loss masks to low-quality/tool-output turns according to AYA's SFT policy.
6. Run PII/secret, prompt-injection, unsafe-operation, duplicate, and benchmark-contamination checks.
7. Evaluate on held-out tasks before expanding beyond the pilot.

This dataset is an SFT source, not a substitute for general-language pretraining.
