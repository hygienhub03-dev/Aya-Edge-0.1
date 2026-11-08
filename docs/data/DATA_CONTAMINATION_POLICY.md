# Evaluation contamination and holdout policy

## BFCL V4 is evaluation-only

Berkeley Function Calling Leaderboard V4 is pinned to evaluator commit `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000` in `source_registry.yaml` and listed in `evaluation_holdout.yaml`. Its categories, prompts, tool/function schemas, expected actions/answers, live examples, and any derived or paraphrased forms must never enter Stage 1, SFT, or RL training. The public benchmark repository is not acquired by the data pipeline. The card/repository license does not change this separation; benchmark integrity takes precedence.

The two Nemotron RL Agentic pivot datasets are later-RL candidates, not Stage 1 corpus sources. Their current acquisition jobs are disabled. Any later approval must create separate, version-pinned data partitions and cannot repurpose the BFCL test set as training data. Instruction/agent synthetic datasets are not independent held-out evaluation evidence.

## Implemented controls

The pipeline has an explicit protected-source registry and rejects protected IDs from training validation. Canonicalized records are deduplicated globally before split construction; exact duplicates are removed, and conservative near duplicates are flagged and grouped so they cannot cross train/validation/test boundaries. Splits are deterministic and hashed by group. Validation checks ID overlap, dedup-group overlap, protected BFCL sources in training, canonical schemas, source approval, explicit token counts, and a non-empty mixture report. A synthetic acceptance test confirms BFCL records in the training split fail validation.

The code cannot detect every benchmark paraphrase, web crawl copy, code/API description overlap, or latent prompt/answer contamination. A source-ID block is effective only when the candidate rows retain their source identity and the source registry lists the protected identifier. If expected actions, examples, or benchmark-derived content are not inspectable, the report must state that contamination could not be ruled out.

## Current audit result

| Check | Observed |
|---|---:|
| BFCL records acquired | 0 |
| Candidate training rows compared against BFCL | 0 |
| External source payloads inspected | 0 |
| Protected-source leakage in synthetic fixture test | Rejected as expected |
| Claim of zero real-data contamination | **Not made** |

There is no external training corpus yet, so no record-level contamination comparison has been performed. Public datasets such as FineWeb and Wikipedia may contain benchmark-related text or API descriptions; synthetic agent/instruction sources may also resemble evaluation examples. Their future use requires source-level overlap review and, where legally and technically possible, exact/normalized/near-duplicate comparisons against the pinned BFCL test records under controlled access. Do not copy or expose BFCL expected answers in reports or training manifests.
