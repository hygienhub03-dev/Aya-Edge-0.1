# AYA Edge dataset research

Reviewed 2026-11-07 against official dataset cards, repository APIs/files, Wikimedia dump/legal/format documentation, and official Berkeley/Gorilla BFCL sources. Revisions below are immutable repository commits except the dated Wikimedia snapshot. No external dataset payloads were downloaded. The repository’s locked identity remains agent-first, balanced reasoning, and hybrid knowledge; the source mix therefore needs general text and knowledge as well as instruction, planning, tool-selection, tool-result interpretation, recovery, and structured-output examples.

## FineWeb — general language and knowledge

FineWeb was selected as a candidate source for broad language, explanations, and factual/educational text. Its current Hugging Face API lists `sample-10BT`, `sample-100BT`, and `sample-350BT` configurations, individual Common Crawl dump configurations, and a default configuration with one `train` split. The viewer reports 25,886,364,489 rows for `default`. The card’s narrative describes 96 crawls from 2013 through April 2024 and about 15 trillion tokens, while current repository metadata includes 2025 crawl configurations. The inventory/size description is therefore version-misaligned and needs validation at the pinned revision before any scale-up. `sample-10BT` denotes a 10-billion-token source configuration and is not itself a small corpus; the acquisition file proposes streaming at most 100 records from it if the license gate is later cleared. [1] [2]

The actual record format is web text with metadata: `text`, `id`, `dump`, `url`, `date`, `file_path`, `language`, `language_score`, and `token_count`. It is not dialogue, SFT, tool-use, or RL data. The preserved URL, crawl, path, date, language, and token metadata help provenance and filtering, but the card does not supply a per-page license manifest. [1]

Pinned revision: `9bb295ddab0e05d785b879661af7260fed5140fc`; configuration `sample-10BT`; split `train`. License evidence: ODC-By 1.0 for database rights, with Common Crawl Terms of Use also applying. ODC-By permits commercial use of covered database rights subject to attribution and notices, but does not license independent rights in individual webpage contents. Common Crawl also defers to source-owner terms and puts content legality assessment on users. FineWeb is a conditional Stage 1 pretraining candidate only after rights review. It is not suitable as-is for SFT or RL, and it is not an independent held-out evaluation set. The source is blocked from acquisition in the current registry. [1] [3] [4]

## Nemotron-Agentic-v1 — agent behavior and tool use

This source was selected for synthetic user/agent/tool trajectories, planning, tool choice, and tool interaction. The card lists two `default`-configuration splits: `interactive_agent` (19,028 examples) and `tool_calling` (316,094 examples), 335,122 total, with an indicated dataset size of about 5.5 GB. The card’s preview exposes `uuid`, `messages`, `license`, `used_in`, `tools`, and `reasoning`; previewed messages include `role` and `content`, and tool definitions contain function/parameter schemas. The dataset viewer’s `/info` endpoint failed on inconsistent nested parameter shapes, and the rendered preview does not establish a complete tool-result schema. We did not download rows to resolve that gap. [5]

Pinned revision: `650d590978ca35c8f1ecea2faf136e5fac421b62`. Both splits declare CC BY 4.0. The card also identifies Glaive Function Calling v2 as an Apache-2.0 component, but does not map that component to rows or fields. It says interactive-agent trajectories are synthetic and names Qwen3-235B-A22B-Thinking-2507, Qwen3-32B, GPT-OSS-120B, and Qwen3-235B-A22B-Instruct-2507 as generation/judging models. Tool sets are collected from public datasets, but those sources are not enumerated per example; NVIDIA Nemotron-Personas-USA is used to seed the user simulator. These unresolved component and source-lineage issues block acquisition despite the top-level CC BY declaration. [5] [6] [7] [8] [9]

The strongest fit is later SFT/tool-use training, with the two splits kept separate and complete trajectories, tool-call arguments, results, `uuid`, `license`, `used_in`, `tools`, and reasoning metadata preserved. It is not general pretraining data or verified RL reward data. Any internally held-out slice would be synthetic and leakage-prone, not independent benchmark evidence. Current registry status: blocked pending component-license and schema review.

## Nemotron-Instruction-Following-Chat-v1 — instruction and structured outputs

This source was selected for instruction following, multi-turn chat, and JSON/schema-constrained responses. Its `default` configuration has `chat_if` (426,009 records) and `structured_outputs` (4,969 records), 430,978 total according to the pinned card. The preview shows fields such as `uuid`, `messages`, `license`, `used_in`, `tools`, `reasoning`, and `capability_target`; message objects may contain `role`, `content`, and `reasoning_content`. The card describes JSON-format-constrained structured outputs. We did not download payload rows, and the viewer did not provide a complete split-specific feature schema. [10]

Pinned revision: `83dcd3aded0d289b0bbc018d3f9af4c5dd4005df`. The card assigns ODC-By 1.0 to `chat_if` and CC BY 4.0 to `structured_outputs`, while the Hugging Face repository-level metadata advertises CC BY 4.0 without subset distinction. The card says chat was refreshed from Nemotron-Post-Training-Dataset-v2 and extended to multi-turn; synthetic generation and filtering use GPT-OSS-120B and Qwen3-family models. The upstream v2 card describes mixed public/open and synthetic prompts, including WildChat ODC-By and StackOverflow CC BY-SA subsets, and notes possible generator-model license obligations. The refreshed v1 card does not map those inherited sources/terms to individual chat records. This subset and upstream discrepancy blocks acquisition until clarified; keep the two subsets separate if later approved. [10] [11]

The intended fit is selective instruction SFT/post-training, not general pretraining, direct RL, or held-out evaluation. The `structured_outputs` subset should be checked against its actual per-record JSON schema and malformed/incomplete outputs should be rejected with reasons. It does not establish tool-use capability merely because a preview includes a `tools` field. Current registry status: blocked pending license-lineage clarification.

## Wikimedia English Wikipedia dump — factual knowledge

Wikipedia was selected as a possible factual and explanatory text source. The dated `enwiki-20261001` snapshot is marked complete and includes `pages-articles-multistream` files plus matching indexes. MediaWiki XML pages carry titles, page IDs, namespace, revision IDs, timestamps, contributor/comment metadata, content model/format, text, and SHA-1 where present. Multistream files use concatenated bzip2 streams; the index maps offsets to page IDs and titles. The snapshot job label and general dump FAQ disagree about which namespaces are included, so actual XML namespace values must be checked before filtering. No XML payload was downloaded. [12] [13] [14] [15]

Pinned snapshot: `enwiki-20261001`, dated 2026-10-01. Wikimedia dump guidance generally identifies CC BY-SA 4.0 and GFDL for original text, with page-level exceptions, imported text under other compatible terms, and independently licensed/non-free media. Commercial use is generally allowed subject to attribution, license notices, change notices, source-specific terms, and ShareAlike for adapted text. Whether ShareAlike applies to model weights or outputs is not resolved by these sources. The dump is not guaranteed to be a complete or transactionally consistent backup, and the XML export format is scheduled to sunset in January 2027. Do not acquire until a page-level rights/attribution filter and an appropriate frozen text-only pilot are reviewed. [13] [16]

If approved, Wikipedia is a Stage 1 knowledge/pretraining source. It is a poor direct SFT or RL source and is not an independent benchmark by default. Current registry status: blocked pending legal, attribution, namespace, and format review.

## Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 — later post-training

This source was requested for later tool-use/RL-stage research, not Stage 1. The card describes synthetic labeling/processing over expert tool-use trajectories from 838 domains and a JSON format compatible with NeMo Gym. Viewer metadata exposes `trajectory_id`, `responses_create_params`, `expected_action`, `scenario`, `num_unique_actions`, `meta_info`, reward statistics, `agent_ref`, and pass-rate fields. The prompt includes tool definitions and messages; trajectory IDs and step/order, tool action, and surrounding context must remain intact. The source does not identify the expert trajectory corpora or give their rights. [17] [18]

Pinned revision: `bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e`, configuration `default`, split `train`. The card declares CC BY 4.0 and commercial use, but upstream trajectory rights remain unknown. The README says 170,320 records and about 3.0 GB; datasets-server reports 96,968 records and about 1.586 GB in memory. There is no validation/test split. The discrepancy and missing rights lineage block acquisition. Candidate only for later behavior-cloning/post-training after parser, rights, and execution-simulator compatibility work; it is not evidence of independently verified reward behavior and not a held-out benchmark. [17] [18] [19]

## Nemotron-RL-Agentic-Function-Calling-Pivot-v1 — later tool-use post-training

This source was requested for later-stage tool-use research. Its JSONL rows are described as NeMo Gym-compatible decision-point examples with `trajectory_id`, `info`, `responses_create_params` (input, tool schemas, parallel-call setting), `expected_action`, and `agent_ref`. `expected_action` may be a message, function call, or batch of function calls. Preserve complete prompt context and parallel-call boundaries; do not convert expected actions into ordinary prose. [20] [21]

Pinned revision: `0ac34ec81bb6976d3746407df306a57ce48f0bd8`, configuration `default`, split `train`. The card declares CC BY 4.0 and separately mentions Apache-2.0 and MIT licenses without scoping them to files or components. The card says 8,458 training records and a validation set, but viewer metadata says 9,620 training records and the pinned repository contains only `train.jsonl`. NeMo Gym’s separately sourced validation artifact is marked license TBD and is excluded. The card’s description of existing expert trajectories conflicts or is underspecified relative to its synthetic-model list, and it does not supply upstream rights. Keep it outside Stage 1 and block acquisition until licensing, provenance, and artifact discrepancies are resolved. [20] [21] [22]

## Berkeley Function Calling Leaderboard — held-out evaluation

BFCL V4 was selected only to evaluate function calling and agentic tool use. It is category-based evaluation data, not train/dev/test data. V4 covers single-turn, multi-turn, live/non-live, language-specific, and agentic scenarios such as web-search and memory. Its evaluator and data formats have changed across benchmark versions, and the current Hugging Face card is stale relative to V4. The official leaderboard identifies BFCL V4 and evaluation code commit `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000`; pin this evaluation version rather than a floating branch. [23] [24] [25]

The repository and card declare Apache-2.0. That license permits commercial redistribution subject to its license, notice, and attribution requirements, without ShareAlike. Berkeley describes some examples as manually authored and live examples as user-submitted function documentation and queries; the reviewed sources do not establish per-record rights for every third-party API description or user contribution. Keep the entire benchmark outside all training mixtures regardless of license status. If test records cannot legally or technically be inspected for contamination checks, report that limitation rather than claiming zero contamination. [24] [25] [26]

## Acquisition decision

The research identifies usable revisions and formats, but it does not clear the requested external data for acquisition. The source registry and acquisition config pin revisions and keep all payload jobs disabled. Licensing ambiguities are recorded as blockers, not converted into approvals. No external records or tokens were acquired, and no source is currently approved for Stage 1 training.

## References

[1]: https://huggingface.co/datasets/HuggingFaceFW/fineweb/raw/9bb295ddab0e05d785b879661af7260fed5140fc/README.md "FineWeb dataset card at pinned revision"
[2]: https://huggingface.co/api/datasets/HuggingFaceFW/fineweb "FineWeb repository API"
[3]: https://opendatacommons.org/licenses/by/1-0/ "Open Data Commons Attribution License 1.0"
[4]: https://commoncrawl.org/terms-of-use "Common Crawl Terms of Use"
[5]: https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1/tree/650d590978ca35c8f1ecea2faf136e5fac421b62 "Nemotron-Agentic-v1 pinned repository"
[6]: https://creativecommons.org/licenses/by/4.0/legalcode.en "Creative Commons Attribution 4.0 International Legal Code"
[7]: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 "Glaive Function Calling v2 dataset card"
[8]: https://www.apache.org/licenses/LICENSE-2.0 "Apache License 2.0"
[9]: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA "Nemotron Personas USA dataset card"
[10]: https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1/raw/83dcd3aded0d289b0bbc018d3f9af4c5dd4005df/README.md "Nemotron Instruction Following Chat card at pinned revision"
[11]: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2 "Nemotron Post-Training Dataset v2 card"
[12]: https://dumps.wikimedia.org/enwiki/20261001/dumpstatus.json "English Wikipedia dump status for 2026-10-01"
[13]: https://dumps.wikimedia.org/legal.html "Wikimedia dump legal information"
[14]: https://meta.wikimedia.org/wiki/Data_dumps/Dump_format "Wikimedia dump format documentation"
[15]: https://meta.wikimedia.org/wiki/Data_dumps/FAQ "Wikimedia dump FAQ"
[16]: https://creativecommons.org/licenses/by-sa/4.0/legalcode "Creative Commons Attribution-ShareAlike 4.0 International Legal Code"
[17]: https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1/raw/bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e/README.md "Nemotron conversational tool-use pivot card at pinned revision"
[18]: https://datasets-server.huggingface.co/info?dataset=nvidia%2FNemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 "Nemotron conversational pivot viewer metadata"
[19]: https://creativecommons.org/licenses/by/4.0/legalcode "Creative Commons Attribution 4.0 International Legal Code"
[20]: https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1/resolve/0ac34ec81bb6976d3746407df306a57ce48f0bd8/README.md "Nemotron function-calling pivot card at pinned revision"
[21]: https://datasets-server.huggingface.co/info?dataset=nvidia%2FNemotron-RL-Agentic-Function-Calling-Pivot-v1 "Nemotron function-calling pivot viewer metadata"
[22]: https://github.com/NVIDIA-NeMo/Gym "NVIDIA NeMo Gym repository and validation artifact references"
[23]: https://gorilla.cs.berkeley.edu/leaderboard.html "Berkeley Function Calling Leaderboard"
[24]: https://github.com/ShishirPatil/gorilla/tree/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/berkeley-function-calling-leaderboard "BFCL evaluation code at pinned commit"
[25]: https://github.com/ShishirPatil/gorilla/blob/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/berkeley-function-calling-leaderboard/bfcl_eval/data/README.md "BFCL data format and version documentation"
[26]: https://github.com/ShishirPatil/gorilla/blob/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/LICENSE "BFCL repository license at pinned commit"
