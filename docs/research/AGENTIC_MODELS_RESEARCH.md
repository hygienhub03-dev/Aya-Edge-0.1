# Agentic Models Research Report

**Project:** AYA Edge 0.1
**Status:** Research synthesis; no production-readiness claim
**Purpose:** Translate tool-use, planning, evaluation, RL, and distillation evidence into testable AYA decisions

## Executive conclusion

AYA should be trained and evaluated as a **typed, verifier-backed interaction policy**, not as a model that merely produces persuasive agent prose. The model should understand intent, plan, select tools, emit schema-valid calls, interpret structured observations, recover from errors, and decide whether completion is justified. The runtime remains authoritative for permissions, validation, execution, state, and completion evidence. [1]

The recommended default is **structured ReAct**: one bounded next action at a time, followed by a machine-readable observation and explicit completion check. Escalate to bounded Tree-of-Thoughts or LATS-like search only for high-branching, high-value tasks with a simulator/verifier and strict budgets. Use verifier-backed SFT first; introduce preference optimization and RL only with held-out gains over the Stage 3 baseline, independent outcome verification, least-privilege tools, and reward-hacking controls. [2–6]

## AYA target and current boundary

The official specification prioritizes agent/tool use, reasoning/planning, instruction following, general language, context/retrieval, and structured output. Its operating principle is **Understand → Plan → Act → Observe → Adapt → Complete**, and it explicitly distinguishes a proposed tool call from actual execution. Full credit requires verified task completion, not a description of steps. [1]

The current scaffold is reported as **85.1M parameters**, while the longer-term architecture target is approximately **310M**. These are distinct; the prototype is not the target model and neither has established agent competence. Stage 0 documentation says instruction following, tool use, long context, expert specialization, and production readiness remain unverified. [7]

## Tool-use model and training evidence

### Typed interaction protocol

A canonical supervised trajectory should encode: system/tool schemas → user request → assistant tool call with name and JSON arguments → runtime tool result with call identifier and status → next assistant call or final answer. This keeps requested action separate from executed action. JSON Schema and strict structured-output mechanisms can enforce syntax for supported schemas, but they do not prove that the selected tool or values are semantically correct. [8,9]

Train separate but composable competencies: tool-needed detection, tool selection, schema validity, argument correctness, observation interpretation, error recovery, clarification/abstention, and verified completion. Include successful calls, empty or contradictory observations, malformed arguments, timeouts, permission failures, unavailable tools, alternate-tool recovery, and honest non-completion. [2,8]

### Toolformer and synthetic trajectories

Toolformer provides a self-supervised route: sample candidate API calls, execute them, retain calls that improve future-token loss, and fine-tune on augmented call/result sequences. It is useful for data generation, call timing, argument learning, and result incorporation, but does not replace curated multi-step trajectories or runtime safety. [10]

ToolLLM/ToolBench documents large synthetic API coverage and execution-path search; API-Bank provides a smaller runnable evaluation environment with annotated dialogues and API calls. Their useful lesson is execution-grounded filtering. Underlying APIs and data may have separate terms, so synthetic volume must not be mistaken for redistributable training data. [11,12]

## Planning and recovery approaches

### Structured ReAct: adopt as default

ReAct interleaves reasoning/decision state, action, and observation. Its reported benchmark improvements are model- and environment-specific, but the behavioral pattern matches AYA’s loop. Use typed state rather than unrestricted prose: goal/constraints → current state → one next decision → validated call → observation envelope → state update. The runtime should return status, provenance, affected state, and retryability; the model should choose retry, correction, alternate tool, clarification, or safe termination. [2,3]

ReAct does not establish long-horizon autonomy. Reasoning traces are not proof of correctness, and the model must not declare completion without an observation, validator, or explicit runtime result. [3]

### Tree of Thoughts and LATS: gated experiments

Tree of Thoughts supports branching, self-evaluation, lookahead, and backtracking; LATS combines search with external feedback and value estimates. These methods can help where an early decision has high downstream impact, but they add inference calls, latency, and evaluator error. For a compact AYA model, use them only behind a risk/branching gate with limits on depth, branches, tool calls, and wall-clock time. [4,5]

### Reflexion-style memory: limited adoption

Reflexion stores verbal feedback for later attempts without updating weights. AYA may use concise, task-scoped episodic summaries after verified failure or low-quality outcomes, with expiry and provenance. Unverified self-critique must not override authoritative runtime errors or user constraints. [6]

## Evaluation design

AYA’s official Agent Completion Score decomposes intent recognition, plan correctness, tool selection, argument correctness, execution outcome, result interpretation, and final completion. [1] The local harness should be the primary gate and should contain no-tool cases, distractor tools, required/optional arguments, multi-step state changes, injected failures, insufficient-information cases, and final-state verification.

Report a vector, not a single opaque score:

1. Tool-need precision/recall over tool, no-tool, and clarify/abstain cases.
2. Exact tool-selection accuracy and wrong-tool rate.
3. Schema validity, required-field accuracy, and semantic argument correctness.
4. Runtime acceptance and expected intermediate-state reachability.
5. Recovery success after injected failures without repeated failed calls.
6. Verified final-state success (`pass^1`) and repeatability (`pass^3` or `pass^k`).
7. Turns/tool calls to success, unnecessary-call rate, policy/minefield violations, and false-completion rate.

Use public suites periodically and record exact revisions, task IDs, decoding, runtime, timeouts, and maximum turns. BFCL is useful for function-call and argument diagnostics; ToolSandbox tests stateful dependencies and unsafe actions; maintained τ³-bench or compatible τ-bench tasks test policy-constrained final state and repeated reliability; AgentBench adds broader environments but is heavier for every iteration. [13–16]

For compact-model promotion, gate on verified completion. A proposed aggregate may be `completion × (selection + argument + execution + recovery)/4`, but the full vector and failure reasons must remain visible; a geometric or gated score is safer than allowing fluent planning to mask execution failure.

## RL, preference optimization, and safety

The primary reward should come from an external verifier of the requested end state: database/state diff, file/test result, schema-valid artifact, or domain oracle. Process terms may reward valid calls, constraints, efficiency, and recovery only after hard safety and permission gates pass. “I completed it” is an untrusted model claim. [17]

Reward hacking can exploit the reward channel, evaluator, simulator, or human perception. Adopt a progression: offline replay/simulator training with deterministic verifiers; shadow evaluation on held-out and adversarially varied tasks; limited online trials with least-privilege tools, action budgets, approval gates for irreversible actions, and immutable audit logs; only then consider broader interaction. Retain the non-RL Stage 3 checkpoint as fallback. [17,18]

DPO is a lower-complexity candidate for preference learning, but it does not solve tool execution or long-horizon correctness by itself. Preferences should be attached to verified episodes and label correctness, constraints, tool validity, safety, usefulness, and verbosity separately. [17]

## Distillation and teacher policy

The official specification names **MiniCPM5-1B** as the intended teacher, subject to exact revision and license verification. The research note reports the OpenBMB repository and model card as Apache-2.0 and identifies the final BF16 release as post-trained with RL + OPD. Exact checkpoint revision, notices, and terms must be recorded. [1,19]

Stage 5 should freeze Stage 4, generate targeted teacher demonstrations, execute/filter them, retain provenance and rejection reasons, construct high-confidence preference pairs, train SFT before DPO/equivalent preference optimization, and gate against Stage 4 on general language, structured output, safety, context retrieval, and agent completion. Teacher outputs are not automatically correct. [19]

Hosted-teacher terms may prohibit automated extraction or training competing models. The cited OpenAI and Anthropic terms therefore require a current contract review before automated collection; prefer a locally runnable, openly licensed teacher or obtain written permission. [19]

## Licensing caveats

- ReAct and Reflexion repositories are identified as MIT; their papers and benchmark/tool dependencies have separate terms. [3,6]
- ToolBench is documented as Apache-2.0, while RapidAPI services and derived material may impose additional conditions. API-Bank’s cited sources do not establish a redistribution license. [11,12]
- BFCL and AgentBench repositories are documented as Apache-2.0; τ-bench’s original repository is MIT, while ToolSandbox uses Apple’s custom permissive license with third-party notices. [13–16]
- Papers may be CC BY or distributed under arXiv terms; that does not automatically license code, weights, datasets, prompts, or generated outputs.
- Teacher model licenses do not automatically grant rights to teacher-generated text, user prompts, benchmark tasks, or third-party source material. Record provenance, terms snapshots, and privacy controls.

## Adopt/reject/test recommendations

| Area | Recommendation | Evidence gate |
|---|---|---|
| Tool protocol | **Adopt** typed call/result envelopes and external runtime validation | JSON/schema validity plus semantic argument and final-state tests |
| Routine planning | **Adopt** structured ReAct as default | Completion, recovery, latency, false-completion rate |
| Deliberative search | **Test selectively** ToT/LATS behind a budget gate | Held-out completion gain exceeds cost/latency and evaluator errors |
| Reflection | **Test bounded** Reflexion-style summaries | Verified failure signal, task scope, expiry, no policy override |
| RL | **Defer broad online use**; start offline/sandboxed | Held-out gains over Stage 3, no safety/reliability regression |
| Preference optimization | **Adopt after verified SFT** | High-confidence pairs and Stage 4 regression gates |
| Teacher distillation | **Test with MiniCPM5-1B only after audit** | Exact revision, license/terms, filtering, contamination and regression checks |
| Public benchmarks | **Use periodically** | Pin revisions; do not treat any suite as complete evidence |

## References

[1]: ../MODEL_SPECIFICATION.md "AYA Edge official model capability specification"
[2]: https://arxiv.org/abs/2210.03629 "ReAct: Synergizing Reasoning and Acting in Language Models"
[3]: https://react-lm.github.io/ "ReAct project page"; https://github.com/ysymyth/ReAct "ReAct repository"
[4]: https://arxiv.org/abs/2305.10601 "Tree of Thoughts"
[5]: https://arxiv.org/abs/2310.04406 "Language Agent Tree Search"
[6]: https://arxiv.org/abs/2303.11366 "Reflexion"; https://github.com/noahshinn/reflexion "Reflexion repository"
[7]: ../stage-0/MODEL_OVERVIEW.md "AYA Edge current model overview"; ../stage-0/STAGE_0_BASELINE.md "AYA Edge Stage 0 baseline"
[8]: https://developers.openai.com/api/docs/guides/function-calling "OpenAI function-calling guide"
[9]: https://developers.openai.com/api/docs/guides/structured-outputs "OpenAI structured outputs guide"
[10]: https://arxiv.org/abs/2302.04761 "Toolformer"
[11]: https://arxiv.org/abs/2307.16789 "ToolLLM"; https://github.com/OpenBMB/ToolBench "ToolBench repository"
[12]: https://aclanthology.org/2023.emnlp-main.187/ "API-Bank"
[13]: https://gorilla.cs.berkeley.edu/leaderboard.html "BFCL V4"; https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard "BFCL repository"
[14]: https://arxiv.org/html/2408.04682v1 "ToolSandbox"; https://github.com/apple-aiml-research/ToolSandbox "ToolSandbox repository"
[15]: https://arxiv.org/html/2406.12045 "τ-bench"; https://github.com/sierra-research/tau-bench "τ-bench repository"
[16]: https://arxiv.org/html/2308.03688v3 "AgentBench"; https://github.com/THUDM/AgentBench "AgentBench repository"
[17]: https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/ "Specification gaming"; https://arxiv.org/abs/2305.18290 "DPO"; https://arxiv.org/abs/2406.12045 "τ-bench"
[18]: https://arxiv.org/abs/1606.06565 "Concrete Problems in AI Safety"; https://arxiv.org/abs/2307.13854 "WebArena"
[19]: https://github.com/OpenBMB/MiniCPM "OpenBMB MiniCPM repository"; https://huggingface.co/openbmb/MiniCPM5-1B "MiniCPM5-1B model card"; https://openai.com/policies/terms-of-use/ "OpenAI Terms"; https://www.anthropic.com/legal/commercial-terms "Anthropic Commercial Terms"
