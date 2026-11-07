# AYA Edge — Stage 5 Distillation Objectives

## Governing specification

This document is subordinate to the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Stage 5 objectives must serve AYA’s agent-first identity and must distinguish generated plans and calls from verified execution.

## 1. Objective hierarchy

### P0 — Verified agent completion

Improve the probability that AYA correctly recognizes intent, forms an executable plan, selects the right tool, emits valid and semantically correct arguments, interprets observations, recovers from failures, and stops only when the requested end state is verified.

### P0 — Trustworthy behaviour

Reduce fabricated completion, unsupported conclusions, unsafe actions, repeated failed calls, policy violations, and silent constraint dropping. Teach clarification or honest non-completion when evidence is insufficient.

### P1 — Instruction and structured communication

Improve multi-constraint instruction following, concise explanations, JSON/XML validity, schema adherence, and consistent conversation state without encouraging unnecessary verbosity.

### P1 — Context and general capability retention

Preserve Stage 4 performance on general language, knowledge, short and long context, reasoning, safety, and structured outputs. Distillation must not collapse AYA into a narrow teacher imitation.

### P2 — Efficiency and consistency

Reduce unnecessary tool calls, retries, and turns to verified success while maintaining correctness, permission boundaries, and recovery quality.

## 2. Required behaviour dimensions

Each candidate trajectory should be labelled independently for:

- Intent and constraint extraction.
- Tool-needed versus no-tool versus clarify/abstain decision.
- Plan validity and dependency ordering.
- Tool selection.
- Schema validity and semantic argument correctness.
- Permission and safety compliance.
- Observation interpretation.
- Recovery choice after timeout, malformed input, denial, contradiction, or unavailable tool.
- Final-state verification and completion honesty.
- User-facing usefulness, relevance, and format.

A single aggregate teacher score is insufficient. Preserve the component labels and failure reasons.

## 3. Distillation targets

Use a typed interaction format containing system/tool schemas, user request, assistant decision or call, runtime result with call identifier and status, state update, and final answer. Include successful and unsuccessful episodes, including no-tool answers, clarification, safe refusal, empty results, contradictory results, permission failure, timeout, invalid arguments, alternate-tool recovery, and verified non-completion.

Do not train hidden chain-of-thought as a correctness guarantee. Distill concise decision state, action rationale where appropriate, typed calls, observations, and completion checks that can be audited by the runtime.

## 4. Teacher-use constraints

MiniCPM5-1B is a reference teacher only after exact revision and legal audit. Record model identifier, revision or commit, artifact digest, tokenizer revision, license, notices, model-card restrictions, access date, inference implementation, and decoding configuration. A teacher answer enters training only after independent validation; if validation cannot establish correctness, retain it only as rejected or quarantined data.

## 5. Preference objectives

Construct pairs from the same prompt and tool environment. Prefer the response with verified correctness, constraint adherence, valid and minimal actions, safe permissions, robust recovery, accurate uncertainty, and useful final communication. Do not prefer a longer, more confident, or more teacher-like response when its execution outcome is weaker or unverifiable. Reject ties, ambiguous labels, contaminated pairs, and pairs whose winner depends only on stylistic taste.

## 6. Data-mixture policy

Start with a measured mixture, not an assumed permanent percentage. Report token and episode counts separately for verified agent trajectories, general instruction/conversation replay, structured-output examples, safety/uncertainty examples, and preference pairs. Any mixture change must record its reason, expected benefit, and evaluation method as required by the governing specification.

## 7. Success definition

The objective is met only when held-out execution-grounded evaluation shows a meaningful improvement over Stage 4 in targeted agent tasks, while all mandatory general, context, structure, safety, and false-completion regression gates pass.

## References

- [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md)
- [Agentic Models Research Report](../research/AGENTIC_MODELS_RESEARCH.md)
- [MiniCPM5-1B model card](https://huggingface.co/openbmb/MiniCPM5-1B)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [DPO](https://arxiv.org/abs/2305.18290)
