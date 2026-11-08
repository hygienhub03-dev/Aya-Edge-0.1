# AYA Edge Agent Evaluation Specification

## Purpose

This specification defines a small, deterministic harness for measuring whether an AYA Edge agent completes a task, rather than only producing a plausible tool call. It is intended for checkpoint regression tests and local development. The harness is dependency-light and uses typed envelopes, a small JSON-schema subset, and a simulated stateful runtime.

The score is reported as an **Agent Completion Score** vector with eight dimensions: `intent`, `plan`, `selection`, `arguments`, `execution`, `observation`, `recovery`, and `final_completion`. Each component is in `[0, 1]`; the vector must be retained even when an aggregate is displayed. A geometric mean is available for dashboards because a zero in a critical dimension should not be hidden by strong performance elsewhere.

## Episode contract

An episode contains a prompt, a fixed tool registry, an initial state, an expected final state, and a maximum step count. The model may emit a `tool_call`, `final`, or `clarify` envelope. Tool calls carry a stable `call_id`, tool name, JSON-compatible arguments, and step index. Tool results expose success or a structured error and are the only observations available from the runtime.

Tool parameters use a deliberately small schema subset: object properties, required fields, primitive types, arrays, enums, minimum values, string minimum length, and `additionalProperties: false`. This is schema-ish validation, not a claim of full JSON Schema compatibility. Semantically wrong values can be schema-valid and are scored separately as argument correctness and final-state completion.

## Scoring methodology

- **Intent:** whether the agent correctly identifies the user’s requested outcome and whether a tool is needed. The implementation gives no automatic intent credit; a rubric or explicit evaluator judgement is required, including for no-tool tasks.
- **Plan:** whether the intended sequence and dependencies are correct. This may be supplied by a deterministic task rubric or a separately judged plan; it is not inferred from fluent prose.
- **Selection:** exact ordered tool-sequence match, including the empty sequence for no-tool tasks.
- **Arguments:** required and normalized argument correctness. A valid schema alone is not sufficient.
- **Execution:** fraction of runtime calls accepted and completed successfully.
- **Observation:** whether the agent consumed the result/error needed to make its next decision. The harness records results so an adapter can score this explicitly.
- **Recovery:** whether an injected failure is handled by adapting or retrying correctly, without repeated invalid calls. Episodes without failure have full recovery credit by default.
- **Final completion:** exact final-state match. This is the primary gate: an episode is not complete merely because the model selected the right tool.

`run_model_in_loop` lets a policy receive the prompt and prior tool results before selecting each next call, retry, final response, or clarification. `score_trace` supplies transparent runtime-derived values and accepts expected tools, expected arguments, expected state, and optional intent/plan/recovery judgments. Unprovided intent and plan judgements score zero rather than being silently treated as correct. For stochastic agents, repeat each episode and report `pass^k` in addition to `pass^1`.

## Required case categories

The synthetic held-out set includes:

1. **No-tool:** answer directly or abstain; unnecessary calls are errors.
2. **Single-step:** one valid call changes one state field.
3. **Multi-step:** an ordered sequence changes multiple dependent fields.
4. **Invalid arguments:** missing, extra, or mistyped fields must be rejected before handler execution.
5. **Injected failure/recovery:** the first call can fail deterministically and a subsequent valid retry must reach the target state.
6. **Insufficient information:** a clarify envelope is preferred over guessing; add these cases when expanding the fixture set.

## Failure taxonomy

Failures should be labeled, not collapsed into a single miss: wrong tool need decision, wrong tool selection, malformed envelope, unknown tool, schema-invalid arguments, semantically wrong arguments, runtime rejection, missed observation, repeated failed call, unsafe/unnecessary call, and final-state mismatch. Keep intermediate labels for diagnosis while using final-state completion as the release gate.

## Leakage controls

The checked-in fixture set is small synthetic data, not a claim of broad benchmark coverage. Keep held-out prompts and expected states separate from training data and do not derive them from generated model traces. Use opaque task IDs, fixed tool/version manifests, deterministic initial states, and a fixture checksum in experiment metadata. Do not put expected tool names or target state in prompts. Freeze the fixture revision before comparing checkpoints, record decoding settings and maximum steps, and avoid evaluating on examples used to tune prompts, handlers, or scoring rules. Any newly added cases should be marked with a new revision rather than silently changing prior labels.

## Reproducibility and reporting

Report fixture revision, task IDs, tool-runtime version, timeout and maximum steps, decoding settings, number of repeats, and all eight score components. Include call count, unnecessary-call rate, failure/recovery counts, and exact final-state success. Preserve raw envelopes and results alongside aggregate summaries so a failed score can be audited.

## Limitations

This harness does not measure open-world knowledge, long-horizon browsing, latency under load, token cost, user satisfaction, policy compliance beyond what fixtures encode, or robustness to adversarial prompt injection. The simulator is intentionally simpler than production APIs: state updates are deterministic, handlers are local, and observations are fully visible. The score's intent, plan, and observation components can still require a rubric or model judge; those judgments must be versioned and audited. A small synthetic held-out set is suitable for regression gates, not for general claims about agent capability. Public suites such as BFCL, ToolSandbox, τ-bench, or AgentBench should be run separately when their operational and licensing requirements are acceptable.
