# AYA Edge — Stage 4 Reward Design

## Governing link

This design implements the outcome-first requirements in the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). The runtime and verifier remain authoritative; the model cannot award itself completion.

## Reward hierarchy

Use a gated, decomposed reward rather than one opaque scalar. Let `G` be the hard-gate indicator. If `G = 0`, the episode is ineligible for a positive completion reward and is labelled with the violated gate.

```text
G = permission × safety × schema × budget × audit_integrity
R = G × (w_outcome V + w_process P + w_efficiency E + w_honesty H)
      − penalties_for_repeated_failure_and_unnecessary_actions
```

The exact weights are an experiment variable and must be recorded. `V` is the dominant term; process terms cannot compensate for an unverified or unsafe outcome.

## Components

| Component | Evidence source | Positive signal | Required guard |
|---|---|---|---|
| Verified outcome `V` | Independent end-state verifier | Goal state, artifact, test, or state diff passes | Never use model claim as evidence |
| Constraint adherence | Task contract and validator | All user constraints satisfied | Missing/ambiguous constraints trigger clarification |
| Tool validity | Runtime schema validator | Known tool and valid arguments | Syntax alone earns no completion credit |
| Semantic argument quality | State-aware checker | Arguments match intent and available facts | Penalise guessed identifiers/values |
| Recovery `P` | Injected-error labels and trace | Correct retry, repair, alternate tool, or safe stop | No credit for repeated identical failures |
| Honest reporting `H` | Verifier vs final response | Accurate success, partial result, or non-completion | Penalise fabricated completion |
| Efficiency `E` | Trace budgets | Fewer unnecessary calls/turns | Applied only after correctness and safety |

Use component-level logs for diagnosis and report both raw and gated metrics.

## Reward-hacking defenses

1. **Independent verification:** score final state, file/test output, schema artifact, or domain oracle from a component separate from the policy.
2. **Held-out verifier variation:** vary task wording, initial state, distractor tools, and verifier implementation; never train directly on the final evaluator.
3. **Counterexamples:** include empty, contradictory, stale, partial, and adversarial observations; include tasks where the shortest path is wrong.
4. **No self-reported success:** phrases such as “done” have zero evidentiary value without a pass result.
5. **Side-effect audit:** record before/after state, tool identity, arguments, permissions, and timestamps; reject hidden mutations.
6. **Budget and repetition limits:** cap calls, retries, tokens, time, and cost; terminate loops deterministically.
7. **Canary tasks:** include known minefields where exploiting a superficial signal would fail the true verifier.
8. **Human review for promotion:** inspect sampled successes, all safety incidents, and high-reward/low-verifier-disagreement traces.
9. **Stage 3 anchor:** compare against the frozen SFT policy; retain Stage 3 if reward gains do not transfer to outcomes.
10. **Reward sensitivity tests:** perturb weights and remove components to detect dependence on a proxy.

## Failure labels

Every negative or ineligible episode receives one or more labels: `permission_denied`, `unsafe_action`, `schema_error`, `wrong_tool`, `bad_arguments`, `observation_misread`, `retry_loop`, `budget_exceeded`, `unverified_completion`, `constraint_violation`, `insufficient_information`, `verifier_disagreement`, or `environment_failure`. Environment failures must not be mislabelled as model failures.

## Non-claims

A reward model, preference label, or verifier pass is not proof of real-world safety or broad competence. Reward design cannot eliminate distribution shift, evaluator bugs, or all specification gaming; it only makes these risks measurable and bounded.

## References

- [Official model capability specification](../../MODEL_SPECIFICATION.md)
- [Agentic models research synthesis](../research/AGENTIC_MODELS_RESEARCH.md)
- [Specification gaming](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
