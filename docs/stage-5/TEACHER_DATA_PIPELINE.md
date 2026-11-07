# AYA Edge — Stage 5 Teacher Data Pipeline

## Governing specification

This pipeline is subordinate to the [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). It must preserve the separation between model-generated tool calls and runtime execution.

## 1. Required record envelope

Every candidate record must have a stable `record_id` and an append-only provenance chain containing:

- Source type: teacher-generated, human-authored, simulator-generated, or transformed.
- Exact teacher name, revision/commit, artifact digest, tokenizer revision, and inference implementation, if applicable.
- License identifier and verbatim license/terms snapshot reference; notices and usage restrictions.
- Collection timestamp, operator/job version, prompt template version, decoding settings, and random seed.
- User/task source identifier, privacy classification, consent or usage basis, and redaction status.
- Tool-schema bundle revision, runtime version, environment/simulator image, permissions, budgets, and execution trace.
- Validator/evaluator version, labels, scores, rejection reasons, transformations, split assignment, and dataset hash.

A missing required provenance field is a quarantine condition, not an invitation to guess.

## 2. Teacher and legal audit gate

Before generation, create `teacher_audit` evidence that verifies from authoritative sources:

1. Exact MiniCPM5-1B checkpoint revision and immutable digest, not merely a family name.
2. Repository/model-card license text, notices, attribution requirements, and whether weights, code, prompts, and generated outputs have separate terms.
3. Current terms for the actual access route, including API or hosted-service restrictions on automated extraction, model training, redistribution, privacy, and competition.
4. Compatibility of all source datasets, tool APIs, benchmark tasks, and generated material with internal use and planned distribution.
5. Approval owner, verification date, source URLs, archived snapshots, and unresolved legal questions.

If the revision or terms cannot be verified, stop automated collection and use a locally runnable, approved artifact or obtain written permission. Do not silently substitute another teacher.

## 3. Pipeline stages

1. **Ingest:** validate envelope, hash raw input, and reject malformed records.
2. **Generate:** request bounded candidate responses and typed calls from the approved teacher.
3. **Execute:** run calls only in a deterministic simulator or least-privilege sandbox; record actual status, state changes, outputs, and errors.
4. **Validate:** check JSON/schema syntax, argument semantics, constraints, safety, intermediate states, final state, and completion claim against an independent verifier.
5. **Filter:** remove privacy leaks, unsafe actions, unsupported claims, duplicates, contamination, contradictions, low-information examples, and unverifiable outcomes.
6. **Review:** send borderline records to qualified review with blind comparison where possible; preserve reviewer IDs and adjudication.
7. **Transform:** normalize only through versioned scripts; retain raw and transformed hashes.
8. **Split:** split by task family/source/template before deduplication leakage can cross train, validation, or test; keep evaluation prompts hidden.
9. **Manifest:** publish accepted, rejected, and quarantined counts, reasons, tokens, hashes, licenses, and versions.
10. **Release:** expose only the approved training artifact and its manifest; retain restricted raw evidence separately.

## 4. Filtering rules

Hard reject or quarantine records with: unverifiable final state; fabricated tool result; invalid schema; wrong tool; semantically incorrect arguments; ignored user constraint; unsafe or unauthorized action; privacy or secret exposure; repeated failed call; unsupported factual claim material to the task; contradictory observation handling; duplicate or near-duplicate leakage; contaminated benchmark content; missing provenance; or unresolved terms.

A fluent answer is not a pass. A successful tool call is not a pass if the final state or user constraints fail. A correct final answer is not a pass if it claims execution without execution evidence.

## 5. Preference-pair construction

Create pairs only from the same prompt, schemas, environment, and budget. Re-run or replay both candidates when possible. Label winner, loser, tie/uncertain, and dimension-level reasons. Require an independently verified outcome difference or a clearly adjudicated safety/constraint difference. Exclude pairs where both are wrong, the outcome is unobservable, or preference is stylistic only.

## 6. Quality and split gates

Before training, verify: no train/evaluation overlap; no source or template leakage across splits; accepted records have complete provenance; rejection reasons are populated; token counts are reproducible; redactions preserve validity; and the accepted set includes hard negatives and honest non-completion cases rather than only easy successes.

## 7. Manifest fields

At minimum record `record_id`, source, teacher revision/digest, license status, prompt hash, task family, schemas, execution trace hash, validator version, outcome status, labels, rejection/quarantine reason, split, token count, transformation version, and dataset hash.

## References

- [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md)
- [Agentic Models Research Report](../research/AGENTIC_MODELS_RESEARCH.md)
- [MiniCPM repository](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM5-1B model card](https://huggingface.co/openbmb/MiniCPM5-1B)
- [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)
- [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms)
