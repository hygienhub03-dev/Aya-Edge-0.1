# Milestone A — First Words (Master Build Directive tracking)

**Branch:** `feature/milestone-a-first-words`  
**Date:** 9 November 2026  
**Authoritative identity:** Agent-first, balanced reasoning, hybrid knowledge, MLA-inspired Fine-Grained MoE with shared experts, 32 768 vocabulary.

This document tracks progress against the AYA Edge Master Build Directive. It is not a claim of completed training or agent competence.

---

## Current repository state (verified)

| Item | Status | Notes |
|------|--------|-------|
| Authoritative model config | `configs/aya_edge_0_2.yaml` | vocab_size=32768, ~310.1M total params target, top-k MoE + shared expert, MLA-inspired attention |
| Production tokenizer contract | Implemented | `aya_edge/tokenizer/bpe.py` — fixed 0–15 specials, 16–271 byte fallback, 272–32767 learned BPE |
| Production tokenizer artifact | **Missing** | No `tokenizer.json` + manifest with real checksums; trainer is approval-gated |
| Development tokenizer | Present | `ByteLevelTokenizer` (~268 IDs) — plumbing only |
| Model ↔ tokenizer alignment | **Blocking mismatch** | Config allocates 32768; only available tokenizer emits ≤268 IDs |
| Approved training corpus | **None** | All acquisition jobs disabled pending license/provenance review |
| First real language training run | Not started | Requires production tokenizer + approved data |
| Reloadable checkpoint from real data | Not present | — |

---

## Milestone A checklist (from Master Build Directive)

- [ ] Production tokenizer finalized and checksummed
- [ ] Model vocabulary alignment confirmed
- [ ] Approved language corpus prepared
- [ ] First genuine language-model training run completed
- [ ] Trained checkpoint saved and reloadable

---

## Immediate blockers (in priority order)

1. **Data licensing / provenance**  
   Resolve open questions in `docs/data/DATASET_LICENSING.md` and the source registry. Enable only sources that pass review. No fabricated or unapproved data may be used for the first run.

2. **Production tokenizer training**  
   Once an approved corpus exists:
   ```bash
   python scripts/train_bpe_tokenizer.py \
     --input data/processed/approved/train.jsonl \
     --input data/processed/approved/validation.jsonl \
     --output artifacts/tokenizer.json \
     --manifest-output artifacts/tokenizer.manifest.json \
     --corpus-manifest data/manifests/approved_corpus.json \
     --approval-record data/manifests/tokenizer_approval.json \
     --implementation-commit <exact-SHA>
   ```
   The resulting artifact must pass `ProductionBPETokenizer` validation (exact 32768 entries, special-token IDs 0–15, complete byte-fallback block).

3. **Training path enforcement**  
   Real (non-dry-run) training must load the production tokenizer artifact and refuse the byte-level development adapter. This is now enforced in `scripts/train.py` on this branch.

---

## What changed on this branch

- `scripts/train.py` now requires `--tokenizer` (path to a production 32k BPE artifact) for any non-dry-run execution and validates model/tokenizer vocabulary alignment before training begins.
- New script `scripts/validate_tokenizer_model_alignment.py` provides a fast, CI-friendly check that fails loudly on mismatches.
- This status document makes the gap between architecture scaffold and actual learning explicit.

---

## Definition of done for Milestone A

A Milestone A claim is valid only when all of the following are true and verifiable from repository artifacts:

1. A production `tokenizer.json` + `tokenizer.manifest.json` exist with real SHA-256 values (no TODO/PENDING).
2. Model config, embeddings, LM head, and tokenizer all report vocab_size == 32768 and identical special-token IDs.
3. An approved, provenance-recorded corpus is present and referenced by the training command.
4. A training run has executed real forward/backward passes, updated parameters, and written a checkpoint that can be reloaded.
5. The checkpoint records: config, seed, Git commit, tokenizer checksum, data manifest, step, and loss metrics.

Until those five items exist, the project remains in the architecture + data-preparation phase. Infrastructure smoke tests are not substitutes for the first real language-learning run.

---

## Next concrete actions

1. Complete license clearance for at least one Stage-1 source (or a legally usable alternative of comparable quality).
2. Acquire → normalize → deduplicate → split → build the approved corpus and its manifest.
3. Produce the tokenizer approval record and train the production BPE.
4. Run the first language training job with the production tokenizer and record the exact Git commit used.
5. Verify the resulting checkpoint can be loaded and produces coherent (even if weak) language-model behaviour on held-out text.

«We are not building an environment that demonstrates that AYA could learn. We are building AYA so that it actually learns.»
