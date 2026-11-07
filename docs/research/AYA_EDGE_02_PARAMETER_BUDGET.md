# AYA Edge 0.2 Parameter Budget

**Status:** Verified for the supplied YAML and current Python implementation.

## Configuration

`V=32,768`, `D=672`, `L=12`, `C=192`, `F=1,024`, `E=10`, `K=2`, and shared expert flag `S=1`.

## Exact formula

Because embeddings and the LM head are tied:

```text
Total = V×D
      + L × [ (2×D×D) + (3×D×C) + (4×D) + (D×E)
              + (3×D×F×(E+S)) ]
      + 2×D
```

Terms are, respectively, tied embedding/head, per-layer attention projections (`Q`, latent down, `K` up, `V` up, output), two LayerNorms, router, all routed plus shared SwiGLU parameters, and final LayerNorm.

```text
= 32,768×672
  + 12 × [2×672×672 + 3×672×192 + 4×672 + 672×10
          + 3×672×1,024×(10+1)]
  + 2×672
= 22,020,096 + 12×24,172,992 + 1,344
= 310,115,904 trainable parameters
```

The count was also verified by instantiating `AyaEdgeForCausalLM(AyaEdgeConfig(...))` and summing `requires_grad` tensors. `embed.weight` and `lm_head.weight` are one tied registered parameter.

## Structural active estimate

The current helper reports **111,935,040**:

```text
Active proxy = non-expert shared parameters
             + L×K×(one routed expert's parameters)
             + L×(shared expert parameters)
```

This estimate assumes exactly K routed experts per layer and includes the shared expert. It is not a FLOP count, not a wall-clock active-parameter measurement, and not a guarantee that a batch exercises exactly that work. It omits token imbalance, padding, routing/indexing, Python dispatch, synchronization, kernel effects, activation memory, optimizer state, and generation-time route variation. It also does not account for capacity drops or a future grouped implementation.

## Budget interpretation

The 310.1M figure is total stored trainable weights for this code/config pair, not an achieved model-size milestone or quality claim. The YAML’s 310M target is within its stated tolerance, but the research status remains `research_candidate_not_production_ready`.

## Reproducibility record

A publishable count must include Git revision, YAML snapshot, Python/PyTorch versions, dtype, tied-weight policy, and the exact counting script. Any future architectural change invalidates this arithmetic until rerun.

## References

- [`configs/aya_edge_0_2.yaml`](../../configs/aya_edge_0_2.yaml)
- [`aya_edge/model.py`](../../aya_edge/model.py), especially `count_parameters` and `count_active_parameters`
- [`aya_edge/config.py`](../../aya_edge/config.py)
- [`AYA_EDGE_02_BASELINE_AUDIT.md`](AYA_EDGE_02_BASELINE_AUDIT.md)
