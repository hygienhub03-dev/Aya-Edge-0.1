# AYA Edge 0.1 — Stage 0 Documentation

This directory contains the Stage 0 documentation for AYA Edge 0.1.

## Purpose

Stage 0 establishes a reproducible, inspectable baseline before larger training runs.

It documents the current architecture, configuration, training diagnostics, known limitations, and requirements for progressing to Stage 1.

## Documentation

| File | Description |
|---|---|
| MODEL_OVERVIEW.md | Project direction and model snapshot |
| ARCHITECTURE.md | Architecture and implementation notes |
| TRAINING_ROADMAP.md | Development and training stages |
| DATA_STRATEGY.md | Initial data mixture and dataset preparation |
| EXPERIMENT_LOG.md | Recorded experiments and measurements |
| STAGE_0_BASELINE.md | Baseline verification and acceptance criteria |

## Important Status

The reported measurements in these documents must be reproduced against the exact Git commit and configuration before being treated as official benchmarks.

The current scaffold has a reported parameter count of approximately 85.1 million.

This is distinct from the longer-term approximately 310M-parameter architecture target.

## Current Development Philosophy

AYA Edge is being developed incrementally.

Each stage must establish measurable progress before the next stage begins.

A successful forward pass or small overfit experiment does not prove general intelligence, reasoning ability, instruction-following capability, or production readiness.

All future claims must be supported by reproducible experiments.
