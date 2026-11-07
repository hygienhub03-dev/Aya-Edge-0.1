# AYA Edge — Stage 2: Mid-Training and Context Development

## Governing specification

This stage plan is subordinate to the repository-level [AYA Edge Official Model Capability Specification](../../MODEL_SPECIFICATION.md). Any change to objectives, data, training, architecture, or evaluation must preserve the approved agent-first identity or document its reason, expected benefit, and evaluation method.

## Purpose

Stage 2 improves AYA Edge's ability to process and retain information across longer input sequences.

The stage builds on the best verified Stage 1 checkpoint.

## Primary objectives

- Increase usable context length.
- Improve long-sequence language modelling.
- Preserve short-context performance.
- Improve contextual information retrieval.
- Reduce long-sequence training instability.
- Measure memory consumption.
- Improve attention efficiency.
- Evaluate context retention.

## Development priorities

1. Establish the current context baseline.
2. Introduce longer sequences gradually.
3. Monitor loss at different sequence lengths.
4. Evaluate information retrieval across positions.
5. Investigate attention and memory bottlenecks.
6. Verify checkpoint compatibility.
7. Compare performance against Stage 1.

## Important distinction

A model configured to accept 4,096 tokens is not automatically capable of using all 4,096 tokens effectively.

Context capacity and context competence must be measured separately.

## Expected deliverables

- Mid-training datasets.
- Context-length training configurations.
- Long-context evaluation datasets.
- Memory and throughput reports.
- Context retention benchmarks.
- Updated checkpoints.
- Stage 2 evaluation report.
