# ADR 0001: Single-node incremental CPU worker

- Status: Accepted
- Date: 2026-08-31

## Context

The target application uses a fixed approximately 1 km² site and sparse tree edits. Users can tolerate exact updates on a seconds-to-minutes timescale. Keeping a GPU server active is unnecessary for this interaction pattern, while whole-domain CPU recomputation wastes work.

## Decision

Use one CPU node with an asynchronous exact worker. Precompute static geometry, compute conservative dirty regions for tree edits, recompute only affected windows when safe, and fall back to full-tile computation otherwise.

## Consequences

Positive:

- low and predictable operating cost;
- no GPU availability dependency;
- simple deployment and debugging;
- direct fit to teaching and live design demonstrations;
- baseline geometry shared across scenarios.

Negative:

- exact updates are not guaranteed to be immediate;
- correctness depends on conservative invalidation and halo logic;
- one worker limits simultaneous classroom throughput;
- significant solver refactoring is required.

## Revisit when

- the site grows substantially;
- exact p95 latency misses the accepted budget after structural optimization;
- concurrent scenario demand requires several workers;
- GPU serverless becomes operationally simpler than CPU refactoring.
