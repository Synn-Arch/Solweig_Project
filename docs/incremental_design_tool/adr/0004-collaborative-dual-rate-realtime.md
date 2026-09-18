# ADR 0004: Dual-rate collaborative real-time analysis

- Status: Accepted as implementation target
- Date: 2026-09-04

## Context

The editor must combine operations from many simultaneous sessions and reflect every accepted edit in the next one-second update period. The current measured exact geometry-edit path is approximately 38 seconds when idle, so a universal claim of exact one-second SOLWEIG recomputation is not supported.

## Decision

Use a deterministic 100 ms epoch reducer and separate deadline-bound fast and asynchronous exact lanes.

The fast lane publishes one of `fast_exact`, `fast_qualified`, or `visual_pending` within the admitted one-second envelope. The exact lane revision-chases the latest canonical workspace state and publishes scientific reconciliation. Every result carries workspace, fast, exact, and exact-base revisions plus result class and provenance.

Accepted operations are never removed by supersession. Supersession compacts exact compute targets only; the authoritative snapshot includes all accepted operations.

## Consequences

- Collaboration and visual/analytical responsiveness are decoupled from the current exact geometry runtime.
- Some edit families can be exact within one second; geometry may initially be qualified or pending.
- A rigorous compensation-validation and out-of-domain mechanism is required.
- Fast-lane CPU capacity must be isolated from exact work.
- A guarantee is valid only inside a measured admission envelope.
- The largest exact-geometry optimization investment is persistent delta visibility/SVF and temporal-state reuse, not more FIFO workers.
