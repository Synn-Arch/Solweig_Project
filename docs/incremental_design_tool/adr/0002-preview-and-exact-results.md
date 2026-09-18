# ADR 0002: Separate immediate preview from exact analysis

- Status: Accepted
- Date: 2026-08-31

## Context

A professional design tool must respond during drag and immediately after drop. The exact scientific solver can take seconds or longer on CPU.

## Decision

Render a client-side preview for tree geometry, direct shadow, and optional approximate cooling. Treat the server result as the only exact scientific state. The UI explicitly represents preview, queued, computing, exact, stale, and failed states.

## Consequences

- Interaction remains fluid even when the worker is busy.
- The frontend must manage two versions of visual state.
- Preview formulas must not be exported or described as SOLWEIG output.
- Stale exact results must be rejected by scene version.

## Revisit when

The exact worker consistently returns below the interaction threshold and preview complexity no longer provides value.
