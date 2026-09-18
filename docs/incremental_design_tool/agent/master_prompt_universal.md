# Master prompt: universal interactive SOLWEIG editing

Use `goal_condition_universal.txt` in a tool field limited to 4,000 characters. This file supplies the expanded operating details.

## Mission

Implement a generic adapter-driven interactive editing system around the current SOLWEIG-GPU scientific pipeline. Do not code the product as a tree-only application. Treat tree editing as the first well-bounded adapter and test fixture.

## Required first action

Audit current code and documentation to enumerate actual editable source inputs, derived nodes, module-level parameters, forcing columns, output products, stateful arrays, cache artifacts, and stage boundaries. Compare that audit with `universal_editing/edit_registry.yaml` and update the registry when repository evidence differs. Do not invent scientific parameters solely for UI convenience.

## Required architecture

Create:

- a versioned dependency graph;
- an `EditAdapter` registry;
- typed source deltas and old/new state;
- per-node spatial and temporal impact plans;
- local/full safety predicates;
- cache reuse declarations;
- a topological execution planner;
- versioned exact result patches;
- preview descriptors that remain explicitly non-scientific;
- capability metadata consumed by the frontend.

The planner must support heterogeneous batches and mixed scope. A local building or vegetation stage can be followed by a global downstream stage caused by a simultaneous weather edit.

## Scientific policy

The current full-domain solver is the oracle. Every enabled adapter needs fresh full-domain comparison. Validate positive and negative dependencies, spatial boundaries, temporal replay, old-state removal, outside-window invariance, cache reuse and stale-result rejection. Full fallback is the correct result whenever locality is uncertain.

Dynamic wind-field response to edited geometry is not implied by using existing directional wind coefficients. Keep it disabled or explicitly labeled until its own scientific pipeline and oracle exist.

## Multi-agent policy

Follow `multi_agent_orchestration_universal.md`. Maximize useful concurrency, isolate writers, and use independent reviewers/verifiers. The lead owns integration and final evidence. Subagents do not merge or push main.

## Completion

Completion requires all `validation_gates.yaml` gates, full test/docs/UI/deployment checks, per-adapter and cross-family scientific differential evidence, and target CPU performance/RSS measurements. A polished interface without validated adapter semantics is incomplete.
