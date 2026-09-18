# Universal SOLWEIG interactive editing

## Scope correction

The incremental system is **not a tree-placement-only feature**. Tree add, move, resize, and delete are a useful reference adapter because their spatial influence is intuitive, but the architecture must support any SOLWEIG input or parameter that can be represented as a validated edit.

The core product is a generic interactive editing engine with four responsibilities:

1. represent an edit as a typed change set;
2. resolve the downstream SOLWEIG dependency graph;
3. determine spatial and temporal invalidation and cache reuse;
4. run only the necessary exact stages, with full-tile fallback when locality cannot be proven safe.

The current public SOLWEIG-GPU pipeline accepts building DSM, DEM, tree/vegetation DSM, land cover, meteorological forcing, optional directional wind coefficients, date/time selection, and model/output parameters. The interactive system should expose a scientifically defensible subset of those inputs through adapters rather than hard-code one component type.

## Documents

- [Editable scope and edit families](editable_scope.md)
- [Dependency graph and invalidation](dependency_graph.md)
- [Edit-adapter contract](edit_adapter_contract.md)
- [Execution pipeline](execution_pipeline.md)
- [Frontend interaction specification](frontend_spec.md)
- [Validation matrix](validation_matrix.md)
- [Collaborative real-time extension](../realtime_collaboration/index.md)

## Agent entry points

- `docs/incremental_design_tool/agent/goal_condition_universal.txt`
- `docs/incremental_design_tool/agent/master_prompt_universal.md`
- `docs/incremental_design_tool/agent/mission_universal.yaml`
- `docs/incremental_design_tool/agent/multi_agent_orchestration_universal.md`
- `docs/incremental_design_tool/agent/goal_condition_realtime_collab.txt`
- `docs/incremental_design_tool/agent/mission_realtime_collab.yaml`

The machine-readable registries are:

- `edit_registry.yaml`
- `dependency_graph.yaml`
- `validation_gates.yaml`
- `../realtime_collaboration/realtime_contract.yaml`

## Product principle

The UI can look like a professional design application, but scientific execution is dependency-driven:

```text
edit source
  -> validate
  -> update source overlay
  -> resolve invalidated dependency nodes
  -> compute local/global and temporal scope
  -> reuse unaffected caches
  -> exact stage execution
  -> versioned result patch
  -> frontend exact-state update
```

A global edit is still interactive if it skips expensive geometry preprocessing. For example, changing air temperature or the selected hour may require a full spatial result but can reuse building, wall, SVF, and visibility caches. Conversely, editing a building may be spatially bounded but invalidate more stages than a weather change.

## Collaborative real-time extension

The universal adapter and dependency model remains authoritative. The collaborative extension adds deterministic multi-session operation reduction, 100 ms micro-epochs, a one-second deadline-bound fast analysis lane, admission control, and asynchronous exact revision reconciliation. It does not replace the edit adapters with a tree-specific queue. See [Collaborative real-time SOLWEIG editing](../realtime_collaboration/index.md).
