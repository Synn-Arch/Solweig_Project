# UEDIT-001..010 gate adjudication (mission_universal)

Date: 2026-09-03
Adjudicator: lead (orchestration agent)
Code state adjudicated: e914c17 (u-e3c merge; full canonical suite 1003 passed /
4 skipped / 1 failed = `tests/test_create_inputs_austin.py::test_create_inputs_austin_smoke`,
the sanctioned pre-existing environment failure, failing on origin/main@4c5e47d),
plus lead-direct ef4d86f (review-nit closure: behavior-preserving constant import
+ one docs disclosure) and 4d9a522 (ledger/worklog rows). A final full-suite run
at the closing HEAD is recorded in the completion audit.

Method: every gate is adjudicated against runtime evidence — tests in the
canonical suite, reviewer runtime-verified failing-first history, red-team
bitwise repros, benchmark artifacts — not against claims. Evidence pointers use
`file:line` / test node ids at the adjudicated state.

## UEDIT-001 — every enabled editor has a registered adapter and schema

**PASS.** The registry is single-sourced: `docs/.../universal_editing/edit_registry.yaml`
(9 adapters) and the parsed mirror in `solweig_gpu/incremental/edit_registry.py`
are kept identical by `test_registry_matches_yaml_entries` (the "do not edit
one without the other" drift-guard — it caught a real drift during u-e3). The
capability document served by `/api/v1/capabilities` derives from the registry
(u-e2 probe D3: derivation verbatim deep-equal snapshot, zero hardcoded
physics), the frontend is registry-driven (u-d2), and unknown adapters are
refused against the document (transport tests). Registry-vs-code parity was
re-verified at HEAD by the u-e2 adapter audit (finding D1: safe/blocked/class/
op/scope lists match exactly), following the u-a reconciliation audit
(`audit/2026-09-02-u-a-audit.md`).

## UEDIT-002 — every scientific adapter resolves valid dependency nodes

**PASS.** The 22-node dependency graph (`universal_editing/dependency_graph.yaml`)
is the planner's only source of order: `test_topological_order_is_valid_and_deterministic`,
plus the planner resolution and no-orphan coverage in `tests/test_incremental_edit_engine.py`
(50/50 at the adjudicated state). Every family edit marks exactly its edited
source nodes (u-d4 mixed-batch HTTP tests), and the u-a audit reconciled the
graph against repo evidence before the engine was built on it.

## UEDIT-003 — local adapters pass full-domain differential fixtures

**PASS.** Per-family bitwise differentials against independent full-domain
twins, all in the canonical suite: landcover (`test_paint_matches_physically_edited_oracle_bitwise`
and the resolved-grid identity family in `tests/test_incremental_lc_integration.py`),
buildings (five differentials from u-d4c/u-d4d, runtime-verified failing-first
by the non-author reviewer), meteorology/parameters (u-e1: nine cross-family
accumulation chains X→Y→Z all bitwise-clean vs full-domain twins,
`/tmp/ue1/repro_accum_chains.py`, real solver stack), and the real-site
differential campaign of U-C T3 (SANITY + U1–U8, evidence file + report).
Tolerances were never relaxed: every assertion is `array_equal`-class or
stronger (fence, watermark, orientation invariants pinned separately).

## UEDIT-004 — global edits reuse unaffected geometry caches

**PASS.** `test_meteorology_edit_keeps_geometry_caches_reusable` (engine) and
the plan stage classification (`capabilities.test.mjs`: changed / recomputed /
reused splits, e.g. a met edit reuses `building_dsm`) pin reuse semantics; the
u-d3 benchmark artifact records the measured reuse behaviour per family
(`benchmarks/2026-09-03-ud-perf.json`). The u-e1 red team explicitly probed
composition shadowing and cache mis-accounting across all families and found
the nine accumulation chains bitwise-clean.

## UEDIT-005 — heterogeneous edit batches produce one valid topological plan

**PASS.** `test_heterogeneous_batch_single_plan_with_mixed_scopes`
(engine-level) and the u-d4/u-d4c HTTP-level mixed-batch tests (mixed family
batches in one request = one job = one version bump, every edited source node
marked; building+met in one batch; post-reset family+tree replay in ONE batch —
u-e3c-reviewer probe P2, bitwise vs baseline+tree+met twin). One batch, one
plan, one publication — pinned end to end.

## UEDIT-006 — stale scene revisions never publish

**PASS.** Two layers, both tested at the adjudicated state:
engine-level `test_stale_scene_revision_refuses_to_plan`; HTTP-level
`test_stale_base_version_conflict_envelope`, `test_if_match_stale_conflict`,
`test_if_match_disagreeing_with_body`, `test_stale_solver_result_never_published`,
`test_reset_with_stale_if_match_conflicts`, and If-Match inside the
idempotency fingerprint. The U-E red-team wave then attacked the routes around
these guards and every attack is closed: the retention-pruning under-route
(u-e1 F1, HIGH) is dead via the durable v5 flag + v6 reset watermark +
reset-aware executor-state rescue, each fix failing-first and bitwise-verified
by both a non-author reviewer and the original red team; the post-reset tree
wedge (NF1) is closed by the voided-tree-base replay; the transport grammar
(u-e2 M1/M2/M3 + u-e3b model-boundary strictness) refuses ambiguity and
coercion with typed envelopes instead of silently resolving it.

## UEDIT-007 — view-only operations enqueue zero scientific jobs

**PASS.** `output_view` operations are answered from published results with
zero solver jobs: view suites in `tests/test_server_capabilities.py`
(`_job_count` unchanged, `len(solver.calls)` pinned), the view-only universal
item path in `tests/test_server_universal_edits.py`, and the u-d3 evidence
pack. Time-index strictness extends to the view body (u-e3b).

## UEDIT-008 — target CPU latency and peak RSS are measured per edit family

**PASS (measured; one borderline disclosed).** `benchmarks/2026-09-03-ud-perf.json`
records per-family latency and peak RSS on the sanctioned fixtures
(PERF-001/002/003, MEM-001 PASS). PERF-002 is borderline (57.5 ms vs the
60 ms target — met, but with thin margin) and is disclosed as such rather
than re-measured away. The benchmark artifact is append-only; the u-d4d
overlap-order disclosure was added additively without touching any
measurement field.

## UEDIT-009 — frontend exposes changed, reused, and recomputed stages

**PASS.** Registry-driven stage display (u-d2): the plan's changed / recomputed
/ reused classification is served and rendered; `capabilities.test.mjs` pins
the classification; the full node suite is 124/124 at the adjudicated state
(rerun at the closing HEAD in the completion audit).

## UEDIT-010 — dynamic wind limitations are disclosed unless independently validated

**PASS.** Dynamic wind stays `scientific_extension`: the disclosure
(`docs/incremental_design_tool/index.md`, "UEDIT-010 disclosure (dynamic
wind)") states the limitation, the registry yaml wind citations were
re-verified line-accurate by u-e2 (L3), and wind selection edits that ARE
safe (roughness-independent coefficient selection) are integrated while the
wind-field computation itself is typed-refused, not silently approximated.

## Register of disclosed residuals (none re-opens a gate)

1. GPU-path differential: standing OPEN by design (intake (j)) — the CPU chain
   is the validated path; documented.
2. `selected_date_time` and DEM edits: typed NOT_INTEGRATED refusals
   (intake (k)); disclosed, never silently accepted.
3. PERF-002 thin margin (57.5 vs 60 ms) — measured PASS, disclosed.
4. Legacy-layout fresh tree-only wedge (snapshot file missing while coverage
   survives, no reset): pre-existing at 57def21, unchanged and not widened;
   modern layout self-heals (u-e1 verify residual (b)).
5. Routing reader/bridge trust-rule alignment deferred: the compound corner
   (attack-C row + fully pruned family ledger + lost generation directory) is
   disclosed in `deployment_operations.md` with the one-reset remedy; outcome
   is bit-identical to the pre-fix legacy misroute (no regression).
6. Pruned-RESET-before-v6 corner: typed `scenario_state_unrecoverable`
   refusal (never wrong science); operator remedy documented.
7. Pre-fix-DB family events that the strict grammar now refuses replay
   loudly inside the job; retention makes this academic; reset heals.
8. Off-grid landcover coordinates pass schema and rasterize to zero cells —
   accepted-but-inert edit, disclosed (u-e2 L5).
9. README snapshot-refresh snippet format drift (indent=1 artifact vs
   indent=2 claim) — documentation-only.
10. Cross-id building overlap-order divergence at overlap cells: disclosed
    semantics with a pinned test (u-d4c/u-d4d), not a defect.

## Verdict

UEDIT-001..010: **10/10 PASS** at the adjudicated state, on runtime evidence,
with every wrong-science attack from two independent red teams closed by
failing-first fixes and verified bitwise by a non-author reviewer plus the
originating red team.
