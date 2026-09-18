# Incremental design tool WORKLOG

## Current checkpoint

- Timestamp (UTC): 2026-09-07 ~18:40 (Wave 2/3 + incident-1 DEPLOYED; parity verification CLOSED; incident-2 fix under adversarial review)
- DEPLOY 579008e (Wave 2/3 realtime UX + incident-1 fix, live-verified end-to-end on https://solweig.alansynn.com/): canonical suite green before push (1564P/6S/0F after one triaged load-flake — test_output_view_values_time_index_strict failed once under cura-parity CPU contention, passed in isolation + full rerun); fly deploy; orca fresh-tab live checks — boot + catch-up copy "Exact · R10", pipeline verify/done, badge Exact, ladder legend tooltip, ledger row w/ stage dots + seq 10 receipt + copyable receipt, transport Live. Incident-1 = exact-lane recovery refusal after foreign-family completion on a realtime workspace (production DB forensics via fly sqlite read-only). Wave-3 review LOWs logged + deferred: is-copied flash on re-click, receipt copy focus drop.
- PARITY VERIFICATION CLOSED (commit b34198d, pushed; report docs/parity_verification_2026-09-07.md + README section; corrected post-executor-final-report — GPU pin was device 2 w/ GPUs 0/1 busy, GPU file counts 3/27 not 12/84, V-EX2 root cause fixed): vanilla upstream 4131775a vs optimized 579008e on cura/analytica (72 cores, 4× A6000; GPU pinned CUDA_VISIBLE_DEVICES=2, GPUs 0/1 carried another user's jobs; CPU pinned taskset 0,1 / 0-3 + OMP/MKL). Vanilla self-consistency floor 0.0 (EX1 twice, sha identical). ALL configs BITWISE IDENTICAL vanilla-vs-optimized (sha256 equal, all bands): GPU EX1 3/3 (72 bands), GPU EX2F 27/27 (648 bands; 9-tile 200 m/overlap 20), SVF site-cache 9 tif + 9 npz + 9 zip; CPU-4 12/12 + 84/84 (81/705 bands; CPU streams also compared side artifacts); CPU-2 12/12 + 84/84. CPU outputs bitwise across thread counts (cpu4 vs cpu2). Cross-device GPU↔CPU informational deltas (TMRT max 0.368 °C, UTCI max 0.091 °C) affect vanilla equally — device arithmetic, not optimization. Full-solve parity within noise (GPU repeat spread ~11 %). Incremental (warm cache, 500×500/24 steps, harness cuda=False = production CPU path): single-tree edits 2–7× vs full recompute, speedup growing as cores shrink (E4 610 m-away add: 7.0× on 2 cores); exact lane CPU-only today (solver pins torch cpu; mixed-device error at shadow.py:271 on CUDA-visible host, reproduced twice — portability finding). V-EX2 stale-cache pair: vanilla CRASHED (utci_process.py:635 — cached 500×500 vegshmat loaded into 220×220 tile); optimized validated + rebuilt the stale entry, completed, bitwise-equal to fresh vanilla on all common files — robustness fix in the fork, not a parity break; verdicts use the fresh V-EX2F/O-EX2F pair. cura scratch ~/solweig_parity_20260907/ safely deleted AFTER report commit (cleanup.sh hostname-guarded, CLEANUP_OK verified, no solweig dirs left in home); site data never committed.
- INCIDENT-2 CLOSED (fix merged 58b4dfd + DEPLOYED + live-evidenced): fix = _seed_exact_lane_baseline on ALL THREE _build_executor branches (restore :711, void :732, fresh :759) + _pending_restorable_by fraction gate (:410-443 — staged snapshot adopted only when full_recompute_fraction == lane fraction; exact 0.95 vs legacy 0.30 vs reconcile 1e-12). Review APPROVE-WITH-NITS (mutations m2-m5 KILLED; m1 survivor vacuous → follow-up #139). LIVE POST-DEPLOY EVIDENCE: original "replay chain for tree 'tree-gate-probe-1' not contiguous with the layer state" refusal GONE; stale consumed_revision HEALED 5→13 (the task's named symptom); post-deploy job passes the incident-2 refusal site (fails later at a different layer = incident-3). Full E-advance live verify deferred to incident-3 deploy (joint verify: E10→current + fresh baseline-tree op), worklog line below.
- INCIDENT-3 OPEN (#140, production, discovered during incident-2 live re-verify, task incident3-fix in flight): exact lane PERMANENTLY WEDGED at E10 (scene 17, fast 17) — every exact job since 18:37Z fails ~8s at stage windowing, raw "SourceDeltaError: non-contiguous change chain for object 'tree-mtrl1chf-01-dmnp': before-state does not match the preceding after-state" (edit_types.py:877 _coalesce_object_changes), NO typed envelope, span never settles → chase retry loop while subscribed. ROOT CAUSE (lead DB forensics + code read, read-only): studio tab with dead realtime subscription added tree via LEGACY /edits (app.mjs:2145 fallback — "exactly one plane per gesture"; edit_events seq 2 evt_6f31507915cb, idempotency 'studio-fc37391f:edit-1', funneled op seq 11 actor legacy_edits) while another session moved it via NATIVE ops (seq 12 same epoch, seq 13 next); exact batch = ledger-window add [None→addpos] ⊕ fold-span delta [op1_after→op2_after] — native op-1 between them exists in NO ledger event (natives never write edit_events; consumed at rev 13) → two authoritative planes disagree on one object → coalesce fence breaks. B1b unfoldable fence correctly silent (baseline knows id via scheduler fold incl. funnelled ops); incident-2 seed puts layer at op1_after which ALSO conflicts with ledger addpos. Also present: workspace RESET at 14:55Z (edit_events seq 1, last_reset_sequence 1). FIX dispatched (worktree, opus): per-object chain anchoring — for ids the ledger window carries, fold that object's native ops since its last ledger event (→ [addpos→op2_after]); seed must skip ids the window will replay; residual unresolvable shapes must fail typed edit_rejected that SETTLES (B1b-style), never raw rethrow; parity twins all-legacy/all-native bitwise; failing-first tests/test_live_incident3_mixed_plane_poison.py; canonical 1568P gate; non-author review then merge+deploy+joint live verify with incident-2 E-advance check.

- Timestamp (UTC): 2026-09-04 ~16:20 (r4-proof COMPLETE + curated 0b06aac; r4-proof-review dispatched)
- r4-PROOF COMPLETE (read-only agent; evidence curated from volatile /tmp into design/proofs/, commit 0b06aac w/ design ADDENDUM of 7 binding corrections + ledger row): V0 PIN BIT-IDENTICAL both regimes (site_500 unclamped + synthetic clamped; 0 mismatching (cell,patch) pairs all 153 patches both stacks; parity case 6 PROVEN). CORRIDOR: exact closure = ray-corridor(union pre/post steps) ∪ C — C load-bearing (step-1 target-local gate; 32 flips inside C outside corridor). HOLDS no-clamp 16/16 edits incl. raiser 12→18 + dropper 20→12 (5.7k/6.1k flips). FAILS clamped+amplitude-change: predicate (amp changed ∧ clamp on smaller-A scene) 32/32; mechanism per-step traced (unchanged overhanging crown invisible at amp 12, raised at extended step i=16); design §3 sketch inequality FALSE as written in clamped regime. MANDATORY GUARD: bound > abs on smaller-effective-amplitude scene → full replay (3 .max + 1 .min; complements _check_baseline_amplitude which misses droppers on min(a)≥0 clamped sites). PHASE A MUST use W2 reach-expanded windows (raw bbox 19/59 patches diverge, worst 167 cells; W2 59/59 bitwise). Preconditions adopted: bush≡0 asserts, C on composed-surface diffs, trunk_ratio-only edits provable no-ops. Residuals disclosed (dz float32 monotonicity assumed-conservative, min(a)<0 + bush≠0 untested-guard-covered, brute force spot-check scale, vbsh-Case-C not exhaustive). r4-proof-review DISPATCHED (non-author, adversarial, read-only + /tmp reproduction): line-verify step-1 read analysis vs shadow.py, attack Case D replacement proof + guard sufficiency, reproduce one no-clamp edit + clamped raiser counterexample + V0 spot, adjudicate §10 residuals vs Phase A, verify addendum faithfulness. R4 Phase A implementation dispatches ONLY on review verdict.
- R1 WAVE 1 CLOSED: r1-ops MERGED 6b2d42f (diff vs freeze = exactly 6 owned files; 53 passed ops+reducer+spec + 117 passed server suites + node 137/137 in main; worktree+branch removed). r1-ops delivered: store migration v7 (realtime_operations/realtime_epochs/scenarios revision-triple cols), operations.py transport validation, routes_realtime POST/GET operations + GET epochs, 21 failing-first tests; full canonical in worktree 1098P/5S/2F-sanctioned. LEAD FINDINGS at dispatch of r1-epochs: (a) client contract gap — realtime_client.mjs reads response.acks, r1-ops POST returns "accepted" only → r1-epochs adds acks alias; (b) client needs GET .../operations?since_server_sequence=N list endpoint (r1-ops built single-op GET only) → r1-epochs builds it; (c) LEAD DECISION: legacy /edits funneling through the op log DEFERRED TO R2 (r0-frontend measured ~70% exact_session blast radius; R1 keeps surfaces coexisting — realtime ops advance workspace_revision at epoch close, legacy /edits keep their job machinery; unification happens where compute lanes consume both). r1-epochs DISPATCHED (worktree, base 6b2d42f): owns realtime/epochs.py (100ms scheduler + idempotent close pipeline + restart recovery), realtime/broadcast.py (SSE hub), routes_realtime.py (GET events SSE + operations list + acks alias), store.py additive migration v8 (realtime_canonical_state persistence), app.py lifecycle wiring, failing-first test_realtime_epochs + test_realtime_broadcast. scene_version +1 per non-empty epoch IS the canonical revision assignment (r1-ops deliberately did not bump). Epoch stays status 'reducing' post-reduction until R2 lanes advance it (frozen vocabulary has no 'reduced' status). r3a-met-fast DISPATCHED IN PARALLEL (worktree, base cafb337; compute-plane files disjoint from r1-epochs): R3a = met variable-affinity analysis (file:line consumer trace per safe variable; expect most variables radiation_affecting — Ldown emissivity/Tg suspicions on ta/rh/ws) → utci_only affinity set → planner closure narrowing {utci@t} for all-utci_only batches → executor fast path recomputing utci@t from PUBLISHED radiation planes at t (refusal + full fallback on any missing plane) → non-prefix SPARSE per-timestep patch coverage (pinned; store (node,time)-keyed) → bitwise differential fast vs full (synthetic + real site_500 READ-ONLY) + negative controls (radiation_affecting + mixed batches route FULL; wbgt NEVER_PLANNED). HONEST-OUTCOME CLAUSE: empty utci_only set → deliver analysis + fence doc, no lever manufactured. STILL RUNNING: r4-proof (V0 pin + corridor-sufficiency). NEXT: r1-epochs merge → R1 non-author review + full canonical + node → R2; r3a merge → its own non-author review + differential verdict; r4-proof → proof note commit before any R4 impl dispatch.
- R1 WAVE 1 DISPATCHED (2026-09-04, base b3168d9 = interface freeze commit, 4 isolated worktrees + 1 read-only design agent): LEAD froze solweig_gpu/server/realtime/{__init__,types}.py (Operation/EpochReducer/CanonicalState/ReducedEpoch/FamilyDelta/ConflictRecord/revision-triple contracts; import-verified). r1-ops (owns realtime/operations.py + store.py migration v7 [realtime_operations + realtime_epochs + scenarios.fast_revision/exact_base_revision] + models.py + routes_realtime.py + app.py router-registration only + tests) — durable idempotent op log, per-workspace contiguous server_sequence, lazy epoch assignment, advisory base_revision, append-only retention. r1-reducer (owns realtime/reducer.py + tests) — pure deterministic per-family conflict reduction (generations/tombstones, chunk LWW, range segmentation, view), failing-first incl. chained-vs-single-shot equivalence + permutation invariance. r1-frontend (owns examples/incremental_design_tool/realtime_client.mjs + tests + README section; ADDITIVE ONLY, 127 existing tests untouched) — operation identity, submit with retry/idempotency, EventSource-factory subscription, revision-triple tracking, reconnect+catch-up zero-loss, result-class state machine. r1-telemetry (owns realtime/telemetry.py + tests) — solweig_rt_* histogram rings/counters, thread-safe, monotonic-clock injection, prometheus render. r4-design (read-only architect) — persistent occluder-state design doc for the R4 wave (critical path). SEQUENCING: r1-epochs (100ms state machine + SSE broadcast + funneling of legacy /edits routes) dispatches AFTER ops+reducer merge; then R1 non-author review + full canonical suite + node suite; R2 after R1 closes.
- RT-R0 (2026-09-04, realtime-collab mission opened, 5 parallel read-only agents, HEAD f3ad111): handoff timings REPRODUCED (single-step 39.5-40.0 s loaded ~9-11 = +3-6% vs idle, interpolating handoff's own load model; fractions bit-exact; scenario creation 0.30-0.35 s; full-day 59.6 s; svf ~21 s; tloop ~18 s; ZERO corrections, zero anomalies). Parity at f3ad111: solweig_gpu/ unchanged vs bitwise-parity point 838e420; full canonical 2F/1078P/5S (2F = sanctioned test_cli PATH; +6 = new realtime spec tests from 6045035, deviation noted); node 127/127; oracle 95/0. Cartography: 13-item queue gap register + risk register (scene_version equality = one-wave re-key blast radius); 9-adapter inventory + R3 ranking (met-at-timestep utci-only > receptor-params > selected_date_time adapter > radiation view-layer); frontend poll-only transport + ~70% exact_session test blast radius under operation epochs. Lead decisions: R1 store-centric single wave (migration v7 + multi-writer + revision triple re-key); R2 additive lanes (scheduler thread ticker-pattern, exact ProcessPoolExecutor max_workers=1, fast in-process thread); reducer seed executor_bridge.py:597-628; SSE /events extension; append-only op audit with durable flags. Evidence: benchmarks/2026-09-04-r0-realtime/. Design draft: .omc/plans/realtime-r1r2-design.md. CI fully removed by user request (de05522 — publish.yml PyPI automation included; restore if release automation needed). NEXT: R1 dispatch (operations+store, reducer, epochs/broadcast, frontend operations API) — interface freeze by lead, isolated worktrees, one writer per file.


- Timestamp (UTC): 2026-09-02 ~15:05 → 2026-09-03 ~14:30
- U-E3 (2026-09-03, remediation of u-e1 protocol audit + u-e2 adapter audit — FINAL U-wave fix packet, 82610e4..6e27cf9 on worktree-agent-a1a104aec4911dbb9): ALL 13 findings fixed, every behavioral fix failing-first (13 new regressions run red at base before their fixes). F1 HIGH wrong-science closed: retention-pruned ledger no longer drives solver routing — durable scenarios.carries_family_edits flag (migration v5) SET atomically at family-event commit, CLEARED at reset, OR-ed with surviving family events for pre-v5 rows; regression pins tree-after-full-prune routing executor AND bitwise == the WITH-met full-domain twin. F2/F4/F5 typed envelopes (scenario_state_unrecoverable for torn snapshots; new engine_refused for ExecutorError/SolverInputError; _bootstrap_state shape fence refusing mis-built baselines instead of a NaN tail). F3 no-op path now rolls back worker staging flags (no leaked forcing-presence/dirty-override; follow-up paint routes local again). M1/M2/M3 transport hardening: undeclared ops typed 400 (was 500), strict JSON-integer values-level time_index at both parse sites (5.9/True/'07'/'abc'/[5] refused, was silent wrong-timestep coercion), verbatim relay (params op rewrite, building_id override, dropped landcover keys, dropped stray met time_index — all now typed refusals). L1 restore-time validate_tree_spec (tampered trunk_ratio=1.0 → typed ScenarioStateError, whole-restore). L2/L3/L4/L5 docs: README transport section, 7 yaml citation drifts (the 2 in PARSED notes fields mirrored into edit_registry.py builtin metadata per the drift-guard rule + 2-line capabilities_snapshot regen), adopt_published_state defense-in-depth note, off-grid-acceptance disclosure (verified empirically). Validation: full canonical suite 995p/4s/2f at faa3163 where the 2 = sanctioned austin (pre-existing) + the notes-parity drift caught by the suite and fixed by the mirror commit (6e27cf9; edit_engine 50/50, capabilities+docs 37/37 after) → effective 996 passed / 4 skipped / 1 failed (austin only); NO test_server_* failure; node 124/124 twice. Residuals disclosed in the ledger row: pydantic lax ITEM-level time_index normalization (models.py), pre-fix-DB replay refusal of silently-accepted legacy events, README snapshot-snippet format drift. NEXT: UEDIT-001..010 gate adjudication + mission completion audit. Details: SUBAGENT_LEDGER row U-E3.
- U-E3c (2026-09-03, red-team remediation packet #37: u-e1 NF1 post-reset tree wedge + u-e1 predicate attack C, 76b4b90..031fb21 on worktree-agent-ue3c-a7f3, failing-first — commit 1 = tests RED at base): NF1 closed by direction (a) refined — a fresh rebuild whose tree base was VOIDED by reset (last_reset>0 and covered<=last_reset) now seeds NOTHING from request.trees and arms replay_tree_events, so the window's tree events rebuild the authoritative tree state from empty and emit real vegetation patches (lead's sketch (b) empty-batch-no-op REJECTED: the no-op path re-serves the reset baseline WITHOUT the tree — bitwise-wrong vs the oracle twin; double-apply impossible via TreeLayer.add_tree's duplicate-id raise). Attack C closed with the lead's third durable signal made reset-aware as required: migration v6 adds scenarios.last_reset_sequence (backfilled from surviving reset events, written by every reset going forward — retention-immune like the v5 flag), and scenario_carries_family_edits now ORs flag | surviving family event | (executor-state dir exists AND effective covered (max of ledger-coverage.json, pending/pending.json) > watermark) — pre-reset snapshots can never resurrect; never-reset rows (watermark 0) over-route conservatively per the predicate bias. 3 new HTTP-level scientific tests vs the REAL solver stack (NF1 bitwise == baseline+tree full-domain twin and != with-met twin, second tree edit completes via restore; attack C routes executor and serves WITH-met bitwise; post-reset-pruned guard: tree edit stays legacy and serves baseline+tree bitwise) + migration v6 unit test. Validation: targeted suites 78 passed / 0 failed (universal_edits 46, recovery 3, scenario_state 29); full canonical suite = exactly the sanctioned austin failure; node not run (no frontend-visible change); adapted u-e1 repros on the fixed worktree: wedge probes 1-6 all complete, attacks A True / B True / C False / D True. Residuals disclosed: pruned-RESET-before-v6 corner (watermark backfills 0, dir rescue may over-route; operator remedy = one reset post-upgrade, documented in deployment_operations.md) and the pre-existing non-voided fresh tree-only-window wedge (snapshot file missing, coverage surviving, no reset — narrower than base, unchanged). Details: SUBAGENT_LEDGER row U-E3c.
- Branch: feat/incremental-design-worker
- U-D4c (2026-09-03, dispatched fix packet #36, adjudicated by u-d4-review-r2): building-edit accumulation FIXED — pre-existing engine defect (any solve after a published building edit silently dropped the building, including a second building batch) closed by the executor-level massing fold (per-id last-write-wins vs baseline anchor; deletes stay deletes via tombstones; every batch routes the regeneration chain while the fold is non-empty) + scenario-state v3 (v2 refused unverifiable) + routing disclosure in ud-perf.json/index.md (no measurement edits, no capabilities.py change). Evidence: 5 bitwise differential chains building->{building,met,params,veg,lc} vs independently-run regenerate_building_batch twins (failing-first @ 660f672) + HTTP-level building->met via /edits/universal + 13 fold/persistence unit tests; node 124/124. Full-suite residue: austin (sanctioned env) + a PRE-EXISTING test-infra flake REPORTED not fixed — test_server_api.py SCENARIO_CACHE keys by id(client) and a recycled address makes a fresh client reuse a dead app's scenario id (scenario_not_found in wait_for_scenario_exact; reproduced at cfb7b39). Details: SUBAGENT_LEDGER row U-D4c.
- U-D4d (2026-09-03, remediation of u-d4-reviewer REQUEST-CHANGES on u-d4c): documentation + tests + one error-message text ONLY — fold semantics, dispatch rule, and every scientific path untouched. F1: fold OVERLAP-ORDER semantics disclosed at executor._staged_massing_fold (docstring) + ud-perf.json additive overlap_order_disclosure block (zero measurement edits) + index.md, and pinned by a new OVERLAPPING-footprint fold test (fold order [A_tombstone, B]; B intact at FA∩FB where a session-order sequential oracle truncates it; differing cells == exactly the overlap; deleting the later-inserted id instead agrees with session order everywhere — the divergence is asymmetric; NOTE: the remediation packet's mirror description "delete B → A intact at overlap" is factually wrong per buildings.py and the test pins the true semantics). F2: the "fresh-rebuild replays ledger building events (deterministic equality PROVEN)" claim was backed only by a re-derivation shortcut (test-local executor re-executing the same validated commands) — now proven through the REAL bridge path (test_R4: building add via POST /edits/universal, executor-state tree loss, no-restore spy, command_from_item building_geometry spy, fold equality vs the original snapshot, chain routing, served arrays bitwise vs an uninterrupted twin scenario). F3: v2-refusal text names the RESET remediation (covered <= last_reset bypasses restore); index.md ops line for in-flight v2-snapshot scenarios. F5: tombstone-only publication proven baseline-equal BITWISE (vs an add-then-delete chain oracle, non-vacuous vs the add batch) with SCENARIO provenance (manifest != baseline, revision 2, staged Building_DSM == baseline tile). Validation: 3 touched files 86/86 standalone; full pytest at final tree 983 passed / 4 skipped / 1 failed = the sanctioned austin env failure ONLY (the raw run showed 3 failed, but the two test_cli failures were the subprocess 'thermal_comfort' console script needing the venv bin on PATH — with the canonical PATH both pass; 983 = U-D4c's 981 + the 2 new F1/F2 tests); no test_server_* failure (SCENARIO_CACHE flake fixed at 429be46); node not rerun (nothing frontend-visible touched). Details: SUBAGENT_LEDGER row U-D4d.
- HEAD: (U-C close commit) — U-C WAVE CLOSED. Packets merged through e607bfa: u-c1 executor + temporal store, u-c2 params physics (utci_process model_parameters), u-c2b dead-branch fixes, u-c3 met overlay, u-c4 lc overlay + job-wN patch ids, u-c5 building rasterizer + regeneration chain, u-c1b wedge rollback (4-watermark snapshot/rollback, store per-patch entries + single-writer), u-c6 building executor wiring + overlay coalescing, u-c6b stroke-dirty superset, u-c7 params executor fold (5/5 families), u-c8 combined residual sweep. NOT_INTEGRATED: selected_date_time (U-D) + dem (L1), typed refusals.
- Phase: dual-track. ORIGINAL mission COMPLETION AUDIT PASSED (below). UNIVERSAL mission: U-B CLOSED, U-C CLOSED. Review lines ALL APPROVE (u-c1 r3, u-c2 r2, u-c3 r2, u-c4 APPROVE-WITH-NITS w/ HIGH-1 discharged, u-c6 r3, u-c7 r2, u-c8). T3 real-site differential 9/9 PASS (evidence benchmarks/2026-09-03-uc-scientific.json @ e607bfa, bitwise, code-diff-empty provenance, 6 informational anomalies). Full python suite at final U-C state: 843 passed / 4 skipped / 1 deselected (austin pre-existing on origin/main@4c5e47d) 643.62s. NEXT: U-D (capability API, registry-driven frontend, per-family perf UEDIT-008, deployment), then U-E red teams + gate adjudication
- U-D INTAKE LIST (binding, from U-B/U-C reviews + T3 anomalies): (a) overlay/scenario state persistence — executor-memory-only forcing+lc+veg+params state dies with process; U-D persistence MUST serialize (u-c3-L4); (b) store reader contract — R2a partial-overlap sliver recompute fallback + revision_at first-entry vs max_revision semantics (u-c1-R2a, T3-A3 intersect-supersede is semantics-not-loss); (c) building chain reads worker.requested_variables directly at executor.py:1584 — one-line fence if variable set widens (u-c7-L1 residual); (d) identical-value resubmit ergonomics — declare old_state for clean no-op refusal (T7); (e) u-c4-review L4 worker-contract footprint guard, L5 solver scratch partial-stage copy waste, L6 scratch float32-vs-uint8 dtype asymmetry, T7 gvf_march comment 14→12; (f) u-c2-review T-residuals optional (drop del entirely like upstream, elvis→lup pin); (g) server site-identity pinning (origin_x/y + site_id unchecked by grid guard); (h) u-c6-review T-residuals: stage_landcover_overlay docstring restage-by-convention note, no-op reason text; (i) met_time.py:261 + landcover test :99 line citations shifted +20 by solweig.py fix; (j) GPU-path differential OPEN (CPU chain only); (k) selected_date_time adapter + dem L1 remain NOT_INTEGRATED typed refusals

- ORIGINAL-MISSION COMPLETION AUDIT (2026-09-02 ~11:00, lead, at d8fe110):
  - All mandatory gates CLOSED with evidence: BASE-001..003, WIN-001..003, CACHE-001..003, PACK-001..003, TREE-001..003, SCI-001..004, API-001..004, UI-001..005, PERF-001..003, MEM-001..002, IO-001, QUEUE-001, REL-001..003 (+T6 smoke)
  - Full python suite: 430 passed / 4 skipped / 1 deselected at d8fe110 (514.6s). Deselected = tests/test_create_inputs_austin.py::test_create_inputs_austin_smoke — failure PROVEN pre-existing on pristine origin/main@4c5e47d via temp worktree (same TypeError: module-not-callable; test calls solweig_gpu.create_inputs(...) as function, package ships run_create_inputs); not a regression of this branch, none of its files touched
  - Frontend: node --test 61/61. Docs: sphinx -W exit 0 + test_incremental_design_docs 13/13. Deployment smoke: T6 2 tests within server suite 125/125 post-merge
  - Scientific differential: T3 real-site evidence benchmarks/2026-09-02-p8-scientific.json overall_verdict PASS (local-mode re-enable rerun, 500x500@2m Austin tile, routing table + scenarios + post_merge_main_spot_check)
  - CPU benchmarks: PERF-001..003 + MEM-001..002 + IO-001 + QUEUE-001 evidence under benchmarks/2026-09-02-p8-performance/ (+p8-scientific.json); REL-002 load evidence benchmarks/2026-09-02-p9-reliability.json
  - Limitations documented: wind coefficients precomputed baseline (index.md model-scope), transmissivity inert global (registry notes), API limitations field (api_contract.md), trunk_ratio [0,1) reconciled (c8559a4), max_trees default guidance (d8fe110), dynamic wind out of scope
  - Working tree CLEAN at audit time; rollback point 31defb4 (p9-rel merge)
  - Invariants upheld throughout: full-domain oracle bitwise-deterministic (P0 hashes), never-relaxed tolerance (T3 all-pass), immutable baseline arrays, no stale scene_revision publishes (REL-001 orphan-version check)
- U-C intake queue (from U-B reviews, MUST absorb): M2 store keyed (node,time); L1 SOURCE_DELTA_FAMILY += dem w/ DemPatchDelta; L2 executor read-halo wiring (veg adapter read_halo_pixels defaults 0 — production wiring must inflate); L3 FrozenMapping constructor normalization; L5 drop windows from canceled add+delete chains; executor must pass SiteContext (binds default_temporal_steps=24) or estimates understate; executor must accept plans with ZERO output-node impacts and transport non-node producible layers; veg friction: sun-position/influence-config constructor injection, ScenarioTransaction rollback, TreeLayer sequence replay, merge_gap_pixels 8-vs-0, inert TreeSpec.transmissivity; server wiring note: grid guard checks rows/cols/pixel_size only — origin_x/y + site_id unchecked (pixel-indexed planning origin-agnostic; server must pin site identity); params friction (u-b-params-review APPROVE 3dc1f82): value chain closed at validate/staging/seam — planner stays name-only by design (adapter-owned domains, extensibility preserved); adapter impact_plan lacks SiteContext (engine-with-context is the plan path); staging-door exception family (EditStateError vs SourceDeltaError) uniformity later-pass; executor must fold kernel_arguments_from_deltas into run_utci_window (10 kernel_arg are module constants today; transVeg/height/leaf-days need body plumbing; anisotropic_sky double-site :613+:767)
- Gates closed: BASE-001..003, WIN-001..003, CACHE-001..003, PACK-001..002, TREE-001..003, SCI-001..004, API-001..004, UI-001..005, PERF-001..003, MEM-001..002, IO-001, QUEUE-001
- Last green validation tier: main tree — node 61/61 (frontend post-merge); universal validator + spec test green @ eda60d1; last full python suite 329 passed/5 skipped @ post-P8-merge
- Rollback commit: d70137a / c159f38 (P5 merge) / 3a8b4b0 / 38a3fd4 / 987b9e2 (P8 close, pre-universal-docs)
- Environment identifier: darwin 25.5.0, repo .venv py3.14.7, GDAL 3.13.3, torch 2.13.0 CPU, numba 0.67.0, Apple M1 Pro 10c/16GB; server deps: fastapi 0.141.1, uvicorn, httpx 0.28.1, zstandard 0.25.0; docs deps: sphinx 9.1.0, myst-parser, nbsphinx, sphinx-rtd-theme, linkify-it-py, furo

## Completed since previous checkpoint

- T5 CLOSED (eb34ade): sphinx -W -b html docs exit 0 in MAIN post-merge + test_incremental_design_docs 13/13. p9-docs fixed 179-warning worktree baseline to 0 (nbsphinx_prolog titles+download path; http/mermaid fences→text; math→{math}; CONTRIBUTING xref→blob URL; docstring blank-lines ×4 files — lead-verified docstring-only, no code change → no oracle rerun needed; ipython3 lexer defensive). Accepted deviation: suppress_warnings=['toc.not_included'] not exclude_patterns (exclusion would break main's links into agent/universal_editing/benchmarks; agent proved empirically + pre-validated merged state). Worktree removed, branch deleted
- U-A CLOSED (eda60d1): u-auditor report committed at universal_editing/audit/2026-09-02-u-a-audit.md (432 lines; sections a-i with file:line evidence); dependency_graph.yaml edges reconciled (removed [dem,walls], [walls,wall_aspect], [dem,building_visibility], [model_parameters,utci]; added [building_dsm,wall_aspect], [relative_geometry,vegetation_visibility], [walls,time_shadow], [wall_aspect,time_shadow], [wind_coefficients,wbgt], [time_shadow,wbgt]); edit_registry.yaml updated per audit §h + lead adjustments (vegetation: no raster_patch, transmissivity EXCLUDED from v1 schema; selected_date_time → adapter_required; model_receptor_parameters safe/blocked sets w/ onlyglobal BLOCKED; landcover valid_classes [1,2,5,6] water FENCED; output_view producible_incremental w/ local-mode-only extras + wbgt never promised; dynamic_wind wind-coeff parity gap documented as hard fence). Validator + spec test green @ eda60d1. u-auditor's earlier "missing file" concern resolved — file was in main tree all along (worktree false-negative); sha256 verified, agent stood down
- P7 RESIDUALS CLOSED: p9-frontend merged (5461313 on worktree branch → merge commit in main). R-1 connect reconnect w/ bounded backoff 1s/2s/4s×3, backlog replays in order on landing, 4xx fails fast; R-2 reset-during-connecting releases backlog + poll-gen bump + held_edits_discarded + mid-baseline-reset race fixed; R-4 exactBadgeForConnection neutral Connecting until verified baseline. Tests 56→61 (node --test 61/61 fail 0 verified in main tree post-merge); stash-verified the 5 new/changed tests fail with fixes reverted. Worktree removed, branch deleted
- P8 MERGED to main (a1a0da4..104b300; lead read of 104b300 diff confirmed night-gate strictly inside `out_window is not None` branch, oracle path untouched). Post-merge main validation ALL GREEN: full suite 329 passed/5 skipped, oracle bitwise 3/3 hashes == corrected baseline, node 56/56
- P8.6 REMEDIATION (104b300 by p8-perf): night-march gate (time-loop march only when altitude>0; night wbgt uses zero plane matching oracle shadow=zeros — verified vs solweig.py:2028-2162 consumption + Lcyl night path); _IdentityPayloadCache threading.Lock; _maybe_sweep stamps after attempt; sweep cutoff timestamp format normalized; api_contract.md idempotency-after-TTL documented. Correctness proof: 151 worker/windowing/server/codec tests, differential inside_max_abs==0.0 across all 24 timesteps incl. 11 night steps; full suite 329 unchanged. RE-MEASURED PERF: run 2 (idle) p50 54.21 s (was 62.75), p95 59.14 s (was 66.74) → PERF-002 now PASSES original 60 s budget; night-gate −10.7 s (24 marches/12.6 s → 13/1.9 s). T3 runner independently proved oracle bitwise at 104b300 (fresh 24-step full-tile == pre-P8 baseline)
- BUDGET ADJUDICATION (lead, cpu_optimization.md): initial table preserved for provenance + measured table added per the doc's own "must be replaced with measured results" clause. PERF-001 production budget 75 s = 54.21 s × 1.35 target-node factor (launch-bound per thread sweep — 1T 88.1/2T 78.6/4T 68.7/8T 74.9) — initial 20 s unreachable with bitwise-exact design (read window physics-bound 0.80 site; profiled floor SVF-replay 22.7 s + define_patch 16.5 s + gvf/sunonsurface ~13 s). PERF-002 kept at 60 s, PASSES. IO-001 "typical" adjudicated = client default single-variable patch (utci-only 2.67 MB PASS); 5.86 MB two-var full-day recorded w/ mitigation follow-ups. Follow-up work documented in priority order: per-patch march windows (−8-10 s), multi-edit window splitting, IO delta/shuffle. NO scientific tolerance changed
- P8 EVIDENCE CURATED: /tmp/p8perf → docs/incremental_design_tool/benchmarks/2026-09-02-p8-performance/ (19 files: summary, 5 measurement scripts, raw perf runs incl. both nightfix runs, mem/queue/s03/routing/thread-sweep reports, logs; npz binaries excluded). T3 harness scripts preserved against /tmp volatility at .omc/artifacts/t3_harness_2026-09-02/
- P8 WORK DONE (p8-perf agent, 5 commits on worktree-p8-perf from d70137a): a1a0da4 re-enabled local mode on real terrain (per-tree LOCAL elevation offset via bounded iterative candidate-region expansion replacing global DEM-relief offset; solar-series-based shadow cap; dirty fractions 0.1239/0.1864/0.2150 at h6/h10/h18 < 0.30 → LOCAL routing restored); 3a3c41d server O(new-versions) compose + identity payload cache + ledger retention; 129212a write-cropped per-cell physics with read-window marches; e274dc9 solweig.py crops only true per-cell Tgwall fields (additive sky_masks/precomputed_shadows/out_slice kwargs, legacy path when None); 8cf2bc1 run_utci_window validates out_window before compute. S03 resize 917.4→63.9 s (window-relative amplitude). IO-001/PACK-003 + MEM/QUEUE measured; inside_max_abs==0.0 claim corrected by T3 to 1-ulp utci drift 4/8 scenarios. Evidence /tmp/p8perf
- LEAD VERIFICATION of P8 (independent, in worktree @ 8cf2bc1): (1) ORACLE BITWISE PASS — run_test.py full-domain run w/ PYTHONPATH=worktree: sha256 UTCI af54933d…/TMRT adb6ca26…/Shadow 925ee0f7… all == corrected baseline → e274dc9+8cf2bc1 oracle-file edits preserve oracle bits; (2) FULL SUITE PASS — 329 passed/5 skipped/0 failed (6:04), matches agent claim; import origin pre-verified solweig_gpu.__file__ = worktree. Lead-own read of a1a0da4 geometry: from-below monotone local-offset iteration + _exact_influence_window per-cell flip mask (Chebyshev, ds≥1, 1 cm slack) — conservative directions correct, no critical issue
- P8 REVIEW VERDICT: APPROVE-WITH-NITS (p8-perf-reviewer). Contract 1-5 ALL PASS (oracle kwargs keyword-only + None-gated with the `_` rebind hazard chased to diffusefraction; write-crop boundary correct — march/raywalk/Tg at READ extent, elementwise after gvf; amaxvalue early-exit verified a strict prefix of oracle steps; exact-influence mask provable superset; identity cache checksum-keyed no poisoning; retention watermark-safe). Tests judged genuinely strong: _assert_bitwise_inside fails on 1 ulp; conservativeness assertion non-vacuous. Both approval conditions (night gate + lock) landed in 104b300. Open → P9: utci_calculator extent-dependence rationale false on CPU — verify on target device; disc-paints-height vs P4 consistency untested directly
- P8 T3 REAL-SITE RERUN: ALL GATES PASS (p5-t3-runner, evidence benchmarks/2026-09-02-p8-scientific.json @ 31927c4). Worktree advanced mid-run to 104b300; runner resolved provenance rigorously (7/8 executed files byte-identical vs 8cf2bc1; night-skip empirically behavior-preserving). LOCAL MODE RESTORED + POLICY-LIVE on real terrain: S01-S04/S06-S08 route local (dirty 0.0522-0.2492, wall 30.6-71.6 s); S05 full by policy 0.3902. 12-px write margins exercised E2E in every local job. Inside-window: utci 1-ulp (2^-17) drift in 4/8 scenarios (bitwise in other 4 + all tmrt + all shadow) — 10× better than pre-P8, inside envelope. Seam: ring max 0.0, outside bands bitwise vs BOTH store AND edited oracle (0 cells). S04 roundtrip bitwise. Anomalies: amaxvalue guard policy-pre-empted E2E (SCI-004 on prior adjudication); no S09-S12 in harness. Void sanity run detected+disclosed by runner
- T5 docs pre-work (lead): sphinx 9.1.0 + myst-parser + nbsphinx + sphinx-rtd-theme + linkify-it-py installed in .venv; docs build runs; 244 warnings = ~207 pre-existing (notebooks/docstrings/xref — untouched by branch, verified vs 4c5e47d) + ~35 ours (15×`http` lexer, 2×`mermaid`, 2×`math`, 14 orphan-toctree). All mechanical; fix in P9/T5
- P9 DISPATCHED (Wave E, 3 parallel isolated worktrees + T3 spot-check): p9-docs (T5: sphinx -W clean — fix our 35 mechanical warnings at source, orphans via conf.py exclude, ~207 pre-existing minimally; test_incremental_design_docs.py must stay green), p9-rel (REL-001 crash-recovery proof test + REL-002 real-time classroom load test w/ evidence JSON + T6 deployment-smoke pytest + 2 P8-review follow-ups: disc-paints-height pin test, stale tan(5°) comment), p9-frontend (P7 residuals R-1 reconnect+backlog-drain / R-2 reset clears backlog / R-4 connecting badge; regression tests). p5-t3-runner re-engaged for post-merge spot-check (sanity+S01+S05 on merged main; harness preserved at .omc/artifacts/t3_harness_2026-09-02/)
- REPO_OVERLAY POLICY DECIDED (lead): FREEZE — solweig_incremental_design_tool_patch/ is the frozen deliverable package (manifest-pinned from upstream 4131775; prototype only). STATUS.md written in the (gitignored) package dir stating frozen status + pointer to live implementation; updating in place would break MANIFEST.txt sha256 integrity. Refreshed packages = new versioned package, not in-place edits. Closes the recon "update-vs-freeze drift risk" item
- P8-T3 FULLY CLOSED: post-merge main spot-check PASS (p5-t3-runner, 804c646 — additive block in 2026-09-02-p8-scientific.json). Merged main @ a408df5 plain checkout: SANITY fresh-oracle bitwise 3/3 (chain now pre-P8 == worktree 104b300 == merged main); S01 LOCAL dirty 0.1600/55.3s, inside utci+tmrt 0.0 + shadow bitwise, identical to 104b300 evidence, outside bitwise vs store; S05 policy-full 0.390208, tile-wide bitwise. Runner detected+discarded a void first attempt (0.5s-wall skip signature) before the graded recompute — provenance discipline held. P8 phase evidence chain complete
- MISSION DIRECTION CHANGE (user, 2026-09-02 ~08:10): patch 0004 "docs(agent): generalize interactive editing beyond tree adapters" applied (committed; artifact preserved .omc/artifacts/).
 Universal editing spec SUPERSEDES tree-only product scope: tree editing = first vegetation_geometry reference adapter; target = adapter-driven engine over versioned dependency graph (22 nodes, 9 adapters) supporting building/DEM/landcover/met/time/wind-selection/model-param/output-view edit families; dynamic wind stays scientific_extension. New mandatory gates UEDIT-001..010, waves A-E, T0-T6 incl. per-adapter differential + cross-family fixtures. Spec validator green (goal 3962 chars, adapters 9, nodes 22); test_universal_interactive_spec.py 1 passed. PLAN: original tree-only mission (mission.yaml P0-P9) CLOSES FIRST (P9 in flight: REL gates, T5/T6, spot-check, residuals); then universal waves — U-A audits (dependency/registry reconciliation vs repo evidence — master prompt required-first-action), U-B graph+planner+registry+harness+adapters (existing P0-P8 assets seed it: SiteCache, packing, TreeLayer→vegetation adapter, ExactWorker→execution core, server, frontend), U-C scientific integration + non-author review + differential, U-D API/frontend/perf/deployment, U-E red teams. p9-docs notified: new universal_editing/ docs join T5 orphan handling
- U-C SPEC RULING (lead, for U-B harness design + U-C grading — from p5-t3-runner's readiness map, point 2): MIXED-SCOPE PATCHES carry per-node write-scope records. ResultPatch extends to entries [(node, spatial_scope, write_windows)] + transport bbox. Grading semantics: outside-window invariance asserted PER-NODE, not per-patch — local nodes bitwise vs prior published store outside their windows; full-scope nodes graded vs edited oracle full-tile (bitwise). A patch containing any full node is full for that node's products only; local-node windowing is not voided by coexisting global stages. Oracles key on (source overlays × forcing × params × time-selection) composite, not tree scenes. Temporal replay becomes a differential axis (replayed-vs-fresh-restart state comparison). Tolerances from deterministic repeatability + adapter noise only — never per-edit-type tuned (validation_matrix.md:67 affirmed)
- P5 CLOSED (task #12): post-merge T3 subset (sanity+S01+S05+S08) on HEAD fcafe1b + fixed repo cache — ALL GATES PASS (evidence benchmarks/2026-09-02-p5-scientific-postmerge.json, committed d70137a). Runner re-derived all verdicts independently from spec citations; S08 containment rule kept sharp (any vs-edited-oracle deviation outside ALL written windows = hard FAIL; deviations confined everywhere). Runner precision dissent accepted: 6.1e-05 = ~16 ulp at 303 K, consistent with multi-op accumulation reordering. Code-invariance cross-check: merged-code oracles bitwise identical to pre-merge for all scenes.
- MAJOR finding → P8 (recorded limitation, benchmarks/2026-09-02-p5-scientific-postmerge.json): local incremental path POLICY-DEAD on real terrain under merged defaults — terrain-aware bounds (global 44.4 m relief offset + 2170 m length cap) inflate one-tree dirty fraction 0.0635→0.3145 > 0.30 → every edit routes mode=full by policy (guard never reached; SCI-004 E2E gap). Outputs remain exact (bitwise vs oracle in this subset — no local solves ran). 23→12 px margins formula-covered by unit tests but UNVERIFIED end-to-end (no local write window produced). Pre-merge 23-px local-mode correctness evidence stands (2026-09-01 run). Full suite incl. scientific: 304 passed/5 skipped @ 7328485.
- P8 DISPATCHED (p8-perf, opus, worktree from d70137a): per-tree LOCAL elevation offset + solar-series shadow cap to re-enable local mode under the correctness contract; PERF-001/002 budgets (p50<20 s, p95<60 s; PERF-003 already ~130-147 s ✓); MEM/IO/QUEUE budgets; PACK-003 diffsh streaming; server O(versions) compose + identity cache + retention; SCI-004 E2E guard test; S03 917 s investigation. Independent T3 rerun after.
- T3 real-site differential COMPLETE (p5-t3-runner, evidence benchmarks/2026-09-01-p5-scientific.json): sanity PASS — worker run_full_tile reproduces corrected baseline GeoTIFFs bitwise (138 s). S01-S08 on real 500×500 site. SCI-001 PASS (inside-window: max UTCI drift 7.2e-5 °C vs 0.25 envelope, Tmrt 3.1e-5 vs 1.0, p99=0, MAE ~1e-8, shadow bitwise-exact everywhere). SCI-003 PASS (ring == interior magnitude — max 7.2e-5, no seam localization). SCI-004 PASS (S05 tall edit: amaxvalue guard fired verbatim, mode=full, full patch bitwise == edited oracle tile-wide).
- GATE DECISION (lead, SCI-002): S08's strict FAIL came from the harness criterion "store == edited oracle bitwise outside window2", which over-reaches the spec — agent_execution_plan.md:241 defines SCI-002 as "Outside-window STORED results remain unchanged" (validation_pipeline.md: "untouched output regions"; tolerances.py: bit-identical to STORED baseline results). Runner's decomposition proves the 2456 differing cells are 100% inside the DISJOINT prior write window w1 (0-89 px from t1, ≥227 px from t2), bit-identical to job1's measured local-vs-full ulp drift; worker discipline (store2 == store1 bitwise outside w2) and oracle scene invariance (addA == sceneAB bitwise outside w2) BOTH hold. SCI-002 graded vs stored results: PASS everywhere. vs_edited_oracle outside-window retained as diagnostic (runner's own recommendation). No tolerance relaxed — the physical invariant is enforced bitwise against its spec-defined reference.
- Recorded observation (evidence file): local window solve differs from full-tile oracle at ≤7.25e-05 abs (≤8 ulp float32, p95/p99=0, 0 above envelope) in ~0.05% of compared cells on the real site (synthetic tiny-site was bitwise); correlates with full-site read windows (terrain amplitude 44.4 m → march reach past site diagonal). S04 add+delete roundtrip same signature (2500 cells, 6.1e-5, NOT tile-wide bitwise for utci/tmrt; shadow bitwise). Deterministic, within envelope, root cause un-investigated (candidate: float32 accumulation-order differences between windowed and full paths). S03 resize job wall 917 s vs ~120 s others → P8 informational.
- REPO CACHE BUG found+fixed by lead (runner anomaly 3, hard blocker as-shipped): cache manifest solar_geometry.altitude_m=0.0 violated oracle altitude rule (median DEM 193.05 → 3.0); merged solver location validation refuses every job. Rebuilt via cache_builder --altitude 3.0: all 26 arrays bitwise identical, validate-only passes. Root-cause fix 7328485: builder default altitude now DERIVED from site DEM via the same torch.median oracle rule (explicit --altitude verbatim); +3 tests (21 cache tests); real-site default build reproduces corrected cache bitwise.
- P5 remediation MERGED (644cbe1 via b372fe2) after T3 evidence provenance: per-site GVF march margins (2 m site: 23→12 px vs oracle 11), terrain-aware influence bounds, oracle lon/lat+altitude validation, supersede recheck per rename, nodata clamps, scratch cleanup.
- P6 CLOSED (review APPROVE, 2026-09-02 ~01:35): R1 REJECT (watermark inversion, restart recovery) remediated in f1f32ca/b31c11b; R2 narrow REJECT (bootstrap brick: _compose_current_state required v0 while bootstrap publishes at target version on no-baseline sites — most natural UI sequence bricked the scenario permanently) fixed in 9b28b23/5e8956d by composing from lowest published version + deterministic regression test; cancel hardened (publish-txn status guard via ResultNotPublishable w/ full rollback; idempotent re-cancel; race-consistent 409). Reviewer independently re-ran its own brick repro against fixed main — clean. Accept/identity negotiation recorded in api_contract.md (58d13b1). Full suite 285 passed/5 skipped + 75 server/codec + node 56/56 all in main tree.
- P7 CLOSED: re-review APPROVE-WITH-NITS (2026-09-02 ~00:15) — all first-review HIGH/MEDIUM resolved with non-vacuous regression tests, new-defect sweep clean, UI-001..005 ALL PASS, 56/56 re-verified in main tree. Deferred to P8/P9 as LOW residuals: R-1 no reconnect after failed connect, R-2 reset-during-connecting doesn't clear backlog, R-3 serve.py Content-Encoding comment wrong (behavior harmless), R-4 connecting badge "Exact". H-2 identity-serving stays P6-side (connected mode fail-closed until then, documented).
- P7 review remediation merged (644033f via 1cda682): all REQUEST-CHANGES findings addressed — H-1 preview layer wired into connected render path + on-canvas PREVIEW pill; M-1 propertyEditor.rebase() for mid-debounce drags (no server-side position revert); M-2 metrics computed from decoded camelCase window (snake_case-manifest tests without window_fraction); M-3 ready-flag + connect-backlog replay; M-4 reset generation bump + pre-getJob/post-decode generation guards (both races tested); M-5 persisted heat grid dims (non-square safe); L-1/3/4/5/6 fixed. 56/56 node tests ×2 worktree + main tree; smoke_proxy extended (auth forwarded, no empty headers). Re-review dispatched to p7-frontend-reviewer.
- P7 merged (87595d7 via 4863e60): examples/incremental_design_tool connected to exact CPU worker — api_client.mjs (injectable-fetch adapter), exact_session.mjs (DOM-free session state machine), one-POST-per-commit with Idempotency-Key+If-Match, 150ms/480ms preview/commit debounce, 3-layer stale rejection, texSubImage2D sub-rect apply, Failed+idempotent-retry, model-scope limitations card. Zstd finding: browser JS DecompressionStream lacks zstd entirely (verified live on Node 26.8.1 + caniuse 2026-09-01) → client negotiates identity via Accept; P6 payload endpoint must honor Accept (added to p6-api-server remediation). 48/48 node tests in worktree AND main tree post-merge; node --check 7 modules. P7 independent review dispatched.
- Transient infra: machine sleep + DNS drop killed p6-api-server, p7-frontend, p5-t3-runner mid-run ~22:00-22:40; all three resumed with preserved state (P6 8 dirty files intact; T3 parts S01-S03 preserved in /tmp/p5sci/parts/).

- SVF-fix review closed (svf-fix-reviewer APPROVE-WITH-NITS; ledger row added): MEDIUM fixed in 3a8b4b0 — corrupt-but-present SVF cache (truncated tif, garbage zip, zip missing svf.tif) now degrades to recompute via try/except RuntimeError instead of crashing (gdal.UseExceptions had made the `ds is None` guards dead code); 3 new fallback tests, 20 passed in the pair suite. LOW items accepted as hardening gaps (only svf.tif zip member extent-checked; CRS not compared). WORKLOG disagreement-rate precision fixed (26.9/31.9/8.3% per stack).
- P5 independent review (p5-core-reviewer): REJECT→fixable — 1 HIGH (GVF_MARCH_PIXELS=22 hard-coded 1 m pixels; oracle marches round(22 m/pixel_size_m) cells; sub-metre sites silently truncated halo+write margin), 3 MEDIUM (one-directional amaxvalue guard on negative-DSM sites; flat-terrain 300 m influence cap vs oracle's amaxvalue/tan march bound; manifest lon/lat never validated against raster centre), LOWs. Positive: veg-SVF replay bitwise-clean, staging/publish machinery unbroken, full-tile scratch provably cache-safe.
- Review remediation by lead in isolated worktree (644cbe1, worktree-p5-review-fixes — held out of main until T3 finishes): GVF_MARCH_METERS + gvf_march_pixels() per-site halo/write margin (2 m site margin 23→12 px vs oracle 11 px); amplitude guard extended to lowering edits on negative-DSM sites; InfluenceConfig.elevation_offset_m (worker sets DEM relief) + length cap raised to oracle march bound; _oracle_location re-derives lon/lat from Building_DSM centre and rejects manifest disagreement >1e-6°; publish-loop supersede recheck before every rename; tree_base nodata clamp in unchanged-veg guard; load_patch rejects missing checksum entries; scratch cleanup; solar row-count mismatch raises; __all__ completed. +11 tests; 220 non-scientific + 8 scientific green in worktree.
- P6 merged (5e90fb6 via 5488b54): solweig_gpu/server/** FastAPI optional-extra ([server]), SQLite WAL store (migrations, idempotency table, optimistic versions), job runner (coalescing+supersede over ExactWorker, injected-solver tests), zstd patch codec, scenario/job/result routes per api_contract.md, restart recovery. 52 tests ×3 flake check; import-guard verified (core never needs fastapi). Agent self-found 6 bugs incl. sqlite executescript-commit migration break.
- P6 reviewer (p6-api-reviewer) + P7 frontend implementer (p7-frontend, owns examples/incremental_design_tool/**) dispatched in parallel.
- Post-merge main-tree validation: full non-scientific 209 passed/4 skipped (pre-P6), +52 server/codec post-merge; scientific 8 passed (8:15).
- P5 committed (86f0888, worktree agent) and merged c159f38: `incremental/{solver,worker,result}.py` + tests/test_incremental_worker.py (+3000 lines). Synthetic evidence: local write-window outputs BITWISE equal to full-tile oracle for add/move/resize/delete (two-job sequences; delete restores baseline bitwise everywhere); outside-window bit-exact vs both store and edited oracle; seam metrics (boundary ring, distance bands) exactly 0.0; unsafe-edit (dirty fraction) and global amaxvalue hazards raise → full-tile fallback with bitwise-correct full patch; superseded jobs publish nothing. 25 fast + 8 scientific worker tests green in worktree; 209 passed / 4 skipped full non-scientific suite post-merge.
- NEW hazard found+guarded by P5 (adopt into later phases): an edit taller than every baseline structure raises site-wide `amaxvalue`, lengthening the GLOBAL shadow-march so cached building-only SVF goes stale far outside any local window (observed 55-cell divergence at resize h6→h8). `solve_window` raises SolverInputError → worker falls back to full tile. Related: write margin = GVF_MARCH_PIXELS+1 = 23 px (GVF shifts Lup/shadow 22 px into results); met file must live inside site dir (cache_builder _relative); svf_calculator ground correction applies only to svfS/W/Sveg/Wveg terms.
- P3 committed (d793bb0): bit-packed SVF shadow stacks — 20 B/px/stack (30.6x vs dense float32), checksum-validated pack/load CLI (`python -m solweig_gpu.incremental.pack_svf`), weighted reductions from packed match dense float64 to ~6e-14 on the real site_500 cache; 18 unit tests; real cache converted to `processed_inputs/SVF/packed_0_0/`.
- P2 (subagent, 127ca28, merged ad2fc57): `incremental/manifest.py` (schema + sha256 validation), `incremental/cache.py` (SiteCache: mmap-only warm path — zero GDAL/rasterio imports on load, read-only memmaps, self-test at load with fixed-stride checksums above 64 MiB), `incremental/cache_builder.py` (deterministic build from processed_inputs incl. solar series; CLI); 18 tests on synthetic fixtures. CACHE-001/002/003 PASS.
- P4 (subagent, f308008, merged 525d515): `incremental/trees.py` (TreeLayer: read-only base vegetation, add/move/update/delete, coalesced batches, window rasterization max-combined, trunk zone = trunk_ratio*height; reduces exactly to vegdem2=trees*0.25+dem for default) + `incremental/spatial_index.py` (world-space uniform grid, influence-footprint buckets, exact disc/AABB tests, deterministic ordering); 13 tests incl. local==full-sliced over 9 windows and boundary-clipping vs fixtures.rasterize_tree. TREE-001/002/003 PASS.
- T3 metrics harness (695a337): `tests/scientific/{metrics,tolerances}.py` — inward-edge-distance masks, boundary ring (0,2), distance bands ((0,2),(3,8),(9,16)), continuous/binary region metrics (float64 upcast, percentiles), inside/ring/outside comparison reports, exact outside-window equality assertion; 28 unit tests. Real-oracle differential scenarios S01-S12 land with P5.
- Merge resolution: `incremental/__init__.py` combined export set (io + cache/manifest + trees/spatial_index + packing); full suite 132 passed after both merges.

## Gate evidence

```text
Gate: WIN-001 (Existing public API remains functional)
Status: PASS
Command: .venv/bin/python -m pytest -q tests/   (PATH=$PWD/.venv/bin:$PATH)
Result: 99 passed, 5 skipped; node --test 22 passed; run_test.py end-to-end still emits baseline outputs
Note: tests/test_cli.py failures are a PATH environment artifact (console script lookup), reproduced without this change

Gate: WIN-002 (Full-window numerical output unchanged within tolerance)
Status: PASS (bitwise, stronger than tolerance)
Command: /usr/bin/time -l .venv/bin/python run_test.py
Result: sha256(UTCI/TMRT/Shadow) identical to P0 baseline (351e18ad…/04d91548…/925ee0f7…); peak RSS 1.24 GB (was 1.41 GB)
Extra: direct run_utci_window(utci,tmrt,shadow) vs oracle GeoTIFF bands: np.array_equal(equal_nan=True) all True

Gate: SCI-001 (Add/move/update/delete match full recomputation)
Status: PASS
Commit: c159f38 worker + b372fe2 remediation merge (subset rerun pending below)
Command: /tmp/p5sci/run_scenario.py S01..S08 (p5-t3-runner harness, real site_500, 24 timesteps)
Result: all 8 scenarios within tests/scientific/tolerances.py envelope: UTCI max abs 7.2e-5 (tol 0.25), Tmrt 3.1e-5 (tol 1.0), MAE ~1e-8, p99 = 0.0, count_above_tolerance = 0; shadow bitwise-exact every scenario. NOT bitwise local==full on real site (≤8 ulp, deterministic) — recorded observation, see notes
Artifacts: docs/incremental_design_tool/benchmarks/2026-09-01-p5-scientific.json

Gate: SCI-002 (Outside-window stored results remain unchanged)
Status: PASS
Commit: c159f38 worker + b372fe2 remediation merge (subset rerun pending below)
Command: /tmp/p5sci/run_scenario.py S01..S08, outside-window exact_diff vs prior published store
Result: store bitwise-unchanged outside every write window in all 8 scenarios (n_different_cells = 0). Harness's additional vs_edited_oracle bitwise sub-check FAILED on S08 only; decomposition (evidence JSON, gates.SCI-002.s08_failure_decomposition): 100% of the 2456/3558 differing cells lie inside the DISJOINT prior write window w1 (0-89 px from t1, ≥227 px from t2), magnitudes bit-identical to job1's measured local-vs-full ulp drift — worker discipline and oracle scene invariance both intact. Graded against the spec definition (stored results; agent_execution_plan.md:241, tolerances.py OUTSIDE_WINDOW_EXACT); vs_edited_oracle retained as diagnostic per runner recommendation. No tolerance relaxed
Artifacts: docs/incremental_design_tool/benchmarks/2026-09-01-p5-scientific.json

Gate: SCI-003 (Boundary error thresholds)
Status: PASS
Commit: c159f38 worker + b372fe2 remediation merge (subset rerun pending below)
Command: /tmp/p5sci/run_scenario.py boundary ring (0,2) + distance bands ((0,2),(3,8),(9,16))
Result: ring max abs == interior magnitude (7.2e-5 UTCI / 3.1e-5 Tmrt vs envelope 0.25/1.0), shadow ring mismatch 0.0 — no seam localization anywhere
Artifacts: docs/incremental_design_tool/benchmarks/2026-09-01-p5-scientific.json

Gate: SCI-004 (Fallback activates for unsafe local jobs)
Status: PASS
Commit: c159f38 worker + b372fe2 remediation merge (subset rerun pending below)
Command: /tmp/p5sci/run_scenario.py S05 (tall edit) 
Result: mode=full; fallback_reason verbatim "scene amaxvalue (236.35) exceeds the cached baseline (228.34); building-only SVF terms would be stale everywhere — full-tile recompute required"; full patch BITWISE equal to edited oracle on the entire tile; wall 147 s
Artifacts: docs/incremental_design_tool/benchmarks/2026-09-01-p5-scientific.json

Gate: WIN-003 (Peak memory decreases when only UTCI is requested)
Status: PASS
Command: .venv/bin/python /tmp/win003_measure.py utci  |  utci,tmrt,kup,kdown,lup,ldown,shadow,ta,wind
Result: output volumes 24 MB (utci) vs 216 MB (all nine); peak RSS 1,033,830,400 B vs 1,058,357,248 B; pre-refactor code allocated all seven output lists unconditionally
Artifacts: docs/incremental_design_tool/benchmarks/2026-09-01-p1-windowing.json
```

## Files changed intentionally

- solweig_gpu/utci_process.py (new run_utci_window core; compute_utci wrapper; landcover None-init + explicit clean re-wrap)
- solweig_gpu/incremental/io.py (new)
- solweig_gpu/incremental/__init__.py (io exports)
- tests/test_incremental_windowing.py (new)
- docs/incremental_design_tool/benchmarks/2026-09-01-p1-windowing.json (new)

## Commands and evidence

```text
.venv/bin/python -m pytest -q tests/test_incremental_{geometry,bitmask,edits,fixtures,windowing,design_docs}.py -> 55 passed
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest -q tests/ -> 99 passed, 5 skipped
node --test -> 22 passed
shasum run through refactored path == P0 baseline hashes (3/3)
```

## Recon findings (all Wave-A agents reported 2026-09-01 ~18:00; see SUBAGENT_LEDGER)

Hazards adopted into later phases:

- **SVF cache staleness — MANIFESTED AND CORRECTED (2026-09-01, post-P4)**: the staged site cache itself was foreign (origin (622396.71, 3356864.35) vs staged tile (621734.71, 3354614.35); earlier run over a different mosaic window silently reused because the cache key is the tile filename only). Fresh recompute disagreed on 26.9% (shadowmat) / 31.9% (vegshadowmat) / 8.3% (vbshmat) of cells. Remediated: staged SVF replaced with wrapper-identical fresh output (stale evidence at /tmp/svf_stale_foreign_evidence/); `utci_process` now validates cache raster extent (size + geotransform within 1e-6) against the Building_DSM tile at both `_svf_cache_exists` and `load_cached_svf_outputs`; oracle re-recorded bitwise-deterministic (new hashes in svf_cache_correction block of the P0 baseline record; Shadow hash unchanged — SVF feeds radiation only); WIN-002 revalidated bitwise 3/3; P3 packed cache regenerated + PACK-002 reverified; P2 site cache now builds (Input_subset/processed_inputs/incremental_cache_0_0). P5 must STILL treat tree edits as invalidating vegetation-dependent SVF (extent validation does not catch content changes from edits).
- **In-place `Tg` mutation** for water cells inside `sunonsurface_2018a` (solweig.py:130): windowed worker must not alias temporal state across window/full instances.
- **Memory**: 4 dense 153-patch stacks (shmat/vegshmat/vbshvegshmat + diffsh) ≈ 584 MiB resident at 500x500 — PACK-003 (streamed diffsh) removes ~146 MiB and P3 packed caches remove more.
- **Frontend**: API adapter seam, exact_result_version tracking, Queued/Superseded/Failed states, texSubImage2D region upload, zstd decision — captured in frontend-auditor report for P6/P7 packets.
- **Validation**: single fixture generator rule (reuse solweig_gpu/incremental/fixtures.py — no second fixture system); pytest markers `scientific`/`nightly`/`benchmark` to register before T3 lands; S01-S12 fixture plan adopted for T3 design.
- `solweig_incremental_design_tool_patch/repo_overlay/` is a frozen mirror of incremental/ — drift risk; decide update-vs-freeze policy before Phase 9 (noted, no action yet).

## Known failures / uncertainty

- test_cli.py needs .venv/bin on PATH (pre-existing env artifact, not a regression; reproduce with `PATH="$PWD/.venv/bin:$PATH"`).
- P2 cache records solar/met arrays float32 per data_model.md; P5 worker must consume them without re-deriving (single source of truth = manifest).
- p2 worktree branched from stale main@4c5e47d (agent confirmed, noted in ledger); merge ad2fc57 resolved cleanly apart from `__init__.py` (lead-resolved). All 132 tests green post-merge.

## Next executable step

```text
1. Dispatch P5 exact local scientific worker (worktree isolation, disjoint files: incremental/{worker,solver,result}.py + tests).
   Packet: reuse run_utci_window (P1), SiteCache (P2), packed SVF (P3), TreeLayer/spatial index (P4);
   hazards: SVF staleness after tree edits, Tg in-place mutation, PACK-003 diffsh streaming, replay-from-0 temporal policy.
2. After P5 merge: independent reviewer (scientific core) + independent T3 runner (real-site differential, tests/scientific harness, S01-S12).
3. Then P6 API/persistence (frontend-auditor gap list), P7 frontend, P8 perf, P9 validation.
```

## Recovery notes

- Oracle outputs snapshot: /tmp/solweig_baseline_run/run1/
- WIN-003 harness: /tmp/win003_measure.py (staged inputs under Input_subset/processed_inputs/)
- Packed real-cache conversion: `.venv/bin/python -m solweig_gpu.incremental.pack_svf --npz Input_subset/processed_inputs/SVF/shadowmats_0_0.npz --out-dir Input_subset/processed_inputs/SVF/packed_0_0`
- fixtures/site_500 regenerable via documented command
- Worktree branches (unmerged refs retained): worktree-agent-a55e8356b1f9069ed (P4, f308008), worktree-agent-ad95b41b391e94b58 (P2, 127ca28)
- Rollback: git checkout d793bb0 / 42bd024 / 05e4ce4
- U-D WAVE CLOSED (2026-09-03, HEAD 748ba7c + ledger 8f86c40): full-suite canonical 983 passed / 4 skipped / 1 failed (austin = sanctioned env, pre-existing on origin/main@4c5e47d); node 124/124 @ 748ba7c. U-D packets: u-d0 residual sweep; u-d1/u-d1b capability API + scenario persistence + site pinning (r2 APPROVE-WITH-NITS); u-d3 UEDIT-007/008 evidence (PERF-001/002/003 + MEM-001 PASS); u-d2/u-d2b registry-driven frontend + UEDIT-009 (r2 APPROVE); u-d4/u-d4b universal transport + plan exposure + deployment (r2 APPROVE after stage-then-adopt coverage protocol); u-d4c/u-d4d building-accumulation defect fix (massing fold + scenario-state v3; r2 APPROVE); lead 429be46 SCENARIO_CACHE weakref flake fix. UEDIT-007 PASS, UEDIT-008 measured+PASS, UEDIT-009 server-served, UEDIT-010 3-site disclosure. Intake items (a)-(i) closed; (j) GPU-path differential standing OPEN (documented); (k) selected_date_time + dem remain typed NOT_INTEGRATED refusals (disclosed). NEXT: U-E — red teams (protocol adversarial + adapter audit incl. pre-committed u-auditor), then UEDIT-001..010 adjudication + mission_universal completion audit
- U-E wave checkpoint 2 (2026-09-03): u-e3 merged 57def21 (13 findings, all failing-first; full suite 996/4/1 austin-only, node 124/124). u-d4-reviewer r1 verdict APPROVE-WITH-NITS (2 residuals) + u-e1 independent verify (F1-F5 fixed bitwise; attack C residual; NF1 NEW MEDIUM pre-existing post-reset tree wedge) — both logged as ledger rows U-E3-review/U-E1-verify. u-e3b lead-direct 76b4b90: v5 migration backfill + strict=True time_index (both model fields) + ops doc subsection; 3 tests RED-at-57def21 → GREEN. Remaining before gate adjudication: u-e3c packet (NF1 post-reset tree wedge + predicate attack C closure via executor-state dir-presence OR / dir backfill; design in executor_bridge+executor; independent review), final full canonical suite, then UEDIT-001..010 + mission_universal completion audit.
- U-E wave checkpoint 3 (2026-09-03, u-e3c line CLOSED): u-e3c merged e914c17 (NF1 voided-tree-base replay + attack C reset-aware dir rescue; failing-first runtime-verified by reviewer at 68536a0 = 3 failed/1 passed as claimed). u-e3c-reviewer APPROVE-WITH-NITS (2 LOW) → both closed lead-direct ef4d86f (STATE_DIRECTORY_NAME constant + compound-corner ops disclosure; reader/bridge trust-rule ALIGNMENT deferred as disclosed residual). u-e1-redteam independent verify: NF1 + attack C FIXED bitwise, original repros healed; lead priority probe (baseline-tree-delete post-reset) UNREACHABLE — three empirical barriers (no baseline tree list; typed 400 unknown tree_id; reachable deletes bitwise-correct); fresh attacks clean (cross-scenario dirs, durable watermark vs event pruning, staged-pending double-guard, modern-layout snapshot wipe self-heals); pruned-RESET residual BETTER than disclosed (typed refusal, never wrong science); legacy-layout wedge unchanged/not widened. Lead full canonical suite @ e914c17: 1003 passed / 4 skipped / 1 failed = austin sanctioned only. Remaining: UEDIT-001..010 adjudication doc + mission_universal completion audit + DONE evaluation.
- UEDIT-001..010 ADJUDICATED 10/10 PASS (2026-09-03, docs/incremental_design_tool/universal_editing/audit/2026-09-03-uedit-adjudication.md): every gate on runtime evidence (canonical suite tests, reviewer failing-first history, red-team bitwise repros, benchmark artifacts); 10-entry disclosed-residual register, none re-opens a gate. NEXT: final full canonical suite + node suite at closing HEAD, mission_universal completion audit, DONE evaluation.
- MISSION_UNIVERSAL COMPLETE (2026-09-03, closing HEAD a4a9fc1): UEDIT-001..010 10/10 PASS (adjudication doc); final full canonical suite 1003 passed / 4 skipped / 1 failed = sanctioned austin env failure only (pre-existing on origin/main); node 124/124; deployment smoke green inside suite; CPU/RSS per family recorded (PERF-002 thin margin disclosed); limitations explicit (10-entry residual register); tree clean. Completion audit: docs/incremental_design_tool/universal_editing/audit/2026-09-03-mission-completion-audit.md. Ledger 117 rows, waves U-A..U-E closed, no STOP condition ever hit.
- U-DEP1 deployment hardening (2026-09-03, worktree base b409d0c, branch worktree-agent-udep1): `python -m solweig_gpu.server` now boots from environment variables (solweig_gpu/server/__main__.py: pure `app_from_env(environ) -> (app, uvicorn_kwargs)` builder + `main()`; SOLWEIG_STATE_ROOT / SOLWEIG_SITE_ID (comma-separated multi-site) / SOLWEIG_CACHE_ROOT (cache resolves to CACHE_ROOT/<site_id>) / SOLWEIG_HOST / SOLWEIG_PORT / SOLWEIG_MAX_TREES_PER_SCENARIO / SOLWEIG_COALESCE_MS / SOLWEIG_REQUESTS_PER_MINUTE / SOLWEIG_EDITS_PER_MINUTE + per-site SOLWEIG_SITE_DIR_<ID> and SOLWEIG_SELECTED_DATE_STR_<ID> where <ID> = site id upper-cased with non-alphanumerics → underscore; unset optional vars keep create_app defaults (only explicitly-set kwargs are passed); invalid value = clear error naming var+value, exit 2; unknown SOLWEIG_* = stderr warning, boots anyway; fastapi/uvicorn/torch imported only after config validates). tests/test_server_main.py: 25 tests FAILING-FIRST (RED at b409d0c: ModuleNotFoundError 'solweig_gpu.server.__main__'; GREEN at head) — 24 unit tests via injected fake app factory (no server booted: exact kwarg mapping, defaults, multi-site, per-site overrides, all invalid-value exits, unknown-var warnings, main() exit codes) + 1 `deployment`-marked subprocess test booting the REAL module against a synthetic site cache (test_server_api.make_site_cache) on an ephemeral port over HTTP. Container artifacts (all new, repo root): Dockerfile two-stage — builder compiles osgeo GDAL bindings against Debian libgdal 3.10.3 (no Linux wheels exist) + torch 2.14.0+cpu from the PyTorch CPU index so CUDA wheels never enter; runtime = python:3.14-slim + libexpat1 + libgdal36 only, non-root uid 10001 with pre-owned /state, EXPOSE 8000, CMD ["python","-m","solweig_gpu.server"]; image 2.86 GB. VERIFIED: container boot with synthetic site-cache RO bind (/site-cache) + named state volume (/state) → /health/live 200, /health/ready 200 (store schema v6, site ok), /api/v1/capabilities 200 (9 adapters, 8 families), state written by uid 10001; mandated data-leak check `find /app -maxdepth 2 -name 'Input_*' -o -name 'ERA-5'` EMPTY; broader sweep clean (benign only: docs/input_data.md filename, tracked synthetic tests/fixtures); paper/ + pip build residue excluded. docker-compose.yml (api + frontend serve.py same-origin proxy; solweig-state RW named volume, ./site-cache:ro bind) — `docker compose config` valid; fly.toml sample (performance-1x 2GB, solweig_state mount, cache provisioned out-of-band, never baked) — TOML valid. Docs truth: deployment_operations.md §Configuration rewritten to (a) implemented env-var table, (b) process-env-only C-lib table (OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS, GDAL_CACHEMAX), (c) not-implemented list (SOLWEIG_STATE_DB/RESULT_ROOT removed — state layout derives from state_root: store.sqlite3 + scenarios/; JOB_COALESCE_MS → COALESCE_MS; WORKER_COUNT/SCIENTIFIC_THREADS/FULL_RECOMPUTE_FRACTION/SCENARIO_TTL/RESULT_TTL never implemented) + new "Container deployment" subsection (RW state vs RO cache volumes, numba cache=False → seconds of JIT per process start). examples README boot block fixed (the previously-failing `python -m solweig_gpu.server` example now shows the working env-var form). VALIDATION: targeted test_server_main+test_deployment_smoke+test_server_api = 95 passed; FULL canonical at head = 1028 passed / 5 skipped / 0 failed (16:33) — the historical "1 sanctioned austin failure" does not reproduce with a properly installed austin CLI. ENV FINDING: the pre-existing .venv was destroyed mid-task by external disk cleanup (host hit 100% and Docker Desktop wedged twice — recovered only after ~79 GB was freed); the rebuilt venv initially lacked pyyaml + austin, which the old venv had installed ad hoc (yaml is required by tests/test_universal_interactive_spec.py's validator; neither dep is in any requirements file) — recommend lead add pyyaml to the test extra or the setup docs.
- U-DEP WAVE (2026-09-03, user-requested research-level CPU deployment): platform research — Supabase compute RULED OUT (Edge Functions Deno/256MB/2s CPU, no Python processes); Fly.io ranked #1 (~$12/mo), Railway #2, Hetzner #3; GT PACE policy-barred. deploy-shape audit found the blockers (no boot entrypoint — README's documented command FAILED; all 12 SOLWEIG_* env vars doc-only; no Dockerfile). u-dep1 packet merged a51c441: solweig_gpu/server/__main__.py env-var boot (app_from_env pure builder + fail-fast + per-site overrides), 25 failing-first tests, two-stage Dockerfile (python:3.14-slim, CPU-only torch, libgdal built in builder, non-root uid 10001, image 2.86GB), docker-compose (api + same-origin frontend proxy, state RW / cache RO volumes), fly.toml, .dockerignore data-policy fence, deployment_operations.md §Configuration rewritten to implemented-truth tables, README boot block fixed. u-dep1-verify (non-author, independent docker rebuild) APPROVE-WITH-NITS → all 4 closed lead-direct 269bcf8 (site-id segment guard RED-first, compose healthcheck + service_healthy, **-anchored dockerignore backstops, README STATE_ROOT, pyyaml→test extra after the audit's disk-full venv rebuild exposed the undeclared dep). Container evidence across both agents: /health/live+/health/ready+/capabilities 200, state owned by non-root, leak finds EMPTY. ENV NOTE: rebuilt .venv (old destroyed by host disk-full during the packet) lacks adhoc earthengine-api → austin test now SKIPS rather than fails; canonical expectation updated to 0-failed/austin-skip on this env.
- W1 PERF WAVE MERGED (2026-09-04, merge ce76d79 into feat/incremental-design-worker; W1 agent worktree branch, never pushed): honest telemetry (additive `read_window_fraction`, `svf_seconds`, `time_loop_seconds` in job metrics; `window_fraction` unchanged), heartbeat ticker (heartbeats flow during gated solves — kills the /health/ready 503 starvation window), causal time-prefix truncation (`resolve_time_stop` server-side + `solve_window(time_stop=)`/`run(time_stop=)`/bridge `all_times` truncation; full-series kept on legacy `compute_utci`, forced-full/fallback, building-regen chain, and PlanExecutor dispatch — node store records full-series coverage, an existing contract test asserts it), trivial dedups (compose/read-window derived once per solve; `cube_window` pre-cropped SVF bundle saves the per-solve shadowmat crop). 22 new tests (tests/test_incremental_perf_wave1.py; RED 14F/3P → 22G) incl. scientific bitwise gates: served step-12 utci/tmrt `np.array_equal(..., equal_nan=True)` on the windowed local path, prefix-truncated `run_utci_window` vs full bitwise, pre-cropped cubes bitwise. Full canonical at W1 head: 1056 passed / 5 skipped (baseline re-verified 1034+5 pre-change). LIVE BENCHMARK on native macOS site_500 after merge (native API restarted on ce76d79, real HTTP scenario + tree add + time_indices [12]): two consecutive jobs `duration_ms` 41514 / 42246 (~41.5-42.2s) vs 71s pre-W1 native = 1.71x; phase split `svf_seconds` ~24s / `time_loop_seconds` ~17.1s / rest ~1s; `read_window_fraction` 0.988 honest (occluder reach 537px > 500px tile); mode local, no fallback. NEXT: W2 agent (worktree) targets svf_seconds ~24s → ≤10s via per-patch march windows + invariant hoist, bitwise-gated; then W4 independent verification at final HEAD.
- DEPLOY-HOTFIX (2026-09-04, live MacBook+tunnel deployment, user-reported "Exact server unreachable"): reproduced in Orca embedded browser. Three real defects, all closed failing-first (node 125→127 green): (1) "?api=/api" DOUBLE-PREFIXED every request (/api/api/v1/... → 404 at connect; the README's own documented proxy URL was broken and no e2e test covered studio-via-proxy) — resolveApiBase() now de-doubles the same-origin proxy prefix; (2) serve.py sent no cache headers so a heuristic browser cache masked the deployed fix (stale app.mjs) — static responses now send Cache-Control: no-cache; (3) the server HONORS Accept negotiation (docs claimed it always serves zstd): an identity-served payload was verified against the manifest's zstd-encoding checksum → deterministic checksum mismatch at connect — decodePatch now verifies against the X-SOLWEIG-Checksum header (digest of the SERVED bytes, already shipped by the server) and decompression follows the served content type, not the manifest compression field. PERF: populated site-cache baseline_results/ via new tools/export_baseline.py (decodes a completed scenario result into <cache>/baseline_results) — scenario creation went from a ~7-minute full-solve per scenario to 0.44s instant materialization (status "exact", no job). Browser end-to-end verified in Orca: connect → "Exact analysis ready", capability-generated editors render, tree Add click → "exact queued" job. Python suite NOT rerun this round: host .venv was destroyed outside the session (disk cleanup); node suite 127/127, no Python source changed (serve.py header + frontend .mjs + tool only).
- W2 PERF WAVE MERGED (2026-09-04, merge 838e420; W2 agent worktree branch, never pushed): solver.py-only change — per-patch march windows (`_patch_march_window`: window expanded by ring reach `ceil(amp·scale/tan(alt))+1` in READ directions only, clamped to read window; bush gate falls back to full-window; scalars zero-padded to read extent so the 19-tuple bundle contract is untouched) + hoisted sky-patch geometry (`_SKY_PATCH_GEOMETRY` memo: create_patches(2), aziintervalaniso, per-ring annulus-weight pairs — once per process). march cell-steps 2.83e9→1.41e9. 18 tests (wave2 suite; RED 18F → 18G) incl. corner-quadrant bitwise + A/B gate + memo counts. Full canonical at W2 head: 1072/5/2, the 2 = pre-existing test_cli env artifact (venv-on-PATH passes; identical at base via stash). Cross-HEAD real-site served hashes identical (tmrt a46d5594ed862d7d, shadow eb918939dca91a34, utci 7c0cbac6178ac13c, artifacts /tmp/w2_bench/{before,after}). LIVE post-merge native: 38.9/38.0s vs W1 41.5s (cumulative pre-wave 71s → 1.87x); svf_seconds live 20.0-20.3 (in-worktree 16.9 @ threads=8). Disclosed residual: ~40% of svf_seconds is kernel-launch overhead in shadow() internals — next lever if ever needed (out of adjudicated scope); refused levers documented (cross-patch batching = float determinism risk, amaxvalue tightening, occluder-bbox windows). NEXT: W4 independent non-author verification at 838e420 (full suite + node + cross-wave differential base 3bd8aba vs head + live bench reproduction + metrics/heartbeat spot-checks), then goal evaluation.
- W4 INDEPENDENT VERIFICATION CONFIRMED (2026-09-04, final HEAD 838e420, non-author w4-verifier, read-only): every W1+W2 claim reproduced with independent evidence, 0 findings above INFO. Suite: full canonical 2F/1072P/5S exact (test_cli env confirmed pre-existing at base 3bd8aba without venv PATH, passes 2/2 with it); node 127/127. BITWISE cross-wave gate PASSED: own harness, real site_500 cache (read-only), identical edit at 3bd8aba vs 838e420 — full-solve AND windowed full-series utci/tmrt/shadow all np.array_equal(equal_nan) base==head; time_stop=13 prefix bitwise == steps 0..12; HTTP-served patch bytes == direct solver planes; rep-to-rep served hashes equal. Metrics honesty independently rederived (read 247005/250000=0.98802, write 73920/250000=0.29568 — exact). Heartbeat: 6 mid-job /health/ready probes all 200. LIVE reproduction: 37.5s/37.8s (svf 19.8s, tloop 17.1s). INFO findings: W2's ledger phrase served-plane hashes = solver-level outputs (clarified in W2 row); default patch carries utci+tmrt only (shadow solver-level); svf 16.9 threads=8-specific (default 18.2). WAVE CLOSED: cumulative live single-tree-edit 71s (native pre-wave) → 37.7s avg = 1.88x, bitwise-identical science, adjudicated steps 1-3 complete; disclosed next levers (NOT taken, out of adjudicated scope): shadow() per-step kernel-launch overhead (~40% of svf_seconds) and delta-SVF reuse for repeat edits. Adjudication journal: session workflows dir w78l8fw6q; artifacts /tmp/w4_bench.py, /tmp/w4_differential.py (verifier, ephemeral).
- R1 WAVE COMPLETE (2026-09-04, HEAD 1d14751): all four R1 packets merged into main — r1-telemetry ca6587a (solweig_rt_* registry, 18 tests), r1-frontend 845d017 (RealtimeClient 797 ln, node 127→137), r1-reducer f12c0fb (DeterministicEpochReducer 912 ln, per-family conflict semantics, 26 tests), r1-ops 6b2d42f (durable idempotent op log, migration v7, /workspaces routes, 21 tests), r1-epochs d5858fd (100ms EpochScheduler + SSE BroadcastHub + zero-loss catch-up + acks alias, migration v8 realtime_canonical_state, 31 tests). Client↔server wire contract verified end-to-end against realtime_client.mjs (acks alias + canonical_revision payload shape + named SSE events + GET ?since_server_sequence). LEAD post-merge (1d14751): test-isolation fix — default-wiring test apps leaked real-thread epoch telemetry into the module-shared singleton, breaking test_realtime_telemetry count asserts in combined runs (bisected per-file + registry-snapshot probe; operations make_rt_app + 2 epochs default-wiring tests now on private TelemetryRegistry); combined realtime suites 102 passed 3×; node 137/137. Non-author review of r1-epochs DISPATCHED (r1-epochs-review, probe-driven, /tmp/r1_review). Full canonical python suite in flight at 1d14751. R1-flagged for R2: legacy /edits bypasses the op log (scene_version can advance outside epoch commits — funnel decision); epochs end status='reducing' awaiting the R2 fast lane; /metrics → default_registry wiring. IN PARALLEL: fly.io demo deploy (user request, cheapest shape, $14/mo ceiling) — single-volume fix committed 07c0e53 (fly mounts max 1 volume/machine: state+cache share solweig_disk 3GB at /data), stale 2GB volumes deleted; performance-1x REFUSED by trial org ("not allowed to use performance machines") → shared-cpu-1x 2GB 1d14751, redeploy in flight; site-cache upload (44M tarball) + smoke after machine boots.
- R1 GATE EVIDENCE (2026-09-04, HEAD 1d14751): full canonical suite 1176 passed / 5 skipped / 0 failed (1153.40s, load 6.22-7.51-8.13); node 137/137; combined realtime suites 102 passed 3x (determinism). Fly demo DEPLOYED + SMOKED LIVE: https://solweig-rt-demo.fly.dev/ — studio 200, capabilities 200 (9 adapters), scenario create instant-exact from baseline_results, result manifest 200; shared-cpu-1x 2GB + suspend-at-zero on solweig_disk 3GB (/data/state + /data/cache/site_500 uploaded out-of-band). R1 remaining: r1-epochs-review verdict.

## R1 CLOSED — review remediation landed (2026-09-04)

- r1-epochs-review verdict APPROVE-WITH-NITS (0C/0H/2M/3L/2T). Lead adjudication: fix all code findings + doc pins NOW, failing-first, rather than carrying debt into R2.
- Remediation MERGED 60b5e48: M1 stranded-epoch live redrive (oldest-first, no epoch-id/revision inversion), M2 stale-baseline typed refusal + fresh-baseline re-fold (EpochBaselineStale in commit txn), L1 SSE subscribe TOCTOU corrective snapshot (privileged push_snapshot), T1 debug-log closed-loop drops, T2 RuntimeError tripwires (strictly-increasing ordering — plain sorted() missed duplicates), L2 + L3 doc pins (missed_events forward-compat; revisions may SKIP under mixed direct-edit + realtime, contiguity not a contract).
- Evidence: 6/6 new tests green (were 6/6 red at 1d14751); combined realtime suites 108 passed ×3 deterministic; test_server_api + test_server_main 99 passed. Host load 14.18 (high but stable).
- R1 COMPLETE. Next: r3a-met-fast arrived (verify → merge → non-author review + differential), then dispatch r2a per .omc/plans decomposition.

## r3a MERGED ae0bccb — R4 Phase A dispatched (2026-09-04)

- r3a-review REQUEST-CHANGES → author remediation (C1 server-bridge sparse scatter at jobs.py/executor_bridge.py, M1 policy consult, M2 docstring-truth, L1 value pin, L2 anti-vacuity) → re-review APPROVE. Fallback boundaries + consumer sweep + E2E regression genuineness all verified by non-author.
- Merge clean by construction (zero file intersection since fork). Worktree+branch removed.
- Perf independently reproduced under load: 128x128 fast 0.717s vs 3.789s establishing; site_500 0.840s vs 127.40s; bitwise utci@12 parity. p99 distributional evidence still open (non-blocking).
- R4 Phase A DISPATCHED (r4a-veg-svf-state, worktree from ae0bccb): packed occluder state + corridor re-march + canonical re-fold, 4 binding conditions from r4-proof-review verbatim in prompt, W2 reach-expanded windows mandate, no server files (r2a owns).
- r2a CODE-COMPLETE at ed5b4ec (canonical 1194/5/0 at branch) — author now rebasing onto ae0bccb (jobs.py overlap with r3a C1 hunks; sparse semantics must survive restructure + regression test). Non-author review after merge.
- In flight: canonical-on-merged-main (bkio1zaby), r2a mandated verification in its worktree (b4470ep69), r2a rebase, r4a Phase A.
- Post-merge canonical on main @ ae0bccb: 1217 passed / 6 skipped / 0 failed (1218.10s, load 9.64). r3a merge verified green.

## r2a MERGED 03d58e2 — r2b + r2a-review dispatched (2026-09-04)

- r2a rebase onto r3a-merged main: auto-merge, zero conflicts; sparse time_indices semantics verified verbatim at both sites; regression test added at compose seam. Canonical at branch 1230/6/0.
- Merged to main 03d58e2 (worktree+branch removed). Post-merge canonical running (nohup /tmp/r2a_merge_canonical.log).
- DISPATCHED in parallel: r2a-review (non-author adversarial: double-apply, loss windows, supersession, segment-skip disclosure adequacy, If-Match) + r2b (fast lane thread + admission + fast result classes v1 + /metrics wiring; tail hook only in epochs.py; store.py off-limits — in-memory fast payloads this wave).
- r4a Phase A still in flight.
- Post-r2a-merge canonical on main: 1230/6/0 (1177.46s, load 4.47).
- r2a-review REQUEST-CHANGES: architecture sound under attack, but C1 pre-gate object blindness (bitwise-wrong 26k+ cells, undisclosed, public-API reachable) + H1 unmapped-epoch permanent wedge. Both fast-follow must-fix before any mixed-traffic demo. r2a-fix DISPATCHED (fresh worktree; gate-open bootstrap + unfolded-op fence + no-op routing + manifest limitations + 5 fence tests).
- In flight: r2b (fast lane), r4a Phase A, r2a-fix.
- r4a Phase A CODE-COMPLETE ff932de (red 34 at base → 36/36; canonical 1253/6/0 at branch). Key deviation flagged: condition-3 amplitude wording (full-tile vs windowed) — agent's value-divergence argument (one-step regimes, vbsh 2.0) plausible; r4a-review dispatched with that as top adjudication priority + bitwise parity, guard trichotomy, corridor sufficiency, state lifecycle, red-line audit, independent perf.
- In flight: r2a-fix, r2b, r4a-review, r4a lead verification (/tmp/r4a_verify.log).
- r4a-review REQUEST-CHANGES (1H/2M/1L/1N; arch/red-lines/corridor-∪C/trichotomy PASS; condition-3 full-tile deviation ADJUDICATED COMPLIANT-AS-IMPLEMENTED): H1 regime-level one-step fence gap — per-window condition-4 fence can't see vbsh==2.0 at unchanged neighbors outside marched windows (reviewer repro 324 cells, svfWaveg 0.1408 silent); M1 march_offsets(cols,rows) transposed (91/306 pairs lose steps non-square); M2 pre-existing W2 replay windowed-amplitude divergence — LEAD-RULED separate packet (NOT r4a blocker); L1 perf framing (corridor ~1.3x slower than deployed-shape W2 replay — win is vs full-tile only); N1 fold constants unprotected (need oracle-anchor test).
- r4a lead verification: 121/121 green combined in worktree (304.97s).
- r4a-remediation DISPATCHED (author resumed in worktree): H1 regime fence (arithmetic eff_post < dz_1(p), batch refusal + baseline-pack companion) + M1 arg order + ADDENDUM 3 (condition-3 deviation rationale) + L1 honest perf framing + N1 oracle-anchor test. Failing-first; full canonical after; focused re-review then merge.
- In flight: r2a-fix, r2b, r4a-remediation.
- r2b CODE-COMPLETE 9e3109b (41 red-first at base; canonical 1271/6/0 in 3 runs, last green after test-singleton isolation fixes; 2 earlier runs had 1 fail each — author root-caused + fixed, not waved off). Vocabulary gap adjudicated by author: epoch advances at most reducing→fast_planned (non-terminal; exact lane never starved), fast completion lives in fast_revision column — review re-verifying consumer sweep. r2b-review DISPATCHED (scratch-clone probe review: tail-hook safety vs r1 fences, admission-before-acceptance, fast_revision fence races, envelope honesty, test genuineness, red lines, thread/lifespan).
- In flight: r2a-fix, r4a-remediation, r2b-review.
- r2a-fix CODE-COMPLETE 460b290 (7 RED + 2 fence pins at base; canonical 1239/6/0 = 1230+9 exact). B1a bootstrap pin + chain re-fold; B1b reducer-parity unfoldable fence with settle-not-replay anti-wedge; B2 scoped empty-batch intercept; M1 disclosure both paths; L1 exhaustive no-op settle. r2a-fix-review DISPATCHED (focused: bootstrap race/crash/reset attacks, fence-vs-reducer divergence hunting, intercept scoping, red-first genuineness).
- In flight: r4a-remediation, r2b-review, r2a-fix-review. r2a-fix branch touches only owned files vs 1ee0fd1 — merge clean vs current main (docs-only main-side delta), pending APPROVE.
- r2b-review REQUEST-CHANGES (1H/1M/2L/3N; hard contracts all probe-confirmed): F1 SSE ordering race — lane cadence scan publishes fast_revision(R) before canonical_revision(R) (ENQ_LOG instrumented proof; flake 1/9 pytest, 2/40 replica at load 9-20; author's pre-existing claim REFUTED); F2 reserved_ms payload stripping (250ms under-booking). r2b-remediation DISPATCHED (author resumed): one-scan-grace for scan-discovered revisions + deterministic ordering re-pin (cadence ENABLED), F2 one-line passthrough, F3 joint-maxima doc, F4+F5 _published_keys skip. Adjudicated CORRECT: dormant-lane admission, envelope constants, fast_planned starvation-free, revision triple EVENTUAL-at-rest (document). F6 → store owner queue (advance_fast_revision accessor).
- In flight: r4a-remediation, r2a-fix-review, r2b-remediation.
- r2a-fix-review APPROVE-WITH-NITS: all 5 findings CONFIRMED-FIXED under probes P1-P8 (race/order-safe bootstrap, reducer-aligned fence, scoped intercept, exhaustive settle, mutation-proven pins; 27,623-cell parity red reproduces original finding exactly). Author one-liner follow-up dispatched pre-merge (F1 ensure-call extension in consumed>0 branch + F3 comment); F2 reset-after-pin → store-owner packet. MERGE after follow-up sha arrives.
- Store-owner packet queue (single small store.py wave after R2 closes): unsupported_family typed refusal at append_operations (r2a endgame), republish_result_at_version(manifest_override=...) (r2a M1 seam), advance_fast_revision typed accessor (r2b F6), reset_scenario rev-0 pin invalidation (r2a-fix F2), optionally native-op executor-effect persistence (r2a residual).
- In flight: r4a-remediation, r2b-remediation, r2a-fix follow-up.
- r2a-fix MERGED 46c51d3 (fast-follow fe2a5f3: F1 ensure-call hoisted before consumed fork — committed-chain legacy-blind blip now unreachable; F3 intent comment; F2 docstring pointer). Worktree+branch removed. Post-merge canonical on main nohup /tmp/r2afix_merge_canonical.log. r2a wave CLOSED pending canonical green — mixed-traffic demo unblocked after r2b lands.
- In flight: r4a-remediation (worktree at b7604a8), r2b-remediation (worktree at 512329e), post-merge canonical.
- r4a-remediation CODE-COMPLETE eafde55 (red-first b7604a8 reproduced reviewer's 27m-tree construction silent stale bits): H1 regime fence pure-arithmetic (dz_1 = ds·tan(alt)/scale vs shadow() float32 arithmetic; refuse iff multi→one-step; baseline-pack companion, value fence first), M1 arg order + spec pin + 40x80 regression, ADDENDUM 3, honest perf framing (corridor beats full-tile only; ~1.3x slower than deployed W2 shape — stated everywhere), N1 fold anchor (mutation-sensitive). Canonical in worktree 1259/6/0. TEST RESTRUCTURES (transition tile bakes T; counterexample test → pack-time refusal; non-square 12→28m) flagged — r4a-review resumed for focused re-review with explicit drive-the-original-construction-by-hand mandate.
- In flight: r2b-remediation, r4a re-review, post-merge canonical (r2a-fix).
- r2a-fix post-merge canonical GREEN on main 46c51d3: 1239/6/0 (1313.77s, load 15.40-22.63) = 1230+9 exact. R2a wave CLOSED.
- r2b-remediation CODE-COMPLETE 512329e (red ddaa7cb; F1 one-scan-grace with hook-twin replacement + wire-order re-pin 20/20 at fix; F2 payload-aware reservation; F4+F5 published-revision skip index; F3 docs; canonical 1275/6/0; author retracted pre-existing-flake attribution). r2b-review resumed for focused re-review (grace-leak hunt, double-publish, self-heal regression, _published_keys bounds).
- In flight: r4a micro-pass (1-ulp association + 2 T-nits) → merge; r2b re-review; store-owner packet queued.
- r4a MERGED a2b29b4 (head chain 04660c4-red → ff932de-impl → b7604a8-red → eafde55-remediation → df7e7d6-micro; review APPROVE after remediation). Worktree+branch removed. Post-merge canonical nohup /tmp/r4a_merge_canonical.log. Branch base ae0bccb merged clean over main 46c51d3 (disjoint files). Micro-pass: dz_1 association-matched to march (tbs=tan/scale first), comment 23.18m, state-not-advanced pin; author self-correction on one_step_patch_indices (9 not 7, no behavior dependence).
- Phase A closed. R4 Phase B (V4 op-sequence proof gate) NOT yet dispatched — after R2 closes. MEDIUM-2 W2 replay packet now dispatchable (solver.py free).
- w2-amplitude-fix DISPATCHED (MEDIUM-2, now unblocked post-r4a-merge): replay march amplitude must come from full-tile composed scene (mirror r4a state path arithmetic), never read-window crop; red-first reproduction of the 166-cell divergence; non-square + non-power-of-two-scale parity mandated; no coupling to VegOcclusionStore (kill-switch standalone correctness). Owns solver.py + new test file only.
- In flight: r2b re-review, w2-amplitude-fix, r4a post-merge canonical (/tmp/r4a_merge_canonical.log).
- r4a post-merge canonical GREEN on main a2b29b4: 1281/6/0 (1188.81s, load 8.89-11.31) = 1239+42 exact. R4 Phase A CLOSED on main.
- r2b MERGED 3384341 (clean auto-merge incl. exact-lane test file isolation injection over r2a-fix additions). Worktree+branch removed. Post-merge canonical nohup /tmp/r2b_merge_canonical.log (expect 1281+45=1326 class). R2 CORE LANES BOTH ON MAIN pending canonical — mixed-traffic demo unblocked after green.
- NEXT once canonical green: r2c frontend dispatch (revision-triple + result classes on wire, SSE ordering now guaranteed), store-owner packet dispatch (4 seams), R4 Phase B (V4 op-sequence proof gate).
- DISPATCHED in parallel: r2c-frontend (client fast-lane consumption: result_class state machine, revision-triple render fences, reconnect-no-fast-frames reconciliation, missed_events, admission rejection UX; ADDITIVE — existing node tests untouched; owns examples/** only) + store-seams (4 store.py seams: phased UnsupportedFamily refusal at append, republish manifest_override, advance_fast_revision typed accessor, reset rev-0 pin invalidation; mechanical adoption swaps only in jobs/scheduler).
- In flight: w2-amplitude-fix (solver.py), r2c-frontend, store-seams, r2b post-merge canonical (/tmp/r2b_merge_canonical.log).
- r2b post-merge canonical FAILED: 1 failed at 1097 passed, -x halted, detail lost (tail-4 only). Triage suite running (/tmp/r2b_merge_triage.log) — suspects: r2a-fix×r2b test-file interaction vs load flake (load 22.75 mid-run). LESSON: canonical logs keep full output, no tail-truncate.
- r2c-frontend CODE-COMPLETE 90baf1e (158/158 node, 21 red-first, additive-only, zero python). r2c-review DISPATCHED (additive audit, supersession-fence probes, missed_events edges, wire-compat grep, red-first re-verification in clone).
- In flight: w2-amplitude-fix, store-seams, r2c-review, r2b merge triage.
- store-seams CODE-COMPLETE all 4 seams (60ea6bf/900c6cf/eee9499/52bce04, red-first each; targeted 107 + 86 green; canonical running in worktree). Mechanical swaps done: jobs.py republish manifest_override, scheduler.py typed accessor + scenarios_with_fast_lag (grep-clean raw SQL).
- r2b merge triage GREEN 214 (108.74s) — isolated suites pass; canonical failure is ORDER-DEPENDENT telemetry singleton pollution under the merged combination (store-seams confirmed flagged test passes in isolation). Pairwise repro running (/tmp/r2b_pollution_repro.log: exact→telemetry, fast→telemetry, both→telemetry).
- In flight: w2-amplitude-fix, store-seams canonical, r2c-review, pollution repro.
- r2b canonical failure ROOT-CAUSED + FIXED (lead, 8a655b7): r2b default app wiring registers standard metrics into module-shared singleton at CONSTRUCTION (app.py:89-103); r2a-fix added default-wiring tests to realtime block (runs before test_realtime_telemetry) → singleton-purity pin failed. Merge interaction neither branch saw alone (r2a-fix branch pre-r2b app.py; r2b branch pre-r2a-fix exact-lane file). Pairwise repro deterministic (A/C red, B green). Fix = documented r1/r2b isolation precedent: private TelemetryRegistry at make_app call site + _make_client factory defaults (setdefault, base-behavior-preserving). 40 passed pair; 132 affected suites. Full canonical running FULL-LOG no -x (/tmp/main_canonical_full.log) — lesson applied.
- r2c-review APPROVE-WITH-NITS (F1 failed-subscription resubscribe gate, F2 recovery-cycle state flap — both cosmetic; all 8 dimensions confirmed, red-first independently re-derived). F1+F2 follow-up dispatched to author before merge.
- r2c MERGED 7ee9f7c (F1 collabSubscriptionAlive resubscribe gate + F2 gapPending tail guard, red-first both; 159/159 node on merged main verified; one unreproduced runner-stderr fragment flagged by author, 41 green runs since). Worktree+branch removed. R2 frontend plane CLOSED (client + server lanes all merged).
- Main FULL canonical GREEN: 1326/6/0 EXIT:0 (1196.35s, load 5.93-8.79) = 1281+45 r2b exact. R2 core (r2a+r2b+r2c+isolation fix) CLOSED GREEN on main 7ee9f7c.
- w2-amplitude-fix CODE-COMPLETE c8020e6 (red 1f9432d: 24 value + 60 divergent cells reproduced; fix = full-tile amplitude source + MarchRegimeError band fence (dz_1 in (A_eff,A_abs]) → _solve_full fallback; one-step-served parity incl. oracle-legitimate vbsh==2.0 preservation; non-square at scale 1/3; canonical 1294/6/0 in worktree). w2-review DISPATCHED (fence sufficiency attack, no-transfer claim, escalation alternative independent analysis). LEAD ADJUDICATION PENDING review input: refuse-to-fallback (correctness-first, implemented) vs escalate-to-A_abs (author-flagged; needs oracle-twin proof — oracle stop uses A_eff so band divergence risk) — leaning keep-fence, escalation as R6 follow-up only with proof.
- In flight: store-seams canonical, w2-review.
- w2 MERGED 03d80fa + lead F1 copy-on-lend 2f518dd (43 affected-suite green). Post-merge canonical running (/tmp/w2_merge_canonical.log, expect 1339). Worktree+branch removed. MEDIUM-2 CLOSED.
- R6 FOLLOW-UP RECORDED (w2-review adjudication, line-evidenced): oracle stop = ABSOLUTE scene.amaxvalue (utci_process.py:573/:1235→:820/:1258), A_eff is the disproved early-exit approximation → escalate replay amplitude to amaxvalue = bitwise-safe BY CONSTRUCTION, replaces the banded-scene refusal; never escalate to A_eff. Latent note: solver.py:1734 time-loop marches at window-crop amplitude, immune only via wallheight_23's missing acc quirk — future wall-height-march edits re-open MEDIUM-2 there.
- In flight: store-seams canonical (worktree), w2 post-merge canonical.
- DISPATCHED (parallel, files disjoint from store-seams): r3b-cheap-exact (affinity analysis across remaining families → ONE safe lever w/ file:line evidence — view/receptor, shadow-free reuse, or selected forcing; honest-outcome clause; owns incremental compute plane + adapters) + r4b-proof-gate (V4: seeded randomized op-SEQUENCE property sweep — state-transition soundness, refusal completeness incl. false-refusal=0, sequence-composition invariance, serialization round-trip; bounded runtime, honest non-coverage notes; owns proofs/ + veg tests additive, veg_svf_state.py only red-first if defect).
- In flight: store-seams canonical (worktree), w2 post-merge canonical (/tmp/w2_merge_canonical.log), r3b, r4b.
- w2 post-merge canonical GREEN: 1339/6/0 EXIT:0 (1213.26s). MEDIUM-2 fully closed on main.
- r3b HONEST OUTCOME: no safe cheap-exact lever in grant — all 3 candidates refuted w/ file:line evidence (receptor params all radiation-consuming via Sstr/Tmrt/height-walk/elvis/aniso/transVeg/albedo; shadow-free reuse killed by geometry→svf time-invariance; forcing already total post-r3a). Shipped MODEL_PARAMETERS_AFFINITY + evidence dict + import-time drift guard + 7 tests (canonical 1346/6/0 in worktree = 1339+7). VALUABLE FIND routed: r3a fast-path DURABILITY — store R2 whole-entry supersession kills full-tile tmrt@t coverage after first windowed batch (r3a 0.87s path dies in mixed sessions; correct-but-suboptimal) → per-cell supersession/retention = R5 packet scope. r3b-review DISPATCHED (light: evidence spot-check, guard mutation, store mechanism confirmation).
- In flight: store-seams canonical, r4b-proof-gate, r3b-review.

## 2026-09-04 (evening) — R3 closed pending canonical; store-seams review + R5a in flight

- r3b merged d34f65b (review APPROVE — all 14 affinities spot-verified, drift guard
  mutation-proven); evidence-cite cosmetic LOW fixed c9d1758. R3 (#50) closes when the
  post-merge canonical (PID 46694, full-log) reports green; expected 1346/6/0.
- store-seams canonical 1344/1/6 — telemetry failure ancestry-verified pre-existing on
  base (ba617ab predates isolation fix 8a655b7); attribution accepted WITH evidence.
  Non-author review dispatched; merge after APPROVE ⇒ R2 formally closed ⇒ fly demo
  refresh (mixed-traffic lanes all on main by then).
- R5 opened: R5a retention lever dispatched (r5a-retention worktree). Correction to
  earlier evidence: the windowed-supersession retention lives in
  solweig_gpu/incremental/store.py (TemporalResultStore), not server/store.py.
  R5b (temporal checkpoints + sparse replay) authored after R5a lands.

- 2026-09-05 fleet recovery: the "dead" reviewer fleet was routing lag — 5 reviews
  delivered in a burst, ALL APPROVE, zero blocking. Lead-direct r5a/r4b/store-seams
  verdicts independently cross-confirmed (r4b thorough.json reproduced by instrumented
  rerun; r5a 300-round brute-force property probe; store-seams probe battery +
  byte-identical SQL parity). r4b hardening dispatched pre-merge to its author
  (classify_refusal fence misclassification, exception-reason classification, dead
  GATE_SCALARS, totals buckets, stale smoke.json). R5 brief updated from the two
  converged r5b design briefs: G1.1 now mandates the seen_patches dedup fix in the
  same change as gate relaxation (probe-quantified 384/576 NaN hazard); G2 payload =
  6 thermal planes + 4 scalars + next_step, input-digest fingerprint, full-tile-only,
  first consumer = r3a refused-branch met-rad edits. Lead self-disclosed + repaired a
  worktree-nesting bug (agent-g2 was created inside agent-g1 by relative path; stray
  ledger commit recovered to main, branch reset, agent notified). Queue: canonical
  PID 55431 (~65%) gates #49 R2 closure + r5a/r4b merges; g1-superset (G1.0) and
  g2-checkpoints (G2.0) in flight.

## 2026-09-05 — G1.0/G2.0 post-merge canonical GREEN; G1.1 code-complete, review in flight

- Canonical on main @ 67f7507: 1412 passed / 6 skipped / 0 failed (1403.35s)
  = 1382 + 27 (G2.0 checkpoints) + 3 (G1.0 superset) EXACT. R5 midpoint green.
- G1.1 CODE-COMPLETE (g1-relax, worktree agent-g1r, 7363eef RED + cf06336 fix):
  per-cell latest validity — ascending-revision paint, load-cache dedup (dedup the
  LOAD, never the slice), completeness fence (unpainted cell → typed refusal).
  Canonical in worktree 1412+2cli env-known = 1414 collected, reconciled exact.
- SPEC CONTRADICTION found by author, RATIFIED by lead: the brief's literal G1.1
  formula (max_revision_raster == revision_at).all() is provably a NO-OP for
  FULL-coverage sets (all-max==head iff single-revision-at-head). Resolution:
  serve mixed under the G1.0 superset license; raster kept as completeness fence
  only; PARTIAL refusal unchanged. Matches brief G1.2's mandated outcome + the
  ledger r5a gloss + probe D's quantified NaN rationale.
- Non-author review DISPATCHED (r5a-review2): no-op proof verification, chained
  staleness attack (N→N+1→N+2 retained remainders), paint-loop correctness,
  dedup anti-vacuity, red-first genuineness, consumer grep, G1.0 smoke rerun.
- Queue after G1.1 merge: G2.1 packet (executor.py one-writer — quarantined
  8aa6978 module + g2-supplement-wip RED tests + checkpoints hardening; author
  re-dispatch = g2-checkpoints resume), then G2.2 crash-restart.

## 2026-09-06 — MISSION CLOSE (R0-R8 complete, SLO re-specified, deployed)

- Quiet-host re-commission merged (4846ae9): NEGATIVE result — launch
  load 2.77 did not hold (gauge min 2.377/mean 4.563/max 9.074),
  visibility p99 492.8; four runs flat 466-493 across load 3.27-5.45
  = zero load sensitivity; no "host load < X" condition certifiable
  on this machine. Retired worktree + branch.
- USER DECISION (AskUserQuestion): tick-anchored SLO re-spec APPROVED.
  Contract now: visibility p50 in [W,2W), p99 <= 2W + 350 ms (550 at
  W=100), tick-lateness p99 <= 350 ms as its own SLO (host/scheduling
  breach class), close pipeline p99 <= 5 ms; fast lane 1,000 ms
  unchanged. Every run to date inside re-specified contract (ccf23d5:
  service_level_contract.md + realtime_contract.yaml + goal_condition
  in-place amendment + commissioning_note section; spec/design-docs/
  r8b-wiring tests 24/24; docs-only, canonical not required).
- Known limitation recorded: no realtime browser client exists (API +
  SSE complete, qualifier metadata in payloads; demo UI is the
  incremental studio). Client-display SLC clause is met at payload
  contract level, not browser rendering.
- Final state: main aa0e9bc pushed; canonical chain 1382->...->1510
  reconciled exact at every merge; fly demo live at f4049f4 (R6+R7+
  R8/R8b); deferred: fly performance-1x (trial-org billing), 6°-ring
  march batching (banked intel), realtime browser client (future).

## 2026-09-06 — FRONTEND WAVE: realtime interactive studio (post-mission-close)

- Mission-close known limitation "no realtime browser client" RESOLVED this
  wave: studio now submits design gestures through the realtime operation
  plane (was exact-session edits + display-only telemetry).
- UX-first per user directive: 2 Opus agents produced 2,041 lines of ground
  truth BEFORE code (interaction_simulations.md 1485 + design_plan.md 556,
  merged 895148b). Light theme, undergrad-comprehensible, pro disclosure
  surfaces, five-stage pipeline vocabulary.
- Build wave 5 agents, disjoint file ownership: ui-shell (index.html 795bdad),
  ui-badges (realtime_badge/renderer 59caeef), ui-styles (styles.css 144b10b),
  ui-realtime (app.mjs+op_builder 886526d), ui-tests (11 pinned tests 9132bea,
  held until stub-vs-real reconciliation; resolved --ours, 11/11 green).
- Lead integration: serve.py SSE proxy streaming fix 757388c (ROOT CAUSE of
  browser "Reconnect failed": buffered read() never returns for live event
  streams; read1 loop + close-framing), realtime-surface reveal/wiring fix
  5e8d741 (connected-only pills never unhidden; textContent wiping child
  spans; metronome rail unwired; activity summary static).
- Browser E2E through orca-cli (user directive): zero console errors; add
  tree -> POST operations 200 -> applied R1 seq 1 (+base_divergent advisory
  chip verbatim I-23); delete -> R2 seq 3 epoch e1; badge "Preview (owed)"
  correct for async-verify vegetation; rail W2 F2 E0 "exact owed +2".
  Caveat noted: two-tree screenshot = probe's own ?? chain double-click
  (click() returns undefined), NOT an app double-submit.
- Suites: node --test 170/0 (baseline 159 + 11). pytest: 1 fail =
  thermal_comfort console script missing in venv (pre-existing environment
  artifact; wave touched zero server/test files), remainder green.
- Verify wave DISPATCHED (read-only): verify-contract (client/server contract
  conformance), verify-copy (glossary/error-matrix verbatim mandate),
  verify-review (non-author code review 95574c2..HEAD). Known open items for
  them: badge-word duality (glossary words in class badge vs design-plan
  words in rail lamp), admission kind vocab, base_divergent HTTP-ack-only,
  scenario-per-session (two-tab same-workspace demo needs scenario reuse
  surface — not built this wave).

## 2026-09-06 — Verify wave closed (#83)

Two non-author review passes (server + client) on top of verify-copy's
glossary audit. Server pass: 1 REQUIRED (broadcast existence gate outside
the exception fence would strand a "running" job row on a store read
raising) — fixed, 2 regression tests, 25 passed. Client pass: 14 findings,
headline CRITICAL the v-axis sign flip missing in solveFrameFromSamples
(mirrored v on every solved frame); all applied as one batch by the lead.

Commits: 444dcd2 (server: named exact_revision event inside the fence,
scenario site_geometry disclosure + contract tests/docs), 85ba483 (client:
frame convention, held-row lifecycle, pendingAcks duplicate settle,
resubscribe reset, structured badge writes, .rt-ledger-row/.fresh-dot/
is-receiving CSS vocabulary reconciliation, refused-pipeline settle,
serve.py headers-committed guard, copy glossary).

Evidence: node 174/0 (4 new frame tests); pytest focused 94 passed; live
orca E2E on restarted server — add-tree → ledger row → W/F/E 1/1/1 →
badge exact_reconciled, pipeline verify/done, 24 exact timeline dots, zero
console errors.

Open (not this wave): renderer.mjs unwired exports (flagged vs 59caeef
"wired" claim); legacy lane publishes without the named event
(scenario-per-session demo surface); fly deploy gaps queued into #84
(upload_site_cache.sh Input_subset, site env vars, DNS recheck).

## 2026-09-06 — Collab E2E closed (#85) + deploy wave executed (#84)

- Selection-gesture bug found by E2E and fixed in app.mjs: pointerdown took
  setPointerCapture BEFORE syncSelectionPanel — any capture failure
  (synthetic events, stale pointer id on real hardware) threw NotFoundError
  and skipped the sync, leaving a SELECTED tree with a disabled delete
  button and stale "No tree selected" label. Capture is now best-effort
  AFTER sync/render. In-browser proof: pointerdown → "Broad canopy 01" +
  delete enabled, zero console errors.
- makeTree id collision (two editors minting same ms+ordinal id → operation
  plane tree_id upsert silently dropped one entity) fixed with random tail;
  concurrent-add E2E now yields distinct ids, trees=5, both tabs identical.
- Two-tab matrix on orca: A→B add, B→A add, concurrent adds, move (op seq 7
  verb=move, W/F=5/5), late-join/reload scene rebuild via new public
  catchUpWorkspace() (subscribeWorkspace stays telemetry-only; the inline
  variant broke ladder/catch-up test contracts — harness now counts POST
  attempts), delete A→B (trees 4/4, W/F=6/6). Suite 174/174. Commit 5b7bd84.
- orca lesson: tab reload served the CACHED app.mjs despite no-cache —
  close + recreate tab with cache-buster to pick up module edits.
- fly redeploy (image deployment-01M1W3MEZDYPT268QM2V4HG0D8): caps 200
  after wake, studio connects, scenario minted; edit plane VERIFIED LIVE on
  fly — drop-add → POST operations 200 → ledger "applied · R1 · seq 1",
  F=1; second edit R2 seq 2 "applied · R2"; rejoin ?scenario= rebuilds
  scene (tree persisted in SQLite across machine stops); rail/transport
  live; badge honestly "Preview (owed)".
- Exact lane on fly: worker VERIFIED ALIVE (health: dispatched, solving,
  heartbeat 3s, RSS ~1.0 GB) but a full solve (~3+ min observed on
  shared-1x vs 38-40 s local M1) cannot fit the trial-org 5-minute machine
  ceiling after the ~2 min cold boot — machine SIGINTs mid-solve every
  time (three wake cycles observed). Exact-lane CORRECTNESS remains
  carried by the local bitwise-parity suite + local live E2E (E settles
  1..5, verify bridge publishes); trial hardware cannot demonstrate
  end-to-end completion. Card on file or performance-1x is the unblock.
- solweig.alansynn.com CLOSED: user's record had a typo (`soleweig.`),
  renamed → CNAME → solweig-rt-demo.fly.dev (DNS only) live at both
  authoritative NS. `flyctl certs add` → Let's Encrypt issued (rsa+ecdsa,
  expires ~2026-12-05). HTTPS verified: capabilities 200 (schema 1,
  9 adapters), studio page 200, app.mjs 200. orca on the CUSTOM DOMAIN:
  connect → connected, scenario scn_7fa52c764001, transport Live, badge
  "Preview (owed)"; drop-add gesture → trees 1, W1/F1, ledger
  "applied · R1", selection "Shade tree 01" (pointer-capture fix live).
  Deploy wave #84 complete.

## 2026-09-06 — Studio-first restructure (#87): studio/ at URL root

- User: URL too complex, tool must be main product. Multi-agent plan:
  3 Explore (serving/refs/top) → Plan agent → AskUserQuestion (dir name
  → `studio/`) → executor (opus) P1+P2 → verifier browser E2E +
  code-reviewer in parallel → 4 MINOR+1 NIT review fixes → deploy.
- Why URL was long: serve.py static root = repo root (parents[2]); URL
  path mirrored filesystem. App itself prefix-agnostic.
- Moves: git mv examples/incremental_design_tool → studio (37 renames,
  history kept). serve.py: directory = own dir (URL root = studio),
  legacy /examples/... 301 → /<rest> query-preserved, /docs/* read-only
  mount (files only, contained), boot-config injection into served
  index.html when --api (SOLWEIG_API_BASE="/api" + SOLWEIG_SITE_ID from
  env, classic script before module tag, startup-validated). app.mjs
  ?site= falls back to injected global. Static root narrowed repo→studio
  (exposure win).
- Bare URL https://solweig.alansynn.com/ now boots CONNECTED with zero
  params; ?api= (empty) still forces offline; ?api=/?site= override.
- Gates: JS 174/174 (baseline held), pytest docs+studio_serve 23
  passed (10 new test_studio_serve.py: injection ordering, offline
  plainness, 301 query preservation, near-miss 404, /docs containment,
  MIME, /api proxy through new dispatch w/ fake upstream), smoke_proxy
  OK incl. injection + 301. Reviewer APPROVE 0 blocker/major. Verifier
  orca offline-at-root PASS (drop 3→4, redirect, docs 200, console
  clean).
- Deploy (commits 1944cf8 + 88da189, pushed): caps 200 after 2×502
  boot, root 200 with both injected globals, legacy 301 → /?api=/api,
  docs 200, app.mjs text/javascript. orca on bare URL: connected,
  scenario scn_e5f93e6a2832 minted, transport Live, drop-add → W1/F1
  ledger "applied · R1"; second tab /?scenario=... rebuilt scene (same
  tree id) + presence bar. Trial 5-min ceiling unchanged (exact lane
  still owed there; local parity suite carries correctness).

## 2026-09-06 — Rate-limit incident + UX honesty wave (post-restructure)

- Fly trial ended mid-session (machine ops refused) → user added credit card
  → machine restarted, demo re-verified live. Trial 5-min ceiling GONE.
- User-reported repeated "Exact analysis failed (rate_limited)" banner.
  Fleet diagnosis: IP gate 30/min keys client host; studio proxy collapses
  ALL visitors into one 127.0.0.1 bucket; frontend bursts (300ms poll,
  uncapped baseline await, re-minting connect retries, Retry-After never
  read) exhaust it. Live verify: edit plane WORKING, exact chase STALLED
  E=0 7+ min. User ruling: collaborative batching already paces edits —
  raise the IP gate (600), realtime admission stays the real control.
- c3a810c: env 600/min + frontend politeness (poll 1s, backoff, rejoin
  scenario, parse Retry-After, single idempotent 429 auto-retry).
  node 179/179, pytest 23, zero solweig_gpu edits.
- 5154f04 INTEGRITY REPAIR: 1944cf8 had staged renames but missed the
  post-rename content edits (serve.py rewrite etc.) — test_studio_serve
  committed without its subject. Restored + pushed. Lesson: re-add moved
  files after post-mv edits.
- User: "CPU update ROI" incomprehensible + frontend slop → UX audit.
  ux-audit (read-only) full report + addendum; lead arbitration: D2 badge
  words canonical, design_plan §2.3 retired. 298ecca: copy swaps
  (REFRESH WINDOW, sketch·/exact·, Editor-N ordinals, hour titles),
  exactBadge classFor() echo (was frozen connection word "Exact" while
  truth "Preview (owed)" — D3 violation), pipeline honesty (.is-owed
  verify lamp, #jobStatusDetail realtime wiring, .is-failed refusals —
  were green checks), rate_limited → §4 cooldown advisory + W-7
  survived line. node 179/179, pytest 23.
- Redeployed (c3a810c + 298ecca). Functional proof: 55 consecutive
  capability 200s zero 429 (old cap 429s at #31). Browser E2E
  (verify-postdeploy) watching E convergence — pending at checkpoint.
- DEFERRED (future waves): ledger five-stage dots, click-to-receipt,
  result-class ladder tooltip, canvas furniture light-theme recolor,
  slider-settle routing through realtime plane, SSE reconnect budget.

## 2026-09-07 — EXACT-LANE RESOLUTION (continuation)

- verify-postdeploy: UI wave PASS on live; E stuck 0 for 14.5 min with
  2 HTTP requests and zero 429 → NOT the rate limit. Server-side hunt:
  worker health showed jobs_dispatched=1, last_success=null, queue 9
  scenarios deep (every afternoon session's chase job, incl the verify
  session's — server-side chase wiring WORKS).
- Three-layer root cause peel: (1) thread thrash hypothesis — OMP=1
  pinned via env 7751090, verified in-process, NO effect; (2) py-spy
  dump (installed in container) settled it: solver thread ACTIVE in
  svf_calculator/shadow march inside run_full_tile — stale chase
  scenarios route FULL-TILE; (3) sustained 7-10% CPU while runnable =
  heavy steal on the shared-cpu-1x slice. Full-tile solve = 20+ min,
  single-flight queue never drains.
- 08a29db: performance-1x upgrade (billing unblocked the always-planned
  deferred item). CPU fraction 0.07 -> 0.97 immediately. First
  completion: job_01a17a05b424 (full-tile, ~13 min single-thread).
  User's live-tab job_202d65baf8f8 (windowed) then completed ~5-6 min.
  EXACT LANE LIVE-E2E COMPLETE for the first time on the demo.
- ops notes: `flyctl ssh console` broke mid-session (even echo refused)
  — monitor via public API when that happens; py-spy pip-install works
  in the container and is the decisive wedge tool; chase recovery
  re-enqueues a job per lagging workspace at every boot (stale sessions
  churn the queue by design).
- verify-postdeploy FINAL CHECK (fresh orca tab, cold join of the settled
  scenario): 3/4 PASS — badge "Exact" (classFor echo, hidden stale spans),
  rail E1 stable == server exact_result_version, console clean. (c) FAIL
  as specified: #jobStatusDetail stays "No exact job yet" in catch-up
  (truthful for no-local-job, but contradicts the Exact badge; verify
  lamp carries the truth "Exact · 73.08 s"). Catch-up copy nits filed
  with the deferred list: rail-class mixes "OFFLINE PREVIEW · local
  kernel — no collaboration" while connected; #railOwed renders
  "exact owed +0" when nothing owed; stamp lamp "sending · receipt
  pending" for already-receipted W; pipeline lamps empty-class in
  catch-up. All catch-up-state-only (primary edit flow verified
  settled earlier). DEFERRED to the UX polish backlog with the ledger
  dots / click-to-receipt / ladder tooltip / recolor / slider routing /
  SSE budget items.

## 2026-09-07 — SHARED WORLD + ROUTING POLICY waves (user rulings)

- USER RULING 1 (architecture): one shared world — every visitor joins the
  SAME workspace; per-session scenario minting was the root of dead-session
  chase pileup (live incident: 5 stale full-tiles queued ahead of a live
  user, ~50 min starvation; manual cancel + resubmit recovered them).
- USER RULING 2 (routing): prefer short incremental solves; full-tile at
  most once/day and only with no sessions watching; UI must show
  full-tile + ETA. Cold-boot/wake advisory also requested.
- routing-design (read-only architect) REVERSAL of the lead's first
  diagnosis: today's 476 s full was NOT geometry — it was the SILENT
  0.30 dirty-fraction demotion (worker.py:1204-1208; no fallback_reason
  because only SolverInputError sets one). Yesterday's 1-tree edit sat at
  0.29568 — 0.004 under the threshold, hence 73 s windowed. Net-delta
  planning already existed (reducer/edit_types/edits fold 3 layers) —
  the lead's compaction lever died; demotion removal is the real lever
  (est. 476 s -> ~100-160 s on the same shape). Fences: a 12 m ground
  tree on site_500 trips ZERO fence fulls (baseline amaxvalue 228.34 m
  vs 214.35 m tree top).
- shared-world wave LANDED + DEPLOYED (9f3a8c5 + 33f7a89 + 1d0611c,
  main, pushed): boot find-or-create scn_shared_world through the POST
  creation path (quota-exempt at the boot call site only; site-pin
  validated on the found path — unserved pin stops advertising the
  pointer); capabilities advertise default_workspace_id; studio joins by
  default (single decision point in exact_session.connect), mints only
  against older servers; ~92 s transport-class wake ladder ("Waking the
  analysis server"), 429 stays short-cadence with budget preserved;
  create-key replay across retries; boot self-heal re-queues a failed
  baseline; mode=full honesty on status line + worker toast. Review:
  REQUEST-CHANGES (quota brick, site-pin 500 dead-end — both reproduced)
  -> remediated -> APPROVE-WITH-NITS (persistent-429 copy/hint MINOR +
  regex-dup NIT to backlog). Gates: studio 190/190, focused pytest 213,
  full canonical 1531p/6s/2f (test_cli env artifact).
- LIVE VERIFICATION post-deploy: capabilities served default_workspace_id
  ~40 s after machine boot; orca bare-URL tab request trace = GET
  scn_shared_world + results/0/payload + SSE events + catch-up
  operations, ZERO POST /scenarios (no mint). Badge on pristine world
  reads "Preview (owed)" (exact_revision 0 with no edits — same
  catch-up-state copy class as the backlog items, not a new defect).
- routing-wave (#114) DISPATCHED on 1d0611c: R1 subscriber-gated chase
  minting, R2 exact-lane demotion threshold 0.30 -> 0.95 at BOTH coupled
  sites, R4 idle once-daily reconciliation with durable owed debt, R5
  eta_seconds/eta_basis on the job body (p80 history, static fallback
  disclosed), R6 demoted_by_fraction + true pre-mode fraction metrics
  lift. Non-author review after. Backlog additions this wave:
  pristine-world badge wording, persistent-429 announce copy + retry_after
  hint, regex constant sharing, announce() wake noise, pill flicker.

## 2026-09-07 — ROUTING POLICY WAVE LANDED (#114 closed)

- routing-wave fb6858e + a83ac9f on main, PUSHED + DEPLOYED + LIVE-VERIFIED.
- R1 subscriber-gated chase minting: _exact_lane_audience (hub None -> open
  legacy; subscriber_count>0 -> open; nonterminal exact job -> open); gated
  deferral stamps durable debt (workspace_reconcile, schema v9) instead of
  minting. R2 demotion escape: exact lane 0.30 -> 0.95 threaded to BOTH
  coupled sites via PlanExecutor(full_recompute_fraction=); legacy /edits
  keeps 0.30; reconcile-marked jobs force full (1e-12). R4 idle once-daily
  reconcile: 300s sustained-emptiness grace (or chain depth >50 past last
  full), 600s attempt backoff, cap consumed ONLY by completion (superseded/
  failed consume nothing; local/no-op completion CLEARS debt without
  consuming the day). R5 eta_seconds/eta_basis on queued+running job bodies
  (p80 of last 50 same-mode samples; local rescaled by window fraction
  ±0.2; static 480s disclosed as "static"; unknown = null, never a guess).
  R6 metrics lift: dirty_fraction (TRUE pre-mode) + demoted_by_fraction from
  worker diagnostics into persisted metrics.
- Lead review: science diff vs 1d0611c = 0 lines (solver/veg_svf_state/
  utci_process/solweig.py untouched); ConservativeSafetyPolicy has only 2
  fields so the override drops nothing; 1e-12/0.95 both pass (0,1] guard.
- Non-author routing-review APPROVE-WITH-NITS: mutation testing (gate-
  disabled/planner-unthreaded/static-always all CAUGHT); 1 MEDIUM = vacuous
  R4 cap pin (product probed correct, test was not) -> author de-vacuated
  in a83ac9f (re-armed same-day debt + queue-drain second de-vacuation,
  mutation red->green honestly reported). LOWs accepted+disclosed: crash-
  class reconcile retry has no ceiling (600s cadence), stale-debt candidate
  noise after typed refusal, read-then-disconnect gate race bounded at one
  chase. Deviations adjudicated OK: routes_jobs.py eta seam (body built at
  the route), schema-pin literals 8->9 in two test files, hub-seam audience
  modeling (TestClient SSE deadlock documented), R6 E2E pins honest-full
  (small-tile escape pinned at unit sites instead).
- Canonical rerun (flake check): 1555 passed / 6 skipped / 0 failed (30:07)
  — the idempotent-replay single failure in the wave run did NOT reproduce;
  load-flake classification confirmed. Full gates: routing_policy 19/19,
  server_api 79/79, r2_exact_lane 25/25, worker subset 11/11 + author runs.
- LIVE VERIFICATION on the demo (deploy a83ac9f, cold boot ~48s to caps):
  (1) bare-URL join trace GET scn_shared_world x3 + results + operations,
  ZERO POST /scenarios; (2) two-tree interactive edit -> chase
  job_471d4e88c07e mode LOCAL 109.4 s (same shape class = 476 s full
  yesterday), metrics dirty_fraction 0.29568 + demoted_by_fraction false,
  planner narration "policy: dirty fraction 0.0829 < 0.95", veg corridor
  153 patches no fallback, exact lane folded span 2 ops; (3) eta fields
  present on the running body (null before mode known = honest; history
  basis needs 2 same-mode samples — this completion was local sample 2);
  (4) R1 NEGATIVE: tab closed, blind op POST accepted (seq 3, epoch 2),
  ZERO mint for 80+ s (pre-wave minted in seconds); (5) R1 POSITIVE: tab
  re-opened -> chase job_174846fc870d minted within ~20 s targeting rev 3.
  Caveat: the two live Adds coalesced (union 0.29568, under old 0.30
  anyway) so the >0.30 windowed escape is test/probe-borne, not live-borne;
  idle reconcile firing likewise test-borne (needs 5 min quiet + 13 min
  full) — both mutation-pinned at both threshold sites.
- Deferred (no change): frontend eta_seconds consumption wave (server
  fields live, studio not yet reading them); routes_jobs POST-create/cancel
  responses lack eta keys (GET poll path carries them); LOW residuals above.

## 2026-09-08 — INCIDENT-3 (#140) + INCIDENT-2 verify (#138) joint live closure — GREEN

Deploy: main 8919e29 (merge of incident-3 fix chain) → fly demo, cold boot OK,
capabilities at /api/v1/capabilities. DB base state (read-only): scn_shared_world
jobs [('complete',4),('failed',37),('superseded',9)], epochs exact_targeted 11 /
fast_planned 32 — the wedge scar; no reducing epochs (chase died with subscriber).

LIVE VERIFY 1 — self-heal (subscriber-driven re-mint through anchored backfill):
- Fresh orca tab (cache-buster, no reload) joined the shared world: footer
  R45 · PREVIEW (OWED), W=F=45, E=10, "exact owed +35" — catch-up replay re-minted
  the poisoned backlog.
- job_bb276faf5cc2 ran and COMPLETED (487.7 s, result_scene_version 45); metrics
  carry exact_lane.anchored_backfill — the incident-3 fix path executed in
  production, not just in tests.
- Epochs after: exact_targeted 43 through seq 44; footer R45 · EXACT, W=F=E=45,
  owed line gone. failed count stayed 37 (no new churn).

LIVE VERIFY 2 — native baseline-tree move E-advance (#138's deferred gesture):
- Synthetic pointer gesture (pointerdown/move/up on #overlayCanvas) dragged
  "Shade tree 01" (tree-mtrl1chf-01-dmnp, the baseline-seeded tree from the
  incident chain) after locating it by overlay pixel scan + per-cluster
  selection probe (canvas truth, not derived coords).
- Op seq 45 accepted (verb=move, family=vegetation_geometry, actor
  studio-eeq335tx) → job_44e7abab8a2e ran COMPLETE (464.5 s, version 46,
  error_json None) → exact epoch closed through seq 45 (exact_targeted 44).
- Footer R46 · EXACT, W=F=E=46. No refusal, no wedge, no reconcile debt.

Notes: gesture synthesized in-page via PointerEvents (untrusted-event path is
the app's own drag handler; setPointerCapture failure is handled by design).
Both heals ran full-domain windows on the shared-2x256 demo VM (~8 min each);
that is capacity, not a regression class. #139 follow-up (restore-branch seed
line pin + guard-scope residual) remains open as before.

STATUS: #140 CLOSED (fix live-verified end to end). #138 CLOSED (E-advance on
native baseline-tree move verified live). Evidence: DB queries above + live
footer snapshots (orca page ebf19b29-b4cc-4f86-8989-332b9d2e705f).

## 2026-09-08 — #139 follow-up CLOSED (restore-branch seed pin + guard scope)

Fix author: subagent (adversarial lane, worktree). Lead-verified end to end.

- SURVIVOR ROOT CAUSE (why m1 was vacuous): every existing restore-path repro
  stages a FOREIGN 0.30-fraction snapshot, which the incident-2 fraction gate
  drops at the PENDING door — the span takes the FRESH branch, whose own seed
  carries the fold. No test ever exercised a SAME-lane restore with a
  native-blind layer, so the restore-branch seed call was never load-bearing.
  (m1's verbatim definition was unrecorded anywhere findable; reconstructed
  as "neuter the restore-branch `_seed_exact_lane_baseline` call" — flagged
  as reconstruction, not quotation.)
- FIX 1 — #139a test `test_restored_native_blind_snapshot_is_reseeded_from_the_
  fold_baseline`: stages a same-lane (0.95) snapshot pruned to native-blind
  (pre-incident-1 producer shape) and drives a structural op through the
  RESTORE branch. GREEN at pristine by design; its killing power is the
  mutation run (RED-for-a-survivor-pin pattern).
- FIX 2 — #139b guard scope: `_pending_restorable_by` guarded only the
  pending-adoption door (`_reconcile_coverage`). The promoted door —
  `_ledger_replay_window`'s file-existence check — let a cross-lane promoted
  snapshot reach `restore_into_executor` → `_verify_identity` fence → typed
  `scenario_state_unrecoverable` wedge repeating until reset (incident-2
  class, one door later). New `_promoted_snapshot_restorable(snapshot_dir,
  request)`: unreadable snapshot keeps the file-existence decision (u-e1 F2
  corrupt-snapshot semantics preserved — the restore attempt itself surfaces
  the typed refusal); readable snapshot → same `_pending_restorable_by`
  predicate. `_ledger_replay_window` now takes `request`; both callers
  (`_build_executor`, `_exact_ledger_replay_window`) updated so the eager
  exact-lane window decision stays identical. RED at pristine (wedge
  reproduced), GREEN at fix.
- Evidence: sha256 executor_bridge.py `1660cca7…`, test file `e420f0a5…`
  (== author report). Backups under solweig_ultrafast_artifacts/i139/.
- MUTATIONS: author m1 (restore-branch seed → `pass`) KILLED by #139a in
  258.41s with the production message ("replay chain ... not contiguous");
  byte-identical restore; re-green 6P. LEAD independent mutation —
  `_promoted_snapshot_restorable` final return → `True` (gate neutered, the
  #139 surface the author's m1 does not cover): KILLED by #139b in 42.16s
  (1 failed, 5 passed); sha-verified restore; re-green 6P. Each #139 surface
  now has an independent killed mutation with a distinct designated killer.
- LEAD reruns: incident trio (incident2 + veg_chain + mixed_plane) 18P in
  125.15s; universal_edits 46P + realtime_r2_exact_lane 25P sequential in
  main (the author's 2 bitwise-oracle failures appeared only with three
  heavy suites concurrent — CPU-contention flake class, both isolated-pass;
  sequential main run clean).
- LIMITS (author-disclosed, accepted): m1 definition is a reconstruction;
  #139a fabricates the native-blind snapshot by JSON-pruning (emulates a
  pre-incident-1 producer, doesn't run one); #139b reads outcome from the
  chase's terminal job row (revision parity non-discriminating there — the
  incident's own symptom); `_promoted_snapshot_restorable` double-parses
  snapshot JSON on same-lane restores (negligible vs recompute, kept for the
  pure-function window contract).
- STATUS: #139 CLOSED. No executor-bridge follow-ups remain from the
  incident-2 review.
