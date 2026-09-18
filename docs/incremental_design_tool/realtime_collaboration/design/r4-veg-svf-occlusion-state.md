# R4 DESIGN — Persistent per-cell directional occlusion state for delta vegetation SVF

> Author: r4-design (read-only architect agent, non-author of the coming
> implementation). Recorded 2026-09-04 at main @ b3168d9. This document is the
> implementation mandate for the R4 wave; the corridor-sufficiency proof note
> (recommendation 4) must be reviewed by the scientific reviewer BEFORE the
> corridor path is enabled by default.

## SUMMARY

The ~20 s svf_seconds phase exists because every veg edit replays the full 153-patch
shadow() march over the read window (0.988 of tile) and folds ten float accumulators
from scratch (solver.py:1103-1321). The march outputs are BINARY per (cell,patch)
classifications (vegshmat, vbshmat — verified binary {0,1}, bitmask.py:8-13), and every
float scalar the physics consumes is an order-dependent re-fold of those bits plus
cell-local corrections. So the correct persistent state is the BITS (~4.77 MiB packed
per stack at site_500), not float accumulators: an edit recomputes bits only along
per-patch CORRIDORS (rays through changed footprint cells); scalars are always
RE-FOLDED in canonical patch order for the consumed window — bitwise-exact by
construction, zero float drift across edits. Removes the replay for first AND repeat
edits: ~20 s -> ~8-12 s (phase A: shadow() on corridor bboxes, zero new arithmetic)
-> ~0.2-0.5 s (phase B: fused corridor kernel, separately parity-gated). No tolerance
relaxation anywhere.

## 1. CURRENT ALGORITHM MAP

Call chain: worker.py:1331-1371 _solve_local -> compose_full_scene_tensors (1341) ->
read_window_for_write_window (1349) -> solve_window(veg_changed=True hardcoded at
1364) -> solver.py:1527-1530 window_svf_bundle(cache, scene, read_window,
veg_changed=True, cube_window=write_window) -> solver.py:872-880
_recompute_veg_svf_window (THE ~20 s phase).

March per (cell,patch) — shadow.py:166-322: while-loop on amaxvalue (245); per step i,
shift (dx,dy) from azimuth (246-253), dz = ds*i*tan(alt)/scale (255); building
running-max f, sh[f>a]=1 (275-277); veg: vegsh2 = (canopy-dz>a) - (trunk-dz>a)
(279-281), vegsh = max(vegsh, vegsh2) then suppression vegsh[vegsh*sh>0]=0 (283-284),
vbshvegsh accumulates AFTER suppression and is RESET at step 1 (286, 288-293, firstvegdem
special + (vegdem2>a) gate); final binary classification (303-316). Step-1 reset is why
vegshmat and vbshmat differ. Read direction: target (r,c) reads occluders at
(r+dx,c+dy) — solver.py:953-975. March inputs a, vegdsm=canopy+a (==a->0 quirk),
vegdsm2=canopy*0.25+a (quirk), bush — all CELLWISE functions of the canopy raster
temp1 (solver.py:508-516), so changed cells under an edit = union of old+new tree
footprints exactly. Bush all-zero on dem>0 sites (vegdem2*vegdem>0); existing gate
solver.py:1022-1032.

Accumulation order parity depends on (solver.py:1230-1267; mirrors shadow.py:556-584):
rings 6/18/30/42/54/66/78/90 deg with 31/30/28/24/19/13/7/1=153 patches
(shadow.py:384-387); per patch index 0..152 ascending, per annulus k ascending within
ring (12/ring): svfveg += w_iso*vegsh_bit; svfaveg += w_iso*vbsh_bit; aniso-weighted
E/S/W/N gated by patch azimuth quadrant. Post: last[vegdem2==0]=3.0459e-4 added to
S/W (1269-1274); clamps >1->1 (1275-1284); svftotal = svf_building -
(1-svfveg)*(1-0.03) (1286-1291, trans hardcoded mirrors shadow.py:612). Scalars
zero-padded to read extent — padding inert, consumers crop at entry
(solver.py:1592-1598; utci_process.py:532-560). This patch-major/annulus-minor order
is handoff §8.2's fence.

Object identity: EXISTS in TreeLayer._live (trees.py:185, 228-230) +
rasterize_tree_patch (trees.py:103-153) — reverse-index source of truth is always
current. LOST at compose: vegetation_rasters_window max-combines (trees.py:342-343);
compose_full_scene_tensors uses only canopy [0] and rebuilds trunk with FLAT 0.25
(solver.py:502, 511) — per-tree trunk_ratio is flattened (oracle parity); per-tree
transmissivity NEVER enters the march — physics applies global transVeg downstream
(utci_process.py:662, 743, 747). Overlap resolves to raster max BEFORE the march — no
per-object tie-breaking inside the march to replicate. Baseline Trees raster has no
identity and needs none (immutable).

W2 composition: _patch_march_reach_pixels (solver.py:929-950), _patch_march_window
(978-1019), _sky_patch_geometry memo (1040-1100), effective_march_amplitude
bit-identical early exit (616-660), read window + 6 deg floor (129, 666-752) — all
untouched; R4 baseline is built at FULL-TILE extent (oracle semantics); W2 machinery
stays the fallback replay.

## 2. DATA STRUCTURES

VegOcclusionState per scenario: vegsh_packed + vbsh_packed (PackedVisibility,
rows x cols x 20 B, bitmask.py verbatim, polarity 1=sky visible per
SHADOW_STACK_POLARITY bitmask.py:25-29); trunk_absent bool mask (drives last,
solver.py:1269-1274); chunk index (64x64 chunks: version u32 + sha256 of chunk packed
bytes + mask); key = {site_id, tile_key, cache_manifest_sha256, scene_revision,
patch_geometry_id, schema_version}.

WHY BITS NOT winning-occluder refcounts: per-(cell,patch) output is a binary
classification of the composed surface profile along one ray. Deletion/overlap
correctness needs the NEW composed classification — the corridor re-march computes it
exactly from the TreeLayer-composed raster (max semantics). A refcount structure would
duplicate the march's classification semantics (step-1 special, suppression ordering)
as a second physics implementation = second parity surface. REFUSED. Scalars NOT
persisted — re-folded per solve at the consumed window through the SAME code path as
the replay (§7); eliminates float drift and ~11 MB/scenario.

Reverse index: DERIVED, not stored. Changed-cell set C = union of old+new footprints
(coalesced old->new pairs, edits.py:74-138). Per patch p with read direction (dr,dc)
and reach R_p: corridor_p = union over s=1..R_p of shift(C, -(dx_s,dy_s)) using
shadow()'s exact per-step offsets (memoized, <= ~42k offsets total). This is
dirty_window_for_edit's old-union-new principle (geometry.py:461-513) lifted from bbox
to (cell,patch) granularity.

Trunk/canopy enters ONLY via composed vegdsm/vegdsm2 (flat 0.25, solver.py:511,
515-516) — corridor re-march inherits oracle semantics incl. the flattening (fence,
§4). Transmissivity is global downstream (transVeg utci_process.py:662; 0.03 in
svftotal solver.py:1286) — bits are transmissivity-INDEPENDENT, so transVeg/phenology
edits do NOT invalidate state. DEVIATION from compute_architecture.md's Delta-SVF key
list (which includes transmissivity/phenology), with reason: those keys matter only if
float scalars were cached; we cache bits only.

Memory arithmetic site_500 (500x500=250,000 px, 153 patches): packed per stack
250,000 x ceil(153/8)=20 B = 4.77 MiB (vs 146 MiB dense float32, pack_svf.py:5-9
precedent); two veg stacks 9.54 MiB; trunk mask 0.24 MiB; chunk metadata ~6 KiB;
per-scenario COW delta single-tree <= ~8 MB worst case; site-level baseline shared
~9.8 MiB; folded scalars 0 (on demand, ~11.3 M MACs at write extent ~20-50 ms).

Persisted vs recomputed: on disk under scenario results root (packed bits + chunk
index + key, checksummed load per pack_svf.py:112-130, atomic rename). Site baseline
packed ONCE from existing P2 cache svf_patches cubes (cache.py:196-197, sliced today
at solver.py:862-863) — ~2-5 s off the request path. UNCERTAIN (flagged): cached cubes
were marched with ABSOLUTE amaxvalue by original svf_calculator; bit-identity vs
effective_march_amplitude replay follows the solver.py:616-660 argument but no test
pins it — V0 below adds one. Recomputed per process: patch geometry (memo exists),
offset tables, folded scalars.

## 3. UPDATE ALGORITHM

Core invariant (corridor sufficiency, to be proven formally in-wave): bits at (t,p)
can change only if ray_p(t) intersects C. Sketch: every per-step comparison reads only
a/vegdsm/vegdsm2 at ray cells; amplitude changes (taller tree raises, deleting tallest
lowers tile amplitude) only extend/truncate TAIL steps — extended steps have dz >
A_old >= vegdsm[o]-a[t] for every UNCHANGED occluder o (flip nothing); shortened steps
have dz > A_new >= vegdsm[o]-a[t] for unchanged o post-composition; either way only
rays through CHANGED cells flip. Step-1 block and suppression re-read only ray cells.
FLAGGED uncertainties to settle by brute force: the step-1 special (shadow.py:288-293)
and suppression re-raise after vegsh[vegsh*sh>0]=0 (283-284) — V2/V3 pin both.

apply_batch pseudo-code:

    guards: bush any nonzero -> FallbackFullReplay (solver.py:1022-1032 precedent);
            key mismatch -> rebuild or replay; touched-chunk checksum mismatch -> FallbackFullReplay
    C = union of footprint cells over (old_tree, new_tree) of each coalesced edit   # trees.py:103-153
    new_canopy = layer.vegetation_rasters_window(full)[0]                           # max-composed, trees.py:323-344
    compose vegdsm/vegdsm2/bush exactly as compose_full_scene_tensors               # solver.py:508-516
    for ring in rings:                              # 8 rings, memoized geometry
        R = _patch_march_reach_pixels(A_eff, scale, ring.altitude)                  # solver.py:929-950
        for p in patches_of(ring):                  # 153 total
            corridor_p = union_{s=1..R} shift(C, -(dx_s,dy_s)) clamped to tile
            if empty: continue                      # untouched patches cost NOTHING
            old = unpack bits(p, corridor_p); new = march_corridor(p, corridor_p, a, vegdsm, vegdsm2)
            if new != old: write bits (batched set_patch_window, bitmask.py:135-161); dirty_scalars |= diff cells
            bump chunk versions over corridor_p u C; recompute sha256
    scene_revision += 1; persist atomically

Heterogeneous batches: forcing/params already route FULL (worker.py:1091-1101);
landcover joins tree windows but does NOT change vegdsm (worker.py:1118-1131) — state
stays valid. BUILDING edits change a (every ray) and route full regeneration
(executor.py:1148-1167, R0 map) — they INVALIDATE state via cache_manifest_sha256 in
the key -> rebuild. That is the separate building/vegetation semantics: buildings
baseline-invalidating, vegetation incremental.

march_corridor phases: (A) run UNTOUCHED shadow() on the bbox of corridor_p per
patch, take corridor cells — identical arithmetic at narrower extent (W2 precedent,
solver.py:1229-1249); diagonal-azimuth bboxes blow up (~200x200 at 6 deg) so only
~2-5x on the dominant ring. (B) fused gather kernel over all (t,p) corridor pairs:
for step s=1..max_reach one batched gather+compare+accumulate over every live pair
with s <= R_p (~100k pairs single-tree, ~25 M (t,p,step) triples vs 1.41e9 cell-steps
today); per-(t,p) must reproduce shadow()'s EXACT op sequence — same dz float32
expression (shadow.py:255), same comparison order, same suppression/max interleaving,
same step-1 block, same edge-clamp semantics (out-of-bounds reads contribute nothing,
equivalent to shadow.py:271-273 clipped fills). Feature-flagged until V4 passes; phase
A is the fallback.

SVF accumulator rule: NEVER subtract-old/add-new on float accumulators — bitwise
safety would require changed patches to be a fold SUFFIX with no rounding propagation,
unprovable per cell in general. Instead: for dirty_scalars n write_window, execute the
FULL 153-patch canonical fold from stored bits (per-cell sum independent of other
cells => bit-identical to replay values at consumed cells; replay's zero-padding is
explicitly inert, solver.py:1295-1302); then last/clamps/svftotal exactly as
solver.py:1269-1291. Cost: 153 vectorized multiply-adds over <= ~74k cells, tens of
ms. This is compute_architecture.md step 5's "same scientific order" achieved by
re-folding, not ordered float deltas.

## 4. PARITY STRATEGY (enumerated)

1. Untouched (t,p) bits reused: BITWISE, contingent on corridor proof + V1/V2.
2. Corridor bits phase A: BITWISE (untouched arithmetic, narrowed extents).
3. Corridor bits phase B: BITWISE ONLY AFTER per-cell op-sequence proof (V4);
   flag-off until then.
4. Folded scalars/svftotal/last/clamps: BITWISE (same ops, same order, per-cell
   independent).
5. Downstream svfbuveg/diffsh/sky_masks/radiation (utci_process.py:743-773;
   solweig.py:566-824): BITWISE (unchanged code, identical inputs).
6. Baseline bits packed from P2 cache: BITWISE PENDING V0 (absolute-vs-relative
   amaxvalue — asserted, not test-pinned).
7. Float accumulator deltas: REFUSED (would need the reviewed tolerance; unnecessary
   under re-fold).
8. Un-flattening per-tree trunk_ratio/transmissivity at solver.py:502,511: OUT OF
   SCOPE / FENCE (changes physics vs oracle). The compute_architecture tolerance
   allowance is NOT used; it remains the escape hatch only if phase B's proof fails in
   a corner (phase A still ships bitwise).

## 5. VALIDATION PLAN

Oracle protocol: W4 pattern (handoff §10.2) — throwaway worktrees, direct solver
level, real site cache READ-ONLY, np.array_equal(equal_nan=True) on
vegshmat/vbshmat/all 10 scalars/svftotal + downstream utci/tmrt/shadow; small-grid
exhaustive brute force on synthetic 60x60 tiles.

- V0: fresh full-tile replay bits vs packed P2 cache bits (pins case 6).
- V1 battery (site_500): add / move / resize / delete / OVERLAP (two trees one ray;
  short tree under tall canopy) / LOW-SUN (6 deg ring corridors, dominant) / BOUNDARY
  (tree at tile corner/edge — clamping equivalence) / REPEAT EDITS (edit->undo->
  re-edit) / AMAXVALUE RAISER and DROPPER (taller-than-max add; delete-the-tallest —
  both directions of §3's amplitude argument).
- V2: randomized edit sequences on synthetic tile, state-vs-replay bitwise after EVERY
  edit (drift test — bits are exact classifications, equality must hold at every
  step).
- V3: fold vs replay for randomized dirty-cell subsets (locks quadrant gates, last,
  clamp ordering).
- V4: phase-B kernel per-(cell,patch) exhaustive vs shadow() on small grids incl.
  edge cells; then V1 re-run with kernel enabled.
- Failure-mode tests: corrupt chunk bytes -> checksum mismatch -> FallbackFullReplay
  and solve still bitwise-correct via today's path; stale key -> rebuild;
  bush-nonzero scene -> fallback; kill during persist -> atomic rename leaves previous
  revision intact.

## 6. IMPACT + RISKS

Today svf_seconds 19.8-20.3 s ~ 153 full-window marches (~1.41e9 cell-steps post-W2);
per handoff §9.1 ~40% is per-step call overhead, 6 deg ring ~65% of phase. Ring
reaches at site_500 (amp 56.4 m, scale 0.5 px/m): 270/88/50/33/22/14/7/1 px;
sum(patches x corridor area) ~ 100k cells vs 1.41e9. ANALYTIC estimate from the
benchmark tree (18 m/11 m), not measured — flag for R4's first profiling gate.

First edit: baseline pack one-time ~2-5 s off request path (server start/scenario
creation); corridor phase A ~8-12 s, phase B ~0.2-0.5 s; re-fold 20-50 ms; bundle
slice+unpack ~10 MB negligible. Repeat edit: same corridor cost (state exists), no
pack. Job-level (single-step, time_stop=13): ~38 s -> ~25-30 s (phase A) -> ~18 s
(phase B), then time_loop (~17 s) dominates and R5/R3 own the rest of the one-second
mandate. Honest claim: R4 removes the veg-SVF replay for first AND repeat edits; it
does NOT alone reach one second end-to-end.

Top 5 risks: (1) corridor-sufficiency proof gap (step-1 special, suppression re-raise,
amplitude extend/truncate) — formal proof note + V2 battery + guard fallback;
(2) fold order divergence — mitigated BY CONSTRUCTION via the fold refactor (one code
path for replay and state-fold; V3 locks it); (3) phase-B kernel arithmetic divergence
(dz float32, edge clamps, while-stop) — feature flag, phase A default, V4 gate;
(4) state lifecycle/corruption — per-chunk checksums, revision-keyed atomic persist,
rebuild-on-mismatch, LRU retention cap; (5) concurrency under R2's exact pool — state
mutation single-writer, committed with exact-lane publication (revision-keyed COW
snapshots mirroring R5 checkpoint keying); today's single worker thread makes it a
design constraint now, not a retrofit.

## 7. INTEGRATION SEAM (oracle paths untouched)

New module solweig_gpu/incremental/veg_svf_state.py (state, corridors, apply_batch,
validation). NO changes to shadow.py (§8.4 fence) or the read-window/amplitude fences
(solver.py:129, 666-752).

Worker seam: worker.py:_solve_local 1331-1371 — after compose (1341), load/validate
state, apply_batch, pass veg_state=state into solve_window. Additive optional
parameter mirroring the STEP 2b dedup precedent (solver.py:1435-1444
scene/required_read_window): default None = today's replay, bit-identical.

Solver seam: window_svf_bundle (solver.py:763) gains veg_state=None; when present and
validating: slice packed bits at cube_window, unpack to float32 cubes (identical
values to solver.py:1248-1249), fold scalars at consumed extent, apply 1269-1291
verbatim, assemble the unchanged 19-tuple (886-906). The veg_changed=False
StaleSvfError guard (832-846) and _check_baseline_amplitude (1358-1386) remain
authoritative and untouched.

Oracle paths unchanged: run_full_tile -> svf_calculator (solver.py:1744-1746;
utci_process.py:1151) stays the authority and fallback target.

R5 interplay: R4 state is the Stage-7 dependency node; its key tuple (site, scenario,
scene_revision, patch-geometry id, baseline checksum) is exactly
compute_architecture.md's versioned-dependency-cache key; R5's per-exact-revision
temporal checkpoints consume the bundle exactly as today (solver.py:1566-1599). State
commits at exact-lane publication so R4/R5 share one revision lineage. Telemetry:
attribute corridor/fold/pack time under svf_seconds in stage_timings
(solver.py:1531-1535) so R0 instrumentation keeps working.

## RECOMMENDATIONS (ordered)

1. Phase A: state + corridor-bbox march + re-fold + the fold refactor of
   _recompute_veg_svf_window to accept precomputed per-patch planes — medium effort,
   ~2x svf phase, zero new march arithmetic, unblocks all.
2. Pin V0 FIRST (trivial cost, removes the largest asserted-not-tested link).
3. Phase B as a separately gated wave with V4 op-sequence proof — the actual
   sub-second svf stage.
4. Write the corridor-sufficiency proof as a reviewable note BEFORE implementation
   (step-1 block + suppression re-raise) — what scientific review will ask for.

## TRADE-OFFS

- A (chosen) persistent bits + corridor re-march + canonical re-fold: bitwise by
  construction, no drift, deletion/overlap exact via recomposition, ~10 MB/scenario,
  transmissivity-independent | corridor cost scales with reach (6 deg ring dominates),
  phase B needs its own parity proof, new persisted state to lifecycle-manage.
- B float accumulator deltas: smallest per-edit compute | breaks the §1.3 order fence
  in general -> needs reviewed tolerance, drift compounds, deletion needs per-ray
  occluder identity the march never had.
- C winning-occluder refcounts per (cell,patch): analytic updates without re-march |
  duplicates march classification semantics (step-1, suppression) as second physics
  implementation, larger memory, worse parity surface than re-marching thin corridors.
- D cached-previous-SVF-cube reuse (handoff §9.2 as written): cheapest to build |
  exactly what compute_architecture.md:64 rejects — cannot prove which object
  determined each direction; wrong under deletion/overlap.

## ADDENDUM (2026-09-04, lead) — proof results (r4-proof): design corrections BEFORE implementation

Proof note + scripts + evidence: `proofs/corridor_sufficiency_note.md` (this
directory) and siblings. Status: pending non-author scientific review before the
corridor path ships enabled; the corrections below are BINDING on the R4
implementation wave regardless of review outcome.

1. **V0 CLOSED — bit-identical, both regimes.** Fresh full-tile replay vs packed
   P2 cache: 0 mismatching (cell,patch) pairs on site_500 (read-only, unclamped
   regime — min(a)>0, relative bound < abs, early exit active) AND on the
   synthetic science site (clamp ACTIVE). The §2 UNCERTAIN
   (absolute-vs-effective amaxvalue) does not manifest as a bit divergence;
   parity case 6 upgrades from "pending V0" to PROVEN.
2. **Corridor closure must be `corridor_p ∪ C` — C itself is load-bearing.**
   shadow()'s step-1 special reads target-LOCAL `vegdsm2[t]`/`a[t]`
   (shadow.py:288-293); 32 observed vegsh flips landed strictly inside C and
   strictly outside the ray corridor. §3 apply_batch's re-march set must use
   `corridor_p ∪ C` (the union the design already uses for chunk versioning),
   not the ray corridor alone.
3. **Corridor sufficiency FAILS in the clamped regime under amplitude change.**
   §3's sketch inequality is false as written when the SMALLER-amplitude scene
   is clamped (`A_short = abs < bound = max(surface) − min(a)` lets an unchanged
   occluder satisfy `vegdsm[o] − a[t] > A_short`): raiser 12→18 and dropper
   20→12 each produced 20 off-closure violating patch/stack entries; a dedicated
   corner probe produced 53. No-clamp sequence: 16/16 edits ZERO violations
   including 12→18 and 20→12 with 5.7k/6.1k flips. Violation predicate
   `(amp changed ∧ clamp on smaller-A scene)` matched 32/32 edit-states.
4. **MANDATORY GUARD** (complements `_check_baseline_amplitude`, which does NOT
   cover droppers on min(a)≥0 clamped sites): if `bound > abs` on the
   smaller-effective-amplitude scene → full replay for the batch. Cost: three
   `.max()` + one `.min()` per scene.
5. **Phase A must use W2 reach-expanded windows, not raw corridor bboxes.**
   Raw bbox diverges from full replay on 19/59 patches (worst 167 cells —
   while-stop uses the WINDOW's shape; out-of-bbox in-tile reads lost);
   `_patch_march_window` reach expansion reproduces 59/59 bitwise.
6. **Preconditions adopted as asserts**: bush ≡ 0 on pre AND post scenes (else
   add bush cells to C); C computed on COMPOSED surface diffs (vegdsm/vegdsm2/
   bush), canopy diff only as a practical superset; trunk_ratio-only edits are
   provable no-ops (compose flattens trunk).
7. **Residuals disclosed** (proof note §10): float32 dz monotonicity assumed
   (failure direction grows the corridor — conservative); min(a)<0 and bush≠0
   untested (guard + existing amplitude check cover); brute force is
   spot-check scale — the analytic argument carries generality; vbsh under Case
   C not exhaustively characterized (guard removes the regime).

## ADDENDUM 2 (2026-09-04, lead) — non-author review VERDICT: APPROVE-WITH-CONDITIONS

r4-proof-review (adversarial, independent): march verified line-by-line, Case D
+ guard attacked analytically (incl. float32-rounding and accumulator attacks),
full evidence re-run — corridor_bruteforce JSON byte-identical to author's, V0
site_500 rerun 0 mismatches both stacks, raiser counterexample reproduced exact.
Phase A is CLEARED to implement under conditions 1-4 below (binding on the R4
wave). Key outcomes:

- **MAJOR (new precondition, condition 4): vbsh is NOT binary in general —
  {0,1,2}.** When the march executes exactly ONE step (dz_1 > A_eff; live at
  the 78° ring on coarse-pixel/low-relief sites — the 4 m-pixel science-recipe
  class; site_500 with A_eff/pixel 28.9 >= 6.65 is safe) and the step-1 raise
  fires, the accumulator was zeroed at index==1 AFTER its only add
  (shadow.py:293) → final 1-(0-1)=2.0; the oracle folds 2w while packed
  bits+re-fold fold 1w. pack_svf/bitmask refuse non-binary LOUDLY (crash, not
  silent corruption). Reviewer constructed it (60x60, 4 m pixels, 12 m
  building, 6 m/8 m crown → 8 cells vbsh==2.0 at patch 148).
- **Guard predicate CORRECTED (condition 1): must include the
  amplitude-changed conjunct** — fallback iff `A_eff(S) != A_eff(S')` AND
  `bound > abs` on the smaller-A_eff scene. Without the conjunct the guard
  fires on EVERY edit of a steady-state clamped site (Case A, proven safe, 0
  violations), silently voiding R4 on exactly the clamped science-site class.
  With it the guard exactly carves the trichotomy: amp-constant = Case A
  proven; amp-changed + smaller-unclamped = Case B proven; amp-changed +
  smaller-clamped = Case C (violations demonstrated). bound==abs boundary
  safe (strict >; empirically resize 18==18 → 0 violations).
- **Conditions for the R4 Phase A wave (verbatim from review):**
  1. Fallback to full replay for an edit batch iff A_eff(S) != A_eff(S') AND
     bound(smaller-scene) > scene_amaxvalue(smaller-scene), with
     bound = max(a.max, vegdsm.max, vegdsm2.max) - a.min() evaluated on the
     FULL-TILE composed pre/post scenes with effective_march_amplitude's exact
     expression; amplitude-constant transitions must NOT trigger the fallback.
  2. The per-patch re-march set is corridor_p (union of pre/post
     global-effective-amplitude march_offsets) ∪ C, with C computed on
     composed surface diffs (vegdsm/vegdsm2/bush); assert bush == 0 on pre AND
     post scenes, else fallback.
  3. Phase A marches run on W2 _patch_march_window reach-expanded windows
     with the windowed effective_march_amplitude — never the raw corridor bbox.
  4. After every corridor re-march (and before baseline packing) assert the
     float vegsh/vbsh planes are within {0,1} (max <= 1); on violation fall
     back to full replay for the batch.
- Note factual slips (3-of-16 not 4; clamped-raiser vbsh_flips=0; §9
  pseudocode conjunct) — author amendment in flight, doc-only non-blocking.
- Residual adjudication: every §10 item BLOCKS-Phase-A: no. dz monotonicity
  now PROVEN (fl(fl(ds*s)*c) with fixed positive float32 constants is
  monotone nondecreasing under round-to-nearest) — the "conservative
  direction" hedge retired. "OOB reads contribute nothing" refined: false
  for min(a)<0, but the contribution is scene-INVARIANT (safety survives via
  _check_baseline_amplitude).
- Negative results (attacked, failed to break): step-1 gate target-locality
  (shadow.py:292 uses UNSHIFTED vegdem2/a — the ∪C foundation is exactly
  right); float32 counterexample to Case D; sh/suppression far-field coupling.

R4 Phase A implementation is UNBLOCKED (dispatch sequenced after the r3a
merge — both waves edit the worker/solver integration seams).

## ADDENDUM 3 (2026-09-04, r4a remediation) — condition 3 deviation, perf framing, regime fence provenance

Recorded by the Phase A implementer after the r4a-review remediation pass
(verdict REQUEST-CHANGES → remediated). Three items:

1. **Condition 3, letter vs as-implemented (adjudicated COMPLIANT-AS-
   IMPLEMENTED by the lead).** The letter (ADDENDUM 2, condition 3) says
   the corridor re-march runs "with the windowed
   effective_march_amplitude". The implementation marches with the POST
   scene's FULL-TILE effective amplitude instead (reach expansion still
   the union bound max(eff_pre, eff_post)). Rationale:
   - The reference the packed bits must reproduce is the full-tile replay
     (`_recompute_veg_svf_window` at window = full tile), whose march
     amplitude is scene-wide. The windowed letter is bit-equal at closure
     cells in multi-step regimes but VALUE-divergent in one-step regimes
     (vbsh 2.0 vs 1.0) — the fold consumes VALUES, so window-cropped
     amplitudes would both trip the condition-4 value fence spuriously
     and mask real 2.0s.
   - Direction analysis: for any read window the replay's marches clamp
     at the read edge, so a corridor march in the one-step regime implies
     the replay is one-step there for any read window; marching a
     SUPERSET of steps is safe (steps the replay would not execute can
     only lower the accumulator monotonically toward the replay's value
     at closure cells — verified bitwise against full-tile replay twins
     across the 42-test suite).
   - Adopting the letter verbatim would import MEDIUM-2's latent
     windowed-amplitude replay defect into the corridor path; the
     as-implemented direction is the safe superset. (MEDIUM-2 itself —
     pre-existing W2 replay windowed-amplitude value divergence — is
     LEAD-RULED DEFERRED, not fixed in this wave.)

2. **HIGH-1 regime fence (condition 4, regime level).** The vbsh==2.0
   hazard is per-REGIME, not per-window: a multi-step → one-step
   amplitude transition (unclamped dropper, Case B — the condition-1
   clamp guard correctly silent) leaves oracle vbsh==2.0 cells at
   UNCHANGED trees outside every re-marched window, where the state's
   packed bits are stale and the fold silently diverges. The remediation
   adds `_regime_transition_reason` (batch refuses iff any patch has
   dz_1 > A_eff(post) while it was multi-step at A_eff(pre); dz_1 is pure
   arithmetic, no march needed) and the same check at baseline pack (a
   baseline already one-step anywhere refuses the pack). Structural
   consequence: any scene whose corridor re-march can produce 2.0 values
   is itself one-step somewhere → refused at pack; the reviewer's
   original apply-time counterexample construction is now precluded at
   pack time (pinned by test restructure, `TestVbshBinaryFence`).

3. **Perf framing (LOW-1, stated honestly).** The corridor path saves
   against the FULL-TILE replay only. At 96×96 @2 m a corridor apply
   (~0.38–0.45 s, all 153 patches re-marched at this amplitude) beat the
   full-tile replay (~0.51–0.59 s) but was ~1.3× SLOWER than the
   deployed-shape W2 windowed replay (read 66×78, accumulate 24×30,
   ~0.32–0.34 s) — at this tile size every patch's closure is non-empty
   and the W2 replay's windows are already tiny. The corridor shape is
   expected to pay off as tiles grow (per-patch march windows scale with
   the closure bbox, not the read window); the path remains
   correctness-first: every refusal routes the full replay.
