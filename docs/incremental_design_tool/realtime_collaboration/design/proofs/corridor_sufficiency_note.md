# R4 Deliverable B — Corridor-Sufficiency Proof Note

**Claim under test** (design `docs/incremental_design_tool/realtime_collaboration/design/r4-veg-svf-occlusion-state.md` §3):
after a vegetation edit with changed-cell set C, the veg-SVF bits at (target, patch) can change
only if the patch's march ray from the target intersects C — including under `amaxvalue` changes.

**VERDICT: the claim holds in the NO-CLAMP regime and FAILS in the CLAMPED regime when the
march amplitude changes.** The failure is not a corridor-definition bug: it is a genuine property
of `effective_march_amplitude` clamping, demonstrated by per-step trajectory traces and reproduced
in a dedicated corner probe. A one-line sufficient condition (guard) restores the claim.

All scripts + evidence: `/tmp/r4_proof/{common.py, v0_pin.py, corridor_bruteforce.py, raiser_repro.py, trajectory_probe.py}`,
results `/tmp/r4_proof/{v0_site500.json, v0_synth.json, corridor_bruteforce.json}`, logs `*.log`.

---

## 1. Corrected statement of the claim

Let, for a scene S and patch p (altitude α, azimuth θ):

- `steps(S,p)` = the executed while-loop steps of `shadow()` (shadow.py:194-255), which depend on
  the **effective march amplitude** `A(S)` (via the `amaxvalue >= dz` stop), `α`, `θ`, scale, and tile size;
- `ray_p(S, t)` = `{ t + (dx_s, dy_s) : s ∈ steps(S,p) }` ∩ tile (the cells the march reads at t).

Define the **C-closure of patch p**:

```
closure_p = { t : ∃ s ∈ steps(S_before,p) ∪ steps(S_after,p),  t + (dx_s, dy_s) ∈ C }  ∪  C
```

i.e. the ray corridor over the **union** of pre/post executed steps, **plus C itself**.

**Claim (corrected).** For every edit whose changed-cell set is C, with pre/post scenes S, S′:

```
bit(t,p) ≠ bit′(t,p)  ⟹  t ∈ closure_p
```

**provided** (amplitude condition) the scene with the SMALLER effective amplitude is unclamped:

```
bound ≤ abs   on  argmin(A(S), A(S′))      where
  bound = max(a.max, vegdsm.max, vegdsm2.max) − min(a)
  abs   = scene.amaxvalue   (absolute amplitude)
```

Equivalently `A_eff = bound` (no clamp) on the smaller-amplitude scene, which guarantees every
occluder relevant at amplitude `A_long` is already reachable/comparable at `A_short`.

The design §3 sketch ("dz > A_old ≥ vegdsm[o] − a[t] suffices") is **false as written** in the
clamped regime; see §5.

## 2. Per-step read analysis (why ray-corridor ∪ C, exactly)

`shadow()` march (shadow.py:275-293), per step s:

| read | expression | in corridor? |
|---|---|---|
| building profile | `a[xc1:xc2, yc1:yc2]` shifted → `temp` → `f`, `sh` | yes — shifted by (dx_s, dy_s) |
| canopy surface | `vegdsm[...]` → `tempvegdem` | yes |
| trunk/bush base | `vegdsm2[...]` → `tempvegdem2` | yes |
| step-1 special | `firstvegdem = tempvegdem − temp` (i.e. vegdsm−a at the ray cell), then **`vegsh = vegsh * (vegdem2 > a)`** — reads `vegdsm2[t]` and `a[t]` AT THE TARGET | **no — target-local** |

- Every per-step comparison operand is a ray cell (t + offset) or the target itself.
- `a[t]` (target's own building+DEM elevation) is only ever an RHS comparison operand; vegetation
  edits never change `a`. Invariant, safe.
- `vegdsm2[t]` (target-local trunk/bush base) **can change** under an edit whose footprint covers
  the target. Hence **C must be included in the closure** — the ray corridor alone is insufficient.
  This is not hypothetical: 3 of the 16 no-clamp edits produced vegsh flips strictly inside C and
  strictly outside the ray corridor (add3: 14 flips; move_t0: 6; delete_corner_sw: 12 —
  `flips_in_C_not_raycorridor` in corridor_bruteforce.json).
- Out-of-bounds ray reads contribute candidate 0 (temp arrays zeroed; the shifted slice is simply
  absent) — no clamped edge reads. That candidate is SCENE-INVARIANT, not inert: it is exactly 0
  regardless of the scene, so it cancels in any pre/post comparison with identical step sets.
  When min(a) < 0 it is not a neutral operand (`0 > a[t]` can hold), so out-of-bounds steps can
  set sh/vegsh — but identically before and after; the only way their effect could differ is a
  changed step set (amplitude change), which `_check_baseline_amplitude` refuses outright on
  min(a) < 0 sites (solver.py:1358-1386). Edge targets are therefore covered by the same
  argument.
- Design §3 `apply_batch` pseudo-code re-marches the ray-corridor set only (`if empty: continue`)
  — it would MISS these target-local flips. The design DOES union `corridor_p ∪ C` for chunk
  versioning; the re-march set must use that same union.

## 3. `sh` gating and suppression ordering (no far-field coupling)

Question: can a changed cell FAR from the target alter the target's bit without being on its ray
(e.g. changed sh suppressing/re-releasing vegsh)?

- `sh[t]` is computed from `f[t] = max(a[t], max_s a[ray_s(t)] − dz_s)` — the SAME ray's building
  profile only. `a` never changes under vegetation edits ⟹ `sh[t]` is **invariant** per executed
  step sequence (it only changes if the step set changes, i.e. amplitude change — covered in §5).
- Suppression `vegsh[vegsh·sh > 0] = 0` (shadow.py:283-284) only **zeroes** vegsh; it can never
  re-raise a bit. So ordering between suppression and later vegsh2 maxima cannot resurrect a bit.
- `vbshvegsh` accumulates `vegsh` after suppression each step (reset at step 1) and is classified
  once (`>0 → 1`, minus final vegsh, complement). Once vegsh has converged at the last common step,
  extended steps add vegsh values already determined by in-corridor reads (or by extended-step reads
  — §5). No cross-target coupling exists anywhere in the loop: all state arrays are indexed by the
  same t being marched.

## 4. Amplitude extend/truncate argument for UNCHANGED occluders

Case A — amplitude constant (`A = A′`): identical step sets; per-step operands identical off C
(ray cells) and at the target (`vegdsm2[t]` — in C if changed). All intermediates
(`f, sh, vegsh, vegsh2, vbshvegsh`) equal off closure. Bits off closure invariant. (Empirically:
every amplitude-constant edit in BOTH sequences, clamped or not, 0 violations.)

Case B — amplitude changes, no clamp on the smaller-amplitude scene `A_short`:
`A_short = bound_short = max_surface − min(a) ≥ vegdsm[o] − a[t]` for every cell o and every
target t (since `a[t] ≥ min(a)`). Split the longer march at the last common step:

- **Truncate direction** (`A_long → A_short`): dropped steps s have `dz_s > A_short`. For any
  target t and dropped-step read o: had o mattered at s, the operand comparison is
  `vegdsm[o] − dz_s > a[t]` ⟸ needs `vegdsm[o] − a[t] > dz_s > A_short` — impossible.
  So dropped steps never flipped a vegsh bit on the A_long scene... except via `vbshvegsh`
  accumulation of a POSITIVE vegsh2 — same impossibility (vegsh2 = +1 requires the same
  inequality). Dropped `sh` updates can only have set sh=1 where `a[o] − dz_s > a[t]`, i.e.
  `a[o] − a[t] > A_short ≥ bound − min(a) + min(a) = max_surface ≥ a[o] − a[t]` — contradiction.
  Hence truncation is bit-safe off C.
- **Extend direction** (`A_short → A_long`): new steps read cells o with
  `vegdsm[o] − a[t] ≤ A_short < dz_s`, so vegsh2 ∈ {0, −1} at those steps: no raise. `sh` may
  newly set (1→0 final polarity) only under the same impossible inequality on `a`. vegsh2 = −1
  steps: `vegsh = max(vegsh, vegsh2)` cannot lower vegsh below 0 (it is already ≥ 0), and
  vbshvegsh classification is settled because a −1 step implies vegsh2 ≤ 0 can only decrement a
  positive accumulator — but the accumulator was already classified >0 iff some earlier step had
  vegsh>0; max(vegsh, −1) = vegsh unchanged. Bit-safe off C.

Case C — amplitude changes AND the smaller-amplitude scene is CLAMPED (`A_short = abs < bound`):
there exist cells o with `surface[o] − min(a) > abs`. For such o and a low target t
(`a[t] ≈ min(a)`), `vegdsm[o] − a[t] > A_short` becomes possible at an extended step with
`A_short < dz_s ≤ A_long` ⟹ vegsh2 = +1 ⟹ **bit flips at t although no ray step of either
executed march intersected C and t ∉ C**. This is the violation mechanism; §5 demonstrates it
per-step.

## 5. Demonstrated counterexample (clamped regime) — mechanism trace

Setup (`/tmp/r4_proof/raiser_repro.py`, `trajectory_probe.py`; 60×60 tile, pixel 2 m, dem 0,
12 m NW block, 4 m SE block, random trees, rng 20260904): one random tree's crown overhangs the
4 m SE block → `vegdsm[(40,47)] = 13.93 > abs 12` → clamp ACTIVE on the pre-edit scene
(bound 13.93+ > abs 12; `A_eff = 12`). Edit: add an 18 m "tall" tree on open ground →
`abs 12 → 18`, clamp released (`A_eff: 12 → 18`, `clamp_before/after = True/False`).

**Violation**: patch 34 (ring altitude 18°, azimuth 40°), target t = (56, 34):

- Full-replay bits: amp 12 → `vegsh[t,p] = True` (sky visible); amp 18 → `False`.
- t's ray over the UNION step set intersects C at **zero** steps; t ∉ C.
- Trajectory probe (`shadow_probe`, exact copy of the march at this cell; target (56,34) is open
  ground, `a[t] = 0.0`): at amp 18, **step i=16 reads (40, 47)** — the unchanged overhanging
  crown, `vegdsm[o] = 13.931` (9.93 m canopy over the 4 m SE block), `a[o] = 4.0`,
  `dz = 13.573`. The raise test is `tempvegdem[t] > a[t]`, i.e.
  `vegdsm[o] − dz = 13.931 − 13.573 = 0.357 > a[t] = 0` → `vegsh2[t] = +1` → vegsh set → final
  bit flips to shadowed. At amp 12 the while-stop (`amaxvalue ≥ dz`) halts before
  dz = 13.573 is ever reached, so step 16 never executes and the crown is invisible to the march.
- Why this is exactly the Case C window. The raise test at a march step s reading cell o is
  `vegdsm[o] − dz_s > a[t]`, and step s executes only while `dz_s ≤ A`. So an off-closure flip
  at an extended step requires an UNCHANGED occluder o with
  `vegdsm[o] − a[t] > dz_s > A_short` — nonempty exactly when
  `max_o (vegdsm[o] − a[t]) > A_short`. Here `vegdsm[o] − a[t] = 13.931 − 0 = 13.931` and
  `A_short = abs = 12` (the clamp: `bound = vegdsm.max − min(a) = 13.931 > abs = 12`), with
  dz_s = 13.573 ∈ (12, 18]. The design §3 sketch inequality "`dz > A_old ≥ vegdsm[o] − a[t]`"
  is the right shape but its premise fails under clamp: `A_old` (= A_short) does NOT dominate
  `vegdsm[o] − a[t]` for every unchanged o — precisely because
  `A_short = abs < bound = max(vegdsm) − min(a)`. In the unclamped regime
  `A_short = bound ≥ vegdsm[o] − min(a) ≥ vegdsm[o] − a[t]` for every o and t, the window is
  empty, and the sketch holds.

Symmetric dropper counterexample (same sequence): deleting the tall tree drops
`abs 18 → 12` with clamp re-engaging AFTER the edit (`clamp_before/after = False/True`) —
same 20 violating patch/stack entries (the mirror image: steps that existed before vanish).

Dedicated corner probe (`corner_probe`, 120×120, designed-clamped scene: 25 m tree ON a 30 m
building → vegdsm 55; dominant 45 m open-ground tree deleted → `abs 45 → 30`, clamp on the
after scene): **53 violating patch/stack entries**, 15118 vegsh flips total (most in-closure;
the off-closure subset concentrated in the 18° ring at targets looking toward the 55 m
tree-on-building from far away — e.g. patch 52 az 256°, 30 cells along row 0). Violating stacks
were exclusively `vegsh`; vbsh violations 0 in the corner probe, and the 60×60 clamped raiser
likewise shows `vbsh_flips = 0` (corridor_bruteforce.json). vbsh is accumulate-then-classify
(post-suppression vegsh summed per step), so single vegsh raises at extended steps often fail
the accumulator's >0 test — but do NOT rely on this: Case C can flip vbsh via
raise-then-suppress (the extended step's raise enters the accumulator; a LATER step's
sh-suppression zeroes final vegsh, so the final fold `1 − (bin(acc) − vegsh_final)` yields 0
where the no-raise fold yields 1 — a flip), and the §9 guard removes the regime regardless.

## 6. Brute-force results (60×60 synthetic tiles, 153 patches, full-tile replay before/after)

Method per edit: full-tile `_recompute_veg_svf_window` replay bits before and after;
flips (`vegsh` cube XOR, `vbsh` cube XOR, per patch) must lie inside
`closure = corridor(C, union steps) ∪ C`; C = canopy raster diff (exact cells that changed).

**NO-CLAMP sequence** (random trees kept entirely off both buildings — clamp inactive on every
scene state; verified per-edit `clamp False/False`):

| edit | C cells | amp | vegsh flips | vbsh flips | violations |
|---|---|---|---|---|---|
| add0 | 9 | 12→12 | 2971 | 50 | **0** |
| add1 | 1 | 12→12 | 890 | 27 | **0** |
| add2 | 9 | 12→12 | 4099 | 70 | **0** |
| add3 | 8 | 12→12 | 1872 | 15 | **0** |
| add_corner_ne | 9 | 12→12 | 2360 | 0 | **0** |
| add_corner_sw | 5 | 12→12 | 1352 | 0 | **0** |
| add_under_base1 | 0 | 12→12 | 0 | 0 | **0** (shadowed add — C=0, no flips) |
| **add_tall_raiser** | 9 | **12→18** | 5793 | 0 | **0** |
| resize_tall_20m | 9 | 18→20 | 1033 | 0 | **0** |
| **delete_tall_dropper** | 9 | **20→12** | 6116 | 0 | **0** |
| move_corner_ne | 18 | 12→12 | 4239 | 0 | **0** |
| move_t0 | 18 | 12→12 | 6100 | 54 | **0** |
| resize_t1_wide | 28 | 12→12 | 8775 | 187 | **0** |
| delete_corner_sw | 2 | 12→12 | 363 | 0 | **0** |
| delete_overlap | 0 | 12→12 | 0 | 0 | **0** |
| move_resize_base2friend | 18 | 12→12 | 6595 | 76 | **0** |

**CLAMPED sequence** (identical RNG stream; random trees may land on/overhang the 4 m SE block):

- ALL amplitude-constant edits (incl. clamped states, `clamp True/True`): **0 violations**.
- `add_tall_raiser` (amp 12→18, clamp True/False): **20 violating patch/stack entries**
  (18° ring, patches 31-53 corridor; e.g. patch 34 cell (56,34); patch 35 (53,31); patch 37
  (45,27),(45,28)) — all vegsh.
- `delete_tall_dropper` (amp 20→12, clamp False/True): **20 violating entries** (mirror set).
- `resize_tall_20m` (amp 18→20, clamp False/False — the 18 m tree set vegdsm.max − min(a) = 18 ≤ abs 18): **0 violations** — amplitude change ALONE is not the trigger; clamp on the
  smaller-amplitude scene is the necessary co-condition, exactly as the corrected claim predicts.

Violation predicate check across all 32 edit-states:
`violations > 0  ⟺  (A_eff changed) ∧ (clamp active on the smaller-A_eff scene)` — 32/32 consistent.

## 7. Phase-A march-window requirement (design §5 phase A wording)

Design phase A says "run UNTOUCHED `shadow()` on the bbox of corridor_p". Measured (representative
single-tree edit, 59 corridor-nonempty patches, full replay as reference):

- **RAW corridor bbox** march (no expansion): 19/59 patches diverge from full replay, worst 167
  cells. Two causes: (i) while-stop `|dx|<sizex` uses the WINDOW's `a.shape`, truncating steps
  that the full tile executes; (ii) reads that fall outside the bbox but inside the tile are lost.
- **Reach-expanded window** (W2 `_patch_march_window`: expand bbox by
  `reach = ceil(A·scale/tan(α)) + 1` in the patch's READ directions, row_dir = −sign(cos),
  col_dir = +sign(sin), clamped to tile): **59/59 patches bitwise identical** to full replay on
  the closure cells.

**Phase A must use the reach-expanded window** (the existing W2 machinery), not the raw bbox.
Also note the windowed `effective_march_amplitude` over the expanded window must be used for the
oracle parity to hold (it was, in the 59/59 check).

## 8. Preconditions and composition quirks (must hold on pre AND post scenes)

1. **bush ≡ 0**: `vegsh` initializes from `bushplant = (bush > 1)`. bush is derived
   (canopy+DEM geometry), nonzero when dem=0 and canopy exists at both layers — on site_500 and
   all fixtures here bush ≡ 0. If bush ≠ 0 anywhere, add bush cells to C (or assert ≡ 0).
2. **Trunk ratio edits are no-ops**: compose flattens trunk to `0.25·canopy` raster unused by the
   march — trunk_ratio-only edits change no march inputs. (Consistent with V0: same solver path.)
3. **`vegdsm[vegdsm == a] = 0` quirk**: a canopy whose vegdsm exactly equals a[t] at some cell
   zeroes it. This is inside compose, so C computed as canopy-raster diff captures any induced
   vegdsm diffs; the corridor argument then applies to the composed surfaces. C should ideally be
   computed on COMPOSED surface diffs (vegdsm, vegdsm2, bush), not raw canopy — canopy diff is a
   superset in practice (max-composition is monotone) but the composed diff is the sound object.
4. **min(a) < 0 sites**: `_check_baseline_amplitude` (solver.py:1358-1386) refuses ANY amplitude
   change when min(a) < 0. My Case B argument used `a[t] ≥ min(a)` which holds for any sign, but
   `bound ≤ abs` with negative min(a) is harder to satisfy; untested empirically — guard covers it.

## 9. Recommended R4 guard

Before applying a corridor-restricted update for an edit batch:

```
smaller_scene = S if A_eff(S) <= A_eff(S') else S'
if A_eff(S) != A_eff(S') and bound(smaller_scene) > smaller_scene.amaxvalue:
    fall back to FULL replay for this patch set (or all patches)
```

The `A_eff(S) != A_eff(S')` conjunct is REQUIRED: an amplitude-CONSTANT edit on a
steady-state clamped site (clamp True/True) is Case A — identical step sets, proven safe,
0 violations on every clamped amplitude-constant edit in §6 — and the bare clamp test would
fire on every edit of such a site, silently voiding the corridor path on exactly the clamped
science-site class.

with `bound = max(a.max, vegdsm.max, vegdsm2.max) − min(a)`. This is cheap (three `.max()` and one
`.min()` per scene), and it exactly complements `_check_baseline_amplitude`, which blocks raisers
(`scene.amax > baseline`) and all amplitude changes on min(a)<0 sites but does NOT block droppers
on min(a) ≥ 0 clamped sites — the exact corner that produced the 20+20+53 violations above.

## 10. What is proven, what is not

Proven (code-level + empirical):
- Per-step read set is ray ∪ {target}; target-local read is vegdsm2[t]/a[t] at step 1 only (§2).
- sh is same-ray, a-invariant; suppression cannot re-raise (§3).
- Amplitude extend/truncate is bit-safe off closure when the smaller-amplitude scene is unclamped
  (§4 Case B argument, backed by 16/16 no-clamp edits incl. 12→18 and 20→12 with 11k+ flips).
- Corridor sufficiency FAILS exactly when (amplitude changed ∧ clamp on smaller-amplitude scene):
  per-step mechanism traced (§5), two-sided brute force (§6), dedicated corner probe (§6 end).
- C must be part of the closure (target-local step-1 gate): 32 in-C-out-of-corridor flips (§2, §6).
- Phase A requires reach-expanded windows: 59/59 vs 19/59 (§7).

NOT proven / residual risks:
- dz monotonicity in the step index — PROVEN (reviewer clarification), not assumed:
  dz_s is computed as `fl32(fl32(ds · s) · c)` with fixed positive float32 constants
  ds and c = tan(α)/scale, and s increases by exactly 1 per step. Multiplication by a
  positive constant is monotone nondecreasing in each argument in exact arithmetic, and
  round-to-nearest preserves monotonicity, so the composition is monotone nondecreasing
  in s. That is exactly what the last-common-step split needs: once the shorter march's
  stop fires at step s* (`A_short < dz_{s*}`), every later step has
  `dz ≥ dz_{s*} > A_short`.
- min(a) < 0 regimes: no fixtures exercised them, but the exposure is bounded — out-of-bounds
  candidates are scene-invariant (§2), so constant-step-set edits compare equal, and amplitude
  changes (the only step-set changers) are refused outright by `_check_baseline_amplitude`
  (solver.py:1358-1386); the §9 guard adds the dropper-on-min(a)≥0 corner.
- bush ≠ 0 sites (no fixtures; precondition 1).
- Brute force is spot-check scale (32 edit-states, 2 tile shapes, 153 patches); the analytic
  argument carries the generality, the brute force the confirmation.
- vbsh-specific corner cases under Case C were not exhaustively characterized (observed vbsh
  violations = 0 off-closure everywhere; do not rely on it — the guard removes the regime).
