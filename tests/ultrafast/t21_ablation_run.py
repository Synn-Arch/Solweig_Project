# SPDX-License-Identifier: GPL-3.0-only
"""T21 ablation driver (DESIGN §807, TASKS T17 protocol).

Modes (fresh process each; artifacts to
``~/Workspace/solweig_ultrafast_artifacts/t21/``):

  parity  — equivalence gates: harness==production identity, restore
            fidelity grid (anchor a x edit r0), independent-oracle
            identity, mutation diagnostics, fence behaviour.
            (correctness-class load)
  memory  — checkpoint memory ledger: exact payload bytes, measured RSS
            delta over a k-anchor sweep, capture/restore costs.
            (memory-class load, brief)
  timing  — window-gated paired full-vs-restore+suffix measurement with
            per-row loadavg and a load guard. RUN ONLY IN AN OPEN
            MEASUREMENT WINDOW (lead-granted).

Usage: python tests/ultrafast/t21_ablation_run.py <mode> [options]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))

import numpy as np  # noqa: E402

from solweig_core.numba_cpu import full_solve as fs  # noqa: E402
from solweig_core.numba_cpu import thermal  # noqa: E402

from tests.ultrafast import t21_suffix_fold as t21  # noqa: E402

ART = t21.ART
LOAD_GUARD = 8.0
N_T = 24


def t_in_is_day(t: int) -> bool:
    """The capture's day/night split (frozen per-t bundle flag)."""
    import numpy as _np
    z = _np.load(t21.CAP / f"t{t:02d}.npz")
    return "sunon_in_shadow" in z.files


def edit_series(series, r0: int, dTa: float = 1.0) -> list:
    """Genuinely perturbed suffix: Ta + dTa from row r0 (T10 family)."""
    return [replace(m, Ta=m.Ta + dTa) if t >= r0 else m
            for t, m in enumerate(series)]


def env_row() -> dict:
    import platform
    row = t21.loadavg_row()
    row.update({"platform": platform.platform(),
                "python": sys.version.split()[0],
                "np": np.__version__})
    return row


def run_window_with_boundaries(cap_set, series, *, dirty_ref=None):
    """Harness fold capturing a Boundary after EVERY step.

    Mirrors run_window step-for-step (same call sequence) while keeping
    deep-copied prefix checkpoints; ``dirty_ref`` (absolute-index
    outputs) turns on stop-at-first-divergence for mutation probes.
    """
    cap_dir = Path(cap_set.cap_dir)
    n = cap_set.n_timesteps
    state = t21.rad_state_from_capture(cap_dir, 0, rows=500, cols=500)
    ci_thread = float(state.CI)
    ci_flavor = False
    site = t21.site_params_from_capture(np.load(cap_dir / "t00.npz"),
                                        cap_set.latitude)
    st = t21.rad_static_from_capture(cap_dir)

    boundaries: dict[int, t21.Boundary] = {}
    outputs: list = []
    ret_ci_series: list = []
    ret_ci_flavors: list = []
    timings_ms: list = []
    for t in range(n):
        t0 = time.perf_counter()
        z = np.load(cap_dir / f"t{t:02d}.npz")
        tg = t21.time_geom_from_capture(z)
        met = series[t]
        t_in = t21.rad_bundle_from_capture(cap_dir, t)
        shadow = (z["sunon_in_shadow"]
                  if t_in.is_day and "sunon_in_shadow" in z.files else None)
        res = t21.recompute_timestep(site, met, tg, ci_thread, shadow,
                                     CI_thread_is_tensor=ci_flavor)
        if t_in.is_day:
            fs._apply_day_overrides(t_in, res, met)
        else:
            fs._apply_night_overrides(t_in, res, met)
        out, state = t21.fused_radiation_timestep(st, t_in, state)
        ci_thread = float(res["ret_CI"])
        ci_flavor = bool(res["CI_flavor_is_tensor"])
        outputs.append(out)
        ret_ci_series.append(ci_thread)
        ret_ci_flavors.append(ci_flavor)
        timings_ms.append((time.perf_counter() - t0) * 1e3)
        boundaries[t + 1] = t21.capture_boundary(
            state, ci_thread, ci_flavor, t + 1, series)
        if dirty_ref is not None:
            if t21.first_output_divergence(out, dirty_ref[t], t) is not None:
                break
    win = t21.WindowResult(outputs, state, ci_thread, ci_flavor,
                           ret_ci_series, ret_ci_flavors, timings_ms,
                           list(range(len(outputs))))
    return win, boundaries


def free_outputs(res) -> None:
    res.outputs.clear()


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------


def run_parity(probes: list[dict]) -> int:
    cap_set = t21.cap_set()
    series = t21.unedited_series()
    jf = ART / "parity_probes.jsonl"

    def emit(**kw):
        row = {"mode": "parity", "ts": time.time(), **env_row(), **kw}
        t21.jsonl(jf, row)
        probes.append(row)
        print(json.dumps({k: row[k] for k in
                          ("kind", "anchor", "r0", "ok", "note")
                          if k in row}, ensure_ascii=False))
        return row

    # -- 1. production identity: harness loop == full_solve_capture -----
    t0 = time.perf_counter()
    prod = fs.full_solve_capture(cap_set, None, profile=t21.PROFILE,
                                 publish_anchor=False)
    prod_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    ref, boundaries = run_window_with_boundaries(cap_set, series)
    ref_ms = (time.perf_counter() - t0) * 1e3

    mism = []
    for t in range(N_T):
        d = t21.first_output_divergence(ref.outputs[t], prod.outputs[t], t)
        if d is not None:
            mism.append((t, d))
    ci_ok = all(t21.f64_bits(a) == t21.f64_bits(b)
                for a, b in zip(ref.ret_ci_series, prod.ret_CI_series))
    fl_ok = ref.ret_ci_flavors == list(prod.ret_CI_flavors)
    prod_final_boundary = t21.capture_boundary(
        prod.final_state, prod.ret_CI_series[-1],
        prod.ret_CI_flavors[-1], N_T, series)
    state_ok = t21.boundary_bits_equal(
        boundaries[N_T], prod_final_boundary)
    emit(kind="harness_vs_production_identity", ok=not mism and ci_ok
         and fl_ok and state_ok,
         plane_mismatches=mism, ci_series_bits_equal=ci_ok,
         flavors_equal=fl_ok, final_state_bits_equal=state_ok,
         prod_wall_ms=round(prod_ms, 1), harness_wall_ms=round(ref_ms, 1),
         median_step_ms={"prod": round(statistics.median(prod.timings_ms), 1),
                         "harness": round(statistics.median(ref.timings_ms), 1)},
         reference_step_ms=[round(m, 1) for m in ref.timings_ms],
         is_day=[bool(t_in_is_day(t)) for t in range(N_T)])
    if mism or not (ci_ok and fl_ok and state_ok):
        print("FATAL: harness loop is not the production fold", file=sys.stderr)
        return 1
    ref.outputs.clear()  # identity served; boundaries stay live

    # -- 2. independent oracle: restored suffix (unedited) == capture ---
    a = 12
    t0 = time.perf_counter()
    suf = t21.run_window(cap_set, series, boundary=boundaries[a],
                         collect_states=True)
    oracle_ms = (time.perf_counter() - t0) * 1e3
    om = []
    for i, t in enumerate(range(a, N_T)):
        z = np.load(t21.CAP / f"t{t:02d}.npz")
        for name in t21.PLANES:
            key = f"ret_{name}"
            if key not in z.files:
                continue
            if not np.array_equal(t21.bits(suf.outputs[i][name]),
                                  t21.bits(z[key])):
                om.append((t, name))
        # carried state vs the oracle's recorded next-state
        st, _cit, _flav = suf.states[i]
        for name in t21.LOOP_PLANES:
            key = f"ret_{name}"
            if key in z.files and not np.array_equal(
                    t21.bits(getattr(st, name)), t21.bits(z[key])):
                om.append((t, "state:" + name))
        if "ret_timeadd__f64" in z.files and (
                t21.f64_bits(st.timeadd)
                != t21.f64_bits(float(z["ret_timeadd__f64"]))):
            om.append((t, "state:timeadd"))
        if "ret_firstdaytime__int" in z.files and (
                int(st.firstdaytime)
                != int(z["ret_firstdaytime__int"][()])):
            om.append((t, "state:firstdaytime"))
    ci_oracle_ok = True
    for i, t in enumerate(range(a, N_T)):
        z = np.load(t21.CAP / f"t{t:02d}.npz")
        if "ret_CI__f64" in z.files:
            if t21.f64_bits(suf.ret_ci_series[i]) != \
                    t21.f64_bits(float(z["ret_CI__f64"])):
                ci_oracle_ok = False
                om.append((t, "ret_CI__f64"))
    emit(kind="restored_suffix_vs_oracle_capture", anchor=a, r0=None,
         ok=not om, mismatches=om[:20], n_mismatch=len(om),
         state_planes_compared=True,
         suffix_wall_ms=round(oracle_ms, 1))

    # -- 3. restore fidelity grid: (anchor a, edit r0 >= a) -------------
    grid = [(4, 8), (8, 8), (4, 12), (8, 12), (12, 12),
            (8, 16), (12, 16), (16, 16)]
    dirty_refs: dict[int, tuple] = {}
    dirty_final: dict[int, t21.Boundary] = {}
    for _, r0 in sorted(grid, key=lambda g: g[1]):
        if r0 in dirty_refs:
            continue
        es = edit_series(series, r0)
        t0 = time.perf_counter()
        dwin, _ = run_window_with_boundaries(cap_set, es)
        wall = (time.perf_counter() - t0) * 1e3
        # memory diet: only the suffix the grid's anchors probe is needed
        min_a = min(a for a, rr in grid if rr == r0)
        for t in range(0, min_a):
            dwin.outputs[t] = None
        dirty_refs[r0] = (dwin, es)
        dirty_final[r0] = t21.capture_boundary(
            dwin.final_state, dwin.ci_thread, dwin.ci_flavor, N_T, es)
        emit(kind="dirty_full_recompute", r0=r0, ok=True,
             wall_ms=round(wall, 1),
             median_step_ms=round(statistics.median(dwin.timings_ms), 1))

    for a, r0 in grid:
        dwin, es = dirty_refs[r0]
        t0 = time.perf_counter()
        suf = t21.run_window(cap_set, es, boundary=boundaries[a])
        suf_wall = (time.perf_counter() - t0) * 1e3
        mism = []
        for i, t in enumerate(range(a, N_T)):
            d = t21.first_output_divergence(suf.outputs[i], dwin.outputs[t], t)
            if d is not None:
                mism.append((t, d))
                break
        ci_ok = all(t21.f64_bits(x) == t21.f64_bits(y)
                    for x, y in zip(suf.ret_ci_series,
                                    dwin.ret_ci_series[a:]))
        fl_ok = suf.ret_ci_flavors == dwin.ret_ci_flavors[a:]
        suf_final = t21.capture_boundary(
            suf.final_state, suf.ci_thread, suf.ci_flavor, N_T, es)
        state_ok = t21.boundary_bits_equal(suf_final, dirty_final[r0])
        emit(kind="restore_fidelity", anchor=a, r0=r0,
             suffix_steps=N_T - a,
             ok=not mism and ci_ok and fl_ok and state_ok,
             first_mismatch=mism[:1], ci_series_bits_equal=ci_ok,
             flavors_equal=fl_ok, final_state_bits_equal=state_ok,
             suffix_wall_ms=round(suf_wall, 1),
             suffix_median_step_ms=round(
                 statistics.median(suf.timings_ms), 1)
             if suf.timings_ms else None)
        suf.outputs.clear()

    # -- 4. mutation diagnostics (counterexample search) ----------------
    es12, dwin12 = dirty_refs[12][1], dirty_refs[12][0]
    for mut in ("M1_ci_thread_f32_collapse", "M2_flavor_flip",
                "M3_timeadd_drop", "M4_firstdaytime_drop",
                "M5_plane_nan", "M5b_plane_1ulp_normal"):
        b_mut = t21.mutated_boundary(boundaries[12], mut)
        div = t21.mutation_first_divergence(cap_set, es12, b_mut,
                                            dwin12.outputs)
        emit(kind="mutation", mutation=mut, anchor=12, r0=12,
             killed=div is not None,
             first_divergence=div)

    # -- 5. fence behaviour ---------------------------------------------
    es = edit_series(series, 12)
    picked = t21.select_anchor(boundaries, es, 12)
    ok_fence = (picked is not None and picked.next_step == 12)
    cand = thermal.CheckpointCandidate(
        scene_revision=1, next_step=16, thermal=True,
        input_fingerprint={"met_prefix": boundaries[16].met_prefix})
    reason, invalidation = thermal.classify_anchor(
        cand, geometry_fingerprint={"scene": "x"},
        met_prefix_digest=lambda k: t21.met_prefix_digest(es, k), r0=12)
    emit(kind="anchor_selection_fence", r0=12, ok=ok_fence,
         selected_next_step=picked.next_step if picked else None,
         classify_anchor_refusal=(reason, invalidation),
         refusal_kind=invalidation)

    # -- 6. access width table (steps AND measured-ms weighted) ---------
    ref_ms_per_step = [float(m) for m in ref.timings_ms]
    anchor_sets = {
        "k0_production_today": [],           # final anchor only (a=24: r0<24 never served)
        "k1": [12],
        "k2": [8, 16],
        "k3": [6, 12, 18],
        "k4": [4, 8, 12, 16, 20],
        "k6": [3, 6, 9, 12, 15, 18, 21],
        "k11": list(range(2, 24, 2)),
        "k23_per_timestep": list(range(1, 24)),
    }
    for label, anchors in anchor_sets.items():
        widths = []
        ms_widths = []
        for r0 in range(1, N_T):
            best = 0
            for a in anchors:
                if a <= r0 and a > best:
                    best = a
            widths.append(N_T - best)
            if ref_ms_per_step is not None:
                ms_widths.append(sum(ref_ms_per_step[best:]))
        row = dict(kind="k_sweep_access_width", label=label,
                   k_anchors=len(anchors), anchor_steps=anchors,
                   mean_refold_steps=round(statistics.mean(widths), 2),
                   max_refold_steps=max(widths),
                   min_refold_steps=min(widths),
                   worst_case_edit_r0=int(np.argmax(widths)) + 1)
        if ms_widths:
            total = sum(ref_ms_per_step)
            row.update(
                fold_ms_full_day=round(total, 1),
                mean_refold_ms=round(statistics.mean(ms_widths), 1),
                max_refold_ms=round(max(ms_widths), 1),
                min_refold_ms=round(min(ms_widths), 1),
                mean_fold_fraction=round(statistics.mean(ms_widths) / total, 4))
        emit(**row)

    ok_all = all(p.get("ok", True) for p in probes
                 if p.get("kind") in ("harness_vs_production_identity",
                                      "restored_suffix_vs_oracle_capture",
                                      "restore_fidelity",
                                      "anchor_selection_fence"))
    print(f"\nPARITY {'PASS' if ok_all else 'FAIL'} — "
          f"{len(probes)} rows -> {jf}")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def run_memory(probes: list[dict]) -> int:
    cap_set = t21.cap_set()
    series = t21.unedited_series()
    jf = ART / "memory_ledger.jsonl"

    def emit(**kw):
        row = {"mode": "memory", "ts": time.time(), **env_row(), **kw}
        t21.jsonl(jf, row)
        probes.append(row)
        print(json.dumps({k: row[k] for k in
                          ("kind", "k_anchors", "rss_kib", "note")
                          if k in row}, ensure_ascii=False))

    emit(kind="payload_math", boundary_payload_bytes=t21.BOUNDARY_PAYLOAD_BYTES,
         boundary_payload_mib=round(t21.BOUNDARY_PAYLOAD_BYTES / 2**20, 3),
         plane_bytes=t21.PLANE_BYTES, per_timestep_mib=round(
             t21.BOUNDARY_PAYLOAD_BYTES / 2**20, 3),
         k24_mib=round(24 * t21.BOUNDARY_PAYLOAD_BYTES / 2**20, 1),
         target_steady_rss_mib=256,
         campaign_steady_rss_mib_pre_t21=286.9)

    rss0 = t21.rss_kib()
    emit(kind="rss_imports", rss_kib=rss0)
    t0 = time.perf_counter()
    _, boundaries = run_window_with_boundaries(cap_set, series)
    wall = (time.perf_counter() - t0) * 1e3
    rss1 = t21.rss_kib()
    emit(kind="rss_after_reference_run", rss_kib=rss1,
         reference_wall_ms=round(wall, 1),
         n_boundaries=len(boundaries))

    b12 = boundaries[12]
    # capture / restore cost (median of 20)
    cap_ms, res_ms = [], []
    for _ in range(20):
        t0 = time.perf_counter()
        bb = t21.capture_boundary(
            t21.rad_state_from_capture(t21.CAP, 0, rows=500, cols=500),
            1.0, False, 12, series)
        cap_ms.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        t21.restore_boundary(bb)
        res_ms.append((time.perf_counter() - t0) * 1e3)
    emit(kind="capture_restore_cost",
         capture_ms_median=round(statistics.median(cap_ms), 3),
         restore_ms_median=round(statistics.median(res_ms), 3))

    held = [t21.capture_boundary(
        t21.rad_state_from_capture(t21.CAP, 0, rows=500, cols=500),
        1.0, False, 12, series)]
    del bb
    for k in (2, 4, 6, 8, 12, 24):
        while len(held) < k:
            held.append(t21.Boundary(
                next_step=held[-1].next_step,
                firstdaytime=held[-1].firstdaytime,
                timeadd=held[-1].timeadd, CI=held[-1].CI,
                ci_thread=held[-1].ci_thread,
                ci_flavor=held[-1].ci_flavor,
                met_prefix=held[-1].met_prefix,
                planes={n: held[-1].planes[n].copy()
                        for n in t21.LOOP_PLANES}))
        rss_k = t21.rss_kib()
        emit(kind="rss_with_k_boundaries", k_anchors=k, rss_kib=rss_k,
             delta_vs_ref_kib=rss_k - rss1,
             measured_bytes_per_anchor=round(
                 (rss_k - rss1) * 1024 / len(held), 0),
             payload_bytes_total=len(held) * t21.BOUNDARY_PAYLOAD_BYTES)

    # steady-RSS projection vs the 256 MiB target
    for k in (2, 4, 6, 8, 12, 24):
        add_mib = k * t21.BOUNDARY_PAYLOAD_BYTES / 2**20
        emit(kind="steady_rss_projection", k_anchors=k,
             added_mib=round(add_mib, 1),
             projected_steady_mib=round(286.9 + add_mib, 1),
             breaches_256_target=bool(286.9 + add_mib > 256))
    print(f"\nMEMORY rows={len(probes)} -> {jf}")
    return 0


# ---------------------------------------------------------------------------
# timing (window-gated)
# ---------------------------------------------------------------------------


def run_timing(probes: list[dict], rounds: int) -> int:
    import os as _os
    cap_set = t21.cap_set()
    series = t21.unedited_series()
    jf = ART / "timing_paired.jsonl"

    def emit(**kw):
        row = {"mode": "timing", "ts": time.time(), **env_row(), **kw}
        t21.jsonl(jf, row)
        probes.append(row)
        print(json.dumps({k: row[k] for k in
                          ("kind", "anchor", "r0", "side", "wall_ms",
                           "load1") if k in row}, ensure_ascii=False))

    def load() -> float:
        return _os.getloadavg()[0]

    # reference boundaries (one pass) + dirty fulls cached per r0
    print("reference pass...")
    _, boundaries = run_window_with_boundaries(cap_set, series)
    pairs = [(4, 8), (8, 8), (8, 12), (12, 12), (12, 16), (16, 16)]
    es_cache: dict[int, list] = {}
    for _, r0 in pairs:
        if r0 not in es_cache:
            es_cache[r0] = edit_series(series, r0)

    rng = random.Random(0xC0FFEE)
    for a, r0 in pairs:
        es = es_cache[r0]
        samples = {"full": [], "suffix": []}

        def one(side):
            t0 = time.perf_counter()
            if side == "full":
                win, _ = run_window_with_boundaries(cap_set, es)
                win.outputs.clear()
                del win
            else:
                win = t21.run_window(cap_set, es, boundary=boundaries[a])
                win.outputs.clear()
                del win
            return (time.perf_counter() - t0) * 1e3

        # warmup x1 each side, then interleaved samples with randomized order
        one("full")
        one("suffix")
        for i in range(rounds):
            order = ("full", "suffix") if rng.random() < 0.5 else ("suffix", "full")
            for side in order:
                if load() > LOAD_GUARD:
                    emit(kind="load_guard_breach", anchor=a, r0=r0,
                         side=side, load1=round(load(), 2), ok=False)
                    return 2
                ms = one(side)
                samples[side].append(ms)
                emit(kind="paired_sample", anchor=a, r0=r0, side=side,
                     wall_ms=round(ms, 1), sample=i, load1=round(load(), 2),
                     suffix_steps=N_T - a)
        emit(kind="paired_summary", anchor=a, r0=r0,
             suffix_steps=N_T - a,
             full_ms_median=round(statistics.median(samples["full"]), 1),
             suffix_ms_median=round(statistics.median(samples["suffix"]), 1),
             ratio=round(statistics.median(samples["suffix"])
                         / statistics.median(samples["full"]), 4),
             rounds=rounds,
             access_width=round((N_T - a) / N_T, 4))
    print(f"\nTIMING rows={len(probes)} -> {jf}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parity", "memory", "timing"])
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    probes: list[dict] = []
    if not t21.capture_present():
        print("t08 capture bundle absent", file=sys.stderr)
        return 3
    if args.mode == "parity":
        return run_parity(probes)
    if args.mode == "memory":
        return run_memory(probes)
    return run_timing(probes, args.rounds)


if __name__ == "__main__":
    raise SystemExit(main())
