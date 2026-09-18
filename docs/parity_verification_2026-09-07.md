# Optimization parity verification — vanilla SOLWEIG-GPU vs this fork (2026-09-07)

Empirical verification that this repository's optimized tree produces **identical physics
outputs** to the vanilla upstream it forked from, at equal full-solve performance, plus the
incremental-recompute speedups the fork exists for. Run on an isolated GPU server
(`cura` / `analytica`) in a single scratch directory; every input byte-identical between
the two trees.

- **Vanilla**: upstream `nvnsudharsan/SOLWEIG-GPU` @ `4131775a`, unmodified.
- **Optimized**: this repository @ `579008e` (`main`, 2026-09-07).

## Environment

| Item | Value |
|---|---|
| Host | `analytica` (Georgia Tech PACE `cura`), 72 cores, 4× NVIDIA RTX A6000 48 GB |
| GPU runs | `CUDA_VISIBLE_DEVICES=2` (single GPU pinned — GPUs 0/1 carried another user's jobs; GPU 2 is the same-model A6000) |
| CPU runs | `CUDA_VISIBLE_DEVICES=""` + `taskset -c 0,1` (2-core) / `taskset -c 0-3` (4-core) + `OMP_NUM_THREADS`/`MKL_NUM_THREADS` matching the pinning |
| torch | 2.14.0+cu126 (both venvs); GDAL/osgeo 3.8.4 (system, `--system-site-packages`) |
| Isolation | `~/solweig_parity_20260907/` — separate checkouts (`vanilla/`, `optimized/`), separate venvs, separate output trees (`runs/`) |

## Method

1. **Same inputs**: `processed_inputs` copied per tree; input rasters verified
   sha256-identical before any run.
2. **Examples**: EX1 — `Input_subset` site, single 600 m tile; EX2F — the same site
   tiled 200 m / overlap 20 (9 tiles). (An initial tiled pair labeled V-EX2/O-EX2 reused
   the shipped 600-tile `processed_inputs` cache at `tile_size=200`: vanilla crashed —
   `RuntimeError` at `solweig_gpu/utci_process.py:635`, loading a cached 500×500 veg
   shade matrix into a 220×220 tile — while the optimized tree validated the stale SVF
   cache, rebuilt the one stale tile entry, and completed, its outputs bitwise-equal to
   fresh-run vanilla on every common file. The verdict tables use the fresh-cache
   V-EX2F/O-EX2F pair; the divergence is a robustness fix in this fork, not a parity
   break.)
3. **Self-consistency first**: vanilla EX1 run twice on GPU to derive the determinism
   noise floor. Result: **0.0** — every band sha256-identical across repeats.
4. **Comparison**: per output file, sha256 then band-wise bitwise compare (`NaN == NaN`
   counts equal), with max/mean abs-delta statistics as fallback detail.
5. **Incremental benchmarks**: the optimized tree's incremental machinery driven
   directly (site cache + `PlanExecutor` via `solweig_bench_incremental.py`, warm cache,
   500×500 grid / 24 time steps): five concrete edits — E1 add tree, E2 move it, E3
   height replace, E4 add a second tree 610 m away, E5 time-prefix recompute — each
   timed as an incremental (windowed) solve AND as a full recompute of the same scene
   state, on three CPU configurations. (The incremental harness initializes torch with
   `cuda=False`: windowed solves are CPU-torch paths, which is also how the production
   demo server runs. GPU incremental timings are therefore not claimed — in fact the
   incremental solver pins `torch.device("cpu")` and its veg-occlusion prep raised a
   mixed-device error on a CUDA-visible host (`shadow.py:271`, reproduced twice during
   bring-up of this bench): the exact lane is CPU-only today, recorded here as a
   portability finding.)

## Result integrity — verdict per configuration

Vanilla vs optimized, same machine, same configuration, same inputs:

| Configuration | Example | Files compared | Verdict |
|---|---|---|---|
| GPU (A6000) | EX1 | 3 (72 bands) | **bitwise identical** (sha256 equal, all bands) |
| GPU (A6000) | EX2F | 27 (648 bands) | **bitwise identical** |
| GPU (A6000) | SVF cache artifacts | 9 tif + 9 npz + 9 zip | **bitwise identical** (zip member-wise) |
| CPU 4-core | EX1 | 12 | **bitwise identical** |
| CPU 4-core | EX2F | 84 | **bitwise identical** |
| CPU 2-core | EX1 | 12 | **bitwise identical** |
| CPU 2-core | EX2F | 84 | **bitwise identical** |

**Verdict: the optimized tree is physics-identical to vanilla on every tested
configuration — GPU, 4-core CPU, and 2-core CPU — for both the single-tile and the
tiled multi-tile path, including the SVF site-cache artifacts.**

File counts are per configuration: the GPU rows count the three output rasters per tile
(Shadow/TMRT/UTCI), while the CPU streams additionally compared side artifacts emitted
alongside, so counts are not comparable across devices.

Cross-device disclosure (informational, not a verdict): GPU outputs vs CPU outputs of the
*same* tree differ in TMRT/UTCI at floating-point-path scale — max abs delta TMRT
0.368 °C / UTCI 0.091 °C, mean ≈ 2×10⁻⁴ °C — and they do so for **vanilla equally**,
so this is device arithmetic, not an optimization artifact. Shadow rasters are bitwise
identical even cross-device. CPU outputs are themselves bitwise-identical across thread
counts (cpu4 vs cpu2, same commit: 12/12 EX1 files equal).

## Performance — full solve (`thermal_comfort` wall seconds)

| Configuration | EX1 vanilla | EX1 optimized | EX2F vanilla | EX2F optimized |
|---|---|---|---|---|
| GPU (A6000) | 64.31 (repeat run: 57.62) | 59.59 | 168.11 | 164.19 |
| CPU 4-core | 90.32 | 83.55 | 183.86 | 185.81 |
| CPU 2-core | 126.53 | 114.09 | 192.84 | 196.29 |

Reading: the vanilla GPU EX1 repeat (64.31 → 57.62, ~11 % spread) bounds the run-to-run
noise on this host; every vanilla-vs-optimized gap above sits inside it. **The optimized
tree keeps full-solve performance at parity** — the fork's changes add incremental
machinery without taxing the full path.

## Performance — incremental recompute (optimized tree only)

Per-edit wall seconds, incremental (windowed) solve vs full recompute of the same scene
state; speedup in parentheses. Dirty fraction = share of the grid the edit's window
touched. CPU configurations as pinned above; "CPU default" = unpinned host threads
(cycle-stealing background, indicative only).

| Edit (dirty fraction) | CPU default inc / full | CPU 4-core inc / full | CPU 2-core inc / full |
|---|---|---|---|
| E1 add tree (0.187) | 23.5 / 54.4 (2.3×) | 20.4 / 65.0 (3.2×) | 27.2 / 112.1 (4.1×) |
| E2 move tree (0.200) | 25.4 / 55.3 (2.2×) | 21.0 / 64.6 (3.1×) | 28.0 / 110.9 (4.0×) |
| E3 height replace (0.296) | 24.6 / 54.3 (2.2×) | 21.0 / 65.1 (3.1×) | 30.1 / 111.8 (3.7×) |
| E4 add second tree 610 m away (0.100) | 16.6 / 53.7 (3.2×) | 14.0 / 64.6 (4.6×) | 16.0 / 111.5 (7.0×) |
| E5 time-prefix recompute (0.246) | 26.1 / 53.4 (2.0×) | — | — |

Reading: a single-tree edit solves in **2–7×** the speed of recomputing the scene, and
the speedup grows as cores shrink (the full solve loses more to thread parallelism than
the windowed solve does) — the regime the CPU-only production server lives in.

## Reproduction

All scripts live in the (now removed) scratch directory `~/solweig_parity_20260907/`:
`run_one.sh` / `run_one_cpu.sh` (single full solve), `solweig_parity_cpu_stream.sh`
(config matrix), `solweig_bench_incremental.py` (incremental benchmarks),
`compare.py` / `compare_all.py` / `compare_cpu_matrix.sh` (bitwise comparison),
`setup_env.sh`, `fix_torch*.sh`. Logs under `logs/`, outputs under `runs/`.
The scratch tree was transferred as a git bundle of `main` @ `579008e` and verified
against the pushed SHA before any run.

## Cleanup

`cleanup.sh` in the scratch root (hostname-guarded to `analytica`) removes the entire
`~/solweig_parity_20260907/` directory and nothing else. **Executed after this report
was committed** — see the worklog entry for this date. Site data was never committed to
any repository.
