# Computation-first optimization roadmap

## Starting evidence

Treat the 2026-09-04 performance handoff as measured evidence that must be reproduced, not as unquestionable truth. Its current reference for a representative native CPU geometry edit is approximately 37.5-39 seconds: about 20 seconds in vegetation SVF replay and 17 seconds in the selected-time causal-prefix loop. The read window is approximately 99% of the tile, so shrinking the read radius or raising the 6-degree sky floor is not an acceptable latency shortcut.

The roadmap is ordered by expected leverage and scientific risk.

## R0 — independent reproduction and telemetry

Parallel agents independently reproduce:

- end-to-end native job;
- SVF and time-loop phase timing;
- read/write fractions;
- exact hashes/parity fixtures;
- behavior under concurrent host load;
- current supersession and queue behavior.

Add histograms for operation ACK, epoch wait, reduction, planning, fast compute, publication, exact queue delay, exact compute, correction, RSS, host load, and analysis class.

Exit: a fresh report confirms or corrects every value used for planning.

## R1 — collaborative operation plane

Implement append-only idempotent operations, deterministic source-specific conflict reduction, workspace revisioning, 100 ms epochs, collaboration broadcast, admission accounting, and replay tests. This phase makes all accepted edits visible within the state SLO without changing scientific math.

Exit: multi-session state p99 and zero-loss/replay gates pass.

## R2 — split fast and exact scheduling

Replace one-job-per-edit behavior with epoch-final dependency plans. Add deadline-ordered fast queue, reserved CPU budget, latest-state exact target compaction, workspace fairness, and stale publication fences.

Exit: exact 38-second work cannot block 1-second canonical/fast publication.

## R3 — cheap exact fast paths

Implement and benchmark:

- view/cache selection;
- receptor/UTCI-only fused kernel;
- selected-hour meteorological updates reusing geometry/SVF;
- precomputed solar and invariant coefficients;
- sparse output patching.

Exit: eligible source families publish `fast_exact` inside deadline.

## R4 — persistent geometry delta visibility/SVF

This is the main scientific performance wave.

- Design object-to-ray reverse index.
- Persist winning/ordered occluder state needed for add, move, resize, and delete.
- Update patch/SVF contribution deltas without a complete 153-patch replay.
- Support heterogeneous building and vegetation edits.
- Validate overlap, transmissivity, low sun, boundaries, and removal.
- Compare bitwise where operation order is preserved; otherwise establish a reviewed numerical tolerance without changing physics.

Exit: repeat geometry edits show a major SVF reduction and pass independent full-domain differential tests.

## R5 — temporal checkpoint and sparse replay

- Identify exact causal state by output and source family.
- Persist copy-on-write state checkpoints.
- Recompute from earliest dirty time and spatial chunks.
- Validate land-cover, meteorology, and geometry history effects.

Exit: time loop no longer evaluates unaffected predecessor states/cells.

## R6 — kernel fusion and compiled hot path

Only after profiling R4/R5:

- fuse shadow march elementwise operations;
- hoist slices/indices and reuse scratch;
- remove Torch/NumPy transitions;
- evaluate Numba, C++/OpenMP, or platform vectorization;
- benchmark thread counts under reserved fast/exact CPU partitions.

Every candidate is A/B tested against the same oracle and fixture.

## R7 — qualified compensated fast models

Build response operators for remaining geometry/surface cases. Train/calibrate only from exact outputs, including cross-family epochs. Add out-of-domain detection and exact-base drift limits.

Exit: admitted fast updates meet one-second p99 with explicit qualification.

## R8 — load, chaos, and commissioning

Run multi-session bursts, exact backlog, process restarts, duplicate/reordered operations, slow clients, admission saturation, host contention, and one-hour classroom soak. Determine the deployment admission envelope and publish achieved SLOs.

## Optimization refusals retained

Without new scientific evidence, do not use:

- shorter read reach or a higher sky-patch floor;
- reordered floating accumulation claimed as bitwise identical;
- silent output precision reduction;
- a surrogate with no out-of-domain and reconciliation mechanism;
- a FIFO exact job for every raw interaction event;
- container performance numbers as the native CPU target.
