#!/bin/zsh
# SPDX-License-Identifier: GPL-3.0-only
# T16 CPU-lane final workload matrix — the exact reproduction commands.
#
# Sampling per docs/incremental_design_tool/ultrafast_bitwise/acceptance_targets.yaml:
#   warmup_min 5 everywhere;
#   cpu-numba-selected-exact: 30 samples (single run ~30-60 s, at/above the
#     ~60 s single-run threshold -> paired_expensive regime >= 30) + 10
#     randomized-order A/B pairs (full-day job vs selected-time job);
#   cpu-numba-full-day: 100 warm in-process samples + 100 cold subprocess
#     samples (single run ~15 s < 60 s -> fast_candidate regime, 100);
#   cpu-numba-full-recompute-warm: 100 samples (single run ~14 s < 60 s).
#
# All commands run serially (timing contamination otherwise). Artifacts
# land under benchmarks/ultrafast/bench/ (worktree) AND mirrored to
# /Users/alansynn/Workspace/solweig_ultrafast_artifacts/t16/cpu/.
set -euo pipefail

PY=/Users/alansynn/Workspace/solweig/.venv/bin/python
ROOT=/Users/alansynn/Workspace/solweig/.claude/worktrees/agent-t16cpu
ART=/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t16/cpu
LOGDIR="$ART/logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

echo "[matrix] $(date -u +%FT%TZ) selected-exact start" | tee -a "$LOGDIR/matrix.log"
$PY benchmarks/ultrafast/run.py bench \
    --variant cpu-numba-selected-exact --suite representative \
    --warmup 5 --repeats 30 --pairs 10 --seed 20260908 \
    > "$LOGDIR/bench_selected_exact.log" 2>&1

echo "[matrix] $(date -u +%FT%TZ) full-day start" | tee -a "$LOGDIR/matrix.log"
$PY benchmarks/ultrafast/run.py bench \
    --variant cpu-numba-full-day --suite representative \
    --warmup 5 --repeats 100 --cold-repeats 100 \
    > "$LOGDIR/bench_full_day.log" 2>&1

echo "[matrix] $(date -u +%FT%TZ) full-recompute-warm start" | tee -a "$LOGDIR/matrix.log"
$PY benchmarks/ultrafast/run.py bench \
    --variant cpu-numba-full-recompute-warm --suite representative \
    --warmup 5 --repeats 100 \
    > "$LOGDIR/bench_recompute_warm.log" 2>&1

echo "[matrix] $(date -u +%FT%TZ) memory start" | tee -a "$LOGDIR/matrix.log"
$PY benchmarks/ultrafast/t16_memory.py --artifacts-dir "$ART" \
    > "$LOGDIR/memory.log" 2>&1

echo "[matrix] $(date -u +%FT%TZ) startup start" | tee -a "$LOGDIR/matrix.log"
$PY benchmarks/ultrafast/t16_startup.py --artifacts-dir "$ART" --runs 6 \
    > "$LOGDIR/startup.log" 2>&1

echo "[matrix] $(date -u +%FT%TZ) DONE" | tee -a "$LOGDIR/matrix.log"
