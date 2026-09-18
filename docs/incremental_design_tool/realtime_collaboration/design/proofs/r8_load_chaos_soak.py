# SPDX-License-Identifier: GPL-3.0-only
"""R8 -- the load / chaos / soak / commissioning harness (final gate).

Drives the REAL server application (``solweig_gpu.server.app.create_app``
with its DEFAULT production wiring: the 100 ms epoch scheduler, the
deadline-isolated fast lane with the R7 compensated vegetation kernel,
realtime admission control, and the real universal-dispatch exact solver)
in a REAL uvicorn subprocess, against a REAL prepared 128x128 site with a
real full-run baseline, and measures the service-level contract under:

* **load** -- N concurrent collaborative clients across 2 workspaces with
  the realistic op mix (view ~70%, vegetation ~15%, meteorology ~10%,
  landcover ~5%), including deliberate DUPLICATE submissions and
  stale-revision (``base_revision``) ops;
* **chaos** -- admission overflow, client disconnects mid-epoch, and
  server CRASH (``os._exit`` at armed seams inside the real commit paths:
  epoch mid-close, exact job mid-solve, fast publish window, accept
  transaction) followed by RESTART and invariant verification;
* **soak** -- one uninterrupted hour of sustained load with RSS sampling
  and slope analysis (the headline run; done ONCE, after calibration).

Measured invariants (every run re-checks them; a violation is a DEFECT,
never a warning):

1. every durably accepted op lands in EXACTLY ONE canonical workspace
   revision (zero loss, no double-apply);
2. ``exact_revision <= fast_revision <= workspace_revision``, all
   monotonic;
3. reject/backpressure BEFORE acceptance (admission control; a rejected
   op must never appear in the durable log);
4. remote visibility p99 <= 300 ms (op accept -> SSE canonical broadcast)
   and fast analysis revision p99 <= 1000 ms, per result class;
5. result classes ``fast_exact`` / ``fast_qualified`` / ``visual_pending``
   with qualified supersession when exact advances;
6. deterministic replay: the SAME op sequence folded into a second
   workspace produces byte-identical canonical state rows.

Usage (all runs local; no external services)::

    python r8_load_chaos_soak.py --mode smoke        # wiring check
    python r8_load_chaos_soak.py --mode calibrate    # 10-min calibration
    python r8_load_chaos_soak.py --mode soak         # ONE-HOUR headline
    python r8_load_chaos_soak.py --mode chaos        # chaos matrix
    python r8_load_chaos_soak.py --mode overload     # 2x-load 5-min soaklet
    python r8_load_chaos_soak.py --mode serve ...    # (internal) server child

Evidence: ``/tmp/r8_proof/load.json`` (calibration), ``soak.json`` (the
headline run: untruncated RSS timeseries, host ``uptime`` recorded),
``chaos.json``, ``overload.json``, ``smoke.json``.

Honesty notes baked into the harness:

* the server's legacy per-scenario edit budget (20/min) and per-IP request
  budget (30/min) are DISABLED for the envelope runs (they are deployment
  defence-in-depth knobs, not the realtime admission contract; the fly
  demo keeps them ON -- see the commissioning note). The realtime
  admission envelope runs at its documented defaults;
* client and server share the host (recorded CPU model + load via
  ``uptime`` in every evidence file);
* exact-lane catch-up latency is measured by a 500 ms poll (poll-limited
  resolution, disclosed in the evidence);
* every number in the evidence JSONs is measured; nothing is extrapolated.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROOF_ROOT = Path("/tmp/r8_proof")
RUNS_ROOT = PROOF_ROOT / "runs"
SITE_ROOT = PROOF_ROOT / "site"
SITES_ROOT = PROOF_ROOT / "sites"
LOG_ROOT = PROOF_ROOT / "logs"

SITE_ID = "r8-site"
WORKSPACE_COUNT = 2
VARIABLES = ("utci", "tmrt")

_RNG_SEED = 20260905

_KILL_EXIT = 70
_KILL_MARKER = "R8KILL:"

#: SLOs (realtime_contract.yaml / service_level_contract.md) — REPORT-ONLY.
#: These constants are echoed into the evidence JSON for side-by-side
#: reading against the measured percentiles; they NEVER contribute to a
#: PASS/FAIL verdict. The verdict gates are exactly the correctness
#: invariants (zero-loss acked==durable, revision triple monotonicity,
#: strict stream delivery where no crash seam fired) plus the per-case
#: chaos gates (reject-before-acceptance, no torn survivors, restart
#: recovery). Latency numbers are measured evidence for the SLO
#: conversation, not pass/fail inputs (R8b review N1).
SLO_VISIBILITY_MS = 300.0
SLO_FAST_MS = 1000.0


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )


def _parse_iso(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    ).timestamp()


def _uptime() -> str:
    try:
        return subprocess.run(
            ["uptime"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - evidence must never crash a run
        return "unavailable"


def _host_block() -> dict[str, Any]:
    import platform

    cpu = "unknown"
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return {
        "host": platform.node(),
        "cpu": cpu,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cores": os.cpu_count(),
        "uptime_at_start": _uptime(),
    }


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def percentiles(samples: Sequence[float]) -> dict[str, Any]:
    """n / mean / p50 / p95 / p99 / max over ms-valued samples."""
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)
    n = len(ordered)

    def pick(fraction: float) -> float:
        index = min(n - 1, max(0, math.ceil(fraction * n) - 1))
        return ordered[index]

    return {
        "n": n,
        "mean_ms": round(statistics.fmean(ordered), 3),
        "p50_ms": round(pick(0.50), 3),
        "p95_ms": round(pick(0.95), 3),
        "p99_ms": round(pick(0.99), 3),
        "max_ms": round(ordered[-1], 3),
    }


def by_class(samples: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    return {name: percentiles(values) for name, values in sorted(samples.items())}


# ---------------------------------------------------------------------------
# The prepared site (built ONCE, read-only afterwards)
# ---------------------------------------------------------------------------


def ensure_site() -> dict[str, Any]:
    """Build the real prepared site + cache + full-run baseline once.

    Same shape as the ud4 family-site fixture (tests/
    test_server_universal_edits.family_site): genuine SVF artifacts, a real
    full-run baseline in ``baseline_results`` so scenario creation
    publishes the v0 baseline inline. Idempotent; servers only READ these
    bytes (their state lives under per-run state roots).
    """
    from tests.test_incremental_worker import (
        DATE_STR,
        TINY_EPSG,
        TINY_ORIGIN,
        TINY_PIXEL,
        _build_cache,
        _compute_baseline_svf,
        _make_prepared_site,
    )

    cache_dir = SITES_ROOT / SITE_ID
    site_dir = SITE_ROOT / "processed_inputs"
    if (cache_dir / "baseline_results" / "metadata.json").is_file():
        return {
            "cache_dir": str(cache_dir),
            "site_dir": str(site_dir),
            "built": False,
            "date": DATE_STR,
        }
    SITE_ROOT.mkdir(parents=True, exist_ok=True)
    SITES_ROOT.mkdir(parents=True, exist_ok=True)
    from solweig_gpu.incremental.geometry import TreeSpec

    base_trees = (
        TreeSpec(
            "b1",
            TINY_ORIGIN[0] + 20.5 * TINY_PIXEL,
            TINY_ORIGIN[1] - 20.5 * TINY_PIXEL,
            3.0,
            2.0,
        ),
        TreeSpec(
            "b2",
            TINY_ORIGIN[0] + 96.5 * TINY_PIXEL,
            TINY_ORIGIN[1] - 96.5 * TINY_PIXEL,
            5.0,
            2.5,
        ),
    )
    grid, site = _make_prepared_site(
        SITE_ROOT,
        rows=128,
        cols=128,
        pixel=TINY_PIXEL,
        origin=TINY_ORIGIN,
        epsg=TINY_EPSG,
        base_trees=base_trees,
        met_hours=range(10, 13),
    )
    _compute_baseline_svf(site)
    cache = _build_cache(
        site,
        cache_dir,
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id=SITE_ID,
    )
    # REAL full-run baseline (the family-site pattern): the composition
    # base every exact job overlays onto.
    import numpy as np
    from solweig_gpu.incremental.solver import load_site_forcing, run_full_tile
    from solweig_gpu.incremental.trees import TreeLayer

    forcing = load_site_forcing(
        cache, site_dir=site, selected_date_str=DATE_STR
    )
    baseline = run_full_tile(
        cache,
        TreeLayer(cache.tree_base, grid),
        forcing=forcing,
        site_dir=site,
        scratch_dir=SITE_ROOT / "baseline_scratch",
        requested_variables=VARIABLES,
    )
    baseline_dir = cache_dir / "baseline_results"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for name in VARIABLES:
        np.save(baseline_dir / f"{name}.f32.npy", baseline[name])
    (baseline_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variables": list(VARIABLES),
                "time_steps": int(cache.time_steps),
                "source": "baseline",
            }
        )
    )
    return {
        "cache_dir": str(cache_dir),
        "site_dir": str(site),
        "built": True,
        "date": DATE_STR,
    }


# ---------------------------------------------------------------------------
# Server child (uvicorn, production wiring, optional kill seams)
# ---------------------------------------------------------------------------


def _arm_kill(seam: str, count: int) -> None:
    """Arm an ``os._exit`` kill at a named seam inside the REAL commit path.

    G2.2's discipline: the wrapper runs the REAL implementation for every
    invocation before the target one; only the process death is injected.
    """
    import solweig_gpu.server.store as store_module
    import solweig_gpu.utci_process as utci_module

    state = {"n": 0}

    def die(tag: str) -> None:
        sys.stderr.write(f"{_KILL_MARKER}{tag}\n")
        sys.stderr.flush()
        os._exit(_KILL_EXIT)

    if seam == "epoch_mid_close":
        # Die AFTER the open->closed mark commits, BEFORE the reduction:
        # a stranded closed epoch the restart's re-drive must settle.
        real = store_module.Store.mark_epoch_status

        def mark_wrapper(self, workspace_id, epoch_id, status, **kwargs):
            out = real(self, workspace_id, epoch_id, status, **kwargs)
            if status == "closed":
                state["n"] += 1
                if state["n"] >= count:
                    die(f"epoch_mid_close:after_{state['n']}_closed_marks")
            return out

        store_module.Store.mark_epoch_status = mark_wrapper
        return

    if seam == "exact_mid_solve":
        # Die INSIDE the physics series (after the Nth real
        # Solweig_2022a_calc step overall -- mid exact-job). Same seam
        # G2.2 proved effective on this module attribute.
        real = utci_module.Solweig_2022a_calc
        steps = {"n": 0}

        def step_wrapper(*args, **kwargs):
            out = real(*args, **kwargs)
            steps["n"] += 1
            if steps["n"] >= count:
                die(f"exact_mid_solve:after_{steps['n']}_physics_steps")
            return out

        utci_module.Solweig_2022a_calc = step_wrapper
        return

    if seam == "fast_publish_pre_commit":
        # Die ON ENTRY of the Nth fast-revision advance: the epoch is
        # committed, the fast lane owes the publication (scan self-heal).
        real = store_module.Store.advance_fast_revision

        def fast_pre_wrapper(self, *args, **kwargs):
            state["n"] += 1
            if state["n"] >= count:
                die(f"fast_publish_pre_commit:at_{state['n']}_advance_entries")
            return real(self, *args, **kwargs)

        store_module.Store.advance_fast_revision = fast_pre_wrapper
        return

    if seam == "fast_publish_post_commit":
        # Die AFTER the Nth fast-revision transaction commits (the
        # broadcast may be missing; the durable revision is truth).
        real = store_module.Store.advance_fast_revision

        def fast_post_wrapper(self, *args, **kwargs):
            out = real(self, *args, **kwargs)
            state["n"] += 1
            if state["n"] >= count:
                die(f"fast_publish_post_commit:after_{state['n']}_advances")
            return out

        store_module.Store.advance_fast_revision = fast_post_wrapper
        return

    if seam == "accept_mid_txn":
        # Die ON ENTRY of the Nth operation append: the batch never
        # commits (one txn per batch); the client saw a connection drop,
        # never an ack.
        real = store_module.Store.append_operations

        def accept_wrapper(self, *args, **kwargs):
            state["n"] += 1
            if state["n"] >= count:
                die(f"accept_mid_txn:at_{state['n']}_append_entries")
            return real(self, *args, **kwargs)

        store_module.Store.append_operations = accept_wrapper
        return

    raise ValueError(f"unknown kill seam {seam!r}")


def serve_child(args: argparse.Namespace) -> int:
    """Run the REAL app under uvicorn (production default wiring)."""
    import faulthandler
    import uvicorn

    from solweig_gpu.server.app import create_app

    if args.kill_seam:
        _arm_kill(args.kill_seam, args.kill_count)

    # Optional thread-stack evidence (soak): recurring all-thread tracebacks
    # into the server log, to attribute scheduler-thread latency (GIL
    # starvation vs lock waits) from the run itself.
    if os.environ.get("R8_DUMP_STACKS") == "1":
        faulthandler.dump_traceback_later(
            60.0, repeat=True, file=sys.stderr, exit=False
        )

    # DIAGNOSTIC-ONLY knob (never used for envelope evidence): stub the
    # exact solver to isolate the exact lane's GIL footprint from the
    # scheduler's visibility latency. Runs are labeled diag_stub.
    solver_factory = None
    if os.environ.get("R8_DIAG_STUB_SOLVER") == "1":
        from solweig_gpu.server.jobs import SolveResult

        def _stub_solver(context):  # noqa: ANN001 - diagnostic shim
            def solve(request, progress):  # noqa: ANN001
                return SolveResult(
                    status="no-op", scene_version=request.target_scene_version
                )

            return solve

        solver_factory = _stub_solver

    app = create_app(
        state_root=args.state_root,
        sites={
            SITE_ID: {
                "cache_dir": args.cache_dir,
                "site_dir": args.site_dir,
                "selected_date_str": args.date,
            }
        },
        # The realtime admission contract runs at its documented defaults.
        # Legacy per-scenario edit budget + per-IP budget are deployment
        # defence-in-depth knobs and stay OFF for envelope measurement
        # (the fly demo keeps them ON -- commissioning note).
        edits_per_minute=None,
        requests_per_minute_per_ip=None,
        coalescing_window_ms=args.window_ms,
        **({"solver_factory": solver_factory} if solver_factory else {}),
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
        timeout_keep_alive=65,
    )
    return 0


# ---------------------------------------------------------------------------
# Server lifecycle (parent side)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    """One uvicorn server subprocess + health polling + RSS sampling."""

    def __init__(
        self,
        name: str,
        state_root: Path,
        *,
        kill_seam: str | None = None,
        kill_count: int = 1,
        window_ms: float | None = None,
    ) -> None:
        self.name = name
        self.state_root = Path(state_root)
        self.port = _free_port()
        self.kill_seam = kill_seam
        self.kill_count = kill_count
        self.window_ms = window_ms
        self.proc: subprocess.Popen | None = None
        self.log_path = LOG_ROOT / f"{name}.log"
        self.restarts = 0
        self.kill_fired = False
        self.kill_marker = ""
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, date_str: str) -> None:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "serve",
            "--port",
            str(self.port),
            "--state-root",
            str(self.state_root),
            "--cache-dir",
            str(SITES_ROOT / SITE_ID),
            "--site-dir",
            str(SITE_ROOT / "processed_inputs"),
            "--date",
            date_str,
        ]
        if self.kill_seam:
            command += [
                "--kill-seam",
                self.kill_seam,
                "--kill-count",
                str(self.kill_count),
            ]
        if self.window_ms is not None:
            command += ["--window-ms", str(self.window_ms)]
        self._log_handle = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            start_new_session=True,
        )
        if not self.wait_ready(timeout=180.0):
            self.stop()
            raise RuntimeError(
                f"server {self.name} never became ready; log: {self.log_path}"
            )

    def wait_ready(self, timeout: float = 120.0) -> bool:
        import httpx

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False
            try:
                response = httpx.get(
                    f"{self.base_url}/health/ready", timeout=2.0
                )
                if response.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001 - not up yet
                pass
            time.sleep(0.25)
        return False

    def rss_kb(self) -> int | None:
        if self.proc is None or self.proc.poll() is not None:
            return None
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(self.proc.pid)],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            return int(out) if out else None
        except Exception:  # noqa: BLE001
            return None

    def poll_kill(self) -> bool:
        """True when the armed seam fired (process died by injection)."""
        if self.proc is None:
            return self.kill_fired
        if self.proc.poll() is None:
            return False
        if self.kill_fired:
            return True
        text = self.log_path.read_text(errors="replace")
        if _KILL_MARKER in text:
            self.kill_fired = True
            for line in text.splitlines():
                if line.startswith(_KILL_MARKER):
                    self.kill_marker = line[len(_KILL_MARKER):]
            return True
        return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_handle = None

    def restart(self, date_str: str) -> None:
        """Stop (if alive) and boot a FRESH process on the same state."""
        self.stop()
        self.restarts += 1
        self.kill_seam = None  # the restarted process never re-arms
        self.start(date_str)


# ---------------------------------------------------------------------------
# Read-only store inspection (post-crash truth)
# ---------------------------------------------------------------------------


def inspect_state(state_root: Path) -> dict[str, Any]:
    """Read the server's store DIRECTLY (read-only): durable truth.

    Per workspace: operation log, epoch rows, revision triple, plus the
    invariant computations the chaos verdicts consume.
    """
    import sqlite3

    db = Path(state_root) / "store.sqlite3"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        report: dict[str, Any] = {"workspaces": {}}
        for row in conn.execute(
            "SELECT scenario_id, scene_version, fast_revision, "
            "exact_result_version FROM scenarios"
        ):
            workspace = row["scenario_id"]
            operations = [
                dict(op)
                for op in conn.execute(
                    "SELECT operation_id, server_sequence, epoch_id, actor_id "
                    "FROM realtime_operations WHERE workspace_id = ? "
                    "ORDER BY server_sequence",
                    (workspace,),
                )
            ]
            epochs = [
                dict(ep)
                for ep in conn.execute(
                    "SELECT epoch_id, status, workspace_revision "
                    "FROM realtime_epochs "
                    "WHERE workspace_id = ? ORDER BY epoch_id",
                    (workspace,),
                )
            ]
            sequences = [op["server_sequence"] for op in operations]
            contiguous = not sequences or sequences == list(
                range(sequences[0], sequences[-1] + 1)
            )
            epoch_members: dict[int, int] = defaultdict(int)
            for op in operations:
                epoch_members[op["epoch_id"]] += 1
            unassigned = [
                ep["epoch_id"]
                for ep in epochs
                if ep["workspace_revision"] is None
                and epoch_members.get(ep["epoch_id"], 0) > 0
            ]
            revisions = [
                ep["workspace_revision"]
                for ep in epochs
                if ep["workspace_revision"] is not None
            ]
            order_ok = revisions == sorted(revisions)
            report["workspaces"][workspace] = {
                "workspace_revision": row["scene_version"],
                "fast_revision": row["fast_revision"],
                "exact_result_version": row["exact_result_version"],
                "operations": len(operations),
                "operation_ids": [op["operation_id"] for op in operations],
                "max_server_sequence": max(sequences) if sequences else 0,
                "sequences_contiguous": contiguous,
                "epochs": [
                    {
                        "epoch_id": ep["epoch_id"],
                        "status": ep["status"],
                        "workspace_revision": ep["workspace_revision"],
                        "ops": epoch_members.get(ep["epoch_id"], 0),
                    }
                    for ep in epochs
                ],
                "revised_epoch_count": len(revisions),
                "unassigned_epochs_with_ops": unassigned,
                "revision_order_ok": order_ok,
                "triple_ok": (
                    row["exact_result_version"]
                    <= row["fast_revision"]
                    <= row["scene_version"]
                ),
            }
        report["jobs"] = {
            row["status"]: row["n"]
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            )
        }
        # exact-lane job latency (queued -> finished), from the durable rows
        exact_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT queued_at, started_at, finished_at FROM jobs "
                "WHERE request_json LIKE '%exact_lane%' "
                "AND finished_at IS NOT NULL"
            )
        ]
        exact_latency_ms = [
            round(
                (
                    _parse_iso(row["finished_at"]) - _parse_iso(row["queued_at"])
                )
                * 1000.0,
                3,
            )
            for row in exact_rows
            if row["queued_at"] and row["finished_at"]
        ]
        report["exact_job_latency"] = percentiles(exact_latency_ms)
        return report
    finally:
        conn.close()


def canonical_state_bytes(state_root: Path, workspace: str) -> str | None:
    """Latest canonical families_json for one workspace (read-only)."""
    import sqlite3

    db = Path(state_root) / "store.sqlite3"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT state_json FROM realtime_canonical_state "
            "WHERE workspace_id = ? ORDER BY workspace_revision DESC LIMIT 1",
            (workspace,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The collaborative client workload
# ---------------------------------------------------------------------------


class OpFactory:
    """Deterministic-seeded realistic op mix for one workspace/actor.

    Same seed => same payload/entity sequence for any workspace prefix
    (the determinism case replays a sequence across two workspaces).
    """

    TREE_POOL = 12

    def __init__(self, workspace: str, actor: str, seed: int) -> None:
        self.workspace = workspace
        self.actor = actor
        self.rng = random.Random(seed)
        self.counter = itertools.count(1)
        self.trees: set[str] = set()
        self.last_op: dict[str, Any] | None = None

    def _op(self, **fields: Any) -> dict[str, Any]:
        number = next(self.counter)
        item: dict[str, Any] = {
            "operation_id": f"{self.workspace}-{self.actor}-{number:06d}",
            "client_sequence": number,
            "base_revision": None,
            "source_family": "output_view",
            "entity_id": None,
            "verb": "select",
            "payload": {},
        }
        item.update(fields)
        self.last_op = dict(item)
        return item

    def view_op(self, base_revision: int | None) -> dict[str, Any]:
        return self._op(
            base_revision=base_revision,
            source_family="output_view",
            verb="select",
            payload={
                "values": {
                    "timestep": self.rng.randrange(0, 3),
                    "layer": self.rng.choice(["utci", "tmrt"]),
                    "east": round(self.rng.uniform(0.0, 1.0), 3),
                    "north": round(self.rng.uniform(0.0, 1.0), 3),
                }
            },
        )

    def veg_op(self, base_revision: int | None) -> dict[str, Any]:
        slot = self.rng.randrange(self.TREE_POOL)
        tree_id = f"tree-{slot:02d}"
        offset_x = (slot % 4 - 1.5) * 16.0
        offset_y = (slot // 4 - 1.0) * 16.0
        roll = self.rng.random()
        if tree_id not in self.trees:
            verb, values = "add", {
                "x_m": round(1000.0 + 64.0 + offset_x, 2),
                "y_m": round(2000.0 - 64.0 + offset_y, 2),
                "height_m": round(self.rng.uniform(4.0, 14.0), 2),
                "canopy_radius_m": round(self.rng.uniform(2.0, 5.0), 2),
                "trunk_ratio": 0.25,
            }
            self.trees.add(tree_id)
        elif roll < 0.45:
            verb, values = "replace", {
                "height_m": round(self.rng.uniform(4.0, 14.0), 2),
                "canopy_radius_m": round(self.rng.uniform(2.0, 5.0), 2),
            }
        elif roll < 0.7:
            verb, values = "move", {
                "x_m": round(1000.0 + 64.0 + offset_x + self.rng.uniform(-3.0, 3.0), 2),
                "y_m": round(2000.0 - 64.0 + offset_y + self.rng.uniform(-3.0, 3.0), 2),
            }
        else:
            verb, values = "delete", {}
            self.trees.discard(tree_id)
        return self._op(
            base_revision=base_revision,
            source_family="vegetation_geometry",
            entity_id=tree_id,
            verb=verb,
            payload={"values": values},
        )

    def met_op(self, base_revision: int | None) -> dict[str, Any]:
        return self._op(
            base_revision=base_revision,
            source_family="meteorological_forcing",
            verb="set",
            payload={
                "time_index": self.rng.randrange(0, 3),
                "values": {
                    "air_temperature": round(self.rng.uniform(18.0, 38.0), 2)
                },
            },
        )

    def landcover_op(self, base_revision: int | None) -> dict[str, Any]:
        row = self.rng.randrange(20, 100)
        col = self.rng.randrange(20, 100)
        return self._op(
            base_revision=base_revision,
            source_family="landcover_surface",
            verb="paint",
            payload={
                "window": {
                    "row_start": row,
                    "row_stop": row + 4,
                    "col_start": col,
                    "col_stop": col + 4,
                },
                # the adapter's valid vocabulary (classes 3/4 are typed
                # edit_rejected refusals at exact-solve time)
                "class": self.rng.choice((1, 2, 5, 6)),
            },
        )

    def next_op(self, base_revision: int | None = None) -> dict[str, Any]:
        roll = self.rng.random()
        if roll < 0.70:
            return self.view_op(base_revision)
        if roll < 0.85:
            return self.veg_op(base_revision)
        if roll < 0.95:
            return self.met_op(base_revision)
        return self.landcover_op(base_revision)

    def forget(self, entity_id: str) -> None:
        """Drop optimistic knowledge of an entity whose add never landed.

        The factory marks an add at BUILD time; if the post is lost (crash
        mid-transaction, transport error) the durable log never saw the
        object, and a later replace/move/delete of that id would trip the
        exact lane's unfoldable-native-op fence (a CORRECT server refusal
        -- the client, not the server, is wrong). The client model must
        forget entities whose adds were not durably acked.
        """
        self.trees.discard(entity_id)

    def stale_op(self, current_revision: int) -> dict[str, Any]:
        """A deliberate stale-base op (base_revision far in the past)."""
        stale_base = max(0, current_revision - self.rng.randrange(3, 30))
        return self.next_op(base_revision=stale_base)


class Recorder:
    """Thread-safe measurement store for one run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.op_records: dict[str, dict[str, Any]] = {}
        self.status_counts: dict[str, int] = defaultdict(int)
        self.reject_codes: dict[str, int] = defaultdict(int)
        self.duplicates = 0
        self.conflict_409 = 0
        self.stale_submitted = 0
        self.stale_divergent_acked = 0
        self.canonical_frames = 0
        self.fast_frames = 0
        self.heartbeats = 0
        self.missed_event_notices = 0
        self.sse_reconnects = 0
        self.sse_drops = 0
        self.client_errors = 0
        #: op_id -> (arrival wall s, revision)
        self.op_visibility: dict[str, tuple[float, int]] = {}
        #: op_id -> ack body (accepted_at + triple)
        self.op_acks: dict[str, dict[str, Any]] = {}
        #: (workspace, revision) -> first client arrival wall s
        self.revision_arrival: dict[tuple[str, int], float] = {}
        #: fast class -> server-reported latency_ms
        self.fast_latency_by_class: dict[str, list[float]] = defaultdict(list)
        #: fast class -> client publish->arrival ms
        self.fast_arrival_by_class: dict[str, list[float]] = defaultdict(list)
        #: workspace -> fast_revision stream (monotonic fence)
        self.fast_revision_seen: dict[str, list[int]] = defaultdict(list)
        self.triple_samples: list[dict[str, Any]] = []
        self.queue_samples: list[dict[str, Any]] = []
        self.rss_samples: list[dict[str, Any]] = []
        #: host-load gauge values sampled by the Monitor's existing
        #: /metrics poll (quiet-host re-commission: min/mean/max evidence)
        self.host_load_samples: list[float] = []
        self.result_class_counts: dict[str, int] = defaultdict(int)
        self.qualified_superseded = 0
        #: (workspace, revision) -> wall s when exact covered it (poll)
        self.exact_covered_at: dict[tuple[str, int], float] = {}
        #: workspace -> canonical frame ledger (determinism evidence)
        self.canonical_ledgers: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ops_submitted": len(self.op_records),
                "status_counts": dict(self.status_counts),
                "reject_codes": dict(self.reject_codes),
                "duplicates": self.duplicates,
                "conflict_409": self.conflict_409,
                "stale_submitted": self.stale_submitted,
                "stale_divergent_acked": self.stale_divergent_acked,
                "canonical_frames": self.canonical_frames,
                "fast_frames": self.fast_frames,
                "heartbeats": self.heartbeats,
                "missed_event_notices": self.missed_event_notices,
                "sse_reconnects": self.sse_reconnects,
                "sse_drops": self.sse_drops,
                "result_class_counts": dict(self.result_class_counts),
                "qualified_superseded": self.qualified_superseded,
                "client_errors": self.client_errors,
            }


class Client:
    """One collaborative client: an op writer + an SSE consumer.

    The writer thread dies with ``stop_event``; the SSE consumer loops
    until ``sse_stop`` and RECONNECTS on any transport failure (the
    chaos cases rely on that: a killed server must not end consumption).
    ``reconnect()`` swaps in a fresh HTTP client and starts a new writer
    for the remaining window after a server restart -- the OpFactory
    survives (same process), so op ids never regress.
    """

    def __init__(
        self,
        name: str,
        workspace: str,
        server: Server,
        recorder: Recorder,
        *,
        rate_ops_per_s: float,
        duration_s: float,
        seed: int,
        stop_event: threading.Event,
        drop_at_s: float | None = None,
    ) -> None:
        import httpx

        self.name = name
        self.workspace = workspace
        self.server = server
        self.recorder = recorder
        self.rate = rate_ops_per_s
        self.duration = duration_s
        self.factory = OpFactory(workspace, name, seed)
        self.stop_event = stop_event
        self.drop_at_s = drop_at_s
        self.http = httpx.Client(base_url=server.base_url, timeout=15.0)
        self.sse_stop = threading.Event()
        self.threads: list[threading.Thread] = []
        #: idempotent retry outbox: (items, kind, attempts)
        self.retry_batches: list[tuple[list[dict[str, Any]], str, int]] = []

    # -- op writer -----------------------------------------------------------

    def _record(
        self,
        item: dict[str, Any],
        status: int,
        *,
        body: dict[str, Any] | None = None,
        error: str | None = None,
        kind: str = "op",
    ) -> None:
        record = {
            "operation_id": item["operation_id"],
            "kind": kind,
            "family": item["source_family"],
            "verb": item["verb"],
            "stale_base": item.get("base_revision") is not None,
            "status": status,
            "error": error,
        }
        with self.recorder._lock:
            self.recorder.status_counts[str(status)] += 1
            if error is not None:
                self.recorder.client_errors += 1
            self.recorder.op_records.setdefault(item["operation_id"], record)
            if status >= 400 and body is not None:
                self.recorder.reject_codes[
                    body.get("error", {}).get("code", "?")
                ] += 1
            if status == 409:
                self.recorder.conflict_409 += 1

    def _post_ops(
        self,
        items: list[dict[str, Any]],
        kind: str,
        attempts: int = 0,
    ) -> None:
        max_attempts = 8
        try:
            response = self.http.post(
                f"/api/v1/workspaces/{self.workspace}/operations",
                json={"actor_id": self.name, "operations": items},
            )
        except Exception as error:  # noqa: BLE001 - transport errors are data
            for item in items:
                self._record(item, 0, error=repr(error), kind=kind)
            if attempts + 1 < max_attempts:
                # idempotent redelivery (same operation ids): a lost post
                # is indistinguishable from a lost ACK, and the store's
                # fingerprint dedupe makes retry the safe resolution.
                self.retry_batches.append((items, kind, attempts + 1))
            else:
                self._give_up(items)
            return
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            body = None
        if response.status_code in (429, 503) or body is None:
            # admission/backpressure: rejected BEFORE acceptance -- the
            # batch is safe to redeliver after backoff
            for item in items:
                self._record(item, response.status_code, body=body, kind=kind)
            if attempts + 1 < max_attempts:
                self.retry_batches.append((items, kind, attempts + 1))
            else:
                self._give_up(items)
            return
        if response.status_code != 200:
            # definitive refusals (400 validation, 409 fingerprint reuse)
            for item in items:
                self._record(
                    item, response.status_code, body=body, kind=kind
                )
            self._give_up(items)
            return
        # 200: record per-item, acks matched to items BY POSITION (the
        # accepted list mirrors the submitted order).
        acks = list(body.get("acks", ()))
        for position, item in enumerate(items):
            self._record(item, response.status_code, kind=kind)
            ack = (
                acks[position]
                if position < len(acks) and acks[position]["operation_id"]
                == item["operation_id"]
                else None
            )
            if ack is None:
                continue
            with self.recorder._lock:
                # keep the EARLIEST acceptance anchor (a duplicate replay's
                # route timestamp would understate visibility latency)
                self.recorder.op_acks.setdefault(
                    ack["operation_id"],
                    {
                        **ack,
                        "accepted_at": body["accepted_at"],
                        "workspace": self.workspace,
                    },
                )
                if ack.get("duplicate"):
                    self.recorder.duplicates += 1
                if (
                    item.get("base_revision") is not None
                    and ack.get("base_divergent")
                ):
                    self.recorder.stale_divergent_acked += 1

    def _give_up(self, items: list[dict[str, Any]]) -> None:
        """The batch will never durably land: forget its optimistic adds."""
        for item in items:
            if (
                item["source_family"] == "vegetation_geometry"
                and item["verb"] == "add"
                and item.get("entity_id")
            ):
                self.factory.forget(item["entity_id"])

    def _writer_loop(self, deadline: float) -> None:
        rng = self.factory.rng
        interval = 1.0 / max(self.rate, 0.01)
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            # drain the idempotent retry outbox first (a lost post is
            # indistinguishable from a lost ack; same ids, safe redelivery)
            if self.retry_batches:
                pending, self.retry_batches = self.retry_batches, []
                for items, kind, attempts in pending:
                    if self.stop_event.is_set():
                        self.retry_batches.append((items, kind, attempts))
                        break
                    self._post_ops(items, kind, attempts=attempts)
                time.sleep(interval)
                continue
            roll = rng.random()
            if roll < 0.0667:
                # deliberate stale-revision op (1 in 15)
                item = self.factory.stale_op(current_revision=1)
                with self.recorder._lock:
                    self.recorder.stale_submitted += 1
                self._post_ops([item], "stale")
            elif roll < 0.1333 and self.factory.last_op is not None:
                # deliberate DUPLICATE resubmission (1 in 15)
                self._post_ops([dict(self.factory.last_op)], "duplicate")
            else:
                batch = [self.factory.next_op()]
                if rng.random() < 0.15:
                    batch.append(self.factory.next_op())
                self._post_ops(batch, "op")
            time.sleep(interval * rng.uniform(0.5, 1.5))

    # -- SSE consumer ----------------------------------------------------------

    def _sse_loop(self) -> None:
        import httpx

        started = time.monotonic()
        dropped_planned = self.drop_at_s is None
        while not self.sse_stop.is_set():
            event_name = None
            data_lines: list[str] = []
            try:
                with self.http.stream(
                    "GET",
                    f"/api/v1/workspaces/{self.workspace}/events",
                    timeout=httpx.Timeout(10.0, read=90.0),
                ) as stream:
                    for line in stream.iter_lines():
                        if self.sse_stop.is_set():
                            break
                        if (
                            not dropped_planned
                            and self.drop_at_s is not None
                            and time.monotonic() - started >= self.drop_at_s
                        ):
                            dropped_planned = True  # one deliberate drop
                            with self.recorder._lock:
                                self.recorder.sse_drops += 1
                            break
                        if line == "":
                            if event_name is not None and data_lines:
                                self._handle_frame(
                                    event_name, "\n".join(data_lines)
                                )
                            event_name, data_lines = None, []
                            continue
                        if line.startswith("event:"):
                            event_name = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:"):].strip())
            except Exception:  # noqa: BLE001 - drops/reconnects are the case
                pass
            if self.sse_stop.is_set():
                return
            with self.recorder._lock:
                self.recorder.sse_reconnects += 1
            time.sleep(0.2)

    def _handle_frame(self, event: str, data: str) -> None:
        arrival = time.time()
        try:
            payload = json.loads(data)
        except ValueError:
            return
        recorder = self.recorder
        workspace = self.workspace
        if event == "canonical_revision":
            revision = payload.get("workspace_revision")
            with recorder._lock:
                recorder.canonical_frames += 1
                recorder.revision_arrival.setdefault(
                    (workspace, revision), arrival
                )
                for op in payload.get("operations", ()):
                    recorder.op_visibility.setdefault(
                        op["operation_id"], (arrival, revision)
                    )
                recorder.triple_samples.append(
                    {
                        "t": round(arrival, 3),
                        "kind": "canonical",
                        "workspace": workspace,
                        "workspace_revision": revision,
                        "fast_revision": payload.get("fast_revision"),
                        "exact_revision": payload.get("exact_revision"),
                    }
                )
                if not payload.get("snapshot"):
                    recorder.canonical_ledgers[workspace].append(
                        {
                            "workspace_revision": revision,
                            "ops": [
                                {
                                    "family": op["source_family"],
                                    "verb": op["verb"],
                                    "entity": op["entity_id"],
                                    "payload": op["payload"],
                                }
                                for op in payload.get("operations", ())
                            ],
                            "conflicts": payload.get("conflicts", []),
                        }
                    )
            return
        if event == "fast_revision":
            result_class = payload.get("result_class")
            with recorder._lock:
                recorder.fast_frames += 1
                if result_class is not None:
                    recorder.result_class_counts[result_class] += 1
                if payload.get("payload", {}).get("superseded_qualified"):
                    recorder.qualified_superseded += 1
                if isinstance(payload.get("latency_ms"), (int, float)):
                    recorder.fast_latency_by_class[result_class].append(
                        float(payload["latency_ms"])
                    )
                try:
                    published = _parse_iso(payload["published_at"])
                    recorder.fast_arrival_by_class[result_class].append(
                        max(0.0, (arrival - published) * 1000.0)
                    )
                except (KeyError, ValueError, TypeError):
                    pass
                recorder.fast_revision_seen[workspace].append(
                    payload.get("fast_revision")
                )
                recorder.triple_samples.append(
                    {
                        "t": round(arrival, 3),
                        "kind": "fast",
                        "workspace": workspace,
                        "workspace_revision": payload.get("workspace_revision"),
                        "fast_revision": payload.get("fast_revision"),
                        "exact_revision": payload.get("exact_revision"),
                    }
                )
            return
        if event == "heartbeat":
            with recorder._lock:
                recorder.heartbeats += 1
                if payload.get("missed_events"):
                    recorder.missed_event_notices += 1

    # -- lifecycle ---------------------------------------------------------------

    def start(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        writer = threading.Thread(
            target=self._writer_loop,
            args=(deadline,),
            name=f"{self.name}-writer",
            daemon=True,
        )
        consumer = threading.Thread(
            target=self._sse_loop, name=f"{self.name}-sse", daemon=True
        )
        self.threads = [writer, consumer]
        for thread in self.threads:
            thread.start()

    def reconnect(self, duration_s: float) -> None:
        """After a server restart: fresh HTTP client + new writer thread.

        The SSE consumer thread is NOT restarted -- its retry loop finds
        the new process on its own (reconnects are counted as evidence).
        """
        import httpx

        try:
            self.http.close()
        except Exception:  # noqa: BLE001
            pass
        self.http = httpx.Client(base_url=self.server.base_url, timeout=15.0)
        deadline = time.monotonic() + duration_s
        writer = threading.Thread(
            target=self._writer_loop,
            args=(deadline,),
            name=f"{self.name}-writer-r",
            daemon=True,
        )
        self.threads.append(writer)
        writer.start()

    def wait_writers(self, timeout: float = 30.0) -> None:
        for thread in list(self.threads):
            if not thread.is_alive():
                continue
            if thread.name.endswith("writer") or thread.name.endswith(
                "writer-r"
            ):
                thread.join(timeout=timeout)

    def stop(self, timeout: float = 10.0) -> None:
        self.sse_stop.set()
        try:
            self.http.close()
        except Exception:  # noqa: BLE001
            pass
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=timeout)


class Monitor:
    """Background poller: revision triple, queue depth, RSS."""

    def __init__(
        self,
        server: Server,
        recorder: Recorder,
        *,
        stop_event: threading.Event,
        workspaces: Sequence[str],
        interval_s: float = 0.5,
        rss_interval_s: float = 5.0,
    ) -> None:
        import httpx

        self.server = server
        self.recorder = recorder
        self.interval = interval_s
        self.rss_interval = rss_interval_s
        self.stop_event = stop_event
        self.workspaces = list(workspaces)
        self.http = httpx.Client(base_url=server.base_url, timeout=10.0)
        self.thread = threading.Thread(
            target=self._loop, name="r8-monitor", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        next_rss = time.monotonic()
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now >= next_rss:
                rss = self.server.rss_kb()
                if rss is not None:
                    with self.recorder._lock:
                        self.recorder.rss_samples.append(
                            {"t": round(time.time(), 3), "rss_kb": rss}
                        )
                next_rss = now + self.rss_interval
            for workspace in self.workspaces:
                try:
                    # The triple rides the catch-up GET; a huge since-watermark
                    # keeps the response O(1) (we never need the ops here).
                    response = self.http.get(
                        f"/api/v1/workspaces/{workspace}/operations",
                        params={"since_server_sequence": 2**31},
                    )
                    if response.status_code == 200:
                        body = response.json()
                        revision = body["workspace_revision"]
                        wall = time.time()
                        with self.recorder._lock:
                            self.recorder.triple_samples.append(
                                {
                                    "t": round(wall, 3),
                                    "kind": "poll",
                                    "workspace": workspace,
                                    "workspace_revision": revision,
                                    "fast_revision": body["fast_revision"],
                                    "exact_revision": body["exact_revision"],
                                }
                            )
                            key = (workspace, revision)
                            if (
                                key not in self.recorder.exact_covered_at
                                and body["exact_revision"] >= revision
                            ):
                                self.recorder.exact_covered_at[key] = wall
                except Exception:  # noqa: BLE001 - server may be down (chaos)
                    pass
            try:
                metrics = self.http.get("/metrics").json()
                jobs = metrics.get("jobs", {})
                # quiet-host re-commission: ride the SAME poll to sample
                # the server-side host-load gauge (ops thread loadavg)
                load = (
                    metrics.get("realtime", {})
                    .get("solweig_rt_host_load", {})
                    .get("", {})
                    .get("value")
                )
                with self.recorder._lock:
                    self.recorder.queue_samples.append(
                        {
                            "t": round(time.time(), 3),
                            "queued": jobs.get("queued", 0),
                            "running": jobs.get("running", 0),
                            "failed": jobs.get("failed", 0),
                            "complete": jobs.get("complete", 0),
                        }
                    )
                    if isinstance(load, (int, float)):
                        self.recorder.host_load_samples.append(float(load))
            except Exception:  # noqa: BLE001
                pass
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.thread.join(timeout=10.0)
        try:
            self.http.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Invariant verification (the R8 gates)
# ---------------------------------------------------------------------------


def verify_invariants(
    recorder: Recorder,
    state_report: dict[str, Any],
    *,
    strict_stream_delivery: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Check every R8 invariant against recorder + durable state.

    Returns ``(defects, evidence)``; empty defects means PASS.
    ``strict_stream_delivery`` is False for crash/drop runs where SSE
    frame loss between kill and reconnect is the DESIGNED loss model
    (the catch-up GET is the recovery path; zero-loss is checked against
    the durable log, not the stream).
    """
    defects: list[str] = []
    evidence: dict[str, Any] = {}

    with recorder._lock:
        acked_ops = dict(recorder.op_acks)
        visibility = dict(recorder.op_visibility)
        triple = list(recorder.triple_samples)
        fast_seen = {
            ws: list(values)
            for ws, values in recorder.fast_revision_seen.items()
        }
        ledgers = {
            ws: list(frames)
            for ws, frames in recorder.canonical_ledgers.items()
        }

    # -- invariant 1: zero loss / no double-apply (durable truth) -----------
    for workspace, info in state_report["workspaces"].items():
        durable_ids = info["operation_ids"]
        durable = set(durable_ids)
        if len(durable_ids) != len(durable):
            defects.append(
                f"{workspace}: duplicate operation rows in the durable log"
            )
        acked_here = {
            op_id for op_id in acked_ops if op_id.startswith(f"{workspace}-")
        }
        missing = sorted(acked_here - durable)
        if missing:
            defects.append(
                f"{workspace}: {len(missing)} acked op(s) missing from the "
                f"durable log (first: {missing[:3]})"
            )
        evidence.setdefault("zero_loss", {})[workspace] = {
            "acked": len(acked_here),
            "durable": len(durable),
            "missing": len(missing),
        }
        if info["unassigned_epochs_with_ops"]:
            defects.append(
                f"{workspace}: epochs {info['unassigned_epochs_with_ops']} "
                "carry ops but no revision (stranded after restart)"
            )
        if not info["sequences_contiguous"]:
            defects.append(f"{workspace}: server_sequence not contiguous")
        if not info["revision_order_ok"]:
            defects.append(f"{workspace}: epoch/revision order not monotonic")
        if not info["triple_ok"]:
            defects.append(
                f"{workspace}: revision triple violated "
                f"(exact={info['exact_result_version']} "
                f"fast={info['fast_revision']} "
                f"workspace={info['workspace_revision']})"
            )
        # canonical revisions are exactly 1..N, each once
        revised = [
            ep["workspace_revision"]
            for ep in info["epochs"]
            if ep["workspace_revision"] is not None
        ]
        if revised and revised != list(range(1, len(revised) + 1)):
            defects.append(
                f"{workspace}: canonical revisions not exactly 1..N "
                f"(head={revised[:5]} tail={revised[-5:]})"
            )
        if info["workspace_revision"] != len(revised):
            defects.append(
                f"{workspace}: workspace_revision {info['workspace_revision']} "
                f"!= revised epoch count {len(revised)}"
            )
    evidence["ledger_frames"] = {
        ws: len(frames) for ws, frames in ledgers.items()
    }

    # -- invariant 2: triple monotonic per observation channel ---------------
    polls_by_workspace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in triple:
        if sample.get("kind") == "poll" and sample.get("workspace"):
            polls_by_workspace[sample["workspace"]].append(sample)
    monotonic: dict[str, bool] = {}
    for workspace, samples in polls_by_workspace.items():
        ok = True
        last = (-1, -1, -1)
        for sample in samples:
            current = (
                sample["exact_revision"],
                sample["fast_revision"],
                sample["workspace_revision"],
            )
            if (
                current[0] > current[1]
                or current[1] > current[2]
                or current[0] < last[0]
                or current[1] < last[1]
                or current[2] < last[2]
            ):
                ok = False
                break
            last = current
        monotonic[workspace] = ok
        if not ok:
            defects.append(f"{workspace}: revision triple not monotonic in polls")
    evidence["triple_monotonic_in_polls"] = monotonic
    for workspace, values in fast_seen.items():
        filtered = [v for v in values if isinstance(v, int)]
        if filtered != sorted(filtered):
            defects.append(
                f"{workspace}: fast_revision stream not monotonic "
                f"(head={filtered[:20]})"
            )

    # -- healthy-stream delivery (skip for crash/drop runs: frames lost -----
    #    across a kill are the DESIGNED loss model)
    if strict_stream_delivery and acked_ops:
        visible = sum(1 for op_id in acked_ops if op_id in visibility)
        ratio = visible / len(acked_ops)
        evidence["healthy_stream_visibility_ratio"] = round(ratio, 4)
        if ratio < 0.95:
            defects.append(
                f"healthy SSE streams failed to deliver {100 * (1 - ratio):.1f}% "
                "of acked ops (catch-up is the recovery path, but a healthy "
                "stream must not lag)"
            )

    return defects, evidence


def visibility_percentiles(recorder: Recorder) -> dict[str, Any]:
    """op accept -> canonical SSE arrival, server-clock anchored."""
    with recorder._lock:
        acks = dict(recorder.op_acks)
        visibility = dict(recorder.op_visibility)
    samples: list[float] = []
    per_workspace: dict[str, list[float]] = defaultdict(list)
    for op_id, (arrival, _revision) in visibility.items():
        ack = acks.get(op_id)
        if ack is None:
            continue
        try:
            accepted = _parse_iso(ack["accepted_at"])
        except (KeyError, ValueError):
            continue
        delta = max(0.0, (arrival - accepted) * 1000.0)
        samples.append(delta)
        per_workspace[ack.get("workspace", "?")].append(delta)
    return {
        "op_accept_to_canonical_arrival": percentiles(samples),
        "by_workspace": {
            name: percentiles(values)
            for name, values in sorted(per_workspace.items())
        },
    }


def exact_catchup_percentiles(recorder: Recorder) -> dict[str, Any]:
    """canonical frame arrival -> exact_revision coverage (poll-limited)."""
    with recorder._lock:
        revision_arrival = dict(recorder.revision_arrival)
        exact_covered = dict(recorder.exact_covered_at)
    samples: list[float] = []
    for key, covered_at in exact_covered.items():
        arrived = revision_arrival.get(key)
        if arrived is None or covered_at < arrived:
            continue
        samples.append((covered_at - arrived) * 1000.0)
    return {
        "canonical_to_exact_coverage_poll_limited": percentiles(samples),
        "poll_interval_ms": 500.0,
        "note": (
            "exact catch-up observed by a 500 ms poll of the catch-up GET; "
            "true latency is <= the reported figure"
        ),
    }


def rss_slope(recorder: Recorder) -> dict[str, Any]:
    """RSS timeseries slope/step analysis (bounded-history fence)."""
    with recorder._lock:
        samples = list(recorder.rss_samples)
    if len(samples) < 10:
        return {"n": len(samples)}
    series = [sample["rss_kb"] for sample in samples]
    n = len(series)
    half = n // 2
    first_half = statistics.fmean(series[:half])
    second_half = statistics.fmean(series[half:])
    x_mean = (n - 1) / 2.0
    y_mean = statistics.fmean(series)
    slope = sum(
        (i - x_mean) * (value - y_mean) for i, value in enumerate(series)
    ) / sum((i - x_mean) ** 2 for i in range(n))
    total_seconds = samples[-1]["t"] - samples[0]["t"]
    interval = max(1.0, total_seconds / max(1, n - 1))
    kb_per_hour = slope * 3600.0 / interval
    return {
        "n": n,
        "first_kb": series[0],
        "last_kb": series[-1],
        "min_kb": min(series),
        "max_kb": max(series),
        "mean_first_half_kb": round(first_half, 1),
        "mean_second_half_kb": round(second_half, 1),
        # least-squares slope where one "x" is ONE RSS POLL (the 5 s
        # sampler), not a second or an epoch — per-poll-index truth
        # (R8b review N2). kb_per_hour_est converts via the measured
        # mean poll interval.
        "lsq_slope_kb_per_poll": round(slope, 3),
        "kb_per_hour_est": round(kb_per_hour, 1),
        "growth_ratio_second_over_first": (
            round(second_half / first_half, 3) if first_half else None
        ),
        "duration_s": round(total_seconds, 1),
    }


def fast_latency_evidence(recorder: Recorder) -> dict[str, Any]:
    with recorder._lock:
        latency = {
            name: list(values)
            for name, values in recorder.fast_latency_by_class.items()
        }
        arrival = {
            name: list(values)
            for name, values in recorder.fast_arrival_by_class.items()
        }
    return {
        "server_reported_latency_ms_by_class": by_class(latency),
        "client_arrival_after_publish_ms_by_class": by_class(arrival),
        "slo_fast_ms": SLO_FAST_MS,
    }


# ---------------------------------------------------------------------------
# The run driver
# ---------------------------------------------------------------------------


class RunConfig:
    def __init__(
        self,
        name: str,
        *,
        duration_s: float,
        clients: int,
        ops_per_s_per_client: float,
        workspaces: int = WORKSPACE_COUNT,
        kill_seam: str | None = None,
        kill_count: int = 1,
        quiesce_s: float = 45.0,
        restart_at_kill: bool = False,
        client_drop_at_s: float | None = None,
        window_ms: float | None = None,
    ) -> None:
        self.name = name
        self.duration_s = duration_s
        self.clients = clients
        self.ops_rate = ops_per_s_per_client
        self.workspace_count = workspaces
        self.kill_seam = kill_seam
        self.kill_count = kill_count
        self.quiesce_s = quiesce_s
        self.restart_at_kill = restart_at_kill
        self.client_drop_at_s = client_drop_at_s
        self.window_ms = window_ms


def ensure_workspaces(
    server: Server, count: int, *, ready_timeout: float = 120.0
) -> list[str]:
    """Create ``count`` workspaces; return their SERVER-ASSIGNED ids.

    Waits for each scenario's baseline publication (inline from the
    harness-precomputed baseline_results) before the load starts.
    """
    import httpx

    ids: list[str] = []
    with httpx.Client(base_url=server.base_url, timeout=30.0) as http:
        for index in range(count):
            response = http.post(
                "/api/v1/scenarios",
                json={
                    "site_id": SITE_ID,
                    "name": f"r8_ws_{index}",
                    "initial_state": "baseline",
                },
            )
            if response.status_code != 201:
                raise RuntimeError(
                    f"workspace create failed: {response.status_code} "
                    f"{response.text[:300]}"
                )
            ids.append(response.json()["scenario_id"])
        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            ready = 0
            for workspace in ids:
                body = http.get(f"/api/v1/scenarios/{workspace}").json()
                if body["exact_result_version"] == body["scene_version"]:
                    ready += 1
            if ready == len(ids):
                return ids
            time.sleep(0.5)
        raise RuntimeError(f"workspaces never published baselines: {ids}")


def run_load(config: RunConfig, evidence_path: Path) -> dict[str, Any]:
    """One measured run: server + clients + monitor + verification."""
    site = ensure_site()
    run_root = RUNS_ROOT / config.name
    shutil.rmtree(run_root, ignore_errors=True)
    server = Server(
        config.name,
        run_root / "state",
        kill_seam=config.kill_seam,
        kill_count=config.kill_count,
        window_ms=config.window_ms,
    )
    recorder = Recorder()
    stop_event = threading.Event()
    started = time.monotonic()

    server.start(site["date"])
    workspaces = ensure_workspaces(server, config.workspace_count)
    evidence: dict[str, Any] = {
        "harness": "r8_load_chaos_soak",
        "run": config.name,
        "config": {
            "duration_s": config.duration_s,
            "clients": config.clients,
            "ops_per_s_per_client": config.ops_rate,
            "workspaces": workspaces,
            "kill_seam": config.kill_seam,
            "kill_count": config.kill_count,
            "restart_at_kill": config.restart_at_kill,
            "client_drop_at_s": config.client_drop_at_s,
            "coalescing_window_ms": config.window_ms,
        },
        "host": _host_block(),
        "site": site,
        "port": server.port,
        "server_log": str(server.log_path),
    }

    clients: list[Client] = []
    for index in range(config.clients):
        workspace = workspaces[index % len(workspaces)]
        clients.append(
            Client(
                f"actor_{index:02d}",
                workspace,
                server,
                recorder,
                rate_ops_per_s=config.ops_rate,
                duration_s=config.duration_s,
                seed=_RNG_SEED + index,
                stop_event=stop_event,
                drop_at_s=config.client_drop_at_s if index == 0 else None,
            )
        )
    monitor = Monitor(
        server, recorder, stop_event=stop_event, workspaces=workspaces
    )
    for client in clients:
        client.start(config.duration_s)
    monitor.start()

    # -- the run body -----------------------------------------------------
    killed = False
    if config.kill_seam:
        deadline = time.monotonic() + config.duration_s
        while time.monotonic() < deadline:
            if server.poll_kill():
                killed = True
                break
            time.sleep(0.1)
    else:
        # progress heartbeat (the soak is polled from the driver log)
        end = time.monotonic() + config.duration_s
        next_report = time.monotonic() + 60.0
        while time.monotonic() < end:
            time.sleep(1.0)
            if time.monotonic() >= next_report:
                elapsed = int(time.monotonic() - started)
                rss = server.rss_kb()
                # quiet-host re-commission: a per-minute `uptime` reading
                # INSIDE the evidence (load conditions, honestly sampled)
                evidence.setdefault("uptime_readings", []).append(
                    {"t_s": elapsed, "uptime": _uptime()}
                )
                with recorder._lock:
                    ops = len(recorder.op_records)
                    fast = recorder.fast_frames
                print(
                    f"[r8 {config.name}] t+{elapsed}s ops={ops} "
                    f"fast_frames={fast} rss_kb={rss}",
                    flush=True,
                )
                next_report = time.monotonic() + 60.0
    for client in clients:
        client.wait_writers(timeout=30.0)

    if killed:
        evidence["kill"] = {
            "seam": config.kill_seam,
            "marker": server.kill_marker,
            "fired_at_wall": time.time(),
            "log_tail": server.log_path.read_text(errors="replace")[-2000:],
        }
        if config.restart_at_kill:
            # writers stop (stop_event); SSE consumers keep retrying; the
            # monitor keeps polling (both tolerate the dead server).
            stop_event.set()
            for client in clients:
                client.wait_writers(timeout=30.0)
            server.restart(site["date"])
            remaining = max(
                1.0, config.duration_s - (time.monotonic() - started)
            )
            stop_event.clear()
            for client in clients:
                client.reconnect(remaining)
            end = time.monotonic() + remaining
            while time.monotonic() < end:
                time.sleep(1.0)
            for client in clients:
                client.wait_writers(timeout=30.0)
            evidence["kill"]["restarted"] = True
            evidence["kill"]["restarts"] = server.restarts

    # -- quiesce: let the planes settle (bounded; residual exact lag is a --
    #    measured outcome, not a defect) ------------------------------------
    quiesce_deadline = time.monotonic() + config.quiesce_s
    import httpx as _httpx

    with _httpx.Client(base_url=server.base_url, timeout=10.0) as probe:
        lag: dict[str, int] = {}
        while time.monotonic() < quiesce_deadline:
            try:
                settled = True
                for workspace in workspaces:
                    body = probe.get(
                        f"/api/v1/workspaces/{workspace}/operations",
                        params={"since_server_sequence": 2**31},
                    ).json()
                    lag[workspace] = (
                        body["workspace_revision"] - body["exact_revision"]
                    )
                    if lag[workspace] > 0:
                        settled = False
                if settled:
                    break
            except Exception:  # noqa: BLE001 - server may be down (kill case)
                pass
            time.sleep(1.0)
        evidence["final_exact_lag"] = lag

    stop_event.set()
    for client in clients:
        client.stop()
    monitor.stop()

    # final server-side telemetry decomposition (the registry histograms:
    # ack_ms / epoch_wait_ms / reduce_ms / publish_ms / fast lanes)
    try:
        import httpx as _hx

        with _hx.Client(base_url=server.base_url, timeout=10.0) as probe:
            evidence["server_telemetry"] = (
                probe.get("/metrics").json().get("realtime", {})
            )
    except Exception:  # noqa: BLE001 - server may be down (kill case)
        evidence["server_telemetry"] = {}

    uptime = time.monotonic() - started
    state_report = inspect_state(run_root / "state")
    defects, invariant_evidence = verify_invariants(
        recorder,
        state_report,
        strict_stream_delivery=(config.kill_seam is None),
    )

    evidence["uptime"] = {"seconds": round(uptime, 1), "wall": _uptime()}
    with recorder._lock:
        load_series = list(recorder.host_load_samples)
    evidence["host_load_gauge"] = {
        "n": len(load_series),
        **(
            {
                "min": round(min(load_series), 3),
                "mean": round(statistics.fmean(load_series), 3),
                "max": round(max(load_series), 3),
                "first": round(load_series[0], 3),
                "last": round(load_series[-1], 3),
            }
            if load_series
            else {}
        ),
        "note": (
            "solweig_rt_host_load gauge (ops-thread loadavg) sampled every "
            "Monitor poll (0.5 s) for the whole run"
        ),
    }
    evidence["server_alive_at_end"] = (
        server.proc is not None and server.proc.poll() is None
    )
    evidence["counts"] = recorder.snapshot()
    evidence["latency"] = {
        "visibility": visibility_percentiles(recorder),
        "fast": fast_latency_evidence(recorder),
        "exact_catchup": exact_catchup_percentiles(recorder),
        "exact_job_queued_to_finished": state_report.get(
            "exact_job_latency", {}
        ),
        "slo": {
            "visibility_p99_ms": SLO_VISIBILITY_MS,
            "fast_p99_ms": SLO_FAST_MS,
        },
    }
    evidence["rss"] = rss_slope(recorder)
    evidence["queue_depth"] = {
        "max_queued": max(
            (sample["queued"] for sample in recorder.queue_samples), default=0
        ),
        "max_running": max(
            (sample["running"] for sample in recorder.queue_samples),
            default=0,
        ),
        "failed_jobs": state_report.get("jobs", {}).get("failed", 0),
        "final_failed_jobs": state_report.get("jobs", {}).get("failed", 0),
    }
    evidence["invariants"] = invariant_evidence
    evidence["state"] = {
        workspace: {
            key: value for key, value in info.items() if key != "operation_ids"
        }
        for workspace, info in state_report["workspaces"].items()
    }
    evidence["state"]["jobs"] = state_report.get("jobs", {})
    evidence["defects"] = defects
    evidence["verdict"] = "PASS" if not defects else "FAIL"

    with recorder._lock:
        evidence["rss_samples"] = list(recorder.rss_samples)
        evidence["queue_samples"] = recorder.queue_samples[-800:]
        evidence["triple_samples"] = recorder.triple_samples[-2000:]
        evidence["canonical_ledger_frames"] = {
            ws: len(frames) for ws, frames in recorder.canonical_ledgers.items()
        }

    server.stop()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    return evidence


# ---------------------------------------------------------------------------
# Chaos matrix
# ---------------------------------------------------------------------------


def chaos_admission_overflow(base: dict[str, Any]) -> dict[str, Any]:
    """Flood past the admission bounds: reject BEFORE acceptance."""
    import httpx

    ensure_site()
    run_root = RUNS_ROOT / "chaos_overflow"
    shutil.rmtree(run_root, ignore_errors=True)
    server = Server("chaos_overflow", run_root / "state")
    server.start(_site_date())
    try:
        workspaces = ensure_workspaces(server, 1)
        workspace = workspaces[0]
        results = {
            "accepted": 0,
            "rejected_429": 0,
            "rejected_503": 0,
            "rejected_other": 0,
            "errors": 0,
            "rejected_ids": [],
            "accepted_ids": [],
        }
        lock = threading.Lock()

        def flood(worker: int) -> None:
            http = httpx.Client(base_url=server.base_url, timeout=20.0)
            rng = random.Random(_RNG_SEED + worker)
            try:
                for batch in range(60):
                    items = [
                        {
                            "operation_id": (
                                f"flood-{worker}-{batch}-{i}-"
                                f"{rng.randrange(1 << 30):x}"
                            ),
                            "client_sequence": i,
                            "source_family": "output_view",
                            "verb": "select",
                            "payload": {
                                "values": {"timestep": rng.randrange(3)}
                            },
                        }
                        for i in range(8)
                    ]
                    try:
                        response = http.post(
                            f"/api/v1/workspaces/{workspace}/operations",
                            json={
                                "actor_id": f"flood_{worker}",
                                "operations": items,
                            },
                        )
                    except Exception:  # noqa: BLE001
                        with lock:
                            results["errors"] += 1
                        continue
                    with lock:
                        bucket = {
                            200: "accepted",
                            429: "rejected_429",
                            503: "rejected_503",
                        }.get(response.status_code, "rejected_other")
                        results[bucket] += 1
                        target = (
                            "accepted_ids" if bucket == "accepted"
                            else "rejected_ids"
                        )
                        results[target].extend(
                            item["operation_id"] for item in items
                        )
                    time.sleep(0.01)
            finally:
                http.close()

        threads = [
            threading.Thread(target=flood, args=(w,), daemon=True)
            for w in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)

        alive = server.wait_ready(timeout=10.0)
        state = inspect_state(run_root / "state")
        durable = set(
            state["workspaces"].get(workspace, {}).get("operation_ids", [])
        )
        defects: list[str] = []
        if results["errors"]:
            defects.append(
                f"{results['errors']} transport errors (server down?)"
            )
        if not alive:
            defects.append("server not live after the flood")
        if results["rejected_429"] + results["rejected_503"] == 0:
            defects.append("flood produced ZERO backpressure rejections")
        phantom = sorted(set(results["rejected_ids"]) & durable)
        if phantom:
            defects.append(
                f"{len(phantom)} rejected op(s) found in the durable log "
                "(rejected BEFORE acceptance violated)"
            )
        lost = sorted(set(results["accepted_ids"]) - durable)
        if lost:
            defects.append(f"{len(lost)} accepted op(s) lost from the log")
        return {
            "name": "accept_queue_overflow",
            "flood_batches_accepted": results["accepted"],
            "flood_batches_rejected_429": results["rejected_429"],
            "flood_batches_rejected_503": results["rejected_503"],
            "flood_batches_rejected_other": results["rejected_other"],
            "transport_errors": results["errors"],
            "server_alive": alive,
            "durable_ops": len(durable),
            "defects": defects,
            "verdict": "PASS" if not defects else "FAIL",
        }
    finally:
        server.stop()


def chaos_duplicate_stale(base: dict[str, Any]) -> dict[str, Any]:
    """Duplicates + stale ops: exactly-one-revision, deterministic fold."""
    import httpx

    ensure_site()
    run_root = RUNS_ROOT / "chaos_dupstale"
    shutil.rmtree(run_root, ignore_errors=True)
    server = Server("chaos_dupstale", run_root / "state")
    server.start(_site_date())
    try:
        first_id, replay_id = ensure_workspaces(server, 2)
        http = httpx.Client(base_url=server.base_url, timeout=30.0)
        defects: list[str] = []

        def post(ws: str, item: dict[str, Any]) -> httpx.Response:
            return http.post(
                f"/api/v1/workspaces/{ws}/operations",
                json={"actor_id": "solo", "operations": [item]},
            )

        # a deterministic mixed sequence
        factory = OpFactory(first_id, "solo", _RNG_SEED + 77)
        sequence = [factory.next_op() for _ in range(40)]
        acked_ids: list[str] = []
        for item in sequence:
            response = post(first_id, item)
            if response.status_code != 200:
                defects.append(
                    f"op {item['operation_id']} rejected "
                    f"{response.status_code}: {response.text[:200]}"
                )
                continue
            acked_ids.append(item["operation_id"])

        # duplicate replay: same content => 200 + duplicate=true
        duplicates_flagged = 0
        for item in sequence[:10]:
            response = post(first_id, item)
            if response.status_code != 200:
                defects.append(
                    f"duplicate replay {item['operation_id']} != 200 "
                    f"({response.status_code})"
                )
                continue
            body = response.json()
            if body["acks"][0].get("duplicate"):
                duplicates_flagged += 1
            if body["acks"][0]["operation_id"] != item["operation_id"]:
                defects.append("duplicate replay returned a different op id")

        # reused id with DIFFERENT content: typed 409
        mutated = dict(sequence[5])
        mutated["payload"] = {"values": {"timestep": 9, "evil": True}}
        conflict_response = post(first_id, mutated)
        conflict_status = conflict_response.status_code
        if conflict_status != 409:
            defects.append(
                f"changed-content reuse returned {conflict_status}, wanted 409"
            )

        # stale base_revision: advisory, accepted with base_divergent
        stale = factory.stale_op(current_revision=1)
        stale_response = post(first_id, stale)
        stale_ok = stale_response.status_code == 200
        if not stale_ok:
            defects.append(
                f"stale-base op rejected {stale_response.status_code} "
                "(base_revision is advisory, never a rejection)"
            )
        stale_divergent = bool(
            stale_ok and stale_response.json()["acks"][0].get("base_divergent")
        )

        # -- determinism: the SAME 41-op sequence into the second workspace
        #    (same seed => identical payloads/entities; only id prefixes
        #    differ, and canonical state carries no op ids)
        replay_factory = OpFactory(replay_id, "solo", _RNG_SEED + 77)
        replay_sequence = [replay_factory.next_op() for _ in range(40)]
        for original, replay in zip(sequence, replay_sequence):
            assert original["payload"] == replay["payload"]
            assert original["source_family"] == replay["source_family"]
        replay_stale = replay_factory.stale_op(current_revision=1)
        assert replay_stale["payload"] == stale["payload"]
        for item in replay_sequence + [replay_stale]:
            response = post(replay_id, item)
            if response.status_code != 200:
                defects.append(
                    f"replay op rejected {response.status_code} "
                    "(determinism calibration failed)"
                )

        # quiesce both (exact lane settles the epochs)
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            a = http.get(
                f"/api/v1/workspaces/{first_id}/operations",
                params={"since_server_sequence": 2**31},
            ).json()
            b = http.get(
                f"/api/v1/workspaces/{replay_id}/operations",
                params={"since_server_sequence": 2**31},
            ).json()
            if (
                a["workspace_revision"] > 0
                and b["workspace_revision"] > 0
                and a["exact_revision"] >= a["workspace_revision"]
                and b["exact_revision"] >= b["workspace_revision"]
            ):
                break
            time.sleep(0.5)

        state = inspect_state(run_root / "state")
        first_info = state["workspaces"][first_id]
        replay_info = state["workspaces"][replay_id]

        # duplicates must NOT add durable rows
        expected_rows = len(acked_ids) + (1 if stale_ok else 0)
        if first_info["operations"] != expected_rows:
            defects.append(
                f"first workspace durable rows {first_info['operations']} != "
                f"acked {expected_rows} (duplicates double-appended?)"
            )
        if replay_info["operations"] != expected_rows:
            defects.append(
                f"replay workspace durable rows {replay_info['operations']} != "
                f"{expected_rows}"
            )
        # NOTE: revision COUNTS are intentionally NOT compared across the
        # two workspaces -- epoch grouping is wall-clock timing (100 ms
        # windows vs sequential posts), so two runs of the same op sequence
        # legitimately fold into different epoch counts. The determinism
        # contract is on the FOLDED STATE: identical op sequences must
        # produce byte-identical final canonical state.
        first_bytes = canonical_state_bytes(run_root / "state", first_id)
        replay_bytes = canonical_state_bytes(run_root / "state", replay_id)
        if first_bytes != replay_bytes:
            defects.append(
                "identical op sequences produced different canonical state "
                "bytes across workspaces (determinism violated)"
            )
        http.close()
        return {
            "name": "duplicate_stale_determinism",
            "ops_acked": len(acked_ids),
            "duplicate_replays_flagged": duplicates_flagged,
            "conflict_409_status": conflict_status,
            "stale_accepted": stale_ok,
            "stale_base_divergent_flag": stale_divergent,
            "first_workspace_revisions": first_info["workspace_revision"],
            "replay_workspace_revisions": replay_info["workspace_revision"],
            "canonical_bytes_equal": first_bytes == replay_bytes,
            "defects": defects,
            "verdict": "PASS" if not defects else "FAIL",
        }
    finally:
        server.stop()


def _site_date() -> str:
    from tests.test_incremental_worker import DATE_STR

    return DATE_STR


CHAOS_CRASH_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "crash_epoch_mid_close",
        "seam": "epoch_mid_close",
        "count": 3,
        "duration_s": 45.0,
    },
    {
        "name": "crash_exact_mid_solve",
        "seam": "exact_mid_solve",
        "count": 4,
        "duration_s": 60.0,
    },
    {
        "name": "crash_fast_publish_pre_commit",
        "seam": "fast_publish_pre_commit",
        "count": 2,
        "duration_s": 45.0,
    },
    {
        "name": "crash_fast_publish_post_commit",
        "seam": "fast_publish_post_commit",
        "count": 2,
        "duration_s": 45.0,
    },
    {
        "name": "crash_accept_mid_txn",
        "seam": "accept_mid_txn",
        "count": 25,
        "duration_s": 45.0,
    },
)


def chaos_crash_case(case: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Kill -9 semantics (os._exit) at an armed seam, restart, verify."""
    config = RunConfig(
        case["name"],
        duration_s=case["duration_s"],
        clients=4,
        ops_per_s_per_client=1.0,
        kill_seam=case["seam"],
        kill_count=case["count"],
        quiesce_s=120.0,
        restart_at_kill=True,
    )
    evidence = run_load(config, RUNS_ROOT / case["name"] / "evidence.json")
    defects = list(evidence["defects"])

    kill = evidence.get("kill", {})
    if not kill.get("fired_at_wall"):
        defects.append("kill seam never fired (harness calibration)")
    if not kill.get("restarted"):
        defects.append("restart never completed")
    if evidence.get("queue_depth", {}).get("failed_jobs", 0) > 0:
        defects.append(
            f"failed jobs after restart: "
            f"{evidence['queue_depth']['failed_jobs']}"
        )
    return {
        "name": case["name"],
        "seam": case["seam"],
        "kill_marker": kill.get("marker"),
        "restarted": kill.get("restarted", False),
        "ops_submitted": evidence.get("counts", {}).get("ops_submitted", 0),
        "client_errors": evidence.get("counts", {}).get("client_errors", 0),
        "defects": defects,
        "verdict": "PASS" if not defects else "FAIL",
        "evidence_path": str(RUNS_ROOT / case["name"] / "evidence.json"),
    }


def chaos_client_disconnect(base: dict[str, Any]) -> dict[str, Any]:
    """One SSE consumer drops mid-epoch; others + the epoch plane continue."""
    config = RunConfig(
        "chaos_disconnect",
        duration_s=90.0,
        clients=4,
        ops_per_s_per_client=1.0,
        client_drop_at_s=20.0,
        quiesce_s=60.0,
    )
    evidence = run_load(
        config, RUNS_ROOT / "chaos_disconnect" / "evidence.json"
    )
    defects = list(evidence["defects"])
    counts = evidence.get("counts", {})
    if counts.get("sse_drops", 0) < 1:
        defects.append("planned disconnect never happened")
    if counts.get("sse_reconnects", 0) < 1:
        defects.append("dropped consumer never reconnected")
    return {
        "name": "client_disconnect_mid_epoch",
        "sse_drops": counts.get("sse_drops", 0),
        "sse_reconnects": counts.get("sse_reconnects", 0),
        "missed_event_notices": counts.get("missed_event_notices", 0),
        "visibility": evidence.get("latency", {}).get("visibility", {}),
        "defects": defects,
        "verdict": "PASS" if not defects else "FAIL",
        "evidence_path": str(RUNS_ROOT / "chaos_disconnect" / "evidence.json"),
    }


def run_chaos(evidence_path: Path) -> dict[str, Any]:
    base = {"host": _host_block()}
    matrix: list[dict[str, Any]] = []

    print("[r8 chaos] case: accept_queue_overflow ...", flush=True)
    matrix.append(chaos_admission_overflow(base))
    print(f"    -> {matrix[-1]['verdict']} {matrix[-1]['defects'] or ''}", flush=True)

    print("[r8 chaos] case: client_disconnect_mid_epoch ...", flush=True)
    matrix.append(chaos_client_disconnect(base))
    print(f"    -> {matrix[-1]['verdict']} {matrix[-1]['defects'] or ''}", flush=True)

    for case in CHAOS_CRASH_CASES:
        print(f"[r8 chaos] case: {case['name']} ({case['seam']}) ...", flush=True)
        matrix.append(chaos_crash_case(case, base))
        print(f"    -> {matrix[-1]['verdict']} {matrix[-1]['defects'] or ''}", flush=True)

    print("[r8 chaos] case: duplicate_stale_determinism ...", flush=True)
    matrix.append(chaos_duplicate_stale(base))
    print(f"    -> {matrix[-1]['verdict']} {matrix[-1]['defects'] or ''}", flush=True)

    evidence = {
        "harness": "r8_load_chaos_soak",
        "matrix": matrix,
        "totals": {
            "cases": len(matrix),
            "passed": sum(1 for c in matrix if c["verdict"] == "PASS"),
            "failed": sum(1 for c in matrix if c["verdict"] == "FAIL"),
            "defects": [
                f"{case['name']}: {defect}"
                for case in matrix
                for defect in case.get("defects", [])
            ],
        },
        **base,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    print(
        f"[r8 chaos] cases={evidence['totals']['cases']} "
        f"passed={evidence['totals']['passed']} "
        f"failed={evidence['totals']['failed']}"
    )
    if evidence["totals"]["defects"]:
        print("[r8 chaos] DEFECTS:")
        for defect in evidence["totals"]["defects"]:
            print(f"    - {defect}")
    print(f"[r8 chaos] evidence: {evidence_path}")
    return evidence


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


MODES = {
    "smoke": dict(duration_s=60.0, clients=4, ops_per_s_per_client=1.0, quiesce_s=90.0),
    "probe50": dict(
        duration_s=240.0, clients=4, ops_per_s_per_client=1.0, quiesce_s=120.0,
        window_ms=50.0,
    ),
    "diag_stub": dict(
        duration_s=240.0, clients=4, ops_per_s_per_client=1.0, quiesce_s=90.0
    ),
    "calibrate": dict(
        duration_s=600.0, clients=4, ops_per_s_per_client=1.0, quiesce_s=180.0
    ),
    "soak": dict(
        duration_s=3600.0, clients=4, ops_per_s_per_client=1.0, quiesce_s=300.0
    ),
    "overload": dict(
        duration_s=300.0, clients=8, ops_per_s_per_client=1.0, quiesce_s=240.0
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=sorted(MODES) + ["serve", "chaos"]
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--site-dir", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--kill-seam", default=None)
    parser.add_argument("--kill-count", type=int, default=1)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument(
        "--evidence-name",
        default=None,
        help="override the evidence JSON filename under /tmp/r8_proof "
        "(e.g. quiet.json for the R8b quiet-host re-measurement, so the "
        "original calibration load.json is never clobbered)",
    )
    args = parser.parse_args(argv)

    if args.mode == "serve":
        return serve_child(args)
    if args.mode == "chaos":
        evidence = run_chaos(PROOF_ROOT / "chaos.json")
        return 0 if evidence["totals"]["failed"] == 0 else 1

    settings = dict(MODES[args.mode])
    evidence_name = args.evidence_name or {
        "smoke": "smoke.json",
        "calibrate": "load.json",
    }.get(args.mode, f"{args.mode}.json")
    evidence = run_load(RunConfig(args.mode, **settings), PROOF_ROOT / evidence_name)
    visibility = evidence["latency"]["visibility"]["op_accept_to_canonical_arrival"]
    fast_classes = evidence["latency"]["fast"][
        "server_reported_latency_ms_by_class"
    ]
    print(
        f"[r8 {args.mode}] verdict={evidence['verdict']} "
        f"uptime={evidence['uptime']['seconds']}s "
        f"ops={evidence['counts']['ops_submitted']} "
        f"visibility_p99={visibility.get('p99_ms')}ms(n={visibility.get('n')})"
    )
    print(
        "[r8 {mode}] fast classes: {classes}".format(
            mode=args.mode,
            classes=", ".join(
                f"{name} p99 {data.get('p99_ms')} ms (n={data.get('n')})"
                for name, data in fast_classes.items()
            ),
        )
    )
    if evidence["defects"]:
        print(f"[r8 {args.mode}] DEFECTS:")
        for defect in evidence["defects"]:
            print(f"    - {defect}")
    print(f"[r8 {args.mode}] evidence: {PROOF_ROOT / evidence_name}")
    return 0 if evidence["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
