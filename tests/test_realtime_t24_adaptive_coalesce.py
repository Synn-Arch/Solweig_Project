# SPDX-License-Identifier: GPL-3.0-only
"""T24: adaptive epoch/debounce scheduling (DESIGN §710), coverage tests.

DESIGN §710 (the scheduling-task mandate): "Single-user latency를 위해 기존
100 ms epoch/debounce를 줄이는 방안은 별도 scheduling task로 평가한다.
Compute target이 100 ms인데 대기 자체가 100 ms이면 target에 도달할 수
없다. Epoch를 adaptive하게 줄이더라도 reducer determinism, idempotency,
conflict resolution은 변하지 않아야 한다."

The debounce half of that mandate ships here behind the SEPARATE flag
``SOLWEIG_RT_ADAPTIVE_COALESCE`` (TASKS T15's "별도 flag와 coverage test"
discipline; the epoch-cadence half already shipped behind T15's own
selected-time flag). Coverage:

* **flag-off pin** — the dispatch wait is exactly the configured window,
  one call, and the burst probes are never consulted (byte-identical
  default behavior);
* **flag-on policy** — isolated job → the 50 ms contract floor; every burst
  indicator (queued work behind, nonterminal peer, open epoch) keeps the
  full window; floor never exceeds the configured window;
* **timing** — an isolated edit dispatches at the floor, not the window
  (deterministic lower bound on the flag-off control);
* **§710 invariants** — the post-sleep supersede layers still act inside
  the shortened window, duplicate submits stay idempotent, and the same
  accepted edit sequence publishes byte-identical results with the flag
  off vs on (the window never feeds semantics);
* **wiring** — ``create_app`` reads the env flag once at construction,
  exactly like the T15 selected-time flag.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.app import create_app
from solweig_gpu.server.jobs import (
    ADAPTIVE_COALESCE_ENV_FLAG,
    ADAPTIVE_COALESCE_FLOOR_MS,
    JobRunner,
    adaptive_coalescing_enabled,
)
from solweig_gpu.server.realtime.epochs import EPOCH_TICK_MIN_MS

from tests.test_server_api import (
    SCENARIO0,
    FakeSolver,
    add_tree,
    make_app,
    make_tree,
    wait_for_job,
)


# ---------------------------------------------------------------------------
# flag surface
# ---------------------------------------------------------------------------


class TestAdaptiveCoalesceFlag:
    def test_flag_parsing(self) -> None:
        off = {"": False, "0": False, "false": False, "no": False, "off": False}
        on = {"1": True, "true": True, "YES": True, " on ": True}
        for raw, expected in {**off, **on}.items():
            assert adaptive_coalescing_enabled({ADAPTIVE_COALESCE_ENV_FLAG: raw}) is (
                expected
            ), f"{raw!r} parsed wrong"
        assert adaptive_coalescing_enabled({}) is False
        assert adaptive_coalescing_enabled({"OTHER": "1"}) is False

    def test_floor_is_the_contract_epoch_floor(self) -> None:
        """One admitted floor, not an invented number: the shrink target is
        the service contract's epoch-duration minimum (the same bound T15's
        adaptive epoch hint uses)."""
        assert ADAPTIVE_COALESCE_FLOOR_MS == EPOCH_TICK_MIN_MS == 50.0

    def test_default_runner_is_flag_off(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = JobRunner(
            app.state.context, solver_factory=lambda _c: solver
        )
        assert runner.adaptive_coalesce is False
        assert runner.adaptive_coalesce_floor_ms == ADAPTIVE_COALESCE_FLOOR_MS


# ---------------------------------------------------------------------------
# effective-window policy (unit level, no timing)
# ---------------------------------------------------------------------------


class TestEffectiveWindow:
    @staticmethod
    def _runner(app, solver, **kwargs) -> JobRunner:
        return JobRunner(
            app.state.context,
            solver_factory=lambda _c: solver,
            coalescing_window_ms=500.0,
            **kwargs,
        )

    @staticmethod
    def _queued_job(client: TestClient, tree_id: str, base: int) -> str:
        submitted = add_tree(client, make_tree(tree_id), base=base)
        assert submitted.status_code == 202, submitted.text
        return submitted.json()["job_id"]

    def test_flag_off_returns_window_without_burst_probes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE PIN (byte-identical default): flag off → the configured
        window, and the open-epoch probe — the only store read unique to
        the adaptive policy — is never consulted."""
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = self._runner(app, solver)  # adaptive_coalesce default off
        store = app.state.context.store
        probed: list[str] = []
        monkeypatch.setattr(
            store, "open_epochs", lambda ws: probed.append(ws) or []
        )
        assert runner._effective_coalescing_window_ms("job_any") == 500.0
        assert probed == [], "flag-off path consulted the burst probe"

    def test_flag_off_zero_window_stays_zero(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = JobRunner(
            app.state.context,
            solver_factory=lambda _c: solver,
            coalescing_window_ms=0.0,
        )
        assert runner.adaptive_coalesce is False
        assert runner._effective_coalescing_window_ms("job_any") == 0.0

    def test_flag_on_isolated_job_uses_floor(
        self, tmp_path: Path
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = self._runner(app, solver, adaptive_coalesce=True)
        with TestClient(app) as client:
            scenario_id = SCENARIO0(client)
            job_id = self._queued_job(client, "iso-tree", base=0)
            store = app.state.context.store
            job = store.require_job(job_id)
            assert job.scenario_id == scenario_id
            # No queued work behind, no nonterminal peer, no open epoch
            # (legacy edits do not open epochs; the funnel is gated on
            # pre-existing realtime operations): the isolated-job branch.
            assert store.open_epochs(scenario_id) == []
            assert runner._effective_coalescing_window_ms(job_id) == 50.0

    def test_flag_on_floor_never_exceeds_window(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = JobRunner(
            app.state.context,
            solver_factory=lambda _c: solver,
            coalescing_window_ms=30.0,
            adaptive_coalesce=True,
        )
        with TestClient(app) as client:
            job_id = self._queued_job(client, "tiny-tree", base=0)
            assert runner._effective_coalescing_window_ms(job_id) == 30.0

    def test_flag_on_queued_work_behind_keeps_full_window(
        self, tmp_path: Path
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = self._runner(app, solver, adaptive_coalesce=True)
        runner._queue.put("job_behind_shim")  # never started: harmless
        try:
            assert (
                runner._effective_coalescing_window_ms("job_any") == 500.0
            ), "work queued behind must keep the full window (burst)"
        finally:
            runner._queue.get_nowait()

    def test_flag_on_nonterminal_peer_keeps_full_window(
        self, tmp_path: Path
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = self._runner(app, solver, adaptive_coalesce=True)
        with TestClient(app) as client:
            scenario_id = SCENARIO0(client)
            job1 = self._queued_job(client, "peer-old", base=0)
            # The newer edit makes job1 superseded at the store layer, but
            # job2 stays nonterminal for the same scenario: a peer ahead.
            job2 = self._queued_job(client, "peer-new", base=1)
            store = app.state.context.store
            assert store.require_job(job2).scenario_id == scenario_id
            # job1's decision sees job2 as a nonterminal peer → full window.
            assert runner._effective_coalescing_window_ms(job1) == 500.0
            # job2 itself has no peer (job1 is terminal): isolated → floor.
            assert runner._effective_coalescing_window_ms(job2) == 50.0

    def test_flag_on_open_epoch_keeps_full_window_and_stays_read_only(
        self, tmp_path: Path
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = self._runner(app, solver, adaptive_coalesce=True)
        store = app.state.context.store
        with TestClient(app) as client:
            scenario_id = SCENARIO0(client)
            job_id = self._queued_job(client, "epoch-tree", base=0)
            # An OPEN epoch = an operation was accepted within the last
            # epoch window (realtime activity): the honest burst signal.
            with store._write() as conn:
                conn.execute(
                    "INSERT INTO realtime_epochs "
                    "(workspace_id, epoch_id, status, opened_at) "
                    "VALUES (?, 0, 'open', ?)",
                    (scenario_id, "2026-01-01T00:00:00.000000Z"),
                )
            assert runner._effective_coalescing_window_ms(job_id) == 500.0
            # The policy only READS the epoch plane (conflict resolution is
            # untouched by the debounce): the row is exactly as inserted.
            rows = store.open_epochs(scenario_id)
            assert [r.epoch_id for r in rows] == [0]
            assert rows[0].status == "open"

    def test_flag_on_missing_job_row_is_conservative(
        self, tmp_path: Path
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = self._runner(app, solver, adaptive_coalesce=True)
        assert (
            runner._effective_coalescing_window_ms("job_gone") == 500.0
        ), "a vanished job must read as burst (the post-sleep scan aborts)"


# ---------------------------------------------------------------------------
# dispatch timing (deterministic bounds only)
# ---------------------------------------------------------------------------


def _last_job_id(app) -> str:
    jobs = app.state.context.store.nonterminal_jobs()
    assert jobs, "no nonterminal job to submit"
    return jobs[-1].job_id


class TestDispatchLatency:
    @staticmethod
    def _dispatch_after(app, solver, window_ms: float = 500.0, **kwargs) -> float:
        """Submit one isolated edit through a fresh runner; return seconds
        from submit to the solver being entered."""
        runner = JobRunner(
            app.state.context,
            solver_factory=lambda _c: solver,
            coalescing_window_ms=window_ms,
            **kwargs,
        )
        solver.entered.clear()
        runner.start()
        try:
            t0 = time.monotonic()
            runner.submit(_last_job_id(app))
            assert solver.entered.wait(timeout=10.0), "solver never started"
            return time.monotonic() - t0
        finally:
            runner.stop()

    def test_isolated_edit_dispatches_at_floor_not_window(
        self, tmp_path: Path
    ) -> None:
        """§710's point, measured: an isolated edit must not pay the full
        500 ms debounce under the flag. The flag-off control's lower bound
        is deterministic (``Event.wait(0.5)`` cannot return early)."""
        solver_on = FakeSolver()
        app = make_app(tmp_path, solver_on, start_worker=False)
        with TestClient(app) as client:
            SCENARIO0(client)
            add_tree(client, make_tree("fast-tree"), base=0)
            adaptive_s = self._dispatch_after(
                app, solver_on, adaptive_coalesce=True
            )
        assert adaptive_s < 0.35, (
            f"adaptive dispatch took {adaptive_s:.3f}s — the isolated job "
            "paid (most of) the full 500 ms window"
        )

        solver_off = FakeSolver()
        app2 = make_app(tmp_path / "off", solver_off, start_worker=False)
        with TestClient(app2) as client:
            SCENARIO0(client)
            add_tree(client, make_tree("slow-tree"), base=0)
            fixed_s = self._dispatch_after(app2, solver_off)
        assert fixed_s >= 0.5, (
            f"flag-off dispatch returned in {fixed_s:.3f}s — the fixed "
            "window's deterministic lower bound was violated"
        )

    def test_supersede_during_shortened_window_still_aborts(
        self, tmp_path: Path
    ) -> None:
        """The post-sleep supersede scan must keep acting on edits that
        land DURING the (now shorter) adaptive wait — the §710 invariants
        live in those layers, not in the wait's duration."""
        solver = FakeSolver()
        # Floor injected at 400 ms so the "edit lands during the window"
        # setup is reliable on a loaded host, while still exercising the
        # adaptive branch (400 < the configured 500). Both jobs flow
        # through the app's own runner (the route's submit path).
        app = make_app(
            tmp_path, solver, start_worker=False, coalescing_window_ms=500.0,
            adaptive_coalesce=True,
        )
        # Test injection (create_app deliberately exposes only the flag):
        # raise the floor to 400 ms so the "edit lands during the window"
        # setup stays reliable on a loaded host, while still exercising the
        # adaptive branch (400 < the configured 500).
        app.state.runner.adaptive_coalesce_floor_ms = 400.0
        with TestClient(app) as client:
            SCENARIO0(client)
            first = add_tree(client, make_tree("pending-tree"), base=0)
            job1 = first.json()["job_id"]
            app.state.runner.start()
            try:
                time.sleep(0.1)  # inside the 400 ms adaptive wait
                second = add_tree(client, make_tree("during-tree"), base=1)
                job2 = second.json()["job_id"]
                detail1 = wait_for_job(
                    client, job1, statuses=("superseded",), timeout=15.0
                )
                detail2 = wait_for_job(client, job2, timeout=15.0)
            finally:
                app.state.runner.stop()
        assert detail1["status"] == "superseded"
        assert detail2["status"] == "complete"

    def test_zero_window_with_adaptive_dispatches_immediately(
        self, tmp_path: Path
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        with TestClient(app) as client:
            SCENARIO0(client)
            add_tree(client, make_tree("nowait-tree"), base=0)
            elapsed = self._dispatch_after(
                app, solver, window_ms=0.0, adaptive_coalesce=True
            )
        assert elapsed < 0.35, (
            f"window=0 with adaptive on still delayed dispatch "
            f"{elapsed:.3f}s"
        )


# ---------------------------------------------------------------------------
# flag-off pin: the run loop's wait call itself
# ---------------------------------------------------------------------------


class TestFlagOffWaitPin:
    def test_wait_call_is_exactly_the_configured_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Byte-identical default behavior, pinned at the call site: with
        the flag off the run loop performs ONE ``_stop.wait`` of exactly
        ``coalescing_window_ms / 1000`` seconds and never consults the
        open-epoch burst probe."""
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        store = app.state.context.store
        probed: list[str] = []
        monkeypatch.setattr(
            store, "open_epochs", lambda ws: probed.append(ws) or []
        )
        runner = JobRunner(
            app.state.context,
            solver_factory=lambda _c: solver,
            coalescing_window_ms=500.0,
        )
        waits: list[float | None] = []
        real_wait = runner._stop.wait

        def spy(timeout: float | None = None) -> bool:
            waits.append(timeout)
            return real_wait(timeout)

        monkeypatch.setattr(runner._stop, "wait", spy)
        with TestClient(app) as client:
            SCENARIO0(client)
            submitted = add_tree(client, make_tree("pin-tree"), base=0)
            job_id = submitted.json()["job_id"]
            runner.start()
            try:
                wait_for_job(client, job_id, timeout=15.0)
            finally:
                runner.stop()
        assert waits == [0.5], (
            f"flag-off dispatch wait calls were {waits!r}: the fixed-window "
            "call pattern changed"
        )
        assert probed == [], "flag-off dispatch consulted the burst probe"

    def test_flag_on_wait_call_is_the_floor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The adaptive counterpart of the pin: an isolated job's single
        wait call is exactly the floor."""
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = JobRunner(
            app.state.context,
            solver_factory=lambda _c: solver,
            coalescing_window_ms=500.0,
            adaptive_coalesce=True,
        )
        waits: list[float | None] = []
        real_wait = runner._stop.wait

        def spy(timeout: float | None = None) -> bool:
            waits.append(timeout)
            return real_wait(timeout)

        monkeypatch.setattr(runner._stop, "wait", spy)
        with TestClient(app) as client:
            SCENARIO0(client)
            submitted = add_tree(client, make_tree("floor-tree"), base=0)
            job_id = submitted.json()["job_id"]
            runner.start()
            try:
                wait_for_job(client, job_id, timeout=15.0)
            finally:
                runner.stop()
        assert waits == [0.05], (
            f"adaptive isolated-job wait calls were {waits!r}: expected "
            "exactly one wait at the 50 ms contract floor"
        )


# ---------------------------------------------------------------------------
# §710 invariants across the flag
# ---------------------------------------------------------------------------


class TestSemanticsAcrossFlag:
    @staticmethod
    def _two_edit_outcomes(tmp_path: Path, adaptive: bool):
        """Run the same accepted two-edit sequence through a runner with
        the given flag; return (final status, published payload bytes).

        The first job's individual outcome (superseded vs complete) is a
        TIMING fact — the supersede layers act on arrival order, and the
        window value legitimately shifts which jobs they catch — so the
        semantic equality asserted downstream is on the FINAL converged
        state: the newest revision's published bytes."""
        solver = FakeSolver()
        sub = tmp_path / ("on" if adaptive else "off")
        # Both jobs flow through the app's own runner (the route's submit
        # path), so the flag under test is the one the deployed wiring
        # would carry.
        app = make_app(
            sub, solver, start_worker=False, coalescing_window_ms=500.0,
            adaptive_coalesce=adaptive,
        )
        runner = app.state.runner
        runner.start()
        try:
            with TestClient(app) as client:
                scenario_id = SCENARIO0(client)
                first = add_tree(client, make_tree("seq-a"), base=0)
                # Land the second edit while the first job is still inside
                # its wait (500 ms off / 50 ms on): the supersede layer,
                # not the window value, must decide the outcome.
                time.sleep(0.02)
                second = add_tree(client, make_tree("seq-b"), base=1)
                job1, job2 = first.json()["job_id"], second.json()["job_id"]
                wait_for_job(
                    client, job1, statuses=("complete", "superseded"),
                    timeout=15.0,
                )
                detail2 = wait_for_job(client, job2, timeout=15.0)
                result = app.state.context.store.get_result(
                    scenario_id, detail2["target_scene_version"]
                )
                assert result is not None
                return detail2["status"], result.payload_bytes()
        finally:
            runner.stop()

    def test_same_sequence_same_outcomes_and_bytes_across_flag(
        self, tmp_path: Path
    ) -> None:
        """Reducer/publication semantics are window-invariant: the same
        accepted sequence converges to the same final status and a
        BYTE-identical published payload with the flag off and on."""
        off = self._two_edit_outcomes(tmp_path, adaptive=False)
        on = self._two_edit_outcomes(tmp_path, adaptive=True)
        assert off[0] == on[0] == "complete"
        assert off[1] == on[1], (
            "published payload bytes differ between flag-off and flag-on "
            "runs of the same accepted sequence — the debounce leaked into "
            "publication semantics"
        )

    def test_duplicate_submit_is_idempotent_under_adaptive(
        self, tmp_path: Path
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = JobRunner(
            app.state.context,
            solver_factory=lambda _c: solver,
            coalescing_window_ms=500.0,
            adaptive_coalesce=True,
        )
        runner.start()
        try:
            with TestClient(app) as client:
                SCENARIO0(client)
                submitted = add_tree(client, make_tree("dup-tree"), base=0)
                job_id = submitted.json()["job_id"]
                runner.submit(job_id)
                runner.submit(job_id)  # the redelivery
                detail = wait_for_job(client, job_id, timeout=15.0)
        finally:
            runner.stop()
        assert detail["status"] == "complete"
        assert len(solver.calls) == 1, (
            f"a redelivered submit solved {len(solver.calls)} times: "
            "idempotency broke under the adaptive debounce"
        )


# ---------------------------------------------------------------------------
# app wiring
# ---------------------------------------------------------------------------


class TestAppWiring:
    def test_create_app_reads_env_flag_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        solver = FakeSolver()
        sites = {"site": {"cache_dir": _site_cache(tmp_path)}}
        monkeypatch.delenv(ADAPTIVE_COALESCE_ENV_FLAG, raising=False)
        app_off = create_app(
            state_root=tmp_path / "s1", sites=sites,
            solver_factory=lambda _c: solver, start_worker=False,
        )
        assert app_off.state.runner.adaptive_coalesce is False

        monkeypatch.setenv(ADAPTIVE_COALESCE_ENV_FLAG, "1")
        app_on = create_app(
            state_root=tmp_path / "s2", sites=sites,
            solver_factory=lambda _c: solver, start_worker=False,
        )
        assert app_on.state.runner.adaptive_coalesce is True

        # An explicit parameter overrides the environment (deployments that
        # pin the behavior must not be flipped by a stray env var).
        app_pinned = create_app(
            state_root=tmp_path / "s3", sites=sites,
            solver_factory=lambda _c: solver, start_worker=False,
            adaptive_coalesce=False,
        )
        assert app_pinned.state.runner.adaptive_coalesce is False

    def test_create_app_flag_off_advertises_the_same_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The advertised coalescing window (route bodies) is untouched by
        the flag: clients see the configured window either way."""
        monkeypatch.delenv(ADAPTIVE_COALESCE_ENV_FLAG, raising=False)
        solver = FakeSolver()
        sites = {"site": {"cache_dir": _site_cache(tmp_path)}}
        for flag in ("", "1"):
            if flag:
                monkeypatch.setenv(ADAPTIVE_COALESCE_ENV_FLAG, flag)
            app = create_app(
                state_root=tmp_path / f"s-{flag or 'off'}", sites=sites,
                solver_factory=lambda _c: solver, start_worker=False,
                requests_per_minute_per_ip=None, edits_per_minute=None,
            )
            assert app.state.coalescing_window_ms == 500.0
            assert app.state.runner.coalescing_window_ms == 500.0


def _site_cache(tmp_path: Path) -> Path:
    from tests.test_server_api import make_site_cache

    return make_site_cache(tmp_path)
