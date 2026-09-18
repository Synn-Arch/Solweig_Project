# SPDX-License-Identifier: GPL-3.0-only
"""T13 GATE 2 — CUDA graphs: capacity buckets + liveness-fenced replay
(TASKS T13, DESIGN 13.4).

Graph replay only becomes legal once the launch GEOMETRY is decoupled
from the logical problem size: capacity buckets fix grid dims once, the
logical ``n`` travels through DEVICE memory (a memcpy node inside the
graph refreshes it from pinned host memory per replay), so one
instantiated graph serves every workload in its bucket.

The two RED witnesses at this surface:

* freed pointer — a captured buffer that has been freed must never be
  replayed; the graph object keeps strong refs and REFUSES once they
  are released (witness 1);
* padding targets — lanes at or beyond the logical ``n`` (the bucket's
  padding region, which shares the output buffer with real lanes) must
  never be written into results (witness 2).

Parity: graph-on must equal graph-off BIT-FOR-BIT — the bucket UTCI
kernel replays the exact elementwise lane math of the plain sparse
kernel on straddling n (n % block != 0, and n == capacity), and the
resident rad_day graph path must reproduce the T12 wrapper bits.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from sw_cuda_harness import (  # noqa: E402
    compare_bits,
    load_host_module,
    require_canonical_runtime,
)

REPO_ROOT = ULTRA_DIR.parents[1]
NATIVE_CUDA = REPO_ROOT / "native" / "cuda"

_T13_MODULES: dict[str, object] = {}


def load_t13(name: str):
    """Load a native/cuda T13 module ONCE, sharing the harness's host
    module copy (single CudaRuntime identity)."""
    if name in _T13_MODULES:
        return _T13_MODULES[name]
    host = load_host_module()
    sys.modules.setdefault("sw_cuda_host", host)
    reg = f"sw_cuda_{name}"
    mod = sys.modules.get(reg)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            reg, NATIVE_CUDA / f"{name}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[reg] = mod
        spec.loader.exec_module(mod)
    _T13_MODULES[name] = mod
    return mod


@pytest.fixture(scope="module")
def rt():
    return require_canonical_runtime()


@pytest.fixture(scope="module")
def host():
    return load_host_module()


@pytest.fixture(scope="module")
def graphs():
    return load_t13("graphs")


@pytest.fixture(scope="module")
def residency():
    return load_t13("residency")


def _utci_vectors(n: int, seed: int = 0x713):
    rng = np.random.default_rng(seed)
    ta = rng.uniform(-5.0, 38.0, n).astype(np.float32)
    rh = rng.uniform(20.0, 95.0, n).astype(np.float32)
    tmrt = rng.uniform(10.0, 75.0, n).astype(np.float32)
    va = rng.uniform(0.15, 12.0, n).astype(np.float32)
    return tuple(np.ascontiguousarray(v) for v in (ta, rh, tmrt, va))


# ---------------------------------------------------------------------------
# capacity ladder (pure)
# ---------------------------------------------------------------------------


class TestCapacityBucket:
    def test_ladder_is_pow2_blocks_and_covers(self, graphs):
        cb = graphs.capacity_bucket
        for block in (256, 512):
            prev = 0
            for n in list(range(1, 4 * block + 3)) + [10 ** 6]:
                cap = cb(n, block=block)
                assert cap % block == 0
                assert cap >= n, (n, cap)
                # power-of-two number of blocks
                blocks = cap // block
                assert blocks & (blocks - 1) == 0, (n, blocks)
                assert cap >= prev  # monotone
                # minimality: previous ladder step must NOT cover n
                if blocks > 1:
                    assert cap // 2 < n, (n, cap)
                prev = cap

    def test_expected_ladder_values(self, graphs):
        cb = graphs.capacity_bucket
        assert cb(1) == 256
        assert cb(256) == 256
        assert cb(257) == 512
        assert cb(1000) == 1024
        assert cb(1024) == 1024
        assert cb(1025) == 2048


# ---------------------------------------------------------------------------
# graph-on == graph-off, straddling n + logical-n updates within a bucket
# ---------------------------------------------------------------------------


class TestUtciGraphParity:
    def test_straddling_n_matches_plain_sparse_bits(self, rt, host, graphs):
        n = 1000  # 1024-capacity bucket, 24 padding lanes
        vecs = _utci_vectors(n)
        want = host.run_utci_sparse(rt, *vecs)
        gu = graphs.GraphedUtci(rt, graphs.capacity_bucket(n))
        got = gu.run(*vecs)
        msg, _ = compare_bits(want, got, f"utci-graph-n{n}")
        assert msg == "PASS", msg
        assert gu.stats()["capacity"] == 1024
        gu.destroy()

    def test_defaults_match_host_lane_model(self, graphs, host):
        # the graph path's lane-model defaults must mirror the T12
        # wrapper constants exactly
        defaults = graphs.UTCI_DEFAULTS
        assert defaults["torch_threads"] == host.UTCI_TORCH_THREADS
        assert defaults["grain"] == host.UTCI_PAR_GRAIN

    def test_logical_n_updates_within_bucket_without_rebuild(self, rt, host,
                                                             graphs):
        gu = graphs.GraphedUtci(rt, graphs.capacity_bucket(1024))
        for n in (1000, 513, 1024, 1, 1024):  # all inside the 1024 bucket
            vecs = _utci_vectors(n, seed=n)
            want = host.run_utci_sparse(rt, *vecs)
            got = gu.run(*vecs)
            msg, _ = compare_bits(want, got, f"utci-graph-n{n}")
            assert msg == "PASS", msg
        st = gu.stats()
        assert st["builds"] == 1, st
        assert st["replays"] == 5, st
        gu.destroy()

    def test_oversized_n_refused(self, rt, graphs):
        gu = graphs.GraphedUtci(rt, 512)
        vecs = _utci_vectors(513)
        with pytest.raises(graphs.GraphError):
            gu.run(*vecs)
        gu.destroy()

    def test_pool_builds_one_graph_per_bucket(self, rt, host, graphs):
        pool = graphs.UtciGraphPool(rt)
        for n in (1000, 513, 900):  # all inside the 1024 bucket
            vecs = _utci_vectors(n, seed=n)
            want = host.run_utci_sparse(rt, *vecs)
            msg, _ = compare_bits(want, pool.run(*vecs), f"pool-n{n}")
            assert msg == "PASS", msg
        st = pool.stats()
        assert st["buckets"] == 1 and st["builds"] == 1, st
        # distinct buckets: 2000 -> 2048, 3000 -> 4096
        for n in (2000, 3000, 2900):
            vecs = _utci_vectors(n, seed=n)
            want = host.run_utci_sparse(rt, *vecs)
            msg, _ = compare_bits(want, pool.run(*vecs), f"pool-n{n}")
            assert msg == "PASS", msg
        st = pool.stats()
        assert st["buckets"] == 3, st  # 1024, 2048, 4096
        assert st["builds"] == 3, st
        pool.destroy()


# ---------------------------------------------------------------------------
# RED witness 2: padding targets written into results
# ---------------------------------------------------------------------------


class TestPaddingFence:
    def test_padding_lanes_never_written(self, rt, host, graphs):
        """The bucket's output buffer is CAPACITY-sized; only out[:n] is
        the result. Poison the tail and demand it survives the replay —
        a kernel that clamps/wraps lanes past n instead of guarding
        writes into the padding region gets caught here."""
        n, cap = 1000, 1024
        gu = graphs.GraphedUtci(rt, cap)
        vecs = _utci_vectors(n, seed=0xBEE)
        poison = np.full(cap, np.nan, dtype=np.float32)
        poison.view(np.uint32)[:] = 0xDEADBEEF
        gu.run_into(*vecs, poison)
        want = host.run_utci_sparse(rt, *vecs)
        msg, _ = compare_bits(want, poison[:n], "pad-head")
        assert msg == "PASS", msg
        tail = poison[n:].view(np.uint32)
        assert int(tail.min()) == 0xDEADBEEF and int(tail.max()) == 0xDEADBEEF, (
            f"padding region clobbered: tail bits "
            f"[0x{int(tail.min()):08x}, 0x{int(tail.max()):08x}]"
        )
        gu.destroy()


# ---------------------------------------------------------------------------
# RED witness 1: graph using a freed pointer
# ---------------------------------------------------------------------------


class TestFreedPointerFence:
    def test_replay_after_buffer_free_refused(self, rt, host, graphs):
        cap = graphs.capacity_bucket(1000)
        gu = graphs.GraphedUtci(rt, cap)
        vecs = _utci_vectors(1000)
        msg, _ = compare_bits(host.run_utci_sparse(rt, *vecs), gu.run(*vecs),
                              "pre-free")
        assert msg == "PASS", msg
        gu.release_buffers()  # frees every captured device buffer
        vecs2 = _utci_vectors(1000, seed=7)
        with pytest.raises(graphs.GraphBufferFreedError):
            gu.run(*vecs2)  # the graph exec outlived its buffers: refuse
        gu.destroy()


# ---------------------------------------------------------------------------
# rad_day graph path (resident runner, replay amortization)
# ---------------------------------------------------------------------------


def _capture_dir() -> Path:
    cap = os.environ.get("SW_T12_T08_CAPTURE", "")
    if not cap or not Path(cap).is_dir():
        pytest.skip("t08 capture fixtures not available (SW_T12_T08_CAPTURE)")
    return Path(cap)


def _day_bundle(cap: Path, t: int) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from solweig_core.numba_cpu.radiation import (
        rad_bundle_from_capture,
        rad_state_from_capture,
        rad_static_from_capture,
    )
    from sw_rad_bundle import build_rad_day_bundle

    st = rad_static_from_capture(cap)
    t_in = rad_bundle_from_capture(cap, t)
    state = rad_state_from_capture(cap, t, rows=st.rows, cols=st.cols)
    assert t_in.is_day
    return build_rad_day_bundle(st, t_in, state)


OUT_NAMES = ("tmrt", "kdown", "kup", "ldown", "lup", "ke", "ks", "kw", "kn",
             "le", "ls", "lw", "ln", "ksidei", "tgout", "lside", "ksided",
             "drad", "kside", "n_lup", "n_e", "n_s", "n_w", "n_n", "n_tg")


class TestRadDayGraph:
    def test_replay_matches_wrapper_bits_and_amortizes(self, rt, host,
                                                       residency, graphs):
        cap = _capture_dir()
        b = _day_bundle(cap, 12)
        want = host.run_rad_day(rt, b)

        from test_cuda_residency import (  # noqa: E402  (sibling gate)
            RAD_DAY_STATIC,
            RAD_DAY_VOLATILE,
        )
        runner = residency.ResidentRadRunner(rt, chunk_rows=128)
        runner.bind({k: b[k] for k in RAD_DAY_STATIC}, static=True)
        runner.bind({k: b[k] for k in RAD_DAY_VOLATILE})

        for rep in range(3):
            lease = runner.run_day(b, download=OUT_NAMES, use_graph=True)
            lease.wait()
            got = lease.publish()
            for name in OUT_NAMES:
                msg, _ = compare_bits(want[name], got[name],
                                      f"rad-day-graph-rep{rep}:{name}")
                assert msg == "PASS", msg
        st = runner.graph_stats()
        assert st["builds"] == 1, st  # one capture amortized over replays
        assert st["replays"] == 3, st
        runner.close()

    def test_cancel_token_with_graph_refused(self, rt, residency):
        """Graph replay cannot be cancelled mid-flight: refusing the
        combination is the honest contract (fenced, not silent)."""
        cap = _capture_dir()
        b = _day_bundle(cap, 12)
        from test_cuda_residency import (  # noqa: E402
            RAD_DAY_STATIC,
            RAD_DAY_VOLATILE,
        )
        runner = residency.ResidentRadRunner(rt, chunk_rows=128)
        runner.bind({k: b[k] for k in RAD_DAY_STATIC}, static=True)
        runner.bind({k: b[k] for k in RAD_DAY_VOLATILE})
        token = residency.CancelToken()
        with pytest.raises(ValueError):
            runner.run_day(b, download=("tmrt",), use_graph=True,
                           cancel_token=token)
        runner.close()
