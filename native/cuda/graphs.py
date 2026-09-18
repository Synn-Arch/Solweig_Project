# SPDX-License-Identifier: GPL-3.0-only
"""T13 — CUDA graphs: capacity buckets + liveness-fenced replay (TASKS
T13, DESIGN 13.4). Torch-free; numpy + ctypes only.

A graph is only replayable when its launch GEOMETRY does not depend on
the logical problem size:

* ``capacity_bucket(n)`` rounds the grid up to a power-of-two number of
  256-thread blocks — the bucket ladder;
* the logical ``n`` travels through DEVICE memory: the graph contains a
  memcpy node (captured from a pinned host word) that refreshes the
  device word on EVERY replay, so one instantiated graph serves every n
  in its bucket;
* the bucket kernel (sw_utci_sparse_bucket_kernel) guards ``k >= n``
  BEFORE any read or write — the padding lanes of the shared output
  buffer are never touched (T13 RED witness 2).

Liveness fence (T13 RED witness 1): the graph object holds STRONG
references to every captured device buffer; ``release_buffers()``
explicitly severs them, and any later replay raises
``GraphBufferFreedError`` instead of dereferencing freed memory.

Parity contract: graph-on == graph-off BIT-FOR-BIT. The bucket kernel is
the same lane math as ``sw_utci_sparse_kernel`` with n loaded from the
device word; tests/ultrafast/test_cuda_graphs.py pins both against the
T12 wrapper on straddling n (n % block != 0) and at exact bucket edges.
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np

__all__ = [
    "GraphError", "GraphBufferFreedError", "capacity_bucket", "GraphedUtci",
    "UtciGraphPool", "UTCI_DEFAULTS",
]


class GraphError(RuntimeError):
    pass


class GraphBufferFreedError(GraphError):
    """Replay attempted after the graph's captured buffers were freed."""


def _host():
    host = sys.modules.get("sw_cuda_host")
    if host is None:
        raise GraphError(
            "native/cuda/host.py must be loaded first (as 'sw_cuda_host')")
    return host


def _res_check(rc: int, lib) -> None:
    if rc != 0:
        msg = "?"
        try:
            msg = lib.sw_res_last_error().decode()
        except Exception:
            pass
        raise GraphError(f"CUDA ABI call failed (rc={rc}): {msg}")


#: lane-model defaults — MUST mirror the T12 wrapper constants (tests
#: assert the equality so a drift between the graph path and the plain
#: sparse path is impossible). Synced from the host module whenever it
#: is already loaded.
UTCI_DEFAULTS = {"torch_threads": 8, "grain": 32768}
try:  # import-time sync when the host copy is available
    _h0 = sys.modules.get("sw_cuda_host")
    if _h0 is not None:
        UTCI_DEFAULTS["torch_threads"] = _h0.UTCI_TORCH_THREADS
        UTCI_DEFAULTS["grain"] = _h0.UTCI_PAR_GRAIN
except Exception:
    pass


def capacity_bucket(n: int, block: int = 256) -> int:
    """Smallest power-of-two number of ``block``-thread blocks covering
    n, as an ELEMENT capacity (blocks * block)."""
    n = int(n)
    block = int(block)
    if n < 1 or block < 1:
        raise GraphError(f"capacity_bucket expects n>=1, block>=1 "
                         f"(got n={n}, block={block})")
    blocks = (n + block - 1) // block
    p = 1
    while p < blocks:
        p <<= 1
    return p * block


class GraphedUtci:
    """One instantiated UTCI sparse graph for a fixed capacity bucket.

    The captured stream sequence is exactly two nodes: the pinned n-word
    memcpy and the bucket kernel launch. Replays read the CURRENT pinned
    word, so ``run`` serves any n <= capacity without rebuilding.
    """

    BLOCK = 256

    def __init__(self, rt, capacity: int, *, torch_threads: int | None = None,
                 grain: int | None = None) -> None:
        host = _host()
        self._host = host
        self._rt = rt
        self._lib = rt.lib
        self._capacity = int(capacity)
        blocks = self._capacity // self.BLOCK
        if (self._capacity % self.BLOCK or blocks < 1
                or blocks & (blocks - 1)):
            raise GraphError(
                f"capacity must be BLOCK*2^k (got {capacity}); use "
                f"capacity_bucket(n)")
        self._torch_threads = int(
            torch_threads if torch_threads is not None
            else host.UTCI_TORCH_THREADS)
        self._grain = int(grain if grain is not None else host.UTCI_PAR_GRAIN)

        self._builds = 1
        self._replays = 0
        self._alive = True

        self._stream = ctypes.c_void_p()
        _res_check(self._lib.sw_stream_create(ctypes.byref(self._stream)),
                   self._lib)
        # captured device buffers (strong refs — the liveness fence)
        self._in_dev = [rt.alloc((self._capacity,), np.float32)
                        for _ in range(4)]
        self._out_dev = rt.alloc((self._capacity,), np.float32)
        self._n_dev = rt.alloc((1,), np.int64)
        # pinned staging (also captured: the n-word memcpy source)
        self._in_pin = []
        for _ in range(4):
            p = ctypes.c_void_p()
            _res_check(self._lib.sw_pinned_alloc(
                ctypes.byref(p), ctypes.c_longlong(self._capacity * 4)),
                self._lib)
            self._in_pin.append(p)
        self._out_pin = ctypes.c_void_p()
        _res_check(self._lib.sw_pinned_alloc(
            ctypes.byref(self._out_pin),
            ctypes.c_longlong(self._capacity * 4)), self._lib)
        self._n_pin = ctypes.c_void_p()
        _res_check(self._lib.sw_pinned_alloc(
            ctypes.byref(self._n_pin), ctypes.c_longlong(8)), self._lib)
        self._n_word = (ctypes.c_longlong * 1).from_address(
            self._n_pin.value)
        self._n_word[0] = 0
        self._event = ctypes.c_void_p()
        _res_check(self._lib.sw_event_create(ctypes.byref(self._event)),
                   self._lib)

        # capture: pinned n-word memcpy + bucket kernel launch
        _res_check(self._lib.sw_stream_sync(self._stream), self._lib)
        _res_check(self._lib.sw_graph_begin_capture(self._stream), self._lib)
        _res_check(self._lib.sw_copy_h2d_async(
            self._n_pin, ctypes.c_void_p(self._n_dev.ptr.value),
            ctypes.c_longlong(8), self._stream), self._lib)
        rc = self._lib.sw_res_utci_sparse_bucket(
            *[ctypes.c_void_p(d.ptr.value) for d in self._in_dev],
            ctypes.c_void_p(self._out_dev.ptr.value),
            ctypes.c_void_p(self._n_dev.ptr.value),
            ctypes.c_int(self._torch_threads),
            ctypes.c_int(self._grain),
            ctypes.c_uint(blocks), self._stream)
        if rc != 0:
            _res_check(rc, self._lib)
        self._graph = ctypes.c_void_p()
        _res_check(self._lib.sw_graph_end_capture(self._stream,
                                                  ctypes.byref(self._graph)),
                   self._lib)
        self._exec = ctypes.c_void_p()
        _res_check(self._lib.sw_graph_instantiate(self._graph,
                                                  ctypes.byref(self._exec)),
                   self._lib)

    # -- staging helpers --------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    def stats(self) -> dict:
        return {"capacity": self._capacity, "builds": self._builds,
                "replays": self._replays, "alive": self._alive}

    def _assert_alive(self) -> None:
        if not self._alive:
            raise GraphBufferFreedError(
                "graph replay refused: captured buffers were freed "
                "(release_buffers) — the exec must never outlive them")

    def _stage_inputs(self, vecs) -> int:
        n = int(np.asarray(vecs[0]).size)
        for v in vecs:
            if int(np.asarray(v).size) != n:
                raise GraphError("utci input vectors must share one length")
        if n > self._capacity:
            raise GraphError(
                f"n={n} exceeds bucket capacity {self._capacity}; use "
                f"capacity_bucket(n)")
        for v, pin in zip(vecs, self._in_pin):
            arr = np.ascontiguousarray(v, dtype=np.float32)
            view = np.ctypeslib.as_array(
                (ctypes.c_float * self._capacity).from_address(pin.value))
            np.copyto(view[:n], arr.reshape(-1)[:n])
        return n

    def _replay(self, n: int) -> None:
        self._assert_alive()
        self._n_word[0] = n
        for pin, dev in zip(self._in_pin, self._in_dev):
            _res_check(self._lib.sw_copy_h2d_async(
                pin, ctypes.c_void_p(dev.ptr.value),
                ctypes.c_longlong(self._capacity * 4), self._stream),
                self._lib)
        _res_check(self._lib.sw_graph_launch(self._exec, self._stream),
                   self._lib)
        self._replays += 1

    # -- public API ---------------------------------------------------------------

    def run(self, ta, rh, tmrt, va) -> np.ndarray:
        """Replay and return out[:n] (graph-on path; bit-equal to
        host.run_utci_sparse)."""
        self._assert_alive()  # BEFORE staging: freed pinned buffers must
        # never be written, not even on the input path
        n = self._stage_inputs((ta, rh, tmrt, va))
        self._replay(n)
        _res_check(self._lib.sw_copy_d2h_async(
            ctypes.c_void_p(self._out_dev.ptr.value), self._out_pin,
            ctypes.c_longlong(self._capacity * 4), self._stream), self._lib)
        _res_check(self._lib.sw_event_record(self._event, self._stream),
                   self._lib)
        _res_check(self._lib.sw_event_sync(self._event), self._lib)
        out_view = np.ctypeslib.as_array(
            (ctypes.c_float * self._capacity).from_address(
                self._out_pin.value))
        return out_view[:n].copy()

    def run_into(self, ta, rh, tmrt, va, out_full: np.ndarray) -> None:
        """Replay into a CAPACITY-sized caller buffer. The caller's
        ``out_full`` is uploaded first (so padding poison is on-device),
        the replay overwrites only out[:n], and the WHOLE capacity is
        downloaded back — padding lanes must survive bit-identically."""
        self._assert_alive()  # freed buffers must not be staged into
        arr = np.ascontiguousarray(out_full, dtype=np.float32)
        if arr.size != self._capacity:
            raise GraphError(
                f"out_full must have exactly capacity={self._capacity} "
                f"elements, got {arr.size}")
        pin_view = np.ctypeslib.as_array(
            (ctypes.c_float * self._capacity).from_address(
                self._out_pin.value))
        np.copyto(pin_view, arr.reshape(-1))
        n = self._stage_inputs((ta, rh, tmrt, va))
        _res_check(self._lib.sw_copy_h2d_async(
            self._out_pin, ctypes.c_void_p(self._out_dev.ptr.value),
            ctypes.c_longlong(self._capacity * 4), self._stream), self._lib)
        self._n_word[0] = n
        for pin_p, dev in zip(self._in_pin, self._in_dev):
            _res_check(self._lib.sw_copy_h2d_async(
                pin_p, ctypes.c_void_p(dev.ptr.value),
                ctypes.c_longlong(self._capacity * 4), self._stream),
                self._lib)
        _res_check(self._lib.sw_graph_launch(self._exec, self._stream),
                   self._lib)
        self._replays += 1
        _res_check(self._lib.sw_copy_d2h_async(
            ctypes.c_void_p(self._out_dev.ptr.value), self._out_pin,
            ctypes.c_longlong(self._capacity * 4), self._stream), self._lib)
        _res_check(self._lib.sw_event_record(self._event, self._stream),
                   self._lib)
        _res_check(self._lib.sw_event_sync(self._event), self._lib)
        np.copyto(arr.reshape(-1), pin_view)

    def release_buffers(self) -> None:
        """Free every captured device + pinned buffer (the graph exec
        survives). Any later replay raises GraphBufferFreedError."""
        if not self._alive:
            return
        self._assert_alive()
        for dev in (*self._in_dev, self._out_dev, self._n_dev):
            dev.free()
        for p in (*self._in_pin, self._out_pin, self._n_pin):
            _res_check(self._lib.sw_pinned_free(p), self._lib)
        self._alive = False

    def destroy(self) -> None:
        if self._exec.value is not None:
            _res_check(self._lib.sw_graph_exec_destroy(self._exec), self._lib)
            self._exec = ctypes.c_void_p(None)
        if self._graph.value is not None:
            _res_check(self._lib.sw_graph_destroy(self._graph), self._lib)
            self._graph = ctypes.c_void_p(None)
        if self._alive:
            self.release_buffers()
        if self._event.value is not None:
            _res_check(self._lib.sw_event_destroy(self._event), self._lib)
            self._event = ctypes.c_void_p(None)
        if self._stream.value is not None:
            _res_check(self._lib.sw_stream_sync(self._stream), self._lib)
            _res_check(self._lib.sw_stream_destroy(self._stream), self._lib)
            self._stream = ctypes.c_void_p(None)

    def __del__(self):  # best effort; destroy() is the sanctioned path
        try:
            self.destroy()
        except Exception:
            pass


class UtciGraphPool:
    """Capacity-bucketed cache of GraphedUtci instances: one instantiated
    graph per bucket, selected by capacity_bucket(n)."""

    def __init__(self, rt) -> None:
        self._rt = rt
        self._pool: dict[int, GraphedUtci] = {}

    def run(self, ta, rh, tmrt, va, *, torch_threads: int | None = None,
            grain: int | None = None) -> np.ndarray:
        n = int(np.asarray(ta).size)
        cap = capacity_bucket(n)
        gu = self._pool.get(cap)
        if gu is None:
            gu = GraphedUtci(self._rt, cap, torch_threads=torch_threads,
                             grain=grain)
            self._pool[cap] = gu
        return gu.run(ta, rh, tmrt, va)

    def stats(self) -> dict:
        builds = sum(g.stats()["builds"] for g in self._pool.values())
        replays = sum(g.stats()["replays"] for g in self._pool.values())
        return {"buckets": len(self._pool), "builds": builds,
                "replays": replays}

    def destroy(self) -> None:
        for gu in self._pool.values():
            gu.destroy()
        self._pool.clear()

    def __del__(self):  # best effort; destroy() is the sanctioned path
        try:
            self.destroy()
        except Exception:
            pass
