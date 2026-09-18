# SPDX-License-Identifier: GPL-3.0-only
"""T13 — GPU residency pool for the radiation kernels (TASKS T13,
DESIGN 13.3). Torch-free; numpy + ctypes only.

The resident runner keeps every STATIC site input on the device (the
~178 MB — diffsh plus the packed cubes and albedo/emissivity planes —
that dominated T12's 26 ms rad_day H2D) and re-uploads only the CHANGED
ROW CHUNKS of per-timestep inputs. Change detection is BIT-EXACT: chunk
bytes are compared as uint8 views (T17: via libc ``memcmp`` — for uint8
views ``memcmp == 0`` is exactly ``np.array_equal``, so NaN payload and
±0.0 differences always transfer; a value-equality shortcut could
silently desynchronize signed zeros and break raw-bit parity).

Async discipline (the two RED witnesses this module fences):

* publish fence — outputs download asynchronously into PINNED host
  slots; an ``OutputLease`` may only be published after its completion
  event fires (``PublishPendingError`` otherwise);
* stale-lease fence — a slot re-downloaded by a later run holds THAT
  run's bits; an older unpublished lease is refused (``LeaseStaleError``)
  even though both events have fired;
* reuse fence — a slot whose event is still pending (e.g. from a
  cancelled lease) is never reacquired (``BufferPendingError``) until
  the event completes.

Profile discipline: the static-set registry is keyed by
``(profile_id, digest)`` — a legacy runtime can never hit a canonical
runner's resident buffers.

Upload sources are PAGEABLE numpy chunks: per CUDA semantics a pageable
async H2D returns once the source has been consumed, so caller arrays
are safe to reuse after ``bind`` returns. Downloads land in pinned
memory because they are genuinely async and event-fenced.

Graphs (DESIGN 13.4): ``run_day(..., use_graph=True)`` captures the
kernel launch into a CUDA graph keyed by the SCALAR DIGEST (the 26
kernel scalars are baked into the launch parameters; only a digest-equal
timestep can replay it). The runner owns the captured device buffers, so
the graph's strong references live exactly as long as the runner
(``close()`` destroys graph execs BEFORE freeing any buffer).
"""
from __future__ import annotations

import ctypes
import hashlib
import sys

import numpy as np

__all__ = [
    "ResidencyError", "PublishPendingError", "BufferPendingError",
    "LeaseStaleError", "CancelToken", "OutputLease", "ResidentRadRunner",
    "register_static_runner", "static_runner",
    "device_alloc_count", "device_alloc_bytes", "mem_info",
]

# T17: libc memcmp as the changed-chunk predicate. For same-shape uint8
# views memcmp==0 is EXACTLY np.array_equal's predicate (uint8 element
# equality IS byte equality — NaN payloads and signed zeros are bytes,
# compared as bytes), at memcmp speed instead of numpy bool temporaries.
_libc = ctypes.CDLL(None)
_memcmp = _libc.memcmp
_memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_memcmp.restype = ctypes.c_int


def _chunk_bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-exact equality of two same-shape chunks (the committed
    ``np.array_equal(a.view(uint8), b.view(uint8))`` predicate, with a
    memcmp fast path for the C-contiguous chunks ``bind`` produces)."""
    av = a.view(np.uint8)
    bv = b.view(np.uint8)
    if av.flags.c_contiguous and bv.flags.c_contiguous:
        return _memcmp(av.ctypes.data, bv.ctypes.data, av.nbytes) == 0
    return bool(np.array_equal(av, bv))


class ResidencyError(RuntimeError):
    pass


class PublishPendingError(ResidencyError):
    """Publish attempted before the async output event completed."""


class LeaseStaleError(ResidencyError):
    """Publish attempted on a lease whose output slot has been
    re-downloaded by a later run — the slot now holds THAT run's bits
    (both events have fired, so the event fence cannot see this; the
    per-slot generation is the fence)."""


class BufferPendingError(ResidencyError):
    """Acquisition of a buffer whose outstanding event is still pending."""


def _host():
    """The T12 host module (single copy — tests register it in
    sys.modules as 'sw_cuda_host' so DeviceArray identity is shared)."""
    host = sys.modules.get("sw_cuda_host")
    if host is None:
        raise ResidencyError(
            "native/cuda/host.py must be loaded first (as 'sw_cuda_host')")
    return host


def _check(rc: int, lib) -> None:
    if rc != 0:
        msg = "?"
        try:
            msg = lib.sw_res_last_error().decode()
        except Exception:
            pass
        raise ResidencyError(f"CUDA ABI call failed (rc={rc}): {msg}")


# ---------------------------------------------------------------------------
# module-level accounting (canonical library counters are monotonic)
# ---------------------------------------------------------------------------


def _canonical_rt():
    host = _host()
    rt = host.get_runtime(host.CANONICAL_CUDA_V1)
    if rt is None:
        raise ResidencyError("canonical CUDA runtime unavailable")
    return rt


def device_alloc_count(rt=None) -> int:
    rt = rt or _canonical_rt()
    return int(rt.lib.sw_alloc_count())


def device_alloc_bytes(rt=None) -> int:
    rt = rt or _canonical_rt()
    rt.lib.sw_alloc_bytes.restype = ctypes.c_longlong
    return int(rt.lib.sw_alloc_bytes())


def mem_info(rt=None) -> tuple[int, int]:
    rt = rt or _canonical_rt()
    free_ = ctypes.c_longlong()
    total_ = ctypes.c_longlong()
    _check(rt.lib.sw_mem_info(ctypes.byref(free_), ctypes.byref(total_)),
           rt.lib)
    return int(free_.value), int(total_.value)


# ---------------------------------------------------------------------------
# static-set registry — (profile_id, digest) keyed, never cross-profile
# ---------------------------------------------------------------------------

_STATIC_REGISTRY: dict[tuple[str, str], "ResidentRadRunner"] = {}


def register_static_runner(rt, digest: str, runner: "ResidentRadRunner"):
    _STATIC_REGISTRY[(rt.profile_id, str(digest))] = runner


def static_runner(rt, digest: str) -> "ResidentRadRunner":
    key = (rt.profile_id, str(digest))
    runner = _STATIC_REGISTRY.get(key)
    if runner is None:
        raise ResidencyError(
            f"no static runner registered for profile={rt.profile_id!r} "
            f"digest={digest!r} (a different profile NEVER hits another "
            f"profile's resident set)")
    return runner


class CancelToken:
    """Advisory cancellation. Cancelling cannot un-launch GPU work; it
    fences the hazard that actually exists: the abandoned lease's output
    slot stays non-reacquirable until its event completes."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class _Bound:
    """One resident input: device buffer + host mirror (the bits last
    uploaded, used for bit-exact changed-chunk detection)."""

    __slots__ = ("dev", "mirror", "dtype")

    def __init__(self, dev, mirror: np.ndarray, dtype) -> None:
        self.dev = dev
        self.mirror = mirror
        self.dtype = dtype


class _Slot:
    """One output name's pinned host download slot + fence event."""

    __slots__ = ("pinned", "view", "event", "row_bytes", "rows", "generation")

    def __init__(self, pinned, view, event, row_bytes, rows) -> None:
        self.pinned = pinned      # c_void_p (pinned plane)
        self.view = view          # numpy (rows, cols) f32 view of pinned
        self.event = event        # c_void_p cudaEvent_t
        self.row_bytes = row_bytes
        self.rows = rows
        self.generation = 0       # bumped on every download into this slot


class OutputLease:
    """Fenced view of one run's async outputs. ``wait()`` blocks on the
    slot events; ``publish()`` REFUSES until they have fired; ``release()``
    abandons the data (the slot stays fenced until the event completes)."""

    def __init__(self, runner: "ResidentRadRunner", names, window,
                 generations: dict) -> None:
        self._runner = runner
        self._names = tuple(names)
        self._window = window
        self._generations = dict(generations)
        self._published = False
        self._released = False

    @property
    def names(self) -> tuple:
        return self._names

    def _slots(self):
        for n in self._names:
            slot = self._runner._slots.get(n)
            if slot is None:
                raise ResidencyError(f"no slot for output {n!r}")
            yield slot

    def wait(self) -> None:
        if self._released:
            raise ResidencyError("lease already released")
        for slot in self._slots():
            if slot.event is not None:
                _check(self._runner._lib.sw_event_sync(slot.event),
                       self._runner._lib)

    def publish(self) -> dict:
        if self._released:
            raise ResidencyError("lease already released")
        if self._published:
            raise ResidencyError("lease already published")
        # stale-lease fence FIRST: a later download of the same slot means
        # the slot holds that run's bits — publishing them under this
        # lease's identity is a stale revision, whatever the event state
        for n in self._names:
            slot = self._runner._slots.get(n)
            if slot is not None and slot.generation != \
                    self._generations.get(n):
                raise LeaseStaleError(
                    f"output slot {n!r} was re-downloaded by a later run "
                    f"(slot generation {slot.generation}, this lease "
                    f"{self._generations.get(n)}); publish the CURRENT "
                    f"lease instead")
        for slot in self._slots():
            if slot.event is not None:
                rc = self._runner._lib.sw_event_query(slot.event)
                if rc == 1:
                    raise PublishPendingError(
                        f"output event still pending — wait() before "
                        f"publish()")
                if rc != 0:
                    _check(-rc if rc < 0 else rc, self._runner._lib)
        out = {}
        if self._names:
            r0 = 0 if self._window is None else int(self._window[0])
            for n in self._names:
                slot = self._runner._slots[n]
                r1 = (slot.rows if self._window is None
                      else int(self._window[1]))
                out[n] = slot.view[r0:r1].copy()
        self._published = True
        return out

    def release(self) -> None:
        """Abandon without publishing. The slot's event stays recorded:
        the pool will refuse reacquisition (BufferPendingError) until it
        completes — the early-reuse fence."""
        self._released = True


class ResidentRadRunner:
    """Device-resident radiation runner: bind static + volatile inputs
    once, then run day/night timesteps moving only changed row chunks
    H2D and requested output windows D2H."""

    def __init__(self, rt, chunk_rows: int = 128) -> None:
        host = _host()
        self._host = host
        self._rt = rt
        self._lib = rt.lib
        self._chunk_rows = int(chunk_rows)
        if self._chunk_rows < 1:
            raise ResidencyError("chunk_rows must be >= 1")
        self._inputs: dict[str, _Bound] = {}
        self._outs: dict[str, object] = {}
        self._slots: dict[str, _Slot] = {}
        self._graphs: dict[str, tuple] = {}
        self._h2d_bytes = 0
        self._graph_builds = 0
        self._graph_replays = 0
        self._closed = False
        self._stream = ctypes.c_void_p()
        _check(self._lib.sw_stream_create(ctypes.byref(self._stream)),
               self._lib)

    # -- properties ----------------------------------------------------------

    @property
    def h2d_bytes(self) -> int:
        return self._h2d_bytes

    def graph_stats(self) -> dict:
        return {"builds": self._graph_builds,
                "replays": self._graph_replays}

    def sync(self) -> None:
        self._assert_open()
        _check(self._lib.sw_stream_sync(self._stream), self._lib)

    def close(self) -> None:
        if self._closed:
            return
        self.sync()
        # graphs first: an exec must never outlive its captured buffers
        for graph, exec_ in self._graphs.values():
            _check(self._lib.sw_graph_exec_destroy(exec_), self._lib)
            _check(self._lib.sw_graph_destroy(graph), self._lib)
        self._graphs.clear()
        for slot in self._slots.values():
            _check(self._lib.sw_event_destroy(slot.event), self._lib)
            _check(self._lib.sw_pinned_free(slot.pinned), self._lib)
        self._slots.clear()
        for dev in self._outs.values():
            dev.free()
        self._outs.clear()
        for bound in self._inputs.values():
            bound.dev.free()
        self._inputs.clear()
        _check(self._lib.sw_stream_destroy(self._stream), self._lib)
        self._closed = True

    def __del__(self):  # best effort; close() is the sanctioned path
        try:
            self.close()
        except Exception:
            pass

    def _assert_open(self) -> None:
        if self._closed:
            raise ResidencyError("runner is closed")

    # -- binding / changed-chunk upload ---------------------------------------

    def _input_dtype(self, name: str):
        for tbl in (self._host._RAD_DAY_INPUTS, self._host._RAD_NIGHT_INPUTS):
            for n, dt in tbl:
                if n == name:
                    return dt
        raise ResidencyError(f"unknown radiation input {name!r}")

    def bind(self, arrays: dict, static: bool = False) -> None:
        """Upload new arrays, moving ONLY the changed row chunks (bit
        comparison per chunk: uint8-view equality; NaN payload and ±0
        differences always transfer). ``static`` is advisory (marks the
        key as scene-invariant for registry bookkeeping)."""
        self._assert_open()
        for name, arr in arrays.items():
            dt = self._input_dtype(name)
            new = np.ascontiguousarray(arr, dtype=dt)
            bound = self._inputs.get(name)
            if bound is None:
                dev = self._rt.alloc(new.shape, dt)
                self._h2d_whole(dev, new)
                self._h2d_bytes += int(new.nbytes)
                self._inputs[name] = _Bound(dev, new, dt)
                continue
            if new.shape != bound.mirror.shape:
                raise ResidencyError(
                    f"input {name!r} changed shape "
                    f"{bound.mirror.shape} -> {new.shape}; rebind requires "
                    f"a fresh runner (fixed site geometry)")
            self._upload_changed(dev=bound.dev, mirror=bound.mirror,
                                 new=new)
            bound.mirror = new

    def _h2d_whole(self, dev, new: np.ndarray) -> None:
        _check(self._lib.sw_copy_h2d_async(
            new.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(dev.ptr.value),
            ctypes.c_longlong(new.nbytes), self._stream), self._lib)

    def _upload_changed(self, dev, mirror: np.ndarray, new: np.ndarray):
        if new.ndim < 2:
            # 1-D tables: single chunk
            if not _chunk_bytes_equal(mirror, new):
                self._h2d_whole(dev, new)
                self._h2d_bytes += int(new.nbytes)
            return
        rows = new.shape[0]
        item_span = int(np.prod(new.shape[1:], dtype=np.int64))
        row_bytes = int(item_span * new.dtype.itemsize)
        for r0 in range(0, rows, self._chunk_rows):
            r1 = min(r0 + self._chunk_rows, rows)
            a = mirror[r0:r1]
            b = new[r0:r1]
            if _chunk_bytes_equal(a, b):
                continue
            height = r1 - r0
            _check(self._lib.sw_copy2d_h2d_async(
                b.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(dev.ptr.value + r0 * row_bytes),
                ctypes.c_longlong(row_bytes), ctypes.c_longlong(height),
                self._stream), self._lib)
            self._h2d_bytes += row_bytes * height

    # -- output slots ----------------------------------------------------------

    def _ensure_outs(self, names, rows: int, cols: int) -> None:
        for name in names:
            if name not in self._outs:
                self._outs[name] = self._rt.alloc((rows, cols), np.float32)

    def _slot(self, name: str, rows: int, cols: int) -> _Slot:
        slot = self._slots.get(name)
        if slot is None:
            n = rows * cols
            pinned = ctypes.c_void_p()
            _check(self._lib.sw_pinned_alloc(
                ctypes.byref(pinned), ctypes.c_longlong(n * 4)), self._lib)
            view = np.ctypeslib.as_array(
                (ctypes.c_float * n).from_address(pinned.value)
            ).reshape(rows, cols)
            event = ctypes.c_void_p()
            _check(self._lib.sw_event_create(ctypes.byref(event)), self._lib)
            slot = _Slot(pinned, view, event, cols * 4, rows)
            self._slots[name] = slot
        return slot

    def _download(self, names, window, rows: int, cols: int) -> dict:
        row_bytes = cols * 4
        r0 = 0 if window is None else int(window[0])
        r1 = rows if window is None else int(window[1])
        if not (0 <= r0 < r1 <= rows):
            raise ResidencyError(f"bad download window {window!r}")
        generations: dict[str, int] = {}
        for name in names:
            slot = self._slot(name, rows, cols)
            if slot.event is not None:
                rc = self._lib.sw_event_query(slot.event)
                if rc == 1:
                    raise BufferPendingError(
                        f"output slot {name!r} still has a pending event "
                        f"(cancelled or unpublish lease); sync first")
                if rc != 0:
                    _check(rc, self._lib)
            slot.generation += 1
            generations[name] = slot.generation
            dev = self._outs[name]
            _check(self._lib.sw_copy2d_d2h_async(
                ctypes.c_void_p(dev.ptr.value + r0 * row_bytes),
                ctypes.c_void_p(slot.pinned.value + r0 * row_bytes),
                ctypes.c_longlong(row_bytes), ctypes.c_longlong(r1 - r0),
                self._stream), self._lib)
            _check(self._lib.sw_event_record(slot.event, self._stream),
                   self._lib)
        return generations

    # -- day -------------------------------------------------------------------

    def _day_arg_list(self, bundle: dict, stream):
        host = self._host
        args = []
        for name, _dt in host._RAD_DAY_INPUTS:
            bound = self._inputs.get(name)
            if bound is None:
                raise ResidencyError(f"input {name!r} never bound")
            args.append(ctypes.c_void_p(bound.dev.ptr.value))
        for name in host._RAD_DAY_OUTPUTS + host._RAD_DAY_NEXT:
            args.append(ctypes.c_void_p(self._outs[name].ptr.value))
        for key, kind in host._RAD_DAY_SCALAR_ARGS:
            if kind == "f":
                args.append(ctypes.c_float(np.float32(bundle[key])))
            elif kind == "d":
                args.append(ctypes.c_double(np.float64(bundle[key])))
            else:
                args.append(ctypes.c_int(int(bundle[key])))
        args.append(ctypes.c_void_p(stream.value))
        return args

    def _day_scalar_digest(self, bundle: dict) -> str:
        h = hashlib.sha256()
        for key, kind in self._host._RAD_DAY_SCALAR_ARGS:
            if kind == "f":
                h.update(np.float32(bundle[key]).tobytes())
            elif kind == "d":
                h.update(np.float64(bundle[key]).tobytes())
            else:
                h.update(np.int64(int(bundle[key])).tobytes())
        h.update(str(id(self)).encode())  # residency generation
        return h.hexdigest()

    def run_day(self, bundle: dict, download=(), window=None,
                cancel_token: CancelToken | None = None,
                use_graph: bool = False) -> OutputLease:
        self._assert_open()
        if use_graph and cancel_token is not None:
            raise ValueError(
                "graph replay cannot be cancelled mid-flight; refuse the "
                "combination (fenced, not silent)")
        if cancel_token is not None and cancel_token.cancelled:
            raise ResidencyError("cancel token already cancelled at dispatch")
        host = self._host
        rows, cols = int(bundle["rows"]), int(bundle["cols"])
        self._ensure_outs(host._RAD_DAY_OUTPUTS + host._RAD_DAY_NEXT,
                          rows, cols)
        if use_graph:
            digest = self._day_scalar_digest(bundle)
            entry = self._graphs.get(digest)
            if entry is None:
                # capture: kernel only (uploads/downloads stay outside the
                # graph); sync first so capture starts from an idle stream
                self.sync()
                _check(self._lib.sw_graph_begin_capture(self._stream),
                       self._lib)
                rc = self._lib.sw_res_rad_day(*self._day_arg_list(
                    bundle, self._stream))
                if rc != 0:
                    _check(rc, self._lib)
                graph = ctypes.c_void_p()
                _check(self._lib.sw_graph_end_capture(self._stream,
                                                      ctypes.byref(graph)),
                       self._lib)
                exec_ = ctypes.c_void_p()
                _check(self._lib.sw_graph_instantiate(graph,
                                                      ctypes.byref(exec_)),
                       self._lib)
                self._graphs[digest] = (graph, exec_)
                self._graph_builds += 1
                entry = (graph, exec_)
            _check(self._lib.sw_graph_launch(entry[1], self._stream),
                   self._lib)
            self._graph_replays += 1
        else:
            rc = self._lib.sw_res_rad_day(*self._day_arg_list(bundle,
                                                              self._stream))
            if rc != 0:
                _check(rc, self._lib)
        generations = self._download(download, window, rows, cols)
        return OutputLease(self, download, window, generations)

    # -- night -------------------------------------------------------------------

    def run_night(self, bundle: dict, download=(), window=None,
                  cancel_token: CancelToken | None = None) -> OutputLease:
        self._assert_open()
        if cancel_token is not None and cancel_token.cancelled:
            raise ResidencyError("cancel token already cancelled at dispatch")
        host = self._host
        rows, cols = int(bundle["rows"]), int(bundle["cols"])
        self._ensure_outs(host._RAD_NIGHT_OUTPUTS, rows, cols)
        args = []
        for name, _dt in host._RAD_NIGHT_INPUTS:
            bound = self._inputs.get(name)
            if bound is None:
                raise ResidencyError(f"input {name!r} never bound")
            args.append(ctypes.c_void_p(bound.dev.ptr.value))
        for name in host._RAD_NIGHT_OUTPUTS:
            args.append(ctypes.c_void_p(self._outs[name].ptr.value))
        args += [ctypes.c_double(np.float64(bundle[k]))
                 for k in host._RAD_NIGHT_F64_SCALARS]
        args += [ctypes.c_int(int(bundle[k]))
                 for k in host._RAD_NIGHT_INT_SCALARS]
        args.append(ctypes.c_void_p(self._stream.value))
        rc = self._lib.sw_res_rad_night(*args)
        if rc != 0:
            _check(rc, self._lib)
        generations = self._download(download, window, rows, cols)
        return OutputLease(self, download, window, generations)
