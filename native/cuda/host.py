# SPDX-License-Identifier: GPL-3.0-only
"""Torch-free ctypes host for the ultrafast CUDA kernels (TASKS T12).

Loads the nvcc-built ``libswcuda.so`` (native/cuda/build.py) and owns
EVERY device buffer (explicit ownership — no hidden allocations). The
profile contract (DESIGN.ko.md 3.1) is enforced here:

* ``canonical_cuda_v1`` — the strict build (``--fmad=false --ftz=false
  --prec-div=true --prec-sqrt=true``). The ONLY profile allowed to claim
  canonical parity vs ``canonical_cpu_v1``.
* ``legacy_cuda_v1`` — the characterization build (aten-default-like
  FMA contraction, UNCERTIFIED). Never mixed with canonical caches: each
  profile loads its own library object from its own build namespace, and
  buffers refuse to cross runtimes (RuntimeMismatch).

Skip-vs-fail discipline: :func:`require_runtime` raises
:class:`SwCudaUnavailable` when no device/nvcc/library exists. Tests
translate that into ``pytest.skip`` LOCALLY; on the GPU host the test
runner exports ``SW_REQUIRE_CUDA=1`` which turns the same condition into
a hard failure — a gate can never silently skip where a GPU exists.
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
_BUILD_PY = PKG_DIR / "build.py"

CANONICAL_CUDA_V1 = "canonical_cuda_v1"
LEGACY_CUDA_V1 = "legacy_cuda_v1"

#: profile -> build variant (native/cuda/build.py VARIANTS)
PROFILE_VARIANTS = {
    CANONICAL_CUDA_V1: "canonical",
    LEGACY_CUDA_V1: "legacy_cuda_v1",
}

_STRICT_REQUIREMENTS = ("--fmad=false", "--ftz=false", "--prec-div=true",
                        "--prec-sqrt=true")

_library_cache: dict[tuple[str, str], tuple[ctypes.CDLL, str]] = {}
_cache_lock = threading.Lock()


class SwCudaUnavailable(RuntimeError):
    """No CUDA device / no nvcc / no build — tests may skip (never on the
    GPU host, where SW_REQUIRE_CUDA=1 promotes this to failure)."""


class RuntimeMismatch(ValueError):
    """Cross-profile buffer operation refused (DESIGN 3.1: no mixing)."""


class CudaError(RuntimeError):
    """A C-ABI call returned non-zero."""


def cuda_required() -> bool:
    return os.environ.get("SW_REQUIRE_CUDA", "") == "1"


def _check(rc: int, lib: ctypes.CDLL) -> None:
    if rc != 0:
        raise CudaError(
            f"libswcuda call failed (rc={rc}): {lib.sw_last_error().decode()}"
        )


class DeviceArray:
    """One owned device buffer. Created only through :class:`CudaRuntime`."""

    __slots__ = ("runtime", "ptr", "size_bytes", "shape", "dtype")

    def __init__(self, runtime: "CudaRuntime", size_bytes: int,
                 shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.runtime = runtime
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.size_bytes = int(size_bytes)
        ptr = ctypes.c_void_p()
        _check(runtime.lib.sw_alloc(ctypes.byref(ptr), ctypes.c_longlong(
            self.size_bytes)), runtime.lib)
        self.ptr = ptr

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def _owner(self, runtime: "CudaRuntime") -> ctypes.c_void_p:
        if runtime is not self.runtime:
            raise RuntimeMismatch(
                "device buffer belongs to "
                f"{self.runtime.profile_id!r}, not {runtime.profile_id!r} "
                "(profiles never share buffers or caches)"
            )
        return ctypes.c_void_p(self.ptr.value)

    def from_numpy(self, host: np.ndarray) -> None:
        arr = np.ascontiguousarray(host)
        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype)
        if arr.nbytes != self.size_bytes:
            raise ValueError(
                f"host array is {arr.nbytes} bytes, buffer is {self.size_bytes}"
            )
        rt = self.runtime
        _check(rt.lib.sw_copy_h2d(
            arr.ctypes.data_as(ctypes.c_void_p), self._owner(rt),
            ctypes.c_longlong(arr.nbytes)), rt.lib)

    def to_numpy(self) -> np.ndarray:
        rt = self.runtime
        out = np.empty(self.shape, dtype=self.dtype)
        _check(rt.lib.sw_copy_d2h(
            self._owner(rt), out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_longlong(out.nbytes)), rt.lib)
        return out

    def free(self) -> None:
        if self.ptr.value is not None:
            rt = self.runtime
            _check(rt.lib.sw_free(self._owner(rt)), rt.lib)
            self.ptr = ctypes.c_void_p(None)

    def __del__(self) -> None:  # best effort; free() is the sanctioned path
        try:
            self.free()
        except Exception:
            pass


class CudaRuntime:
    """One loaded library + device context for ONE profile."""

    def __init__(self, profile_id: str, *, arch: str | None = None,
                 build: bool = True) -> None:
        if profile_id not in PROFILE_VARIANTS:
            raise ValueError(
                f"unknown profile_id {profile_id!r}; known: "
                f"{sorted(PROFILE_VARIANTS)}"
            )
        self.profile_id = profile_id
        self.variant = PROFILE_VARIANTS[profile_id]
        if build:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "sw_cuda_build", _BUILD_PY)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            try:
                self.lib_path = mod.build(self.variant, arch=arch)
            except (FileNotFoundError, RuntimeError) as exc:
                raise SwCudaUnavailable(f"build failed: {exc}") from exc
        else:
            self.lib_path = mod_path(self.variant)
            if not self.lib_path.is_file():
                raise SwCudaUnavailable(
                    f"{self.lib_path} not built (call build first)"
                )
        key = (self.profile_id, str(self.lib_path))
        with _cache_lock:
            cached = _library_cache.get(key)
        if cached is not None:
            self.lib, self.build_flags = cached
            n_dev = int(self.lib.sw_device_count())
            if n_dev <= 0:
                raise SwCudaUnavailable("no CUDA device visible (sw_device_count)")
            return
        try:
            lib = ctypes.CDLL(str(self.lib_path))
            lib.sw_last_error.restype = ctypes.c_char_p
            lib.sw_build_flags.restype = ctypes.c_char_p
            lib.sw_device_count.restype = ctypes.c_int
            lib.sw_timer_end.restype = ctypes.c_float  # xmm0, not EAX
        except OSError as exc:
            raise SwCudaUnavailable(f"cannot load {self.lib_path}: {exc}") from exc
        if int(lib.sw_device_count()) <= 0:
            raise SwCudaUnavailable("no CUDA device visible (sw_device_count)")
        self.build_flags = lib.sw_build_flags().decode()
        with _cache_lock:
            _library_cache[key] = (lib, self.build_flags)
        self.lib = lib

    # -- ABI plumbing -------------------------------------------------------

    def device_name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        _check(self.lib.sw_device_name(0, buf, 256), self.lib)
        return buf.value.decode()

    def build_info(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "variant": self.variant,
            "library": str(self.lib_path),
            "build_flags": self.build_flags,
            "abi_version": int(self.lib.sw_abi_version()),
        }

    def assert_canonical_strict(self) -> None:
        """Refuse to certify anything but the strict canonical build."""
        if self.profile_id != CANONICAL_CUDA_V1:
            raise RuntimeMismatch(
                f"{self.profile_id!r} is not the canonical profile"
            )
        missing = [f for f in _STRICT_REQUIREMENTS if f not in self.build_flags]
        if missing:
            raise RuntimeMismatch(
                f"canonical profile loaded a non-strict build "
                f"({self.build_flags}); missing {missing}"
            )

    def sync(self) -> None:
        _check(self.lib.sw_sync(), self.lib)

    # -- buffers ------------------------------------------------------------
    def alloc(self, shape, dtype=np.float32) -> DeviceArray:
        dtype = np.dtype(dtype)
        n = int(np.prod(shape)) if isinstance(shape, (tuple, list)) else int(shape)
        return DeviceArray(self, max(n, 1) * dtype.itemsize, shape, dtype)

    def to_device(self, host: np.ndarray) -> DeviceArray:
        arr = np.ascontiguousarray(host)
        dev = self.alloc(arr.shape, arr.dtype)
        dev.from_numpy(arr)
        return dev

    # -- timing ---------------------------------------------------------------
    def timer_begin(self) -> None:
        _check(self.lib.sw_timer_begin(), self.lib)

    def timer_end(self) -> float:
        ms = float(self.lib.sw_timer_end())
        if ms < 0:
            raise CudaError(
                f"timer failed: {self.lib.sw_last_error().decode()}"
            )
        return ms


def mod_path(variant: str) -> Path:
    return PKG_DIR / "build" / variant / "libswcuda.so"


_RUNTIME_CACHE: dict[str, CudaRuntime] = {}
_RUNTIME_LOCK = threading.Lock()


def get_runtime(profile_id: str = CANONICAL_CUDA_V1,
                *, arch: str | None = None) -> CudaRuntime:
    """Cached runtime per profile (separate namespaces per DESIGN 3.1)."""
    with _RUNTIME_LOCK:
        rt = _RUNTIME_CACHE.get(profile_id)
        if rt is None:
            rt = CudaRuntime(profile_id, arch=arch)
            _RUNTIME_CACHE[profile_id] = rt
        return rt


def require_runtime(profile_id: str = CANONICAL_CUDA_V1,
                    *, arch: str | None = None) -> CudaRuntime | None:
    """Runtime or None when CUDA is unavailable.

    Honors SW_REQUIRE_CUDA=1: on the GPU host an unavailable runtime is a
    hard failure, never a skip.
    """
    try:
        rt = get_runtime(profile_id, arch=arch)
        rt.device_name()  # force a real device query
        return rt
    except SwCudaUnavailable as exc:
        if cuda_required():
            raise
        print(f"SW CUDA unavailable ({profile_id}): {exc}")
        return None


# ---------------------------------------------------------------------------
# primitive parity probes (thin launchers; bit equality judged in tests)
# ---------------------------------------------------------------------------

def run_exp(rt: CudaRuntime, x_f32: np.ndarray) -> np.ndarray:
    xd = rt.to_device(np.ascontiguousarray(x_f32, dtype=np.float32))
    yd = rt.alloc(xd.shape, np.float32)
    rc = rt.lib.sw_run_probe_expf(ctypes.c_void_p(xd.ptr.value),
                                  ctypes.c_void_p(yd.ptr.value),
                                  ctypes.c_longlong(xd.size))
    _check(rc, rt.lib)
    return yd.to_numpy()


def run_log(rt: CudaRuntime, x_f32: np.ndarray) -> np.ndarray:
    xd = rt.to_device(np.ascontiguousarray(x_f32, dtype=np.float32))
    yd = rt.alloc(xd.shape, np.float32)
    rc = rt.lib.sw_run_probe_logf(ctypes.c_void_p(xd.ptr.value),
                                  ctypes.c_void_p(yd.ptr.value),
                                  ctypes.c_longlong(xd.size))
    _check(rc, rt.lib)
    return yd.to_numpy()


def run_powf(rt: CudaRuntime, x_f32: np.ndarray, e_f32: np.ndarray) -> np.ndarray:
    xd = rt.to_device(np.ascontiguousarray(x_f32, dtype=np.float32))
    ed = rt.to_device(np.ascontiguousarray(e_f32, dtype=np.float32))
    yd = rt.alloc(xd.shape, np.float32)
    rc = rt.lib.sw_run_probe_powf(ctypes.c_void_p(xd.ptr.value),
                                  ctypes.c_void_p(ed.ptr.value),
                                  ctypes.c_void_p(yd.ptr.value),
                                  ctypes.c_longlong(xd.size))
    _check(rc, rt.lib)
    return yd.to_numpy()


def run_opmath_pow(rt: CudaRuntime, x_f32: np.ndarray, e: int) -> np.ndarray:
    xd = rt.to_device(np.ascontiguousarray(x_f32, dtype=np.float32))
    yd = rt.alloc(xd.shape, np.float32)
    rc = rt.lib.sw_run_probe_opmath_powf(ctypes.c_void_p(xd.ptr.value),
                                         ctypes.c_void_p(yd.ptr.value),
                                         ctypes.c_longlong(xd.size),
                                         ctypes.c_int(e))
    _check(rc, rt.lib)
    return yd.to_numpy()


def run_arith(rt: CudaRuntime, a_f32: np.ndarray, b_f32: np.ndarray):
    ad = rt.to_device(np.ascontiguousarray(a_f32, dtype=np.float32))
    bd = rt.to_device(np.ascontiguousarray(b_f32, dtype=np.float32))
    outs = [rt.alloc(ad.shape, np.float32) for _ in range(4)]
    rc = rt.lib.sw_run_probe_arith(
        *[ctypes.c_void_p(d.ptr.value) for d in (ad, bd, *outs)],
        ctypes.c_longlong(ad.size))
    _check(rc, rt.lib)
    return [o.to_numpy() for o in outs]


def run_contract(rt: CudaRuntime, a_f32, b_f32, c_f32) -> np.ndarray:
    dev = [rt.to_device(np.ascontiguousarray(v, dtype=np.float32))
           for v in (a_f32, b_f32, c_f32)]
    yd = rt.alloc(dev[0].shape, np.float32)
    rc = rt.lib.sw_run_probe_contract(
        *[ctypes.c_void_p(d.ptr.value) for d in (*dev, yd)],
        ctypes.c_longlong(dev[0].size))
    _check(rc, rt.lib)
    return yd.to_numpy()


def run_maximum(rt: CudaRuntime, a_f32, b_f32) -> np.ndarray:
    ad = rt.to_device(np.ascontiguousarray(a_f32, dtype=np.float32))
    bd = rt.to_device(np.ascontiguousarray(b_f32, dtype=np.float32))
    yd = rt.alloc(ad.shape, np.float32)
    rc = rt.lib.sw_run_probe_maximum(ctypes.c_void_p(ad.ptr.value),
                                     ctypes.c_void_p(bd.ptr.value),
                                     ctypes.c_void_p(yd.ptr.value),
                                     ctypes.c_longlong(ad.size))
    _check(rc, rt.lib)
    return yd.to_numpy()


def run_narith(rt: CudaRuntime, a_f32, b_f32):
    ad = rt.to_device(np.ascontiguousarray(a_f32, dtype=np.float32))
    bd = rt.to_device(np.ascontiguousarray(b_f32, dtype=np.float32))
    outs = [rt.alloc(ad.shape, np.float32) for _ in range(4)]
    rc = rt.lib.sw_run_probe_narith(
        *[ctypes.c_void_p(d.ptr.value) for d in (ad, bd, *outs)],
        ctypes.c_longlong(ad.size))
    _check(rc, rt.lib)
    return [o.to_numpy() for o in outs]


def run_is_libm_lane(rt: CudaRuntime, i_arr: np.ndarray, n_arr: np.ndarray,
                     T_arr: np.ndarray, grain_arr: np.ndarray) -> np.ndarray:
    dev = [rt.to_device(np.ascontiguousarray(v, dtype=t))
           for v, t in ((i_arr, np.int64), (n_arr, np.int64),
                        (T_arr, np.int32), (grain_arr, np.int32))]
    yd = rt.alloc(dev[0].shape, np.int32)
    rc = rt.lib.sw_run_probe_is_libm_lane(
        *[ctypes.c_void_p(d.ptr.value) for d in (*dev, yd)],
        ctypes.c_longlong(dev[0].size))
    _check(rc, rt.lib)
    return yd.to_numpy()


# ---------------------------------------------------------------------------
# march (frozen T03 step tables consumed as DATA: dx/dy int32, dz bits f32)
# ---------------------------------------------------------------------------

def run_march_svf_shadow(rt: CudaRuntime, a: np.ndarray, vegdem: np.ndarray,
                         vegdem2: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                         dz: np.ndarray) -> tuple[np.ndarray, ...]:
    """CUDA svf_shadow march; returns (sh, vegsh, vbshvegsh) planes.

    ``dz`` must be the table's raw float32 bits (step_tables.dz_bits viewed
    as float32) — bit-discipline input, never recomputed.
    """
    rows, cols = a.shape
    planes = [rt.to_device(np.ascontiguousarray(p, dtype=np.float32))
              for p in (a, vegdem, vegdem2)]
    cols_dev = [rt.to_device(np.ascontiguousarray(dx, dtype=np.int32)),
                rt.to_device(np.ascontiguousarray(dy, dtype=np.int32)),
                rt.to_device(np.ascontiguousarray(dz, dtype=np.float32))]
    outs = [rt.alloc((rows, cols), np.float32) for _ in range(3)]
    rc = rt.lib.sw_run_march_svf_shadow(
        *[ctypes.c_void_p(d.ptr.value) for d in (*planes, *cols_dev, *outs)],
        ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(len(dx)))
    _check(rc, rt.lib)
    return tuple(o.to_numpy() for o in outs)


def run_march_wallheight23(rt: CudaRuntime, a: np.ndarray, vegdem: np.ndarray,
                           vegdem2: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                           dz: np.ndarray,
                           dzprev: np.ndarray) -> tuple[np.ndarray, ...]:
    """CUDA wallheight_23 march; returns (sh, vegsh, vbshvegsh) planes."""
    rows, cols = a.shape
    planes = [rt.to_device(np.ascontiguousarray(p, dtype=np.float32))
              for p in (a, vegdem, vegdem2)]
    cols_dev = [
        rt.to_device(np.ascontiguousarray(v, dtype=np.int32 if i < 2 else np.float32))
        for i, v in enumerate((dx, dy, dz, dzprev))
    ]
    outs = [rt.alloc((rows, cols), np.float32) for _ in range(3)]
    rc = rt.lib.sw_run_march_wallheight23(
        *[ctypes.c_void_p(d.ptr.value) for d in (*planes, *cols_dev, *outs)],
        ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(len(dx)))
    _check(rc, rt.lib)
    return tuple(o.to_numpy() for o in outs)


# ---------------------------------------------------------------------------
# SVF fold (T07 packed bit state + frozen constant tables consumed as DATA)
# ---------------------------------------------------------------------------

def run_fold(rt: CudaRuntime, veg_bytes: np.ndarray, vbsh_bytes: np.ndarray,
             vegdem2: np.ndarray, svf_building: np.ndarray, tables: dict,
             *, cell_mask: np.ndarray | None = None,
             out_base: np.ndarray | None = None) -> np.ndarray:
    """CUDA SVF fold; returns the (rows, cols, 11) output stack in
    FOLD_OUTPUT_NAMES order (svfveg, svfEveg, ..., svfNaveg, svftotal).

    ``veg_bytes``/``vbsh_bytes`` are the packed bitplanes uint8 arrays
    (rows, cols, 20). ``tables`` carries the FROZEN T07 constants as DATA:
    w_iso/w_aniso (float32 (8, 12) views of the captured bits), ring
    (int32 (153,)), na (int32 (8,)), dir_e/dir_s/dir_w/dir_n (int8 (153,)),
    last_const, one_minus_trans (float32 scalars).

    ``cell_mask`` (uint8/bool plane) folds ONLY masked cells; ``out_base``
    (rows, cols, 11 float32) pre-fills the output so every unmasked cell is
    returned bit-identical to its base (affected-chunk-only contract).
    """
    rows, cols = vegdem2.shape
    veg = rt.to_device(np.ascontiguousarray(veg_bytes, dtype=np.uint8))
    vbsh = rt.to_device(np.ascontiguousarray(vbsh_bytes, dtype=np.uint8))
    planes = [rt.to_device(np.ascontiguousarray(p, dtype=np.float32))
              for p in (vegdem2, svf_building)]
    tables_dev = [
        rt.to_device(np.ascontiguousarray(tables[k], dtype=t))
        for k, t in (("w_iso", np.float32), ("w_aniso", np.float32),
                     ("ring", np.int32), ("na", np.int32),
                     ("dir_e", np.int8), ("dir_s", np.int8),
                     ("dir_w", np.int8), ("dir_n", np.int8))
    ]
    if cell_mask is not None:
        mask_dev = rt.to_device(
            np.ascontiguousarray(cell_mask).astype(np.uint8))
        mask_ptr = ctypes.c_void_p(mask_dev.ptr.value)
        do_mask = 1
    else:
        mask_ptr = ctypes.c_void_p(None)
        do_mask = 0
    if out_base is not None:
        base = np.ascontiguousarray(out_base, dtype=np.float32)
        if base.shape != (rows, cols, 11):
            raise ValueError(
                f"out_base must be {(rows, cols, 11)}, got {base.shape}")
        out_dev = rt.to_device(base)
    else:
        out_dev = rt.alloc((rows, cols, 11), np.float32)
    rc = rt.lib.sw_run_fold(
        *[ctypes.c_void_p(d.ptr.value)
          for d in (veg, vbsh, *planes, *tables_dev)],
        mask_ptr, ctypes.c_void_p(out_dev.ptr.value),
        ctypes.c_float(np.float32(tables["last_const"])),
        ctypes.c_float(np.float32(tables["one_minus_trans"])),
        ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(do_mask))
    _check(rc, rt.lib)
    return out_dev.to_numpy()


# ---------------------------------------------------------------------------
# radiation (T08 fused day/night kernels; bundle dicts of numpy DATA)
# ---------------------------------------------------------------------------

#: upload order == sw_run_rad_day ABI order (pointers first, scalars last)
_RAD_DAY_INPUTS = (
    ("buildings", np.float32), ("aspect", np.float32), ("wallbol", np.float32),
    ("alb_grid", np.float32), ("emis_grid", np.float32),
    ("svfbuveg", np.float32), ("diffsh", np.float32),
    ("sh_pb", np.uint8), ("veg_pb", np.uint8), ("vbsh_pb", np.uint8),
    ("sun_pb", np.uint8), ("shd_pb", np.uint8),
    ("dp_rank", np.int64), ("guard_true", np.int8),
    ("shadow", np.float32), ("sunwall", np.float32),
    ("albshadow", np.float32), ("alb", np.float32), ("Lup_pre", np.float32),
    ("gvflup_extra", np.float32),
    ("lv2", np.float32), ("ster", np.float32), ("psin", np.float32),
    ("pcos", np.float32), ("lumChi", np.float32), ("lsky_d2", np.float32),
    ("lsky_s2", np.float32),
    ("card_e", np.int8), ("card_s", np.int8), ("card_w", np.int8),
    ("card_n", np.int8),
    ("ccos_e", np.float32), ("ccos_s", np.float32), ("ccos_w", np.float32),
    ("ccos_n", np.float32),
    ("walk_az_low", np.float32), ("walk_az_high", np.float32),
    ("walk_az_branch", np.int32), ("walk_dy", np.int32), ("walk_dx", np.int32),
    ("jE", np.int8), ("jS", np.int8), ("jW", np.int8), ("jN", np.int8),
    ("F_sh", np.float32), ("Tg_plane", np.float32),
    ("m_lup_in", np.float32), ("m_e_in", np.float32), ("m_s_in", np.float32),
    ("m_w_in", np.float32), ("m_n_in", np.float32), ("m_tg_in", np.float32),
)
_RAD_DAY_OUTPUTS = (
    "tmrt", "kdown", "kup", "ldown", "lup", "ke", "ks", "kw", "kn",
    "le", "ls", "lw", "ln", "ksidei", "tgout", "lside", "ksided", "drad",
    "kside",
)
_RAD_DAY_NEXT = ("n_lup", "n_e", "n_s", "n_w", "n_n", "n_tg")
_RAD_DAY_F32_SCALARS = (
    "ks_sun", "ks_shd", "radI", "radD", "radG", "sinalt", "cosalt",
    "ta273", "Lwall32", "Ta32",
    "w1_0", "w1_1", "w1_2", "w1_3", "w1_4", "w1_5",
)
_RAD_DAY_F64_SCALARS = ("veg64", "shd64", "sun64")
_RAD_DAY_INT_SCALARS = ("rows", "cols", "n_patches", "kside_n", "fd_eq_1",
                        "branch2", "sun_stride")
#: scalar marshalling in EXACT kernel parameter order — the C ABI is
#: positional, so the float/double groups must interleave exactly as the
#: kernel signature declares them (7 f32, 3 f64, 9 f32, then ints).
_RAD_DAY_SCALAR_ARGS = (
    ("ks_sun", "f"), ("ks_shd", "f"), ("radI", "f"), ("radD", "f"),
    ("radG", "f"), ("sinalt", "f"), ("cosalt", "f"),
    ("veg64", "d"), ("shd64", "d"), ("sun64", "d"),
    ("ta273", "f"), ("Lwall32", "f"), ("Ta32", "f"),
    ("w1_0", "f"), ("w1_1", "f"), ("w1_2", "f"), ("w1_3", "f"),
    ("w1_4", "f"), ("w1_5", "f"),
    ("rows", "i"), ("cols", "i"), ("n_patches", "i"), ("kside_n", "i"),
    ("fd_eq_1", "i"), ("branch2", "i"), ("sun_stride", "i"),
)

_RAD_NIGHT_INPUTS = (
    ("sh_pb", np.uint8), ("veg_pb", np.uint8), ("vbsh_pb", np.uint8),
    ("night_Lup", np.float32),
    ("ster", np.float32), ("psin", np.float32), ("pcos", np.float32),
    ("lsky_d2", np.float32), ("lsky_s2", np.float32),
    ("card_e", np.int8), ("card_s", np.int8), ("card_w", np.int8),
    ("card_n", np.int8),
    ("ccos_e", np.float32), ("ccos_s", np.float32), ("ccos_w", np.float32),
    ("ccos_n", np.float32),
)
_RAD_NIGHT_OUTPUTS = ("tmrt", "ldown", "lside", "le", "ls", "lw", "ln")
_RAD_NIGHT_F64_SCALARS = ("veg64", "shd64")
_RAD_NIGHT_INT_SCALARS = ("rows", "cols", "n_patches")


def run_rad_day(rt: CudaRuntime, b: dict) -> dict:
    """CUDA fused daytime radiation step; ``b`` carries every kernel input
    as numpy DATA (frozen bundle bits + march outputs + static planes).
    Returns {output name: plane} for the 19 Solweig returns plus the 6
    next-state planes (n_lup..n_tg)."""
    rows, cols = int(b["rows"]), int(b["cols"])
    devs = [rt.to_device(np.ascontiguousarray(b[k], dtype=dt))
            for k, dt in _RAD_DAY_INPUTS]
    outs = [rt.alloc((rows, cols), np.float32)
            for _ in _RAD_DAY_OUTPUTS + _RAD_DAY_NEXT]
    args = [ctypes.c_void_p(d.ptr.value) for d in (*devs, *outs)]
    for key, kind in _RAD_DAY_SCALAR_ARGS:
        if kind == "f":
            args.append(ctypes.c_float(np.float32(b[key])))
        elif kind == "d":
            args.append(ctypes.c_double(np.float64(b[key])))
        else:
            args.append(ctypes.c_int(int(b[key])))
    _check(rt.lib.sw_run_rad_day(*args), rt.lib)
    planes = [o.to_numpy() for o in outs]
    out = dict(zip(_RAD_DAY_OUTPUTS, planes[:len(_RAD_DAY_OUTPUTS)]))
    out.update(dict(zip(_RAD_DAY_NEXT, planes[len(_RAD_DAY_OUTPUTS):])))
    return out


def run_rad_night(rt: CudaRuntime, b: dict) -> dict:
    """CUDA fused nighttime radiation step; returns the 7 night planes."""
    rows, cols = int(b["rows"]), int(b["cols"])
    devs = [rt.to_device(np.ascontiguousarray(b[k], dtype=dt))
            for k, dt in _RAD_NIGHT_INPUTS]
    outs = [rt.alloc((rows, cols), np.float32)
            for _ in _RAD_NIGHT_OUTPUTS]
    args = [ctypes.c_void_p(d.ptr.value) for d in (*devs, *outs)]
    args += [ctypes.c_double(np.float64(b[k])) for k in _RAD_NIGHT_F64_SCALARS]
    args += [ctypes.c_int(int(b[k])) for k in _RAD_NIGHT_INT_SCALARS]
    _check(rt.lib.sw_run_rad_night(*args), rt.lib)
    return dict(zip(_RAD_NIGHT_OUTPUTS, [o.to_numpy() for o in outs]))


# ---------------------------------------------------------------------------
# UTCI (T09 CPU canonical utci.py; torch chunk-layout lane model exposed)
# ---------------------------------------------------------------------------

#: torch elementwise chunk-layout parameters (math_compat defaults; t08
#: capture manifest torch_threads=8, empirical torch-2.14 grain 32768).
#: Hardcoded so the CUDA runtime stays numba-free — tests assert equality
#: with the CPU module's DEFAULT_TORCH_THREADS/TORCH_PAR_GRAIN.
UTCI_TORCH_THREADS = 8
UTCI_PAR_GRAIN = 32768


def run_utci_dense(rt: CudaRuntime, ta_plane: np.ndarray,
                   rh_plane: np.ndarray, tmrt_plane: np.ndarray,
                   va_plane: np.ndarray,
                   torch_threads: int = UTCI_TORCH_THREADS,
                   grain: int = UTCI_PAR_GRAIN) -> np.ndarray:
    """CUDA dense UTCI; returns the -999-filled (rows, cols) f32 plane.

    Two launches around a host prefix sum: the count kernel writes
    per-row valid counts, ``offs`` is cumsum'd here (exactly the CPU
    kernel's offsets array), and the fill kernel writes only valid lanes
    into the caller-visible -999 prefill.
    """
    planes = [rt.to_device(np.ascontiguousarray(p, dtype=np.float32))
              for p in (ta_plane, rh_plane, tmrt_plane, va_plane)]
    rows, cols = planes[0].shape
    counts = rt.alloc((rows,), np.int64)
    _check(rt.lib.sw_utci_count(
        *[ctypes.c_void_p(d.ptr.value) for d in planes],
        ctypes.c_void_p(counts.ptr.value),
        ctypes.c_int(rows), ctypes.c_int(cols)), rt.lib)
    offs = np.zeros(rows + 1, dtype=np.int64)
    offs[1:] = np.cumsum(counts.to_numpy())
    n = int(offs[-1])
    offs_dev = rt.to_device(offs)
    out_dev = rt.to_device(np.full((rows, cols), np.float32(-999.0),
                                   dtype=np.float32))
    _check(rt.lib.sw_utci_fill(
        *[ctypes.c_void_p(d.ptr.value) for d in planes],
        ctypes.c_void_p(offs_dev.ptr.value),
        ctypes.c_void_p(out_dev.ptr.value),
        ctypes.c_longlong(n), ctypes.c_int(int(torch_threads)),
        ctypes.c_int(int(grain)), ctypes.c_int(rows), ctypes.c_int(cols)),
        rt.lib)
    return out_dev.to_numpy()


def run_utci_sparse(rt: CudaRuntime, ta_v: np.ndarray, rh_v: np.ndarray,
                    tmrt_v: np.ndarray, va_v: np.ndarray,
                    torch_threads: int = UTCI_TORCH_THREADS,
                    grain: int = UTCI_PAR_GRAIN) -> np.ndarray:
    """CUDA sparse UTCI over the pre-compacted valid vectors (masked-select
    order); bit-identical per element to the dense route on the same valid
    set, including odd-length tails."""
    planes = [rt.to_device(np.ascontiguousarray(v, dtype=np.float32))
              for v in (ta_v, rh_v, tmrt_v, va_v)]
    n = int(planes[0].size)
    out = rt.alloc((n,), np.float32)
    _check(rt.lib.sw_utci_sparse(
        *[ctypes.c_void_p(d.ptr.value) for d in planes],
        ctypes.c_void_p(out.ptr.value),
        ctypes.c_longlong(n), ctypes.c_int(int(torch_threads)),
        ctypes.c_int(int(grain))), rt.lib)
    return out.to_numpy()
