# SPDX-License-Identifier: GPL-3.0-only
"""T13 GATE 1 — GPU residency pool: changed-chunk transfer + async
publish/cancel fencing + profile-keyed cache (TASKS T13; DESIGN 13.3).

The resident runner keeps the site's STATIC kernel inputs on device
(diffsh 153 MB, packed cubes, albedo/emissivity planes — the ~178 MB that
dominated T12's 26 ms rad_day H2D) and re-uploads only the CHANGED row
chunks of the per-timestep inputs, downloading only requested output
windows. Every async path is fenced:

* publish fence — an async D2H output lease may NOT be published before
  its completion event fires (RED witness 3);
* cancel fence — a cancelled job's buffer may NOT be reacquired while
  its outstanding event is still pending (RED witness 4);
* profile fence — the static-set registry is keyed by
  (profile_id, digest): a different profile must NEVER hit another
  profile's resident buffers (RED witness 5).

Parity: the resident rad_day/rad_night paths must reproduce the T12
host-wrapper results bit-for-bit on the full site (the T12 gates pinned
those against the oracle capture digests — this file pins resident ==
wrapper, so the oracle chain is transitive through run_rad_day).
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
    require_runtime,
)

REPO_ROOT = ULTRA_DIR.parents[1]
NATIVE_CUDA = REPO_ROOT / "native" / "cuda"

_T13_MODULES: dict[str, object] = {}


def load_t13(name: str):
    """Load a native/cuda T13 module ONCE, sharing the harness's host
    module copy (single CudaRuntime identity — buffers never cross
    module copies)."""
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
def residency():
    return load_t13("residency")


def _capture_dir() -> Path:
    cap = os.environ.get("SW_T12_T08_CAPTURE", "")
    if not cap or not Path(cap).is_dir():
        pytest.skip("t08 capture fixtures not available (SW_T12_T08_CAPTURE)")
    return Path(cap)


def _site_bundles(cap: Path, t: int):
    sys.path.insert(0, str(REPO_ROOT))
    from solweig_core.numba_cpu.radiation import (
        rad_bundle_from_capture,
        rad_state_from_capture,
        rad_static_from_capture,
    )
    from sw_rad_bundle import build_rad_day_bundle, build_rad_night_bundle

    st = rad_static_from_capture(cap)
    t_in = rad_bundle_from_capture(cap, t)
    state = rad_state_from_capture(cap, t, rows=st.rows, cols=st.cols)
    if t_in.is_day:
        b = build_rad_day_bundle(st, t_in, state)
    else:
        b = build_rad_night_bundle(st, t_in)
    return st, t_in, state, b


#: static (scene-invariant) rad_day bundle inputs — resident candidates
RAD_DAY_STATIC = ("buildings", "aspect", "wallbol", "alb_grid", "emis_grid",
                  "svfbuveg", "diffsh", "sh_pb", "veg_pb", "vbsh_pb", "alb")
RAD_DAY_VOLATILE = ("sun_pb", "shd_pb", "dp_rank", "guard_true", "shadow",
                    "sunwall", "albshadow", "Lup_pre", "gvflup_extra",
                    "lv2", "ster", "psin", "pcos", "lumChi", "lsky_d2",
                    "lsky_s2", "card_e", "card_s", "card_w", "card_n",
                    "ccos_e", "ccos_s", "ccos_w", "ccos_n",
                    "walk_az_low", "walk_az_high", "walk_az_branch",
                    "walk_dy", "walk_dx", "jE", "jS", "jW", "jN",
                    "F_sh", "Tg_plane", "m_lup_in", "m_e_in", "m_s_in",
                    "m_w_in", "m_n_in", "m_tg_in")
RAD_DAY_OUTPUTS = ("tmrt", "kdown", "kup", "ldown", "lup", "ke", "ks", "kw",
                   "kn", "le", "ls", "lw", "ln", "ksidei", "tgout", "lside",
                   "ksided", "drad", "kside")
RAD_DAY_NEXT = ("n_lup", "n_e", "n_s", "n_w", "n_n", "n_tg")
RAD_NIGHT_STATIC = ("sh_pb", "veg_pb", "vbsh_pb")
RAD_NIGHT_VOLATILE = ("night_Lup", "ster", "psin", "pcos", "lsky_d2",
                      "lsky_s2", "card_e", "card_s", "card_w", "card_n",
                      "ccos_e", "ccos_s", "ccos_w", "ccos_n")
RAD_NIGHT_OUTPUTS = ("tmrt", "ldown", "lside", "le", "ls", "lw", "ln")


def _make_runner(residency, rt, bundle, *, chunk_rows: int = 128):
    runner = residency.ResidentRadRunner(rt, chunk_rows=chunk_rows)
    runner.bind({k: bundle[k] for k in RAD_DAY_STATIC}, static=True)
    runner.bind({k: bundle[k] for k in RAD_DAY_VOLATILE})
    return runner


def _is_day(b: dict) -> bool:
    return "ks_sun" in b  # only the day bundle carries day scalars


# ---------------------------------------------------------------------------
# parity: resident == T12 host wrapper (== oracle, transitively)
# ---------------------------------------------------------------------------


class TestResidentParity:
    def test_rad_day_full_site_matches_wrapper_bits(self, rt, host,
                                                    residency):
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 12)
        assert _is_day(b), "capture t12 must be a day timestep"

        want = host.run_rad_day(rt, b)
        runner = _make_runner(residency, rt, b)
        lease = runner.run_day(b, download=RAD_DAY_OUTPUTS + RAD_DAY_NEXT)
        lease.wait()
        got = lease.publish()
        failures = []
        for name in RAD_DAY_OUTPUTS + RAD_DAY_NEXT:
            msg, _ = compare_bits(want[name], got[name], f"resident:{name}")
            if msg != "PASS":
                failures.append(msg)
        assert not failures, "\n".join(failures[:6])
        runner.close()

    def test_rad_night_full_site_matches_wrapper_bits(self, rt, host,
                                                      residency):
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 5)
        assert not _is_day(b), "capture t05 must be a night timestep"

        want = host.run_rad_night(rt, b)
        runner = residency.ResidentRadRunner(rt, chunk_rows=128)
        runner.bind({k: b[k] for k in RAD_NIGHT_STATIC}, static=True)
        runner.bind({k: b[k] for k in RAD_NIGHT_VOLATILE})
        lease = runner.run_night(b, download=RAD_NIGHT_OUTPUTS)
        lease.wait()
        got = lease.publish()
        for name in RAD_NIGHT_OUTPUTS:
            msg, _ = compare_bits(want[name], got[name], f"resident:{name}")
            assert msg == "PASS", msg
        runner.close()

    def test_windowed_download_matches_full_slice(self, rt, host, residency):
        """Output windows (row-range D2H) must be bit-identical to the
        matching slice of the full download — publication moves only the
        dirty rectangle."""
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 12)
        runner = _make_runner(residency, rt, b)
        full = runner.run_day(b, download=("tmrt", "lup"))
        full.wait()
        full_planes = full.publish()
        win = runner.run_day(b, download=("tmrt", "lup"),
                             window=(120, 264))
        win.wait()
        win_planes = win.publish()
        for name in ("tmrt", "lup"):
            msg, _ = compare_bits(
                full_planes[name][120:264, :], win_planes[name],
                f"window:{name}")
            assert msg == "PASS", msg
        runner.close()


# ---------------------------------------------------------------------------
# changed-chunk-only transfer accounting
# ---------------------------------------------------------------------------


class TestChangedChunkTransfer:
    def test_rebind_same_bundle_transfers_zero_bytes(self, rt, residency):
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 12)
        runner = _make_runner(residency, rt, b)
        runner.run_day(b, download=()).wait()
        before = runner.h2d_bytes
        runner.bind({k: b[k] for k in RAD_DAY_VOLATILE})
        runner.bind({k: b[k] for k in RAD_DAY_STATIC}, static=True)
        after = runner.h2d_bytes
        assert after == before, (
            f"unchanged re-bind moved {after - before} bytes "
            f"(must be 0 — array-equal chunks never transfer)"
        )
        runner.close()

    def test_row_window_edit_transfers_only_that_window(self, rt, residency):
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 12)
        runner = _make_runner(residency, rt, b, chunk_rows=128)
        runner.run_day(b, download=()).wait()
        before = runner.h2d_bytes

        # a 40-row edit inside chunk rows [128, 256)
        edited = {k: b[k].copy() for k in ("shadow", "Tg_plane", "m_tg_in")}
        for arr in edited.values():
            arr[150:190] += np.float32(0.25)
        runner.bind(edited)
        after = runner.h2d_bytes
        moved = after - before
        rows, cols = b["shadow"].shape
        want = 3 * 128 * cols * 4  # one 128-row chunk per edited plane
        assert moved == want, (
            f"row-window edit moved {moved} bytes, expected exactly "
            f"{want} (one changed chunk per edited plane)"
        )
        b2 = dict(b)
        b2.update(edited)
        lease = runner.run_day(b2, download=("tgout",))
        lease.wait()
        got = lease.publish()
        assert got["tgout"].shape == (rows, cols)
        runner.close()


# ---------------------------------------------------------------------------
# RED witness 3: publish before async output completion
# ---------------------------------------------------------------------------


class TestPublishFence:
    def test_publish_before_completion_refused(self, rt, residency):
        """Async D2H queued behind the ~13 ms rad_day kernel: publishing
        immediately must raise — the fence, not luck, protects the
        caller (DESIGN 13.3 'never publish before async completion')."""
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 12)
        runner = _make_runner(residency, rt, b)
        lease = runner.run_day(b, download=("tmrt",))
        with pytest.raises(residency.PublishPendingError):
            lease.publish()
        lease.wait()
        planes = lease.publish()  # after the event: must succeed
        assert planes["tmrt"].shape == b["shadow"].shape
        runner.close()


# ---------------------------------------------------------------------------
# stale-lease fence: an UNPUBLISHED lease whose event completed must not
# publish bits from a LATER download of the same slot
# ---------------------------------------------------------------------------


class TestStaleLeaseFence:
    def test_unpublished_lease_refused_after_slot_redownloaded(self, rt,
                                                                residency):
        """Lease 1 is left unpublished past its event completion; the same
        output slot is then re-downloaded by lease 2. Lease 1's publish()
        must be REFUSED — the slot now holds lease 2's bits, and publishing
        them under lease 1's identity is a stale-revision hazard ('never
        publish stale revisions'), not a data race the event fence can
        see (both events have fired)."""
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 12)
        runner = _make_runner(residency, rt, b)
        lease1 = runner.run_day(b, download=("tmrt",))
        lease1.wait()  # event complete; deliberately NOT published
        lease2 = runner.run_day(b, download=("tmrt",))
        lease2.wait()  # slot re-downloaded; new event also complete
        with pytest.raises(residency.LeaseStaleError):
            lease1.publish()
        planes = lease2.publish()  # the CURRENT lease publishes fine
        assert planes["tmrt"].shape == b["shadow"].shape
        runner.close()


# ---------------------------------------------------------------------------
# RED witness 4: early reuse of a cancelled buffer
# ---------------------------------------------------------------------------


class TestCancelFence:
    def test_cancelled_buffer_not_reacquired_while_pending(self, rt,
                                                            residency):
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 12)
        runner = _make_runner(residency, rt, b)
        token = residency.CancelToken()
        lease = runner.run_day(b, download=("tmrt",), cancel_token=token)
        token.cancel()
        lease.release()  # cancelled job returns its buffer to the pool

        # the buffer's D2H event is still queued behind the kernel: the
        # pool must refuse to hand the same slot to a new job
        with pytest.raises(residency.BufferPendingError):
            runner.run_day(b, download=("tmrt",))
        # once everything completes, acquisition works again
        runner.sync()
        lease2 = runner.run_day(b, download=("tmrt",))
        lease2.wait()
        lease2.publish()
        runner.close()


# ---------------------------------------------------------------------------
# RED witness 5: different-profile cache hit
# ---------------------------------------------------------------------------


class TestProfileKeyedRegistry:
    def test_other_profile_never_hits_static_registry(self, rt, host,
                                                      residency):
        rt_leg = require_runtime(host.LEGACY_CUDA_V1)
        digest = "deadbeef" * 8
        with pytest.raises((residency.ResidencyError, host.RuntimeMismatch)):
            residency.static_runner(rt, digest)  # nothing stored -> miss
        runner = residency.ResidentRadRunner(rt)
        runner.bind({"shadow": np.zeros((4, 4), np.float32)}, static=True)
        residency.register_static_runner(rt, digest, runner)
        got = residency.static_runner(rt, digest)
        assert got is runner
        with pytest.raises((residency.ResidencyError, host.RuntimeMismatch)):
            residency.static_runner(rt_leg, digest)


# ---------------------------------------------------------------------------
# stress / leak
# ---------------------------------------------------------------------------


class TestStressLeak:
    def test_repeated_cycles_stable_allocations_and_memory(self, rt, host,
                                                            residency):
        cap = _capture_dir()
        _, _, _, b = _site_bundles(cap, 12)
        runner = _make_runner(residency, rt, b)
        lease = runner.run_day(b, download=RAD_DAY_OUTPUTS)
        lease.wait()
        lease.publish()
        base_alloc = residency.device_alloc_count()
        base_free, _base_total = residency.mem_info()

        for i in range(60):
            token = residency.CancelToken() if i % 3 == 0 else None
            lease = runner.run_day(b, download=RAD_DAY_OUTPUTS,
                                   cancel_token=token)
            if token is not None and i % 6 == 0:
                token.cancel()
                lease.release()
                # the cancelled slot's event may still be outstanding —
                # the pool would (correctly) refuse reacquisition, so the
                # stress loop drains before the next cycle
                runner.sync()
            else:
                lease.wait()
                lease.publish()
        runner.sync()
        assert residency.device_alloc_count() == base_alloc, (
            "steady-state cycles must not grow device allocations "
            f"(delta {residency.device_alloc_count() - base_alloc})"
        )
        free2, _t2 = residency.mem_info()
        drift = base_free - free2
        assert abs(drift) < 8 << 20, f"device free memory drifted {drift} B"
        runner.close()


# ---------------------------------------------------------------------------
# purity
# ---------------------------------------------------------------------------


class TestPurity:
    def test_residency_modules_torch_free(self):
        for name in ("residency", "graphs", "dispatch_cost"):
            p = NATIVE_CUDA / f"{name}.py"
            if not p.is_file():
                pytest.fail(f"missing T13 module {p}")
            assert "import torch" not in p.read_text(), name


# ---------------------------------------------------------------------------
# T17: the changed-chunk predicate itself (CPU-only — the T17 optimization
# swapped np.array_equal-on-uint8 views for a libc memcmp fast path; this
# gate pins that the predicate is STILL bit-exact: it can never classify
# differing bits as equal, which is what keeps the upload set — and the
# device state — raw-bit faithful. The full-domain proof that the swap
# changed zero upload bytes and zero output bits lives on the GPU lane
# (h2d_bytes 543,574,853 identical committed-vs-optimized; 24/24 per-t
# digests == committed == cube pins) — recorded in WORKLOG_ULTRAFAST.md.
# ---------------------------------------------------------------------------


def _load_chunk_bytes_equal():
    spec = importlib.util.spec_from_file_location(
        "sw_cuda_residency_predicate_gate", NATIVE_CUDA / "residency.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._chunk_bytes_equal


class TestChunkBytesEqualPredicate:
    @pytest.fixture
    def chunk_eq(self):
        return _load_chunk_bytes_equal()

    @staticmethod
    def _pair(kind):
        a = np.zeros(4096, dtype=np.float32)
        b = a.copy()
        if kind == "equal":
            return a, b, True
        if kind == "value":
            b[7] = np.float32(1.5)
            return a, b, False
        if kind == "signed_zero":
            a[3] = np.float32(0.0)
            b[3] = np.float32(-0.0)
            return a, b, False
        if kind == "nan_payload":
            aw = a.view(np.uint32)
            bw = b.view(np.uint32)
            aw[11] = 0x7F800001  # signaling NaN payload
            bw[11] = 0x7FC00000  # quiet NaN payload
            return a, b, False
        raise AssertionError(kind)

    def test_named_cases_match_uint8_view_equality(self, chunk_eq):
        """equal/value/signed_zero/nan_payload: the predicate's verdict is
        EXACTLY np.array_equal on the uint8 views — ±0.0 and distinct NaN
        payloads are 'different' (they must trigger a re-upload)."""
        for kind in ("equal", "value", "signed_zero", "nan_payload"):
            a, b, expected = self._pair(kind)
            got = chunk_eq(a, b)
            assert got is expected or got == expected, (kind, got)
            ref = bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))
            assert got == ref, (kind, got, ref)

    def test_non_contiguous_fallback_is_bit_exact(self, chunk_eq):
        """Row-strided 2-D chunks take the np.array_equal fallback — the
        reachable non-contiguous case (the bind path produces C-contiguous
        chunks; committed code's uint8 view has the same last-axis
        contiguity requirement, so the fallback — not a relaxation — is
        the T17 answer for strided inputs)."""
        a = np.arange(64, dtype=np.float32).reshape(8, 8)
        b = a.copy()
        av = a[::2]
        bv = b[::2]
        assert not av.flags.c_contiguous and av.flags["C_CONTIGUOUS"] is False
        assert chunk_eq(av, bv)
        assert np.array_equal(av.view(np.uint8), bv.view(np.uint8))
        b[0, 1] = np.float32(-0.0)
        a[0, 1] = np.float32(0.0)
        assert not chunk_eq(av, bv)
        assert not np.array_equal(av.view(np.uint8), bv.view(np.uint8))

    def test_randomized_never_equalizes_differing_bits(self, chunk_eq):
        """Fuzz: whenever ANY bit differs, the predicate says not-equal —
        a value-equality shortcut here would silently desynchronize the
        device mirror and break raw-bit parity (the T17 fence)."""
        rng = np.random.default_rng(20260908)
        for _ in range(64):
            a = rng.standard_normal((16, 16)).astype(np.float32)
            b = a.copy()
            if rng.integers(0, 2):
                flat = rng.integers(0, b.size)
                word = b.reshape(-1)[flat].view(np.uint32)
                word ^= np.uint32(1) << np.uint32(rng.integers(0, 32))
            expected = bool(
                np.array_equal(a.view(np.uint8), b.view(np.uint8))
            )
            assert chunk_eq(a, b) == expected

    def test_counterexample_shape(self, chunk_eq):
        """Sanity of the gate itself: an actually-equal pair passes and a
        one-ULP flip fails — so a `return True` (or value-equality) mutant
        of the predicate cannot survive this class."""
        a = np.array([1.0, 2.0, np.nan], dtype=np.float32)
        same = a.copy()
        assert chunk_eq(a, same)
        flipped = a.copy()
        flipped[0] = np.nextafter(np.float32(1.0), np.float32(2.0))
        assert not chunk_eq(a, flipped)
