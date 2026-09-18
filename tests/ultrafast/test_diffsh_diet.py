# SPDX-License-Identifier: GPL-3.0-only
"""T14 Change F gates: the diffsh dense-cube diet (RSS budget).

``rad_static_from_capture`` used to materialize the 153-patch dense
``diffsh`` cube — plus three transient unpacked bit cubes (sh/veg/vbsh,
459 MB more at site_500) — pushing a single-scenario static load to a
~730 MiB RSS peak and a 24-step job to ~1.3 GiB, against
``acceptance_targets.yaml`` budgets of 256 MiB steady / 768 MiB job
peak.

The diet: the runtime static carries the PACKED cubes only; ``diffsh``
is recomputed per cell inside the kernels and per patch inside the numpy
mirrors as

    ds = f32(sh_bit) - (f32(1.0) - f32(veg_bit)) * f32(1.0 - 0.03)

which is bit-identical to the dense expression (operands are exactly
0.0/1.0; same op order; same constant) — proven by the raw-bit pin.

Gates:
* **Witnesses (RED first)** — default static has NO dense diffsh; the
  steady RSS (static + t00 bundle + state, nothing else) fits 256 MiB;
  the full 24-step job peak fits 768 MiB.
* **Raw-bit pin** — post-diet full solve reproduces the pre-change
  freeze (``diffsh_diet_pin.json``, cross-checked dev-env == clean-env
  before any edit): per-timestep plane digests, CI series, anchor
  fingerprint.
* **Dense equivalence** — the CUDA lane's opt-in dense materialization
  is bit-equal to the per-patch mirror planes on every patch.
* **Mutation killer** — a mutated ``DIFFSH_C97`` (0.03 instead of
  fl32(1.0-0.03)) compiled into a fresh-cache subprocess diverges the
  solve digests; the pin has teeth.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from solweig_core.numba_cpu import radiation  # noqa: E402
from solweig_core.numba_cpu.radiation import (  # noqa: E402
    rad_bundle_from_capture,
    rad_state_from_capture,
    rad_static_from_capture,
)

CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t14/compact_capture")
PIN = ULTRA_DIR / "diffsh_diet_pin.json"
PROFILE = "site_500-default"
LATITUDE = 30.312645
STEADY_BUDGET_MIB = 256.0
JOB_PEAK_BUDGET_MIB = 768.0
#: post-diet geometry-lineage token (the dense diffsh left the
#: fingerprint vocabulary; the packed cubes it derives from stay in —
#: see test_full_solve_matches_prechange_pin)
POST_DIET_GEOMETRY_DIGEST = (
    "d4e357fcf08f9fe4baee94d2ccc57a666848d95d7758b8067ee8b8d31f618025"
)

pytestmark = pytest.mark.skipif(
    not (CAP / "t00.npz").is_file() or not PIN.is_file(),
    reason="compact capture or pre-change pin absent",
)

_STEADY_SCRIPT = r"""
import platform, resource, sys
from pathlib import Path
import numpy as np  # noqa: F401
from solweig_core.numba_cpu.radiation import (
    rad_bundle_from_capture, rad_state_from_capture, rad_static_from_capture,
)
cap = Path(sys.argv[1])
st = rad_static_from_capture(cap)
t_in = rad_bundle_from_capture(cap, 0)
state = rad_state_from_capture(cap, 0, rows=st.rows, cols=st.cols)
assert st.rows > 0 and t_in is not None and state is not None
raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
div = 1048576.0 if platform.system() == "Darwin" else 1024.0
print("@@RSS@@", raw / div)
"""

_JOB_SCRIPT = r"""
import platform, resource, sys
from pathlib import Path
import numpy as np  # noqa: F401
from solweig_core.numba_cpu import full_solve as fs
cap_set = fs.CaptureSet(cap_dir=sys.argv[1], profile="site_500-default",
                        latitude=30.312645)
res = fs.full_solve_capture(cap_set, profile="site_500-default")
assert len(res.outputs) == int(res.n_timesteps)
raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
div = 1048576.0 if platform.system() == "Darwin" else 1024.0
print("@@RSS@@", raw / div)
"""

_MUTATE_SCRIPT = r"""
import json, sys
import numpy as np
import solweig_core.numba_cpu.radiation as radiation
radiation.DIFFSH_C97 = np.float32(0.03)  # the mutation under test
from solweig_core.numba_cpu import full_solve as fs
cap_set = fs.CaptureSet(cap_dir=sys.argv[1], profile="site_500-default",
                        latitude=30.312645)
res = fs.full_solve_capture(cap_set, profile="site_500-default")

def digest(arr):
    a = np.ascontiguousarray(arr)
    if a.dtype == np.float32:
        a = a.view(np.uint32)
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()

per_t = []
for out in res.outputs:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(out):
        v = out[k]
        if hasattr(v, "tobytes"):
            h.update(k.encode()); h.update(digest(v).encode())
    per_t.append(h.hexdigest())
print("@@DIGESTS@@", json.dumps(per_t))
"""


def _run(script: str, *args, extra_env: dict | None = None,
         cache_dir: Path | None = None) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if cache_dir is not None:
        env["NUMBA_CACHE_DIR"] = str(cache_dir)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", script, *map(str, args)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line.split(" ", 1)[1])
    raise AssertionError(f"no @@ marker in output:\n{proc.stdout[-2000:]}")


@pytest.fixture(scope="module")
def pin() -> dict:
    return json.loads(PIN.read_text())


@pytest.fixture(scope="module")
def solve_digests(pin):
    """One post-diet solve; digests computed exactly like the freezer."""
    import hashlib

    from solweig_core.numba_cpu import full_solve as fs

    cap = fs.CaptureSet(cap_dir=str(CAP), profile=PROFILE, latitude=LATITUDE)
    res = fs.full_solve_capture(cap, profile=PROFILE)
    per_t = []
    for out in res.outputs:
        h = hashlib.sha256()
        for k in sorted(out):
            v = out[k]
            if hasattr(v, "tobytes"):
                a = np.ascontiguousarray(v)
                if a.dtype == np.float32:
                    a = a.view(np.uint32)
                h.update(k.encode())
                h.update(hashlib.sha256(a.tobytes()).hexdigest().encode())
        per_t.append(h.hexdigest())
    return res, per_t


class TestDietWitness:
    def test_default_static_has_no_dense_diffsh(self):
        st = rad_static_from_capture(CAP)
        assert st.diffsh is None, (
            "runtime static still materializes the dense diffsh cube "
            f"({None if st.diffsh is None else st.diffsh.nbytes / 2**20:.0f} MiB)"
        )
        # the packed cubes ARE present (kernels consume them)
        assert st.shmat_packed is not None and st.vegshmat_packed is not None

    def test_steady_rss_within_budget(self, tmp_path):
        rss = _run(_STEADY_SCRIPT, CAP, cache_dir=tmp_path)
        assert rss <= STEADY_BUDGET_MIB, (
            f"steady RSS {rss:.1f} MiB exceeds the 256 MiB budget"
        )

    def test_job_peak_rss_within_budget(self, tmp_path):
        rss = _run(_JOB_SCRIPT, CAP, cache_dir=tmp_path)
        assert rss <= JOB_PEAK_BUDGET_MIB, (
            f"24-step job peak RSS {rss:.1f} MiB exceeds the 768 MiB budget"
        )


class TestParityPin:
    def test_full_solve_matches_prechange_pin(self, pin, solve_digests):
        res, per_t = solve_digests
        assert int(res.n_timesteps) == pin["n_timesteps"]
        assert per_t == pin["per_t"], (
            "post-diet per-timestep digests diverged from the pre-change "
            f"freeze at {[i for i, (a, b) in enumerate(zip(per_t, pin['per_t'])) if a != b][:5]}"
        )
        import hashlib

        ci = [
            hashlib.sha256(repr(v).encode()).hexdigest()
            for v in res.ret_CI_series
        ]
        assert ci == pin["ci_series"]
        # Anchor lineage: met_prefix (forcing rows) is untouched by the
        # diet. The three geometry keys are a lineage TOKEN produced by
        # geometry_fingerprint_of_capture over vars(st) — the diet removed
        # the dense diffsh array from that vocabulary (the packed sh/veg
        # cubes it derives from remain hashed), so the token value changes
        # while meaning the same scene. Re-frozen here, deliberately, so
        # any FURTHER vocabulary drift still fails loudly.
        fp = res.anchor_fingerprint
        assert fp["met_prefix"] == pin["anchor_fingerprint"]["met_prefix"]
        for key in ("composed_scene", "resolved_landcover",
                    "model_parameters"):
            assert fp[key] == POST_DIET_GEOMETRY_DIGEST, (
                f"{key} drifted from the re-frozen post-diet token"
            )

    def test_dense_materialization_bit_equal_per_patch(self):
        """The CUDA lane's dense diffsh == the mirrors' per-patch planes,
        every patch, raw bits (the diet changed WHERE ds is computed, not
        its value)."""
        st_dense = rad_static_from_capture(CAP, dense_diffsh=True)
        st_packed = rad_static_from_capture(CAP)
        n_patches = st_dense.diffsh.shape[2]
        problems = []
        for idx in range(n_patches):
            plane = radiation._diffsh_plane(st_packed, idx)
            dense = np.ascontiguousarray(st_dense.diffsh[:, :, idx])
            if not np.array_equal(plane.view(np.uint32),
                                  dense.view(np.uint32)):
                problems.append(idx)
                if len(problems) >= 3:
                    break
        assert not problems, f"patch planes diverged at {problems}"


class TestMutationKiller:
    def _digests_under_mutation(self, tmp_path, expression):
        script = _MUTATE_SCRIPT.replace(
            "np.float32(0.03)  # the mutation under test", expression
        )
        return _run(script, CAP, cache_dir=tmp_path / "mutcache")

    def test_mutated_constant_diverges(self, pin, tmp_path):
        """DESIGNATED KILLER: DIFFSH_C97 -> fl32(0.03) (a gross mis-cleanup)
        must change the solve digests; survival would mean the pin is
        blind to diffsh."""
        digests = self._digests_under_mutation(
            tmp_path, "np.float32(0.03)  # the mutation under test"
        )
        bad = [i for i, (a, b) in enumerate(zip(digests, pin["per_t"]))
               if a != b]
        assert bad, "mutation SURVIVED: every timestep digest still matched"

    def test_literal_cleanup_is_f32_flat(self):
        """DOCUMENTED FLATNESS (mutation that cannot exist): replacing
        fl32(1.0 - 0.03) with fl32(0.97) is value-identical — the f64
        subtract and the literal land on the same f32 (0x3f7851ec), so
        that cleanup is NOT a mutation surface. Recorded so nobody
        mistakes it for an untested seam."""
        a = np.float32(1.0 - 0.03).view(np.uint32)
        b = np.float32(0.97).view(np.uint32)
        assert a == b == np.uint32(0x3F7851EC)
