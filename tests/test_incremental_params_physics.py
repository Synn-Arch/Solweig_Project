# SPDX-License-Identifier: GPL-3.0-only
"""U-C packet 2: model-parameter physics plumbing in ``run_utci_window``.

The ``model_receptor_parameters`` adapter stages typed
``ModelParameterDelta`` records and its executor seam
(:func:`solweig_gpu.incremental.adapters.model_parameters.
kernel_arguments_from_deltas`) folds them into a flat ``name -> value``
mapping. These tests prove that mapping actually reaches the physics:

* **Default bit-identity** — ``model_parameters=None`` (and ``{}``, and an
  explicit all-defaults mapping) reproduce the pre-plumbing code bitwise,
  verified against a committed pre-change output baseline recorded from
  this exact fixture (``fixtures/params_physics_prechange_default.npz``)
  plus an in-test twin call (machine-independent determinism check).
* **Wiring, not tolerance** — every probed parameter class must *change*
  the outputs physics says it must (``kernel_arg`` sample: ``albedo_b``,
  ``cyl``; ``loop_local`` sample: ``transVeg``, ``height``; leaf-days via a
  DOY-sensitive override), while overrides that keep a parameter inside its
  default leaf-on window leave the outputs bitwise identical. Identical
  output where physics demands change is a plumbing failure, and a
  half-wired override that reaches only one of ``anisotropic_sky``'s two
  documented sites fails the kernel-argument capture below.
* **Windowed differential** — an override run through ``out_window``
  (window == tensors, oracle extent) stays bitwise identical to the
  full-tile run inside the window, with the override active.
* **Seam validation** — blocked names never reach the physics: the fold
  seam refuses them with a typed error, and the ``run_utci_window`` seam
  itself refuses every name outside ``PLUMBING_CLASSIFICATION`` so a
  mistyped override can never silently miss the physics.

Two overrides (``cyl=False`` and ``anisotropic_sky=0``) used to crash
inside ``solweig.py``'s previously DEAD branches — latent GPU-port bugs
that predated the plumbing packet (both values were hardcoded module
constants, so those branches had never executed). The U-C2 review ruling
repaired both branches upstream-faithfully (the ``torch.where`` on a
Python-bool condition became upstream's scalar if/else again; the
memory-cleanup ``del`` of loop-local names is gated on the loop that
binds them), so the tests below now assert the *fixed* behaviour: the
cyl=0 cardinal direct beam matches the upstream Kside_veg_v2022a formula
inside a DERIVED two-rounding tolerance, the anisotropic_sky=0 isotropic
longwave path runs to completion, and all four cyl/anisotropic_sky Sstr
gates (solweig.py:2286-2297) execute end to end. A zero-side suite pins
that decoupled outputs stay bitwise identical under each override.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import solweig_gpu.utci_process as utci_process
from solweig_gpu.utci_process import load_cached_svf_outputs, run_utci_window

from solweig_gpu.incremental import MODEL_PARAMETERS_BLOCKED
from solweig_gpu.incremental.adapters.model_parameters import (
    ADAPTER_ID,
    PARAMETER_DEFAULTS,
    PLUMBING_CLASSIFICATION,
    before_values_from_deltas,
    kernel_arguments_from_deltas,
)
from solweig_gpu.incremental.edit_types import (
    ModelParameterChange,
    ModelParameterDelta,
    SourceDeltaError,
)
from solweig_gpu.incremental.geometry import TreeSpec
from solweig_gpu.incremental.solver import (
    load_landcover_classes,
    load_site_forcing,
)

from tests.test_incremental_adapters_model_params import (
    context,
    param_command,
    make_adapter,
)
from tests.test_incremental_worker import (
    DATE_STR,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
)

# ---------------------------------------------------------------------------
# Fixture: a small real-physics site (real svf_calculator sky-view field)
# ---------------------------------------------------------------------------

ROWS = COLS = 64
PIXEL = 2.0
ORIGIN = (1000.0, 2000.0)
EPSG = 32616
#: One tree south-west of the building cluster: vegetation must be present
#: for ``transVeg`` and the leaf-day parameters to have any effect.
FIXTURE_TREE = TreeSpec(
    "p-tree",
    ORIGIN[0] + 18.5 * PIXEL,
    ORIGIN[1] - 46.5 * PIXEL,
    8.0,
    3.0,
)
#: DOY 172 forcing (``_met_row``) is leaf-on under the default
#: ``firstdayleaf=97 < 172 < lastdayleaf=300`` window, so leaf-day overrides
#: can flip the mask either way (DOY-sensitive fixture).
MET_HOURS = range(10, 14)
VARIABLES = ("utci", "tmrt", "kup", "kdown", "lup", "ldown", "shadow")

#: Pre-change baseline: default-path outputs of ``run_utci_window`` recorded
#: from the pre-plumbing code (no ``model_parameters`` argument existed) on
#: this fixture. Like the WIN-002 gate this is a machine-local bitwise
#: reference; the in-test twin calls below cover reproducibility everywhere
#: else. Regenerate ONLY from unmodified physics via the documented gate.
PRECHANGE_NPZ = (
    Path(__file__).parent / "fixtures" / "params_physics_prechange_default.npz"
)

#: The real kernel, captured before any monkeypatching, for direct replays.
_KERNEL = utci_process.Solweig_2022a_calc

#: ``Solweig_2022a_calc`` return names in order (solweig.py return
#: statement; the utci_process.py call site unpacks this exact sequence).
_KERNEL_RETURN_NAMES = (
    "Tmrt", "Kdown", "Kup", "Ldown", "Lup", "Tg", "ea", "esky", "I0",
    "CI", "shadow", "firstdaytime", "timestepdec", "timeadd", "Tgmap1",
    "Tgmap1E", "Tgmap1S", "Tgmap1W", "Tgmap1N", "Keast", "Ksouth",
    "Kwest", "Knorth", "Least", "Lsouth", "Lwest", "Lnorth", "KsideI",
    "TgOut1", "TgOut", "radI", "radD", "Lside", "L_patches", "CI_Tg",
    "CI_TgG", "KsideD", "dRad", "Kside",
)


def replay_kernel_call(call: dict, **overrides) -> dict:
    """Re-run one captured kernel call and return its outputs by name.

    Tensor arguments are cloned so replays cannot couple through in-place
    state (``run_utci_window`` already proved single-call determinism);
    ``precomputed_shadows``/``out_slice``/``sky_masks`` keep their None
    defaults, so a full-tile replay runs the march internally — the
    bit-identical path per the windowed-differential commentary.
    """
    args = {
        name: (value.clone() if isinstance(value, torch.Tensor) else value)
        for name, value in call.items()
    }
    args.update(overrides)
    outputs = _KERNEL(**args)
    return dict(zip(_KERNEL_RETURN_NAMES, outputs))


def upstream_cardinal_selection(azimuth: float, t: float = 0.0) -> dict:
    """Upstream ``Kside_veg_v2022a`` direct-beam windows, verbatim.

    The four scalar if/else guards in UMEP's
    ``SOLWEIGpython/Kside_veg_v2022a.py`` (``### Kside with weights ###``)
    with the kernel's ``t = 0.`` — the exact conditions the GPU port's
    restored if/else evaluates.
    """
    return {
        "east": azimuth > (360 - t) or azimuth <= (180 - t),
        "south": azimuth > (90 - t) and azimuth <= (270 - t),
        "west": azimuth > (180 - t) and azimuth <= (360 - t),
        "north": azimuth <= (90 - t) or azimuth > (270 - t),
    }


def build_physics_fixture(root: Path) -> SimpleNamespace:
    """One prepared site with a real SVF field and 4 daytime timesteps."""
    grid, site = _make_prepared_site(
        root,
        rows=ROWS,
        cols=COLS,
        pixel=PIXEL,
        origin=ORIGIN,
        epsg=EPSG,
        base_trees=(FIXTURE_TREE,),
        met_hours=MET_HOURS,
    )
    _compute_baseline_svf(site)
    cache = _build_cache(
        site,
        root / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="params-physics",
    )
    forcing = load_site_forcing(
        cache, site_dir=site, selected_date_str=DATE_STR
    )

    building = site / "Building_DSM" / "Building_DSM_0_0.tif"

    def tensor(path: Path) -> torch.Tensor:
        t, _dataset = utci_process.load_raster_to_tensor(str(path))
        return t

    return SimpleNamespace(
        site=site,
        grid=grid,
        forcing=forcing,
        inputs={
            "a": tensor(building),
            "temp1": tensor(site / "Trees" / "Trees_0_0.tif"),
            "temp2": tensor(site / "DEM" / "DEM_0_0.tif"),
            "walls": tensor(site / "walls" / "walls_0_0.tif"),
            "dirwalls": tensor(site / "aspect" / "aspect_0_0.tif"),
        },
        landcover=tensor(site / "Landcover" / "Landcover_0_0.tif"),
        lc_class=load_landcover_classes(),
        svf_bundle=load_cached_svf_outputs(str(building), "0_0"),
    )


def run_kwargs(
    fx: SimpleNamespace,
    *,
    out_window=None,
    model_parameters=None,
) -> dict:
    """Per-run kwargs with fresh mutable tensors.

    ``run_utci_window`` clamps ``temp1`` in place, so the spatial inputs are
    cloned per run; the SVF bundle is only ever sliced (views), never
    mutated, so it is shared.
    """
    kwargs = dict(
        a=fx.inputs["a"].clone(),
        temp1=fx.inputs["temp1"].clone(),
        temp2=fx.inputs["temp2"].clone(),
        walls=fx.inputs["walls"].clone(),
        dirwalls=fx.inputs["dirwalls"].clone(),
        svf_bundle=fx.svf_bundle,
        met_file=fx.forcing.met_table,
        altitude=fx.forcing.altitude,
        azimuth=fx.forcing.azimuth,
        zen=fx.forcing.zen,
        jday=fx.forcing.jday,
        dectime=fx.forcing.dectime,
        altmax=fx.forcing.altmax,
        location=fx.forcing.location,
        scale=1.0 / PIXEL,
        landcover_grid=fx.landcover.clone(),
        lc_class=fx.lc_class,
        windcoeff=None,
        windcoeff_by_dir=None,
        time_start=0,
        time_stop=None,
        requested_variables=VARIABLES,
        save_wbgt=False,
    )
    if out_window is not None:
        kwargs["out_window"] = out_window
    if model_parameters is not None:
        kwargs["model_parameters"] = dict(model_parameters)
    return kwargs


def run_physics(
    fx: SimpleNamespace, *, model_parameters=None, out_window=None
) -> dict[str, np.ndarray]:
    return run_utci_window(
        **run_kwargs(
            fx, out_window=out_window, model_parameters=model_parameters
        )
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def max_abs_delta(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray], variable: str
) -> float:
    b = before[variable].astype(np.float64)
    a = after[variable].astype(np.float64)
    both = ~np.isnan(b) & ~np.isnan(a)
    if not both.any():
        return 0.0
    return float(np.max(np.abs(np.where(both, a - b, 0.0))))


def assert_changed(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    variables,
    *,
    label: str,
) -> None:
    """Wiring assertion: physics says these outputs must move — identical
    output means the override never reached the kernel (no tolerance)."""
    for variable in variables:
        delta = max_abs_delta(before, after, variable)
        assert delta > 0.0, (
            f"{label}: {variable} is bitwise unchanged "
            f"(max|delta| = {delta}); the override never reached the physics"
        )


def assert_bitwise(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    *,
    label: str,
) -> None:
    for variable in VARIABLES:
        assert np.array_equal(
            before[variable], after[variable], equal_nan=True
        ), f"{label}: {variable} differs"


def record_kernel_arguments(monkeypatch, fx: SimpleNamespace) -> list[dict]:
    """Capture the per-timestep arguments actually handed to the kernel.

    The recorder wraps the real kernel, so the physics is unchanged; every
    captured call dict maps the kernel signature's parameter names to the
    values ``run_utci_window`` forwarded.
    """
    original = utci_process.Solweig_2022a_calc
    positional = [
        name
        for name, p in inspect.signature(original).parameters.items()
        if p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    recorded: list[dict] = []

    def recorder(*args, **kwargs):
        recorded.append(dict(zip(positional, args)))
        return original(*args, **kwargs)

    monkeypatch.setattr(utci_process, "Solweig_2022a_calc", recorder)
    return recorded


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> SimpleNamespace:
    return build_physics_fixture(tmp_path_factory.mktemp("params_physics"))


@pytest.fixture(scope="module")
def default_run(fx) -> dict[str, np.ndarray]:
    return run_physics(fx)


# ---------------------------------------------------------------------------
# (a) Default path: bitwise identical to the pre-change code
# ---------------------------------------------------------------------------


class TestDefaultPathBitIdentical:
    def test_none_matches_prechange_baseline(
        self, default_run: dict[str, np.ndarray]
    ) -> None:
        """Default-path outputs equal the committed pre-change recording."""
        with np.load(PRECHANGE_NPZ) as baseline:
            assert set(baseline.files) == set(VARIABLES)
            for variable in VARIABLES:
                assert np.array_equal(
                    default_run[variable],
                    baseline[variable],
                    equal_nan=True,
                ), f"{variable}: default path drifted from the pre-change code"

    def test_twin_call_empty_and_explicit_defaults_bitwise(
        self, fx, default_run
    ) -> None:
        """In-test twins: None == rerun == {} == all-defaults mapping."""
        assert_bitwise(default_run, run_physics(fx), label="rerun(None)")
        assert_bitwise(
            default_run,
            run_physics(fx, model_parameters={}),
            label="empty mapping",
        )
        assert_bitwise(
            default_run,
            run_physics(fx, model_parameters=dict(PARAMETER_DEFAULTS)),
            label="explicit PARAMETER_DEFAULTS mapping",
        )


# ---------------------------------------------------------------------------
# (b) Each plumbing class actually changes the physics
# ---------------------------------------------------------------------------


class TestOverridesReachThePhysics:
    def test_kernel_arg_albedo_b(self, fx, default_run) -> None:
        after = run_physics(fx, model_parameters={"albedo_b": 0.35})
        assert_changed(
            default_run, after, ("tmrt", "kup", "kdown"), label="albedo_b"
        )

    def test_kernel_arg_cyl(self, fx, default_run, monkeypatch) -> None:
        """cyl=False reaches the kernel AND runs the repaired cyl=0 branch.

        ``cyl`` was a hardcoded module constant (``cyl = True``) before the
        plumbing packet, so ``Kside_veg_v2022a``'s ``cyl == 0`` branch was
        dead code in the GPU port: its ``torch.where(python_bool, Tensor,
        Tensor)`` conditions were a numpy-era idiom that torch 2.x rejects
        (TypeError). The U-C2 review ruling restored upstream UMEP's scalar
        if/else, so the override now completes and moves Tmrt/UTCI (the
        box receptor weights the cardinal fluxes differently than the
        cylinder); the exact cardinal values are pinned against the
        upstream formula in TestDeadBranchFixes below.
        """
        recorded = record_kernel_arguments(monkeypatch, fx)
        after = run_physics(fx, model_parameters={"cyl": False})
        assert recorded
        for call in recorded:
            assert call["cyl"] is False, (
                f"kernel received cyl={call['cyl']!r}; the override must be "
                "forwarded (wiring), independent of the branch repair"
            )
        assert_changed(
            default_run, after, ("tmrt", "utci"), label="cyl=False"
        )

    @pytest.mark.parametrize(
        "override",
        ({"Fside": 0.3}, {"Fup": 0.1}, {"Fcyl": 0.35}),
        ids=("Fside", "Fup", "Fcyl"),
    )
    def test_kernel_arg_weighting_fractions(
        self, fx, default_run, override
    ) -> None:
        after = run_physics(fx, model_parameters=override)
        assert_changed(
            default_run, after, ("tmrt",), label=str(override)
        )

    def test_loop_local_trans_veg(self, fx, default_run) -> None:
        after = run_physics(fx, model_parameters={"transVeg": 0.4})
        assert_changed(
            default_run, after, ("tmrt", "kdown"), label="transVeg"
        )

    def test_loop_local_height(self, fx, default_run) -> None:
        after = run_physics(fx, model_parameters={"height": 2.0})
        assert_changed(default_run, after, ("tmrt",), label="height")

    def test_leaf_days_doy_sensitive(self, fx, default_run) -> None:
        # 200 > DOY 172 flips leaf-off (psi 0.03 -> 0.5 under the tree)
        leaf_off = run_physics(fx, model_parameters={"firstdayleaf": 200})
        assert_changed(
            default_run, leaf_off, ("tmrt", "kdown"), label="firstdayleaf=200"
        )
        # 150 still leaves DOY 172 inside [firstdayleaf, lastdayleaf): the
        # leaf mask is unchanged, so the outputs must be bitwise identical
        # (DOY-sensitivity proven both ways).
        still_on = run_physics(fx, model_parameters={"firstdayleaf": 150})
        assert_bitwise(
            default_run, still_on, label="firstdayleaf=150 (leaf still on)"
        )
        later_end = run_physics(fx, model_parameters={"lastdayleaf": 310})
        assert_bitwise(
            default_run, later_end, label="lastdayleaf=310 (leaf still on)"
        )


# ---------------------------------------------------------------------------
# (b0) The ZERO side of the differential: decoupled outputs stay bitwise
# ---------------------------------------------------------------------------


class TestZeroSideOfTheDifferential:
    """Overrides leave their PHYSICALLY DECOUPLED outputs bitwise intact.

    The wiring assertions above prove each override moves what physics
    couples it to; these pin the opposite side — the outputs that must NOT
    move stay BITWISE identical (no tolerance): a single flipped bit on
    the zero side means the override leaked into state it must not touch
    (reviewer-verified decoupling table, U-C2 review LOW).

    * ``albedo_b`` is ground SHORTWAVE reflectivity: it enters Kdown/Kup/
      Kside and the gvf albedo walk, never the longwave emissivity chain
      (``emis_grid``/``ewall``/``esky``) nor the shadow march.
    * ``ewall``/``elvis`` are LONGWAVE emissivities (wall emission and the
      esky correction, solweig.py:2046): no shortwave budget term and no
      shadow term sees them.
    * ``height`` only scales the GVF walk reach (``first``/``second``);
      Kdown's direct+diffuse+reflection terms and the shadow march are
      reach-independent. (``kup``/``lup`` legitimately move through the
      gvf walk products.)
    * ``absK``/``absL``/``Fside``/``Fup``/``Fcyl`` enter only the Sstr
      radiant-flux sum; every stored radiation field precedes Sstr.
    * ``cyl`` only switches the Kside direct-beam geometry and the Sstr
      merge; every stored field is computed before either.
    """

    def test_albedo_b_leaves_longwave_and_shadow_bitwise(
        self, fx, default_run
    ) -> None:
        after = run_physics(fx, model_parameters={"albedo_b": 0.35})
        for variable in ("lup", "ldown", "shadow"):
            np.testing.assert_array_equal(
                after[variable],
                default_run[variable],
                err_msg=f"albedo_b: {variable} is shortwave-decoupled",
            )

    @pytest.mark.parametrize(
        "override", ({"ewall": 0.85}, {"elvis": 1}), ids=("ewall", "elvis")
    )
    def test_longwave_emissivities_leave_shortwave_bitwise(
        self, fx, default_run, override
    ) -> None:
        after = run_physics(fx, model_parameters=override)
        for variable in ("kup", "kdown", "shadow"):
            np.testing.assert_array_equal(
                after[variable],
                default_run[variable],
                err_msg=f"{override}: {variable} is longwave-decoupled",
            )

    def test_height_leaves_kdown_and_shadow_bitwise(
        self, fx, default_run
    ) -> None:
        after = run_physics(fx, model_parameters={"height": 2.0})
        for variable in ("kdown", "shadow"):
            np.testing.assert_array_equal(
                after[variable],
                default_run[variable],
                err_msg=f"height: {variable} is GVF-reach-decoupled",
            )

    @pytest.mark.parametrize(
        "override",
        (
            {"absK": 0.6},
            {"absL": 0.9},
            {"Fside": 0.3},
            {"Fup": 0.1},
            {"Fcyl": 0.35},
            {"cyl": False},
        ),
        ids=("absK", "absL", "Fside", "Fup", "Fcyl", "cyl"),
    )
    def test_sstr_only_parameters_leave_every_field_bitwise(
        self, fx, default_run, override
    ) -> None:
        after = run_physics(fx, model_parameters=override)
        for variable in ("kup", "kdown", "lup", "ldown", "shadow"):
            np.testing.assert_array_equal(
                after[variable],
                default_run[variable],
                err_msg=f"{override}: {variable} precedes Sstr",
            )


# ---------------------------------------------------------------------------
# (c) anisotropic_sky: the override replaces BOTH documented sites
# ---------------------------------------------------------------------------


class TestAnisotropicSkyBothSites:
    """``anisotropic_sky`` is DOUBLE-SITED: the loop-local assignment
    (``anisotropic_sky = 1``, pre-plumbing utci_process.py:613) is the
    single source of the value forwarded as a kernel argument at the
    ``Solweig_2022a_calc`` call. The captured kernel argument can only be
    the override if that assignment was replaced: a naive kernel-arg-only
    override resolved *before* the loop-local line would be silently
    overwritten back to 1, and the capture below would fail — that is the
    "loop-local site matters" construction.

    The anisotropic_sky=0 *kernel* path used to crash: with the loop
    hardcoded to 1, ``Kside_veg_v2022a``'s isotropic path never ran, and
    its cleanup ``del temp_vegsh, temp_vbsh, temp_sh`` referenced names
    only the anisotropic loop binds (UnboundLocalError). The U-C2 review
    ruling gated that GPU-port-only ``del`` on the loop that binds the
    names, so the override now runs the isotropic physics end to end (the
    isotropic Ldown itself is pinned in TestDeadBranchFixes below).
    """

    def test_override_replaces_loop_local_and_kernel_arg(
        self, fx, monkeypatch
    ) -> None:
        recorded = record_kernel_arguments(monkeypatch, fx)
        # default control: the recorder sees the loop-local default 1
        run_physics(fx)
        assert recorded
        for call in recorded:
            assert call["anisotropic_sky"] == 1
        recorded.clear()

        after = run_physics(fx, model_parameters={"anisotropic_sky": 0})
        assert recorded, "the kernel was never called"
        for call in recorded:
            # BOTH sites replaced: the forwarded kernel argument carries the
            # override, and its only source is the (replaced) loop-local
            # assignment — value 1 here would mean the loop-local site won.
            assert call["anisotropic_sky"] == 0, (
                f"kernel received anisotropic_sky="
                f"{call['anisotropic_sky']!r}"
            )
        # the repaired isotropic path runs to completion with finite physics
        for variable in ("tmrt", "kup", "kdown", "lup", "ldown", "shadow"):
            assert np.isfinite(after[variable]).all(), variable

    def test_identity_override_is_bitwise(self, fx, default_run) -> None:
        after = run_physics(fx, model_parameters={"anisotropic_sky": 1})
        assert_bitwise(
            default_run, after, label="anisotropic_sky=1 (identity)"
        )


# ---------------------------------------------------------------------------
# (c2) The repaired dead branches: upstream-faithful values, not just "runs"
# ---------------------------------------------------------------------------


class TestDeadBranchFixes:
    """The two pre-existing solweig.py dead branches, now repaired.

    Both branches were unreachable while ``cyl = True`` and
    ``anisotropic_sky = 1`` were hardcoded module constants. The U-C2
    review ruling fixed them upstream-faithfully rather than fencing the
    parameters; these tests pin the FIXED branches against upstream UMEP
    semantics, so a future edit that changes the repaired physics fails
    here even though every default-path test stays green.
    """

    def test_cyl_false_cardinal_direct_matches_upstream_formula(
        self, fx, monkeypatch
    ) -> None:
        """cyl=False direct beam == upstream Kside_veg_v2022a if/else.

        Under ``anisotropic_sky = 0`` the kernel's cardinal flux is exactly
        one addition away from the direct term (solweig.py:806/809/812/815,
        ``Keast = KeastI + KeastDG``), so replaying one captured timestep
        with cyl flipped isolates the repaired branch:

        * selection — the direction(s) whose upstream azimuth window
          contains the sun carry the beam; every other direction must be
          BITWISE identical between the two runs (its ``KeastI`` is the
          same zeros tensor in both);
        * value — the expected term is reproduced with the exact internal
          op chain (``radI * shadow * cos(altitude*deg2rad) *
          sin(aziX*deg2rad)``, fp32 0-dim trig, ``t = 0.``), so it equals
          the internal ``KeastI`` bitwise, and the only rounding left is

              diff = Keast(cyl=False) - Keast(cyl=True)
                   = fl(fl(a + b) - b)      with b = KeastDG (0 + b == b)

          Each ``fl`` contributes a relative error of at most the unit
          roundoff u = 2**-24 (the cardinal fluxes are float32), so the
          standard two-rounding bound gives
          ``|diff - a| <= (2u / (1 - 2u)) * (|a| + |b|)`` — DERIVED, not
          fitted; the observed deviations are half-ulp of the sum (a
          single fl(a+b) rounding), ~2x inside the bound.
        """
        u = 2.0 ** -24  # float32 unit roundoff
        gamma2 = 2.0 * u / (1.0 - 2.0 * u)
        t = 0.0  # the kernel's local time parameter (solweig.py `t = 0.`)
        offsets = {"east": 0.0, "south": -90.0, "west": -180.0, "north": -270.0}
        deg2rad = torch.tensor(torch.pi / 180.0)  # solweig.py Kside_veg

        recorded = record_kernel_arguments(monkeypatch, fx)
        run_physics(fx, model_parameters={"anisotropic_sky": 0})
        assert recorded

        calls = list(recorded)
        # The fixture's morning azimuths (20-56 deg, 347 deg) only select
        # the east/west/north windows; two synthetic afternoon azimuths
        # exercise the remaining south selection (and a second two-face
        # case), so all four upstream if/else guards are covered selected
        # AND unselected across this test.
        for synthetic_azimuth in (150.0, 285.0):
            synthetic = dict(calls[-1])
            synthetic["azimuth"] = np.float64(synthetic_azimuth)
            calls.append(synthetic)

        saw_selected = {d: False for d in offsets}
        for call in calls:
            azimuth = float(call["azimuth"])
            selected = upstream_cardinal_selection(azimuth, t)
            base = replay_kernel_call(call, cyl=True)
            box = replay_kernel_call(call, cyl=False)
            # Kside_veg receives altitude.item() (a Python double) and
            # tensorizes it at solweig.py:636 -> default dtype float32;
            # radI is the kernel's recomputed value when onlyglobal == 1,
            # i.e. the RETURNED radI, not the met-file argument.
            altitude = torch.tensor(float(call["altitude"]))
            rad_i = base["radI"]
            shadow = base["shadow"]
            for direction, offset in offsets.items():
                key = "K" + direction
                if not selected[direction]:
                    assert torch.equal(box[key], base[key]), (
                        f"azimuth={azimuth}: {key} is outside the upstream "
                        "direct-beam window, so cyl cannot change it "
                        "(expected bitwise identity)"
                    )
                    continue
                saw_selected[direction] = True
                azi = azimuth + offset + t
                expected = (
                    rad_i
                    * shadow
                    * torch.cos(altitude * deg2rad)
                    * torch.sin(azi * deg2rad)
                )
                deviation = (box[key] - base[key] - expected).abs()
                tolerance = gamma2 * (expected.abs() + base[key].abs())
                assert bool((deviation <= tolerance).all()), (
                    f"azimuth={azimuth}: {key} direct beam deviates from the "
                    f"upstream formula by max {float(deviation.max()):.3e} "
                    f"over the derived bound {float(tolerance.max()):.3e}"
                )
            # cyl=False kills the cylinder-side direct term by definition
            # (upstream: KsideI = shadow * 0).
            assert bool((box["KsideI"] == 0).all())

        assert all(saw_selected.values()), (
            "every cardinal direction must be exercised with the direct "
            f"beam somewhere in this test (got {saw_selected})"
        )

    def test_anisotropic_zero_isotropic_longwave_path(
        self, fx, default_run
    ) -> None:
        """anisotropic_sky=0 runs the isotropic Ldown path and completes.

        The Ldown gate (solweig.py:2231) swaps Martin & Berdahl-style
        ``Lcyl_v2022a`` sky patches for the Jonsson et al. (2006) formula,
        and the diffuse split swaps the Perez luminance weighting for
        ``dRad = radD * svfbuveg`` — both must move their outputs (ldown,
        kdown) while the shadow march, which neither switch touches, stays
        bitwise identical. Before the repair this path died inside
        ``Kside_veg_v2022a`` on the ungated cleanup ``del``.
        """
        after = run_physics(fx, model_parameters={"anisotropic_sky": 0})
        for variable in ("tmrt", "kup", "kdown", "lup", "ldown", "shadow"):
            assert np.isfinite(after[variable]).all(), variable
        assert_changed(
            default_run,
            after,
            ("tmrt", "ldown", "kdown"),
            label="anisotropic_sky=0",
        )
        # zero side: the shadow march is switch-independent
        np.testing.assert_array_equal(
            after["shadow"], default_run["shadow"]
        )
        # the UTCI polynomial's undefined-domain NaN mask is a property of
        # the fixture's climate, not of the sky-diffuse switch
        np.testing.assert_array_equal(
            np.isnan(after["utci"]), np.isnan(default_run["utci"])
        )

    @pytest.mark.parametrize(
        "cyl, anisotropic_sky",
        ((True, 1), (True, 0), (False, 1), (False, 0)),
        ids=("cyl1_aniso1", "cyl1_aniso0", "cyl0_aniso1", "cyl0_aniso0"),
    )
    def test_all_four_cyl_aniso_sstr_gates(
        self, fx, default_run, cyl, anisotropic_sky
    ) -> None:
        """Every (cyl, anisotropic_sky) Sstr gate executes end to end.

        The four-receptor gates at solweig.py:2286-2297 (cylinder with
        isotropic sky, cylinder with Perez/Martin-Berdahl anisotropy, and
        the standing-box else for both cyl=0 cases) plus the L-side merge
        at :2278 must all complete with finite physics and move Tmrt
        (each gate weights the cardinal fluxes differently). Spelling the
        defaults explicitly ((True, 1)) must stay bitwise identical to the
        default run — the two switches are pure selectors.
        """
        after = run_physics(
            fx,
            model_parameters={"cyl": cyl, "anisotropic_sky": anisotropic_sky},
        )
        for variable in ("tmrt", "kup", "kdown", "lup", "ldown", "shadow"):
            assert np.isfinite(after[variable]).all(), variable
        np.testing.assert_array_equal(
            after["shadow"], default_run["shadow"]
        )
        if (cyl, anisotropic_sky) == (True, 1):
            assert_bitwise(
                default_run, after, label="explicit default switches"
            )
        else:
            assert_changed(
                default_run,
                after,
                ("tmrt",),
                label=f"cyl={cyl}, anisotropic_sky={anisotropic_sky}",
            )


# ---------------------------------------------------------------------------
# Unchanged parameters keep their module defaults
# ---------------------------------------------------------------------------


class TestUnchangedParametersKeepDefaults:
    # kernel-forwarded parameters (loop-local firstdayleaf/lastdayleaf are
    # consumed by the leaf mask, never forwarded; the leaf-day no-op tests
    # above pin their defaults bitwise)
    _MODULE_DEFAULTS = {
        "albedo_b": "albedo_b",
        "absK": "absK",
        "absL": "absL",
        "ewall": "ewall",
        "Fside": "Fside",
        "Fup": "Fup",
        "Fcyl": "Fcyl",
        "cyl": "cyl",
        "elvis": "elvis",
    }

    def test_partial_mapping_leaves_other_kernel_args_alone(
        self, fx, monkeypatch
    ) -> None:
        recorded = record_kernel_arguments(monkeypatch, fx)
        run_physics(fx, model_parameters={"height": 2.0})
        assert recorded
        for call in recorded:
            for arg, module_name in self._MODULE_DEFAULTS.items():
                assert call[arg] == getattr(utci_process, module_name), (
                    f"{arg} forwarded {call[arg]!r}; the mapping carried no "
                    f"{arg} override, so the module constant must be used"
                )
            assert call["anisotropic_sky"] == 1  # loop-local default


# ---------------------------------------------------------------------------
# (d) Blocked/unknown names never reach the physics
# ---------------------------------------------------------------------------


class TestSeamRefusesBlockedAndUnknownNames:
    def test_fold_seam_refuses_blocked_names_typed(self) -> None:
        for name in sorted(MODEL_PARAMETERS_BLOCKED):
            with pytest.raises(SourceDeltaError) as excinfo:
                kernel_arguments_from_deltas(
                    [
                        ModelParameterDelta(
                            source_node_id="model_parameters",
                            adapter_id=ADAPTER_ID,
                            parameters=(ModelParameterChange(name, None, 2),),
                        )
                    ]
                )
            # executor-facing error is TYPED, not string-matched dispatch
            assert type(excinfo.value) is SourceDeltaError, name

    def test_physics_seam_refuses_unknown_names(self, fx) -> None:
        for name in ("patch_option", "albedo_g", "scale", "walllimit"):
            with pytest.raises(ValueError, match="unknown model parameters"):
                run_physics(fx, model_parameters={name: 2})
        with pytest.raises(ValueError, match="unknown model parameters"):
            run_physics(fx, model_parameters={"transmissivity": 0.4})

    def test_physics_seam_names_mirror_plumbing_classification(self) -> None:
        """The physics seam accepts exactly the classified parameter set,
        and every duplicated default equals its cited code constant."""
        assert set(utci_process._MODEL_PARAMETER_NAMES) == set(
            PLUMBING_CLASSIFICATION
        )
        for name, module_name in {
            "albedo_b": "albedo_b",
            "ewall": "ewall",
            "absK": "absK",
            "absL": "absL",
            "Fside": "Fside",
            "Fup": "Fup",
            "Fcyl": "Fcyl",
            "cyl": "cyl",
            "elvis": "elvis",
            "firstdayleaf": "firstdayleaf",
            "lastdayleaf": "lastdayleaf",
        }.items():
            assert PARAMETER_DEFAULTS[name] == getattr(
                utci_process, module_name
            ), name
        # loop-local defaults are literals in the plumbing comments; keep
        # the adapter's documented defaults pinned to them
        assert PARAMETER_DEFAULTS["transVeg"] == 3.0 / 100.0
        assert PARAMETER_DEFAULTS["height"] == 1.1
        assert PARAMETER_DEFAULTS["anisotropic_sky"] == 1


# ---------------------------------------------------------------------------
# (e) before_value=None resolves from PARAMETER_DEFAULTS
# ---------------------------------------------------------------------------


class TestBeforeValueResolution:
    def test_undeclared_before_resolves_from_defaults(self) -> None:
        """No prior edit: before is None and the executor resolves each
        parameter from the documented scenario defaults (the only
        authoritative fallback short of a parameter store)."""
        edit = make_adapter().validate(
            param_command(
                "update",
                edit_id="params-e",
                new_state={"albedo_b": 0.35, "transVeg": 0.4},
            ),
            context(),
        )
        for change in edit.delta.parameters:
            assert change.before_value is None
        folded = kernel_arguments_from_deltas([edit.delta])
        before = before_values_from_deltas([edit.delta])
        assert before == {}  # nothing declared
        # executor rule: before.get(name) if not None else PARAMETER_DEFAULTS
        resolved = {
            name: (
                before[name]
                if before.get(name) is not None
                else PARAMETER_DEFAULTS[name]
            )
            for name in folded
        }
        assert resolved == {"albedo_b": 0.2, "transVeg": 0.03}

    def test_declared_before_wins_over_defaults(self) -> None:
        edit = make_adapter().validate(
            param_command(
                "update",
                edit_id="params-e2",
                old_state={"albedo_b": 0.3},
                new_state={"albedo_b": 0.35},
            ),
            context(),
        )
        before = before_values_from_deltas([edit.delta])
        assert before == {"albedo_b": 0.3}  # the declared value is used


# ---------------------------------------------------------------------------
# Windowed differential: override active through the out_window path
# ---------------------------------------------------------------------------


class TestWindowedDifferential:
    OUT_WINDOW = (12, 52, 12, 52)

    @pytest.mark.parametrize(
        "override",
        (
            {"albedo_b": 0.35},  # kernel_arg
            {"transVeg": 0.4},  # loop_local
        ),
        ids=("kernel_arg", "loop_local"),
    )
    def test_windowed_run_with_override_matches_full_tile(
        self, fx, override
    ) -> None:
        full = run_physics(fx, model_parameters=override)
        windowed = run_physics(
            fx, model_parameters=override, out_window=self.OUT_WINDOW
        )
        r0, r1, c0, c1 = self.OUT_WINDOW
        for variable in VARIABLES:
            np.testing.assert_array_equal(
                windowed[variable],
                full[variable][:, r0:r1, c0:c1],
                err_msg=f"{variable}: windowed override run differs",
            )
