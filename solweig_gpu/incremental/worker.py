# SPDX-License-Identifier: GPL-3.0-only
"""Exact incremental worker job lifecycle (Phase 5, Stages 1-10).

The :class:`ExactWorker` executes one design-tool job against a loaded site
cache and a :class:`~solweig_gpu.incremental.trees.TreeLayer` holding the
user's pending tree edits:

1. coalesce the layer's edits (:func:`coalesce_tree_edits`);
2. compute conservative dirty windows (:func:`dirty_windows_for_batch`);
   a staged-but-unpublished scenario forcing overlay (u-c3,
   :meth:`ExactWorker.stage_forcing_overlay`) or model-parameter payload
   (u-c7, :meth:`ExactWorker.stage_model_parameters`) is pending work on
   its own and forces the recompute set to the full tile — both are
   site-global — while a staged-but-unpublished land-cover paint overlay
   (u-c4, :meth:`ExactWorker.stage_landcover_overlay`) joins the dirty
   windows with its patch footprints — a paint is windowed;
3. choose local vs. full recomputation (:func:`choose_recompute_mode`);
   unsafe jobs fall back to the standard full-tile path — never to relaxed
   tolerances;
4. build the read window (write window + exactness halo) and drive
   :mod:`solweig_gpu.incremental.solver`;
5. stage and atomically publish versioned result patches (Stage 10), with
   cooperative supersession checks between stages: if the scene revision
   moved on, the job publishes nothing.

The worker never mutates the shared baseline cache: all writes go to patch
directories under the caller-provided results root. Its publication
watermarks (the edit sequence ack, the forcing/land-cover published
overlays, and the published model-parameter payload) advance only on a
completed publish, and the executor pairs every
dispatch with a :attr:`ExactWorker.publication_watermarks` snapshot it can
hand back through :meth:`ExactWorker.rollback_pending` — the failure-path
hook that keeps worker-publish and executor-commit lined up: both survive a
batch, or both roll back (u-c1b M1).
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field, replace
from math import floor, radians, tan
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from solweig_gpu.incremental.cache import SiteCache
from solweig_gpu.incremental.checkpoints import thermal_fingerprint
from solweig_gpu.incremental.adapters.landcover import LandCoverOverlay
from solweig_gpu.incremental.adapters.met_time import ForcingOverlay
from solweig_gpu.incremental.adapters.model_parameters import (
    PLUMBING_CLASSIFICATION,
    check_parameter_value,
)
from solweig_gpu.incremental.edits import (
    CoalescedEditBatch,
    coalesce_tree_edits,
    dirty_windows_for_batch,
)
from solweig_gpu.incremental.edit_types import EditStateError
from solweig_gpu.incremental.geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    RecomputeMode,
    SunPosition,
    TreeSpec,
    choose_recompute_mode,
    local_elevation_offset_m,
    merge_windows,
    tree_influence_bounds_m,
)
from solweig_gpu.incremental.result import (
    ResultPatch,
    discard_staging,
    publish_staged_patch,
    stage_patch,
    new_job_id,
)
from solweig_gpu.incremental.solver import (    SiteForcing,
    compose_full_scene_tensors,
    gvf_march_pixels,
    load_site_forcing,
    read_window_for_write_window,
    resolve_landcover_overlay,
    run_full_tile,
    solve_window,
    solve_with_core_adapter,
    SolverInputError,
)
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.veg_svf_state import (
    VegOcclusionState,
    VegOcclusionStateError,
    VegOcclusionStore,
)

__all__ = [
    "ExactWorker",
    "JobOutcome",
    "SupersededJobError",
    "WorkerError",
    "discard_published_patches",
]

#: Orphan publication cleanup (u-c1b M4) is best-effort: a patch directory
#: that cannot be removed is reported here, loudly — rev-globbing readers
#: must be able to see that a leftover survived the cleanup.
_LOG = logging.getLogger(__name__)


class WorkerError(RuntimeError):
    """Worker-level failure (bad configuration, failed publish)."""


class SupersededJobError(RuntimeError):
    """The scene revision moved on while the job was in flight."""


def guard_patch_window(
    window: RasterWindow, grid: RasterGrid, *, label: str
) -> None:
    """L4 (u-c4-review item 4): the worker's patch-footprint contract.

    Every window the worker hands to the solver or publishes in a patch —
    write AND read — must be a non-empty window fully inside the site
    grid's half-open bounds. The windowing pipeline derives windows from
    in-grid dirty windows and clamps its expansions, so this guard is an
    ASSERTION, not correction: a window that escapes the grid means an
    upstream dirty-window derivation is broken, and reading or publishing
    it would touch cells that do not exist on the site. Refused loudly as
    a :class:`WorkerError` instead.
    """
    if window.is_empty:
        raise WorkerError(f"{label} window is empty: {window}")
    if (
        window.row_start < 0
        or window.col_start < 0
        or window.row_stop > grid.rows
        or window.col_stop > grid.cols
    ):
        raise WorkerError(
            f"{label} window {window} escapes the site grid "
            f"({grid.rows}x{grid.cols}); the worker must not read or write "
            "outside the declared patch footprint"
        )


def guard_read_covers_write(
    read_window: RasterWindow, write_window: RasterWindow
) -> None:
    """L4 (u-c4-review item 4): reads must cover the declared writes.

    The patch declares ``write_window`` as its footprint; the solver runs on
    ``read_window`` (the write window plus the GVF march and halo reach). A
    read window that does not CONTAIN the write window means the solve never
    looked at every cell it claims to publish — an assertion failure of the
    windowing contract, refused loudly.
    """
    if not (
        read_window.row_start <= write_window.row_start
        and read_window.col_start <= write_window.col_start
        and read_window.row_stop >= write_window.row_stop
        and read_window.col_stop >= write_window.col_stop
    ):
        raise WorkerError(
            f"read window {read_window} does not contain the declared write "
            f"window {write_window}; the solve must cover every cell it "
            "publishes"
        )


@dataclass(frozen=True)
class PublicationWatermarks:
    """The worker's publication watermarks as one snapshot (u-c1b M1).

    ``acked_sequence`` is the edit-sequence watermark; the two overlay
    fields are the forcing/land-cover overlays the last PUBLISHED job ran
    under, and ``model_parameters`` (u-c7) is the folded kernel-argument
    mapping that job ran under. A failed executor batch rewinds all four
    together through :meth:`ExactWorker.rollback_pending` so the worker's
    notion of "already published" can never drift past the executor's
    committed state.
    """

    acked_sequence: int
    forcing_overlay: ForcingOverlay | None
    landcover_overlay: LandCoverOverlay | None
    model_parameters: Mapping[str, Any] | None = None


@dataclass
class JobOutcome:
    """What happened to one worker job."""

    status: str  # "published" | "superseded" | "no-op"
    mode: str | None  # "local" | "full" | None for no-op
    job_id: str
    scene_revision: int
    write_windows: tuple[RasterWindow, ...] = ()
    read_windows: tuple[RasterWindow, ...] = ()
    patch_paths: tuple[Path, ...] = ()
    fallback_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def published(self) -> bool:
        return self.status == "published"


class ExactWorker:
    """Drive exact local (or full-tile fallback) evaluation for one scenario."""

    def __init__(
        self,
        cache: SiteCache,
        layer: TreeLayer,
        *,
        site_dir: str | Path,
        results_root: str | Path,
        selected_date_str: str,
        influence_config: InfluenceConfig | None = None,
        requested_variables: Sequence[str] = ("utci", "tmrt", "shadow"),
        revision_provider: Callable[[], int] | None = None,
        scratch_root: str | Path | None = None,
        core_adapter: bool = False,
    ) -> None:
        self.cache = cache
        self.layer = layer
        self.site_dir = Path(site_dir).resolve()
        self.results_root = Path(results_root)
        self.selected_date_str = selected_date_str
        self.influence_config = influence_config or InfluenceConfig()
        self.requested_variables = tuple(requested_variables)
        #: T02 opt-in seam: route every LOCAL solve through
        #: ``solve_with_core_adapter`` (the solweig_core ABI/dispatch
        #: boundary around the SAME numerical path). ``False`` (default)
        #: keeps the direct ``solve_window`` call byte-identical to the
        #: pre-T02 worker; the adapter route is asserted bit-identical by
        #: tests/ultrafast/test_adapter_roundtrip.py.
        self._core_adapter = bool(core_adapter)
        self.scratch_root = (
            Path(scratch_root) if scratch_root is not None else self.results_root / ".scratch"
        )
        self._revision_provider = revision_provider or (lambda: self.scene_revision)
        self.scene_revision = 0
        self._forcing: SiteForcing | None = None
        #: R4 Phase A: persistent vegetation SVF occluder state. ``True``
        #: (the default) enables the packed-bits corridor path for
        #: vegetation edits; every typed refusal
        #: (:class:`~solweig_gpu.incremental.veg_svf_state.\
        #: VegOcclusionFallback`) routes the batch to the proven replay /
        #: full-tile paths. Tests and operational rollbacks can disable the
        #: path wholesale by flipping this attribute.
        self._veg_occlusion_enabled = True
        self._veg_store: VegOcclusionStore | None = None
        #: State prepared by the local solve of the CURRENT job; committed
        #: by the exact-publication seam only after the publish succeeds.
        self._pending_veg_state = None
        #: G2.1 item 3: FULL-TILE thermal state captured by ``_solve_full``
        #: of the CURRENT job (``{"final_state", "fingerprint"}``), handed to
        #: the executor's publication hook — never persisted by the worker.
        #: Cleared at the start of every ``run``: a capture left behind by a
        #: superseded/failed job must never leak into a later revision's
        #: checkpoint (checkpoints commit at PUBLICATION only).
        self._pending_thermal_capture: dict[str, Any] | None = None
        #: Scenario forcing overlay (u-c3) staged for the NEXT job. The
        #: overlay rides the job: ``forcing()`` materializes it through
        #: ``load_site_forcing(overlay=...)`` (the masked re-check seam), so
        #: both the local and the full solve consume the RESOLVED table.
        #: ``None`` (never staged) is the legacy byte-identical path.
        self._forcing_overlay: ForcingOverlay | None = None
        #: The overlay under which results were last PUBLISHED. A staged
        #: overlay that differs from it marks the job pending (mirroring
        #: ``_acked_sequence``), so a forcing-only batch — no tree edits to
        #: coalesce — still recomputes and publishes instead of no-op'ing
        #: the overlay away. Publication consumes the staged overlay.
        self._published_forcing_overlay: ForcingOverlay | None = None
        #: Forcing pending by PRESENCE, not by value (u-c6b, u-c6-review
        #: met analog): a batch that carries forcing deltas alongside other
        #: real deltas marks the job pending even when the fold resolves
        #: value-equal to the published overlay — the executor sets this
        #: through :meth:`stage_forcing_overlay`'s ``batch_pending`` door.
        #: Without it a mixed batch (vegetation + a value-equal forcing
        #: fold) would route LOCAL on the tree windows alone while the
        #: forcing payload rides unacknowledged; with it the job recomputes
        #: the forcing closure (the full tile — forcing is site-global).
        #: A forcing-ONLY batch never sets it, so the u-c1b wedge retry
        #: (worker watermarks stuck ahead of the executor state) still
        #: finds nothing pending and raises the loud error.
        self._forcing_presence_pending: bool = False
        #: Scenario land-cover paint overlay (u-c4) staged for the NEXT job.
        #: Unlike forcing, a paint is WINDOWED: the staged overlay's patch
        #: windows join the job's dirty windows (plus the standard write
        #: margin), so a paint-only batch recomputes locally instead of
        #: forcing a full tile. The solver seam
        #: (``solve_window``/``run_full_tile``) materializes the resolved
        #: class grid from it. ``None`` (never staged, or reset) is the
        #: legacy byte-identical path.
        self._landcover_overlay: LandCoverOverlay | None = None
        #: The land-cover overlay under which results were last PUBLISHED
        #: (the u-c3 forcing watermark discipline, mirrored): a staged
        #: overlay that differs from it marks the job pending, and
        #: publication consumes it. A ``None`` staged against a published
        #: overlay is the drop-overlay reset — its recompute region is the
        #: previously painted footprint.
        self._published_landcover_overlay: LandCoverOverlay | None = None
        #: Explicit dirty windows for the staged paint overlay (u-c6,
        #: u-c4-review M3; u-c6b superset rule): when the executor stages
        #: an overlay it also hands the batch's dirty footprint — the
        #: STROKE windows of the batch's committed paint patches UNION the
        #: symmetric-difference windows vs the PUBLISHED overlay state —
        #: so the executed scope always covers the plan's stroke-footprint
        #: promise (the planner sees only deltas, never resolved values)
        #: while a paint-only job still never recomputes footprints the
        #: session painted EARLIER (only the current stroke dirties; the
        #: value-diff is normally a subset of the stroke, the union is for
        #: safety). ``None`` (never set — direct API callers) keeps the
        #: derived-from-patches behaviour, byte-identical to before.
        self._landcover_dirty_override: tuple[RasterWindow, ...] | None = None
        #: Folded model-parameter kernel arguments (u-c7) staged for the
        #: NEXT job: committed ``model_parameters`` deltas fold over the
        #: published state (last-write-wins per name; the parameters are
        #: time-invariant, so there is no time dimension to the payload).
        #: Like the forcing overlay it is site-global — a parameter change
        #: couples the radiation closure at EVERY cell, so a pending
        #: payload routes the job full-tile — and it rides the job: both
        #: solve paths forward it into ``run_utci_window``. ``None`` (never
        #: staged) is the legacy byte-identical path.
        self._model_parameters: dict[str, Any] | None = None
        #: The model parameters under which results were last PUBLISHED
        #: (the u-c3 watermark discipline, mirrored): a staged payload that
        #: differs from it marks the job pending — a params-only batch
        #: carries no tree edits to coalesce and must never be no-op'd
        #: away — and publication consumes the staged payload.
        self._published_model_parameters: dict[str, Any] | None = None
        #: Sequence watermark of edits already consumed by a published job.
        #: Without it, ``add -> publish -> delete`` would coalesce to a no-op
        #: while the published add patch is still in the result store; the
        #: delete job would then never recompute the removed tree's
        #: influence. Only edits newer than this watermark form a job batch.
        self._acked_sequence = 0

        grid = RasterGrid(
            rows=cache.rows,
            cols=cache.cols,
            pixel_size_m=cache.pixel_size_m,
            origin_x_m=cache.manifest.origin_x_m,
            origin_y_m=cache.manifest.origin_y_m,
        )
        self.grid = grid
        if layer.grid.rows != grid.rows or layer.grid.cols != grid.cols:
            raise WorkerError("tree layer grid does not match the site cache")
        #: Extra cells added to every dirty window to form the write window.
        #: A scene change alters Lup/shadow at the changed cells, and the
        #: ground-view-factor integration shifts those values into results up
        #: to ``gvf_march_pixels`` cells away, so the published patch must
        #: cover the dirty window plus that propagation reach (plus one for
        #: clamping). The oracle's march is ``round(22 m / pixel_size_m)``
        #: cells, so the margin is derived per site, never hard-coded.
        self.write_margin_pixels = gvf_march_pixels(cache.pixel_size_m) + 1

    # ------------------------------------------------------------------
    # Job lifecycle (Stages 1-10)
    # ------------------------------------------------------------------

    def bump_scene_revision(self) -> int:
        """Advance the worker's notion of the current scene revision."""
        self.scene_revision += 1
        return self.scene_revision

    @property
    def publication_watermarks(self) -> PublicationWatermarks:
        """Snapshot of the four publication watermarks (u-c1b M1, u-c7).

        The executor captures this BEFORE dispatching a job and hands it
        back through :meth:`rollback_pending` if the batch fails — an
        explicit per-batch pairing, so the rewind can never accidentally
        target a stale snapshot from an earlier run (e.g. when the batch
        failed before :meth:`run` was even entered).
        """
        return PublicationWatermarks(
            acked_sequence=self._acked_sequence,
            forcing_overlay=self._published_forcing_overlay,
            landcover_overlay=self._published_landcover_overlay,
            # Hand a COPY of the published mapping (publish-by-reference
            # hygiene, u-c7-review TRIVIAL): the snapshot the executor pairs
            # with a dispatch must not alias live worker state.
            model_parameters=(
                dict(self._published_model_parameters)
                if self._published_model_parameters is not None
                else None
            ),
        )

    def rollback_pending(self, watermarks: PublicationWatermarks) -> None:
        """Rewind the publication watermarks to an explicit snapshot.

        The M1 wedge fix (u-c1b): the worker advances ``_acked_sequence``
        and the forcing/land-cover published overlays the moment ITS
        publish loop completes, but the executor commits its own state
        only after recording the outcome. An executor failure in between
        (store rejection, reconciliation, metadata read) rolled the
        executor back while the worker stayed consumed — every retry then
        found "nothing pending" and raised forever. The executor calls
        this hook on its failure path, right after rolling back its own
        state, so the commit points line up: worker-publish and
        executor-commit survive together or roll back together.

        Pass the snapshot taken from :attr:`publication_watermarks`
        immediately before the failed batch was dispatched. Restoring a
        snapshot taken after a SUCCESSFUL commit would rewind published
        watermarks and is a caller bug.
        """
        self._acked_sequence = watermarks.acked_sequence
        self._published_forcing_overlay = watermarks.forcing_overlay
        self._published_landcover_overlay = watermarks.landcover_overlay
        self._published_model_parameters = (
            dict(watermarks.model_parameters)
            if watermarks.model_parameters is not None
            else None
        )
        # A rewound batch's explicit dirty windows and presence flag are
        # stale with it.
        self._landcover_dirty_override = None
        self._forcing_presence_pending = False

    def adopt_published_state(self, watermarks: PublicationWatermarks) -> None:
        """Adopt watermarks for a batch published OUTSIDE :meth:`run` (u-c6).

        The building regeneration path publishes full-tile results through
        the executor (the chain runs its own solve), so the worker's
        publication watermarks must advance to the state those results
        depict: the tree edits the chain consumed (``acked_sequence``) and
        the overlays the chain ran under. Without this, a later windowed
        batch would see the staged overlays as pending forever (forcing
        would force every later job full-tile). Monotone in the sequence
        watermark; the overlay fields mirror the publish commit points.

        Defense-in-depth contract (u-e1 L4): callers must pass FENCED
        objects — a :class:`PublicationWatermarks` built from what the
        adopting chain actually published (the executor's rollback path
        copies on receipt for the same reason). This method TRUSTS the
        watermark payload by design (the publishing chain is the only
        authority on what it consumed); it re-copies the mapping but
        performs no independent verification, so an unfenced caller
        aliasing live staged state could smuggle a later mutation past
        the watermark. No behavior change — documentation only.
        """
        self._acked_sequence = max(
            self._acked_sequence, watermarks.acked_sequence
        )
        self._published_forcing_overlay = watermarks.forcing_overlay
        self._published_landcover_overlay = watermarks.landcover_overlay
        self._published_model_parameters = (
            dict(watermarks.model_parameters)
            if watermarks.model_parameters is not None
            else None
        )
        self._landcover_dirty_override = None
        self._forcing_presence_pending = False

    def _check_not_superseded(self, target_revision: int) -> None:
        current = int(self._revision_provider())
        if current != target_revision:
            raise SupersededJobError(
                f"job targeted scene revision {target_revision} but the store "
                f"is at revision {current}"
            )

    def forcing(self) -> SiteForcing:
        """Materialize the job's forcing (baseline + staged overlay, u-c3).

        Single materialization point for both solve paths: with a staged
        overlay the loader resolves it copy-on-write against the baseline
        met table and the masked cross-check verifies the resolve (an
        undeclared-cell tamper is refused here, never trusted). Without one
        this is the legacy loader call, byte-identical.
        """
        if self._forcing is None:
            self._forcing = load_site_forcing(
                self.cache,
                site_dir=self.site_dir,
                selected_date_str=self.selected_date_str,
                overlay=self._forcing_overlay,
            )
        return self._forcing

    def stage_forcing_overlay(
        self,
        overlay: ForcingOverlay | None,
        *,
        batch_pending: bool = False,
    ) -> None:
        """Stage the scenario's effective forcing overlay for the next job.

        The executor's u-c3 seam: committed forcing deltas fold into the
        overlay the scenario carries, and the staged overlay rides the job
        (it is consumed at the next :meth:`forcing` materialization). A
        staged overlay that differs from the last PUBLISHED one marks the
        job pending, so a forcing-only batch still recomputes — the overlay
        is never silently dropped by a no-op. ``None`` restores the baseline
        forcing (drop-overlay reset semantics). Staging does not touch the
        published watermark: an unpublished or failed job leaves the
        previously published forcing state intact.

        ``batch_pending`` (u-c6b, u-c6-review met analog) marks the
        forcing payload pending by PRESENCE, not by value: the executor
        sets it when the batch carries forcing deltas alongside other real
        deltas, so a fold that merely resolves value-equal to the
        published overlay still routes the job full-tile (the forcing
        closure — forcing is site-global) instead of leaving the payload
        to ride a LOCAL tree-window job unacknowledged. The executor
        never sets it for a forcing-ONLY batch (the whole-batch value
        no-op is intercepted one level up, and a wedged retry must keep
        finding nothing pending). Publication/rollback/ adoption consume
        the flag with the overlay.
        """
        if overlay is not None and not isinstance(overlay, ForcingOverlay):
            raise WorkerError(
                "forcing overlay must be a ForcingOverlay or None, got "
                f"{type(overlay).__name__}"
            )
        self._forcing_overlay = overlay
        self._forcing_presence_pending = bool(batch_pending)
        self._forcing = None  # re-materialize lazily with the staged overlay

    def stage_landcover_overlay(
        self,
        overlay: LandCoverOverlay | None,
        *,
        dirty_windows: Sequence[RasterWindow] | None = None,
    ) -> None:
        """Stage the scenario's effective land-cover overlay for the next job.

        The executor's u-c4 seam: committed paint deltas fold into the
        overlay the scenario carries, and the staged overlay rides the job
        (``solve_window``/``run_full_tile`` materialize the resolved class
        grid from it on both solve paths). A staged overlay that differs
        from the last PUBLISHED one marks the job pending, so a paint-only
        batch still recomputes — the overlay is never silently dropped by a
        no-op. ``None`` restores the baseline class grid (drop-overlay reset
        semantics). Staging does not touch the published watermark: an
        unpublished or failed job leaves the previously published land-cover
        state intact.

        Restage-by-convention invariant (U-D, u-c4-review (h)): staging
        REPLACES the worker's staged overlay wholesale — the staged state is
        assigned, never merged — so restaging the same overlay (or staging a
        successor) is always safe and can never double-paint. Paint
        accumulation happens ONLY across PUBLISHED batches: the executor
        folds committed paint deltas into the scenario's overlay (per-cell,
        last-write-wins) before handing it here, and the worker adopts the
        staged overlay as the published watermark exactly when a job
        publishes. An overlay staged, restaged, or replaced while still
        unpublished simply overwrites staged state.

        Overlays are only stageable on sites whose cache carries a
        land-cover raster: without one the physics runs with
        ``landcover_grid=None`` and the paint could never reach it — refused
        here rather than silently dropped. Patch windows must fit the site
        grid (they become the job's dirty windows).

        ``dirty_windows`` (u-c6, u-c4-review M3; u-c6b superset rule)
        optionally carries the batch's ACTUAL dirty footprint — the
        executor's union of the batch's STROKE windows (the plan's
        write-window promise basis) with the symmetric difference between
        the staged overlay and the published overlay state — replacing
        the derived-from-patch-windows footprint. Only the CURRENT stroke
        dirties: no job recomputes footprints the session painted earlier,
        and the executed scope always covers the plan's promise (never
        less). The default ``None`` keeps the derived behaviour for
        direct API callers.
        """
        if overlay is not None and not isinstance(overlay, LandCoverOverlay):
            raise WorkerError(
                "landcover overlay must be a LandCoverOverlay or None, got "
                f"{type(overlay).__name__}"
            )
        if overlay is not None:
            if "landcover" not in self.cache:
                raise WorkerError(
                    "site cache has no landcover raster; a paint overlay "
                    "could never reach the physics — refusing the overlay"
                )
            for index, patch in enumerate(overlay.patches):
                window = patch.window
                if (
                    window.row_start < 0
                    or window.col_start < 0
                    or window.row_stop > self.cache.rows
                    or window.col_stop > self.cache.cols
                ):
                    raise WorkerError(
                        f"landcover overlay patch {index} window {window} is "
                        f"outside the site grid ({self.cache.rows}x"
                        f"{self.cache.cols}); refusing the overlay"
                    )
        override: tuple[RasterWindow, ...] | None = None
        if dirty_windows is not None:
            windows = tuple(dirty_windows)
            for index, window in enumerate(windows):
                if not isinstance(window, RasterWindow):
                    raise WorkerError(
                        "dirty_windows must be RasterWindow records, got "
                        f"{type(window).__name__} at index {index}"
                    )
                if window.is_empty:
                    continue  # an empty entry is a no-op, not corruption
                if (
                    window.row_start < 0
                    or window.col_start < 0
                    or window.row_stop > self.cache.rows
                    or window.col_stop > self.cache.cols
                ):
                    raise WorkerError(
                        f"dirty window {index} {window} is outside the site "
                        f"grid ({self.cache.rows}x{self.cache.cols}); "
                        "refusing the staged dirty footprint"
                    )
            override = tuple(
                window for window in windows if not window.is_empty
            )
        self._landcover_overlay = overlay
        self._landcover_dirty_override = override

    def stage_model_parameters(
        self, model_parameters: Mapping[str, Any] | None
    ) -> None:
        """Stage the scenario's effective model parameters for the next job.

        The executor's u-c7 seam: committed model parameter deltas fold
        (over the published state, last-write-wins per name) into this
        mapping via ``kernel_arguments_from_deltas``, and the staged
        payload rides the job — both solve paths forward it into
        ``run_utci_window``. A staged payload that differs from the last
        PUBLISHED one marks the job pending and routes it full-tile (the
        parameters are site-global: every cell's radiation closure reads
        them), so a params-only batch still recomputes — the payload is
        never silently dropped by a no-op. ``None`` (or an empty mapping)
        stages the legacy parameter-free run. Staging does not touch the
        published watermark: an unpublished or failed job leaves the
        previously published parameters intact.

        The fence mirrors the overlay staging doors: the payload must be a
        ``name -> scalar`` mapping whose names are the documented physics
        seam names (``PLUMBING_CLASSIFICATION``, mirrored by
        ``run_utci_window``'s accepted set) and whose values pass the
        adapter's documented per-parameter domain — type spelling,
        finiteness, bounds — through the public
        :func:`~solweig_gpu.incremental.adapters.model_parameters.\
check_parameter_value` wrapper (u-c7-review MEDIUM: a right-kind scalar
        with a wrong value used to stage and forward unchecked). An unknown
        name or an out-of-domain value would only
        surface deep inside the solve — refused here, before any state
        advances. The staged copy is frozen at stage time so a caller
        mutating its mapping afterwards cannot rewrite a job's physics.
        """
        if model_parameters is None:
            self._model_parameters = None
            return
        if not isinstance(model_parameters, Mapping):
            raise WorkerError(
                "model parameters must be a name -> value mapping or None, "
                f"got {type(model_parameters).__name__}"
            )
        unknown = sorted(set(model_parameters) - set(PLUMBING_CLASSIFICATION))
        if unknown:
            raise WorkerError(
                f"unknown model parameters {unknown}; accepted names are "
                f"{sorted(PLUMBING_CLASSIFICATION)}; refusing the staged "
                "payload"
            )
        staged: dict[str, Any] = {}
        for name, value in model_parameters.items():
            # bool is an int subclass; check it first so flags stay flags.
            # Float-kind values are coerced to a plain Python double (the
            # fold seam's dtype hygiene: np.float64 passes isinstance but
            # its dtype rules can diverge from a plain double inside the
            # physics — parameter differentials must stay bitwise-stable).
            if isinstance(value, bool):
                staged[name] = value
            elif isinstance(value, float):
                staged[name] = float(value)
            elif isinstance(value, int):
                staged[name] = int(value)
            else:
                raise WorkerError(
                    f"model parameter {name!r} value must be a scalar: "
                    "accepted are Python bool, int, and float (np.float64 "
                    "is accepted too — it is a float subclass and is "
                    "coerced to a plain double); NumPy scalars such as "
                    "np.float32, np.bool_, and np.int64 are NOT accepted "
                    "(their dtype rules diverge from Python scalars inside "
                    f"the physics — convert with .item()); got "
                    f"{type(value).__name__}"
                )
            # Value-domain fence (u-c7-review MEDIUM): the kind checks above
            # accept any scalar of the right kind, so {albedo_b: True},
            # {albedo_b: 50.0}, and {transVeg: inf} would all stage and ride
            # into the physics unchecked. Apply the adapter's own documented
            # domain check (public wrapper; never the private helper) to the
            # COERCED value — the exact object the job would forward.
            try:
                check_parameter_value(name, staged[name])
            except EditStateError as error:
                raise WorkerError(str(error)) from error
        # An empty mapping is the parameter-free run, normalized to None so
        # it can never mark a job pending (or force a full recompute) with
        # no parameter content behind it.
        self._model_parameters = staged or None

    def _sun_positions(self) -> list[SunPosition]:
        solar = np.asarray(self.cache.solar)
        return [
            SunPosition(altitude_deg=float(row[0]), azimuth_deg=float(row[1]))
            for row in solar
        ]

    def _minimum_positive_sun_altitude_deg(self) -> float | None:
        """Lowest altitude the oracle actually marches shadows at.

        ``compute_utci``/``run_utci_window`` only run the direct-shadow march
        for timesteps with positive solar altitude on the modeled date, so
        the true corridor floor is the series' minimum positive altitude —
        not a config guess. A config guess that is LOWER stays conservative;
        one that is HIGHER would silently truncate real corridors.
        """
        altitude = np.asarray(self.cache.solar)[:, 0]
        positive = altitude[altitude > 0.0]
        if positive.size == 0:
            return None
        return float(positive.min())

    def dirty_windows(self) -> list[RasterWindow]:
        batch = self.pending_batch()
        if batch is None or not batch.edits:
            return []
        config = self._site_influence_config()
        offsets = self._batch_elevation_offsets(batch, config)
        scalar = dirty_windows_for_batch(
            batch,
            grid=self.grid,
            sun_positions=self._sun_positions(),
            config=config,
            elevation_offset_for_tree=(
                lambda tree: offsets[_tree_position_key(tree)]
            ),
        )
        # Exact per-cell backstop: the scalar bounds above use per-tree local
        # relief and closed-form corridors; this mask evaluates the oracle's
        # own march-flip condition at every cell, closing any residual hole
        # (e.g. a deep basin outside the iterated candidate region).
        trees = [
            tree
            for edit in batch.edits
            for tree in (edit.old_tree, edit.new_tree)
            if tree is not None
        ]
        exact = self._exact_influence_window(trees, config)
        if exact is not None:
            scalar = merge_windows([*scalar, exact], gap_pixels=0)
        return scalar

    # ------------------------------------------------------------------
    # Terrain-aware influence bounds (per-tree local relief)
    # ------------------------------------------------------------------

    def _site_influence_config(self) -> InfluenceConfig:
        """Influence policy adjusted to this site's forcing and march bounds.

        Adjustment over the caller's config, conservative:

        * ``maximum_shadow_length_m`` is raised to at least the oracle's own
          march bound (``amaxvalue / tan(alt)``) at the lowest altitude the
          oracle can actually march: the minimum positive solar altitude in
          the cached series for direct-shadow corridors (the oracle only
          marches while the sun is up on the modeled date), and the
          configured lowest sky-patch altitude (6 degrees for
          ``patch_option=2``: 31 of the 153 patches sit at 6 degrees, and
          their ``vegsh`` flips feed ``svfveg`` bitwise, so the bound cannot
          drop below the patch floor even though the band carries only ~4
          percent of the hemispherical solid-angle weight). Any cap below
          those bounds could silently truncate true influence; the grid
          clamp and the full-recompute fraction check bound the work.

        The vertical amplitude is NOT inflated globally here: per-tree local
        relief is derived in :meth:`_influence_elevation_offset` (the global
        DEM-relief offset used previously inflated every small tree's bounds
        to whole-hillside windows and forced every real-site edit to
        full-tile recomputation by policy).
        """
        a = np.asarray(self.cache.building_dsm)
        dem = np.asarray(self.cache.dem)
        canopy = np.asarray(self.cache.tree_base)
        canopy = np.where(canopy < 0.0, 0.0, canopy)
        baseline_amaxvalue = float(max(a.max(), (canopy + dem).max()))

        series_min = self._minimum_positive_sun_altitude_deg()
        if series_min is None:
            # No positive altitude in the cached series: fall back to the
            # configured guess for the direct-sun corridor.
            sun_floor = self.influence_config.minimum_direct_sun_altitude_deg
        else:
            # The direct-sun corridor is capped by the ACTUAL lowest positive
            # solar altitude in the cached series (the oracle only marches
            # while the sun is up on the modeled date), not the config guess:
            # a series whose sun never drops below e.g. 45 degrees must not
            # carry 5-degree shadow lengths.
            sun_floor = series_min
        shadow_bound = baseline_amaxvalue / tan(radians(sun_floor))
        svf_bound = baseline_amaxvalue / tan(
            radians(self.influence_config.lowest_sky_patch_altitude_deg)
        )
        return replace(
            self.influence_config,
            minimum_direct_sun_altitude_deg=sun_floor,
            maximum_shadow_length_m=max(
                self.influence_config.maximum_shadow_length_m,
                shadow_bound,
                svf_bound,
            ),
        )

    def _tree_cell(self, tree: TreeSpec) -> tuple[int, int] | None:
        """Grid cell containing the tree centre, or None if off-site."""
        col = floor((tree.x_m - self.grid.origin_x_m) / self.grid.pixel_size_m)
        row = floor((self.grid.origin_y_m - tree.y_m) / self.grid.pixel_size_m)
        if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
            return int(row), int(col)
        return None

    def _influence_elevation_offset(
        self, tree: TreeSpec, config: InfluenceConfig
    ) -> float:
        """Per-tree local relief for this tree's influence bounds.

        The oracle's march compares the canopy top (``a[tree] + height``,
        the ``vegdsm`` datum the march actually uses) against each target
        cell's building-DSM elevation, so the tree's vertical amplitude is
        its height plus the drop to the lowest target surface it can reach —
        a *local* quantity. The deterministic iteration lives in
        :func:`geometry.local_elevation_offset_m`; when it cannot stabilise
        (monotone slope to the site edge) the honest bound is the drop to
        the global minimum surface, which typically routes that edit to a
        full-tile recompute.
        """
        a = np.asarray(self.cache.building_dsm)
        cell = self._tree_cell(tree)
        if cell is None:
            return config.elevation_offset_m
        base = float(a[cell[0], cell[1]])
        global_offset = max(0.0, base - float(a.min()))
        if global_offset == 0.0:
            return 0.0

        sun_positions = self._sun_positions()

        def bounds_for_offset(offset: float) -> RasterWindow:
            bounds = tree_influence_bounds_m(
                tree,
                sun_positions,
                replace(config, elevation_offset_m=offset),
            )
            return self.grid.world_bounds_to_window(
                min_x_m=bounds[0],
                min_y_m=bounds[1],
                max_x_m=bounds[2],
                max_y_m=bounds[3],
            )

        def region_minimum(window: RasterWindow) -> float:
            return float(
                a[
                    window.row_start : window.row_stop,
                    window.col_start : window.col_stop,
                ].min()
            )

        offset = local_elevation_offset_m(
            base_elevation_m=base,
            bounds_for_offset=bounds_for_offset,
            region_minimum_m=region_minimum,
        )
        return global_offset if offset is None else offset

    def _batch_elevation_offsets(
        self, batch: CoalescedEditBatch, config: InfluenceConfig
    ) -> dict[tuple[str, float, float], float]:
        offsets: dict[tuple[str, float, float], float] = {}
        for edit in batch.edits:
            for tree in (edit.old_tree, edit.new_tree):
                if tree is None:
                    continue
                key = _tree_position_key(tree)
                if key not in offsets:
                    offsets[key] = self._influence_elevation_offset(tree, config)
        return offsets

    def _exact_influence_window(
        self, trees: Sequence[TreeSpec], config: InfluenceConfig
    ) -> RasterWindow | None:
        """Window of every cell whose march comparison this edit can flip.

        For one tree at cell ``c`` with target-datum base ``a[c]``, the
        oracle's march flips a comparison at target cell ``t`` only when
        ``a[c] + height - a[t] > dist(t, c) * tan(altitude)`` for some
        marched altitude (sky patches down to 6 degrees, or a positive solar
        altitude from the forcing series). Reach is monotone decreasing in
        altitude, so the minimum marched altitude bounds every march; the
        march's diagonal step factor ``ds >= 1`` makes a Chebyshev distance
        bound conservative. Evaluating the condition at every cell (one
        vectorised site pass per tree) gives the exact influence footprint,
        which both validates the iterated scalar bounds and closes their
        residual holes (a deep basin outside the iterated candidate region).
        Returns None when no tree centre is on the grid.
        """
        a = np.asarray(self.cache.building_dsm)
        rows, cols = self.grid.rows, self.grid.cols
        pixel = self.grid.pixel_size_m

        altitudes = [config.lowest_sky_patch_altitude_deg]
        series_min = self._minimum_positive_sun_altitude_deg()
        if series_min is not None:
            altitudes.append(series_min)
        # The marched altitudes are exactly the SVF sky patches (down to the
        # configured patch floor, 6 degrees for patch_option=2) plus the
        # positive solar altitudes of the cached series — the config's direct
        # sun-altitude guess is deliberately NOT consulted here: the actual
        # series replaces the guess, and a lower guess would only inflate
        # every window without buying exactness.
        tan_min = tan(radians(min(altitudes)))

        row_index = np.arange(rows, dtype=np.float64)[:, None]
        col_index = np.arange(cols, dtype=np.float64)[None, :]
        combined = np.zeros((rows, cols), dtype=bool)
        on_grid = False
        # 1 cm slack: float32 march arithmetic near the flip threshold must
        # fall on the "influenced" side of this float64 mask.
        slack_m = 0.01
        for tree in trees:
            cell = self._tree_cell(tree)
            if cell is None:
                continue
            on_grid = True
            row0, col0 = cell
            # The rasterized canopy paints the whole footprint disc at the
            # tree's height, so the tallest occluder cell is the highest
            # surface under the crown, not the centre cell.
            reach_px = int(
                np.ceil(
                    (tree.canopy_radius_m + config.safety_margin_m) / pixel
                )
            )
            r_lo = max(0, row0 - reach_px)
            r_hi = min(rows, row0 + reach_px + 1)
            c_lo = max(0, col0 - reach_px)
            c_hi = min(cols, col0 + reach_px + 1)
            base = float(a[r_lo:r_hi, c_lo:c_hi].max())
            cheb_m = (
                np.maximum(
                    np.abs(row_index - row0), np.abs(col_index - col0)
                )
                * pixel
            )
            flip = (base + tree.height_m - a) > (cheb_m * tan_min - slack_m)
            footprint = cheb_m <= (
                tree.canopy_radius_m + config.safety_margin_m
            )
            combined |= flip | footprint
        if not on_grid:
            return None
        rows_true, cols_true = np.nonzero(combined)
        window = RasterWindow(
            int(rows_true.min()),
            int(rows_true.max()) + 1,
            int(cols_true.min()),
            int(cols_true.max()) + 1,
        )
        return window.align(config.block_size_pixels).clamp(rows=rows, cols=cols)

    def pending_batch(self) -> CoalescedEditBatch | None:
        """Coalesce the edits not yet consumed by a published job."""
        pending = tuple(
            edit
            for edit in self.layer.edits
            if edit.sequence > self._acked_sequence
        )
        if not pending:
            return None
        return coalesce_tree_edits(pending)

    def _landcover_dirty_windows(self) -> list[RasterWindow]:
        """Dirty windows of the pending land-cover overlay state (u-c4).

        A paint's write scope IS its footprint (the adapter's spatial
        semantics), so the dirty windows are the staged overlay's patch
        windows — the same set the planner merges into the plan's write
        windows, merged with the same ``merge_windows`` so worker coverage
        of the plan's promise is structural. A staged ``None`` against a
        published overlay is the drop-overlay reset: the recompute region
        is the previously painted footprint.

        u-c6 (u-c4-review M3) + u-c6b superset rule: when the executor
        stages an explicit ``dirty_windows`` footprint with the overlay,
        that footprint — the batch's STROKE windows union the symmetric
        difference between the staged and published overlay states —
        replaces the derived one. The stroke is the plan's promise basis
        (the planner scopes write windows from the delta patches, never
        from resolved values), so the executed scope can never undercut
        the plan; the union with the value-diff keeps the batch cost
        proportional to what THIS batch actually touched, and the
        accumulated overlay stays coalesced (O(edited area)), so no job
        ever recomputes footprints the session painted earlier.
        """
        if self._landcover_dirty_override is not None:
            if not self._landcover_dirty_override:
                return []
            return merge_windows(
                list(self._landcover_dirty_override), gap_pixels=0
            )
        overlay = (
            self._landcover_overlay
            if self._landcover_overlay is not None
            else self._published_landcover_overlay
        )
        if overlay is None:
            return []
        return merge_windows(
            [patch.window for patch in overlay.patches], gap_pixels=0
        )

    def run(
        self,
        *,
        job_id: str | None = None,
        target_revision: int | None = None,
        time_stop: int | None = None,
    ) -> JobOutcome:
        """Execute the latest pending edit batch and publish result patches.

        ``target_revision`` is the scene revision the job computes against;
        it defaults to the worker's current revision. Between stages the
        worker re-checks the revision provider: a superseded job discards its
        staging area and publishes nothing.

        ``time_stop`` (perf wave 1 STEP 2a, default ``None`` = full series)
        bounds the LOCAL solve's causal time-prefix replay at
        ``t = 0..time_stop``: every computed step is bit-identical to the
        full-series run (the temporal state chains only carry forward), and
        the published patch self-describes its truncated coverage. The
        full-tile fallback keeps the full series (the legacy orchestrator
        has no time seam; its patch simply covers a superset).
        """
        job_id = job_id or new_job_id()
        target = int(
            target_revision if target_revision is not None else self.scene_revision
        )
        scenario_root = self.results_root / self.layer.scenario_id
        staging_root = scenario_root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)

        batch = self.pending_batch()
        # A staged overlay not yet published is pending work on its own: a
        # forcing-only, paint-only, or params-only batch carries no tree
        # edits to coalesce, and without these flags its job would no-op
        # the payload away (u-c3/u-c4/u-c7 hazard). Forcing is pending by
        # PRESENCE too (u-c6b): a mixed batch re-staging a value-equal
        # fold still owes the forcing closure a recompute.
        forcing_pending = self._forcing_presence_pending or (
            self._forcing_overlay != self._published_forcing_overlay
        )
        landcover_pending = (
            self._landcover_overlay != self._published_landcover_overlay
        )
        model_parameters_pending = (
            self._model_parameters != self._published_model_parameters
        )
        if (
            (batch is None or not batch.edits)
            and not forcing_pending
            and not landcover_pending
            and not model_parameters_pending
        ):
            discard_staging(staging_root)
            # (h) evidence-based no-op reason: name WHICH pending source was
            # checked and found empty, so a no-op outcome is diagnosable from
            # the diagnostics alone instead of guessing which door returned it.
            return JobOutcome(
                status="no-op",
                mode=None,
                job_id=job_id,
                scene_revision=target,
                diagnostics={
                    "reason": (
                        "no pending edits: no batch is staged and the "
                        "forcing, land-cover, and model-parameter overlays "
                        "all match their published watermarks"
                        if batch is None
                        else "no pending edits: the staged batch carries "
                        "no edits and no overlay or parameter changes are "
                        "pending"
                    ),
                    "batch_sequences": (
                        None
                        if batch is None
                        else (batch.first_sequence, batch.last_sequence)
                    ),
                    "forcing_overlay_pending": forcing_pending,
                    "landcover_overlay_pending": landcover_pending,
                    "model_parameters_pending": model_parameters_pending,
                },
            )

        started = time.time()
        #: Patch directories this job has already renamed into the results
        #: root. Hoisted out of the try so the superseded handler can see
        #: (and clean up) a PARTIALLY published set (u-c1b M4).
        patch_paths: list[Path] = []
        try:
            self._check_not_superseded(target)  # checkpoint 1: before windowing

            if forcing_pending or model_parameters_pending:
                # Forcing is site-global (the planner's
                # full_downstream_only ruling): every cell's physics reads
                # the resolved table, so the recompute set is the full tile
                # — the union of tree dirty windows and a forcing change is
                # always the whole site. Model parameters are site-global
                # the same way (u-c7): the folded kernel arguments couple
                # the radiation closure at every cell, and the planner
                # scopes the params dirty set FULL, so the worker routes
                # the whole tile rather than re-deriving a windowed subset.
                dirty = [self.grid.full_window]
            else:
                tree_dirty = self.dirty_windows()
                # A paint is windowed (the registry's nominal scope):
                # its footprint joins the tree dirty windows — merged
                # with the same gap-0 merge the planner uses, so the
                # published write windows structurally cover the plan's
                # promise. The join is gated on the executor's staged
                # FOOTPRINT, not on overlay inequality (u-c6-review
                # round-2 F1): a same-value repaint of ALREADY-PUBLISHED
                # cells folds overlay-equal (``landcover_pending`` is
                # False) yet its stroke is real dirty work the plan
                # promised as a write window — dropping it under-published
                # the batch and ``_reconcile_scope`` refused it forever.
                # An empty staged override short-circuits to [] inside
                # ``_landcover_dirty_windows``, so no-paint batches are
                # unaffected.
                landcover_dirty = (
                    self._landcover_dirty_windows()
                    if (landcover_pending or self._landcover_dirty_override)
                    else []
                )
                if landcover_pending or self._landcover_dirty_override:
                    dirty = merge_windows(
                        [*tree_dirty, *landcover_dirty], gap_pixels=0
                    )
                else:
                    # No paint join: keep ``dirty_windows()`` verbatim —
                    # merging here would change the published window layout
                    # for tree-only batches (published bits must not move).
                    dirty = tree_dirty
            if not dirty:
                discard_staging(staging_root)
                # (h) evidence-based no-op reason: name which pending source
                # produced no dirty windows instead of a bare "coalesced".
                pending_tree_edits = 0 if batch is None else len(batch.edits)
                no_op_parts: list[str] = []
                if pending_tree_edits:
                    no_op_parts.append(
                        f"{pending_tree_edits} pending tree edit(s) produced "
                        "no dirty windows (cancelled add+delete pairs drop "
                        "at coalescing)"
                    )
                if landcover_pending or self._landcover_dirty_override:
                    no_op_parts.append("staged land-cover footprint is empty")
                if not no_op_parts:
                    no_op_parts.append(
                        "dirty-window derivation returned no windows"
                    )
                return JobOutcome(
                    status="no-op",
                    mode=None,
                    job_id=job_id,
                    scene_revision=target,
                    diagnostics={
                        "reason": "edits coalesced to no-op: "
                        + "; ".join(no_op_parts),
                        "batch_sequences": (
                            None
                            if batch is None
                            else (batch.first_sequence, batch.last_sequence)
                        ),
                    },
                )
            write_windows = tuple(
                window.expand(self.write_margin_pixels).clamp(
                    rows=self.grid.rows, cols=self.grid.cols
                )
                for window in dirty
            )
            # L4 (u-c4-review item 4): assert the published footprints stay
            # inside the site grid before mode choice or any solve sees them.
            for window in write_windows:
                guard_patch_window(window, self.grid, label="write")

            self._check_not_superseded(target)  # checkpoint 2: before solving

            union = merge_windows(write_windows)[0]
            mode = choose_recompute_mode(
                union,
                self.grid,
                full_recompute_fraction=self.influence_config.full_recompute_fraction,
            )
            # Routing observability (exact-lane policy wave): flag a FULL
            # mode whose cause is the fraction threshold — the dirty union
            # is a strict SUBSET of the tile, so windows would have covered
            # it had the threshold not demoted the solve. A whole-tile
            # dirty union (forcing/model-parameter pending overlays) is a
            # legitimate full, and the SolverInputError fallback below
            # clears the bit (its cause rides ``fallback_reason``).
            demoted_by_fraction = (
                mode is RecomputeMode.FULL and union.area < self.grid.area_pixels
            )

            forcing = self.forcing()
            fallback_reason: str | None = None
            # Per-stage solve telemetry (perf wave 1 STEP 1a): filled by the
            # local solve seams; the legacy full-tile orchestrator has no
            # per-stage seams, so its jobs simply omit the fields.
            stage_timings: dict[str, float] = {}
            # R4 Phase A occluder-state telemetry (refusals included).
            self._pending_veg_state = None
            # G2.1 item 3: any thermal capture a previous (superseded or
            # failed) job left pending dies here — a stale state must never
            # be committed under the revision the executor is about to
            # publish (superseded jobs persist nothing).
            self._pending_thermal_capture = None
            veg_state_diag: dict[str, object] = {}
            if mode is RecomputeMode.FULL:
                # Full-tile path keeps the FULL series (documented deviation):
                # the legacy compute_utci orchestrator has no time seam, and
                # a full patch covering a superset of any truncated request
                # is always composition-safe.
                patches = self._solve_full(job_id, target, forcing)
            else:
                try:
                    patches = self._solve_local(
                        job_id, target, forcing, write_windows,
                        time_stop=time_stop, stage_timings=stage_timings,
                        veg_state_diag=veg_state_diag,
                    )
                except SolverInputError as error:
                    # Unsafe local job: fall back to the standard full path.
                    # Correctness is preserved by recomputation, never by
                    # relaxing tolerances.
                    fallback_reason = str(error)
                    stage_timings.clear()
                    patches = self._solve_full(job_id, target, forcing)
                    mode = RecomputeMode.FULL
                    demoted_by_fraction = False

            staged = [stage_patch(patch, staging_root) for patch in patches]

            self._check_not_superseded(target)  # checkpoint 3: before publish

            for staging_dir, patch in zip(staged, patches):
                # Re-check before every rename: a revision bump mid-publish
                # must leave no partially published patch set.
                self._check_not_superseded(target)
                patch_paths.append(
                    publish_staged_patch(staging_dir, scenario_root, patch.scene_revision)
                )
            # Publish succeeded: consume the batch so a later delete of a
            # published tree forms its own job instead of coalescing away,
            # and consume the staged overlay the results were computed under.
            if batch is not None:
                self._acked_sequence = max(
                    self._acked_sequence, batch.last_sequence
                )
            self._published_forcing_overlay = self._forcing_overlay
            self._forcing_presence_pending = False
            self._published_landcover_overlay = self._landcover_overlay
            # Copy at publish (u-c7-review TRIVIAL): binding the staged dict
            # itself would alias the two watermarks, and rollback/adopt —
            # which copy on receipt — would paper over an in-place mutation
            # of the staged mapping only by luck. The published watermark is
            # a frozen copy of what the job ran under.
            self._published_model_parameters = (
                dict(self._model_parameters)
                if self._model_parameters is not None
                else None
            )
            # R4 Phase A exact-publication seam: the publish completed, so
            # the occluder bits this job prepared now describe the current
            # scene revision — commit them here and only here.
            self._commit_veg_state()
            elapsed = time.time() - started
            return JobOutcome(
                status="published",
                mode=mode.value,
                job_id=job_id,
                scene_revision=target,
                write_windows=tuple(patch.write_window for patch in patches),
                read_windows=tuple(patch.read_window for patch in patches),
                patch_paths=tuple(patch_paths),
                fallback_reason=fallback_reason,
                diagnostics={
                    "dirty_windows": [
                        (w.row_start, w.row_stop, w.col_start, w.col_stop)
                        for w in dirty
                    ],
                    "dirty_fraction": union.area / self.grid.area_pixels,
                    "demoted_by_fraction": demoted_by_fraction,
                    "elapsed_seconds": elapsed,
                    "batch_sequences": (
                        None
                        if batch is None
                        else (batch.first_sequence, batch.last_sequence)
                    ),
                    "forcing_overlay_pending": forcing_pending,
                    "landcover_overlay_pending": landcover_pending,
                    "model_parameters_pending": model_parameters_pending,
                    **({"veg_state": dict(veg_state_diag)} if veg_state_diag else {}),
                    **(
                        {"stage_timings": {k: round(v, 3) for k, v in stage_timings.items()}}
                        if stage_timings
                        else {}
                    ),
                },
            )
        except SupersededJobError:
            # A revision bump mid-publish renames patch directories before
            # the checkpoint can stop the loop: those directories are
            # orphans of a job that will never complete (the outcome
            # reports nothing published), so they are removed here rather
            # than left for rev-globbing readers to trust (u-c1b M4).
            if patch_paths:
                discard_published_patches(tuple(patch_paths))
            discard_staging(staging_root)
            return JobOutcome(
                status="superseded",
                mode=None,
                job_id=job_id,
                scene_revision=target,
                diagnostics={"reason": "scene revision advanced mid-flight"},
            )
        except BaseException:
            # u-c1-review R4: the supersession handler above used to be the
            # ONLY cleanup path, so a NON-supersede failure between two
            # window publishes (a rename failing, a metadata read raising,
            # ...) skipped the orphan cleanup entirely — the already-renamed
            # patch directories of a job that will never complete stayed in
            # the results root as never-current revisions a rev-globbing
            # reader could trust. ANY failure after the first rename gets
            # the same best-effort discard, with survivors reported loudly;
            # the original exception stays the actionable one and
            # propagates (the executor's rollback pairs with it).
            if patch_paths:
                leftovers = discard_published_patches(tuple(patch_paths))
                for leftover in leftovers:
                    _LOG.error(
                        "orphan patch directory %s of failed job %s survived "
                        "best-effort cleanup (u-c1-review R4); it carries a "
                        "scene revision that never became current and must "
                        "not be read",
                        leftover,
                        job_id,
                    )
            discard_staging(staging_root)
            raise

    # ------------------------------------------------------------------
    # Solve drivers
    # ------------------------------------------------------------------

    def _patch_common(self, job_id: str, target_revision: int) -> dict:
        return {
            "job_id": job_id,
            "scene_revision": target_revision,
            "site_id": self.cache.site_id,
            "tile_key": self.cache.tile_key,
            "cache_manifest_sha256": self.cache.metadata()["manifest_sha256"],
            "model_version": self.cache.model_version,
            "variables": self.requested_variables,
        }

    def _veg_occlusion_store(self) -> VegOcclusionStore:
        """Lazily build the per-scenario occluder-state store (R4 Phase A).

        Persistent state lives under ``<results_root>/.veg_state`` — an
        operational artifact of the result store, never of the read-only
        site cache.
        """
        if self._veg_store is None:
            self._veg_store = VegOcclusionStore(
                self.cache, state_root=self.results_root / ".veg_state"
            )
        return self._veg_store

    def _prepare_veg_state(
        self, scene
    ) -> tuple[VegOcclusionState | None, dict]:
        """Load-or-build the packed occluder state for the live scene.

        Telemetry-only failure: the store already routes every typed
        refusal (:class:`VegOcclusionFallback`) to ``(None, reason)``, and
        this seam additionally refuses loudly-but-non-fatally on state
        errors or a cache without SVF patch cubes — the local solve then
        runs today's proven replay (``veg_state=None``) unchanged. Nothing
        here may block a job.
        """
        if not self._veg_occlusion_enabled:
            return None, {"enabled": False, "fallback_reason": "disabled"}
        patches = self.cache.svf_patches
        for name in ("vegshadowmat", "vbshmat"):
            if name not in patches:
                return None, {
                    "enabled": False,
                    "fallback_reason": f"cache carries no {name} patch cubes",
                }
        try:
            state, telemetry = self._veg_occlusion_store().prepare(scene)
        except (VegOcclusionStateError, KeyError, ValueError) as error:
            return None, {"enabled": False, "fallback_reason": str(error)}
        telemetry["enabled"] = state is not None
        return state, telemetry

    def _commit_veg_state(self) -> None:
        """Exact-publication seam (R4 Phase A).

        Called by :meth:`run` exactly when a locally solved job's publish
        completed: the packed bits then describe the scene revision that
        just became current, so the store advances and (with a state root)
        persists atomically. A refused, failed, or superseded job never
        reaches here — the next job's ``prepare`` catches up from the last
        committed scene. TODO(r2a): the server publication layer should
        call this same seam once ``jobs.py`` / ``executor_bridge.py``
        merge, so executor commit points and state revisions cannot drift.
        A persist failure is logged, never fatal: the state stays correct
        in memory and the next job rebuilds if the worker dies.
        """
        state, self._pending_veg_state = self._pending_veg_state, None
        if state is None or self._veg_store is None:
            return
        try:
            self._veg_store.commit(state)
        except OSError:
            _LOG.warning(
                "vegetation occluder state persist failed; continuing "
                "with in-memory state",
                exc_info=True,
            )

    def take_pending_thermal_capture(self) -> dict[str, Any] | None:
        """Ownership-transfer handoff of the current job's thermal capture.

        The executor's publication hook calls this exactly once after a
        batch fully published: the returned ``{"final_state",
        "fingerprint"}`` becomes a warm-state checkpoint via
        :func:`~solweig_gpu.incremental.checkpoints.thermal_checkpoint`.
        Any other outcome (refused, failed, superseded) never reaches the
        hook, and the capture is cleared at the start of every ``run`` —
        the ownership transfer guarantees a capture is consumed at most
        once and can never leak across jobs.
        """
        capture, self._pending_thermal_capture = (
            self._pending_thermal_capture,
            None,
        )
        return capture

    def stage_thermal_capture(
        self, final_state: Mapping[str, Any], input_fingerprint: Mapping[str, str]
    ) -> None:
        """Executor-side staging of a warm anchor (G2.1 item 4).

        The met warm-sparse consumer solves OUTSIDE ``run`` (it never
        dispatches a worker job), so it stages its suffix-split anchor
        here right after its own publish succeeds; the publication hook
        commits it exactly like a ``_solve_full`` capture. Any worker-run
        job that follows clears it at the start of ``run`` — a capture is
        never committed under a revision it did not serve.
        """
        self._pending_thermal_capture = {
            "final_state": final_state,
            "fingerprint": dict(input_fingerprint),
        }

    def _thermal_fingerprint_for(
        self, forcing: SiteForcing, final_state: Mapping[str, Any]
    ) -> dict[str, str]:
        """Digest the inputs that determined ``final_state`` (capture side).

        The consumer side (G2.1 item 4) recomputes the SAME spelling over
        its CURRENT inputs through the one shared function; any drift
        between the sides would silently accept a stale warm state.

        Theoretical note (r5 review N1): at ``next_step == 1`` the
        met-prefix digest covers only row 0, while the replay's first step
        also derives ``timestepdec`` from rows 0 AND 1 (utci_process.py:
        855-857, ``timestepdec = dectime[1] - dectime[0]``). Unreachable
        today -- a ForcingOverlay edits physical columns, never ``dtime``
        -- so the digest gap cannot yet admit a stale warm state.
        """
        scene = compose_full_scene_tensors(self.cache, self.layer)
        if self._landcover_overlay is not None:
            resolved_landcover: np.ndarray | None = resolve_landcover_overlay(
                self.cache, self._landcover_overlay
            )
        elif "landcover" in self.cache:
            resolved_landcover = np.asarray(self.cache.landcover)
        else:
            resolved_landcover = None
        return thermal_fingerprint(
            building_dsm=scene.a.numpy(),
            canopy=scene.canopy.numpy(),
            dem=scene.dem.numpy(),
            resolved_landcover=resolved_landcover,
            model_parameters=self._model_parameters,
            met_prefix=forcing.met_table[: int(final_state["next_step"])],
        )

    def _solve_local(
        self,
        job_id: str,
        target_revision: int,
        forcing: SiteForcing,
        write_windows: tuple[RasterWindow, ...],
        *,
        time_stop: int | None = None,
        stage_timings: MutableMapping[str, float] | None = None,
        veg_state_diag: MutableMapping[str, object] | None = None,
    ) -> list[ResultPatch]:
        scene = compose_full_scene_tensors(self.cache, self.layer)
        # R4 Phase A: packed occluder bits for the edited scene. A refusal
        # yields ``None`` and the solve replays the 153-patch march below,
        # bitwise the pre-R4 path. Corridor/pack seconds ride the SVF stage
        # telemetry (telemetry never blocks the solve). Perf honesty (review
        # LOW-1): the corridor path saves vs the FULL-TILE replay only — at
        # 96x96 @2 m it is ~1.3x SLOWER than the deployed-shape W2 windowed
        # replay (read 66x78, acc 24x30); it is correctness-first routing,
        # not a small-tile speedup.
        veg_state, veg_telemetry = self._prepare_veg_state(scene)
        self._pending_veg_state = veg_state
        if veg_state_diag is not None:
            veg_state_diag.update(veg_telemetry)
        if stage_timings is not None:
            stage_overhead = float(veg_telemetry.get("corridor_seconds", 0.0)) + float(
                veg_telemetry.get("pack_seconds", 0.0)
            )
            if stage_overhead:
                stage_timings["svf_seconds"] = (
                    stage_timings.get("svf_seconds", 0.0) + stage_overhead
                )
            # T19b: the corridor re-marches already ran through the lane
            # router inside this prepare; lift their counts here so the
            # manifest witness covers BOTH march sites (solve_window then
            # ADDS the replay/fold-side delta per window — accumulate, the
            # same stage dict is shared across windows).
            for timing_key, telemetry_key in (
                ("svf_march_routed", "march_routed"),
                ("svf_march_refused", "march_refused"),
            ):
                marches = veg_telemetry.get(telemetry_key)
                if marches:
                    stage_timings[timing_key] = (
                        stage_timings.get(timing_key, 0.0) + float(marches)
                    )
        # The patch's time bound follows the arrays actually produced (the
        # solve truncates to the causal prefix only when asked to).
        patch_time_stop = (
            int(time_stop) if time_stop is not None else self.cache.time_steps
        )
        patches: list[ResultPatch] = []
        for index, write_window in enumerate(write_windows):
            read_window = read_window_for_write_window(
                write_window, self.cache, scene, forcing
            )
            # L4 (u-c4-review item 4): the local solve reads and writes
            # only inside in-grid windows, and every read covers its write.
            guard_patch_window(write_window, self.grid, label="write")
            guard_patch_window(read_window, self.grid, label="read")
            guard_read_covers_write(read_window, write_window)
            if self._core_adapter:
                # T02 opt-in route: the solweig_core ABI/dispatch boundary
                # around the SAME numerical path. The adapter composes the
                # scene itself from its validated canopy view, so the
                # caller-composed ``scene`` above (kept for the halo guard)
                # is intentionally NOT forwarded — one compose per route,
                # bit-identical values either way.
                arrays = solve_with_core_adapter(
                    self.cache,
                    self.layer,
                    read_window=read_window,
                    write_window=write_window,
                    forcing=forcing,
                    requested_variables=self.requested_variables,
                    veg_changed=True,  # pending tree edits always change vegetation
                    landcover_overlay=self._landcover_overlay,
                    model_parameters=self._model_parameters,
                    time_stop=time_stop,
                    stage_timings=stage_timings,
                    veg_state=veg_state,
                )
            else:
                arrays = solve_window(
                    self.cache,
                    self.layer,
                    read_window=read_window,
                    write_window=write_window,
                    forcing=forcing,
                    requested_variables=self.requested_variables,
                    veg_changed=True,  # pending tree edits always change vegetation
                    landcover_overlay=self._landcover_overlay,
                    model_parameters=self._model_parameters,
                    time_stop=time_stop,
                    stage_timings=stage_timings,
                    scene=scene,
                    required_read_window=read_window,
                    veg_state=veg_state,
                )
            # One patch per write window. A job with DISJOINT dirty windows
            # (e.g. a tree edit far from a scenario paint footprint — u-c4
            # unions both into one job's windows) publishes several patches,
            # and the patch store keys directories by job id: every patch
            # needs its own. Single-window jobs (the common case) keep the
            # bare job id, byte-identical to the pre-u-c4 layout.
            patch_job_id = (
                job_id if len(write_windows) == 1 else f"{job_id}-w{index}"
            )
            patches.append(
                ResultPatch(
                    mode="local",
                    write_window=write_window,
                    read_window=read_window,
                    arrays=arrays,
                    time_start=0,
                    time_stop=patch_time_stop,
                    **self._patch_common(patch_job_id, target_revision),
                )
            )
        return patches

    def _solve_full(
        self, job_id: str, target_revision: int, forcing: SiteForcing
    ) -> list[ResultPatch]:
        # u-c7-review LOW (requested-variable symmetry): the LEGACY
        # full-tile path (no model parameters) routes through
        # ``compute_utci``, which hardwires save_kup/kdown/lup/ldown=False
        # — it can NEVER produce the flux variables, while the direct
        # model-parameter path carries every ``run_utci_window`` variable
        # and refuses a missing one loudly ("did not produce"). Asking the
        # legacy orchestrator for a flux variable used to pass silently
        # and simply drop the variable from the published patch; it is
        # refused with a typed error at this door instead — the production
        # door every caller routes through. (The solver level stays
        # permissive for the same request: the sanctioned oracle-helper
        # pattern requests the full vocabulary through both branches.)
        if not self._model_parameters:
            unproducible = tuple(
                name
                for name in self.requested_variables
                if name in ("kup", "kdown", "lup", "ldown")
            )
            if unproducible:
                raise WorkerError(
                    "legacy full-tile run cannot produce "
                    f"{list(unproducible)}: compute_utci writes only "
                    "utci/tmrt/shadow (its save_kup/save_kdown/save_lup/"
                    "save_ldown switches are hardwired False); request "
                    "utci/tmrt/shadow only, or stage model parameters "
                    "(the direct path carries every run_utci_window "
                    "variable) — refusing the request instead of "
                    "silently dropping the variables"
                )
        scratch = self.scratch_root / job_id
        try:
            solved = run_full_tile(
                self.cache,
                self.layer,
                forcing=forcing,
                site_dir=self.site_dir,
                scratch_dir=scratch,
                requested_variables=self.requested_variables,
                landcover_overlay=self._landcover_overlay,
                model_parameters=self._model_parameters,
                return_final_state=True,
            )
            arrays, final_state = solved
            # G2.1 item 3: capture the FULL-TILE final thermal state with
            # the input digests computed over EXACTLY the inputs this run
            # consumed (the scene the solver composed, the resolved land
            # cover, the folded parameters, the forcing rows 0..next_step).
            # The capture rides the job: the executor's publication hook
            # commits it ONLY after the batch fully publishes — a
            # superseded job leaves it cleared by the next ``run``.
            self._pending_thermal_capture = {
                "final_state": final_state,
                "fingerprint": self._thermal_fingerprint_for(
                    forcing, final_state
                ),
            }
        finally:
            # The arrays are in memory; the scratch prepared-site copy is
            # disposable. Clean it up on success AND failure so the scratch
            # root never accumulates one full-site copy per job.
            shutil.rmtree(scratch, ignore_errors=True)
        full_window = self.grid.full_window
        # L4 (u-c4-review item 4): the full patch declares the whole site as
        # both footprint and read extent — assert it really is in-grid.
        guard_patch_window(full_window, self.grid, label="write")
        return [
            ResultPatch(
                mode="full",
                write_window=full_window,
                read_window=full_window,
                arrays={
                    name: np.ascontiguousarray(array) for name, array in arrays.items()
                },
                time_start=0,
                time_stop=self.cache.time_steps,
                **self._patch_common(job_id, target_revision),
            )
        ]


def _tree_position_key(tree: TreeSpec) -> tuple[str, float, float]:
    return (tree.tree_id, tree.x_m, tree.y_m)


def discard_published_patches(patch_paths: Sequence[Path]) -> tuple[Path, ...]:
    """Best-effort removal of a failed/superseded job's patch directories.

    The u-c1b M4 rule: a patch directory renamed into the results root by
    a batch that never completed is an orphan — it carries a revision that
    was never current, and rev-globbing readers would happily read it.
    Removal failures never raise (the caller is already on a failure
    path); a leftover is logged loudly instead. Returns the paths that
    survived the cleanup.
    """
    leftovers: list[Path] = []
    for path in patch_paths:
        try:
            if path.is_dir():
                shutil.rmtree(path)
        except OSError as error:
            _LOG.error(
                "orphan patch directory %s survived best-effort cleanup "
                "(u-c1b M4): %s; it carries a never-current scene revision "
                "and must not be read",
                path,
                error,
            )
        if path.exists() and path not in leftovers:
            leftovers.append(path)
    return tuple(leftovers)
