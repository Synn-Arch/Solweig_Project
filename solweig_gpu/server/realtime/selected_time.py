# SPDX-License-Identifier: GPL-3.0-only
"""Selected-time streaming and coverage tracking (T15, behind a SEPARATE flag).

``SOLWEIG_RT_SELECTED_TIME_STREAMING`` gates every behavior in this
module; with the flag unset the server behaves byte-identically to
before (pinned by tests). The flag adds:

* **Request-cut disclosure** — a solve published under a client's
  ``requested_result.time_indices`` subset computed values for its causal
  time prefix and then DISCARDED every unrequested plane. The durable
  record for that revision therefore does not carry the edit's effect at
  its uncovered times. Such a record discloses a
  ``time_coverage.complete=false`` manifest block (DESIGN §11.4) so the
  revision can never be mistaken for a full-coverage publication.
* **The composition guard** — composing a history with an UNHEALED
  request-cut record would silently keep the cut revision's uncovered
  times at their stale values: an accepted edit's effect dropped from
  every later composition. :func:`guard_composition` refuses with a
  typed :class:`TimeCoverageGap`; the solve paths catch it and heal by
  recomputing/rebuilding FULL coverage for the healing publication. A
  later record heals the gap exactly when it covers the FULL site window
  and every time step (a bootstrap full, a reset's re-published baseline,
  an exact-lane reconcile full).
* **The coverage surface** — :func:`coverage_report` answers, per
  variable, the times the server can serve bitwise-correctly at the
  current revision (the guard's own model: everything except unhealed
  request-cut gaps), and the ``selected_time`` SSE frames carry the same
  summary so a streaming client composes ``(variable, t)`` coverage
  exactly (DESIGN §15) instead of trusting a revision number.

Native sparse patches (the r3a met fast path) are NOT request-cut: their
manifest time indices are their exact change set and they carry no
``time_coverage`` block; the guard never trips on them. Historical
records published before this flag existed carry no block either and are
treated as native for the same reason — the block is the request-cut
marker going forward (disclosed as a limitation: a pre-flag request-cut
record cannot be distinguished retroactively).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from solweig_gpu.server.store import Store, StoreError


def _parse_store_stamp(stamp: str) -> datetime:
    """Parse the store's ``_now_utc`` wall-clock stamp to an aware UTC time."""
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )

__all__ = [
    "ENV_FLAG_NAME",
    "MANIFEST_KEY",
    "SelectedTimeDemand",
    "TimeCoverageGap",
    "coverage_report",
    "disclosure_block",
    "enabled",
    "guard_composition",
    "initial_frame",
    "manifest_covers_full_time",
    "record_window_is_full_site",
]

#: The SEPARATE flag (T15): unset/empty/"0"/"false" keep the pre-T15
#: behavior byte-identical.
ENV_FLAG_NAME = "SOLWEIG_RT_SELECTED_TIME_STREAMING"

#: Manifest key carrying the request-cut disclosure block.
MANIFEST_KEY = "time_coverage"

_TRUTHY = {"1", "true", "yes", "on"}


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether selected-time streaming is on (read at call time)."""
    source = os.environ if environ is None else environ
    return str(source.get(ENV_FLAG_NAME, "")).strip().lower() in _TRUTHY


def disclosure_block(time_indices: Sequence[int], time_steps: int) -> dict[str, Any]:
    """The request-cut disclosure block for a manifest.

    ``complete`` is False by construction: this block is only written for
    records whose durable payload was cut to a strict subset of the
    site's time axis by a client request.
    """
    covered = sorted({int(t) for t in time_indices})
    complete = len(covered) >= int(time_steps) and covered == list(range(int(time_steps)))
    return {
        "complete": bool(complete),
        "time_indices": covered,
        "time_steps": int(time_steps),
    }


def _record_missing_times(
    manifest: Mapping[str, Any], time_steps: int
) -> set[int]:
    """Times a request-cut record left uncovered (empty when not cut)."""
    block = manifest.get(MANIFEST_KEY)
    if not isinstance(block, Mapping) or block.get("complete") is not False:
        return set()
    covered = {int(t) for t in block.get("time_indices", ())}
    return set(range(int(time_steps))) - covered


def manifest_covers_full_time(manifest: Mapping[str, Any], time_steps: int) -> bool:
    """Whether a record's manifest covers every time step of the site."""
    indices = [int(t) for t in manifest.get("time_indices", ())]
    return len(indices) == int(time_steps) and set(indices) == set(range(int(time_steps)))


def record_window_is_full_site(manifest: Mapping[str, Any], rows: int, cols: int) -> bool:
    """Whether a record's write window is the entire site grid."""
    window = manifest.get("window") or {}
    return (
        int(window.get("row_start", -1)) == 0
        and int(window.get("col_start", -1)) == 0
        and int(window.get("row_stop", -1)) == int(rows)
        and int(window.get("col_stop", -1)) == int(cols)
    )


def _heals_gap(manifest: Mapping[str, Any], time_steps: int, rows: int, cols: int) -> bool:
    """A record heals prior request-cut gaps iff it recomputed the WHOLE
    scene: full site window AND every time step."""
    return manifest_covers_full_time(manifest, time_steps) and record_window_is_full_site(
        manifest, rows, cols
    )


def _gap_times(
    manifests: Sequence[Mapping[str, Any]],
    variables: Sequence[str],
    time_steps: int,
    rows: int,
    cols: int,
) -> dict[str, set[int]]:
    """Walk the durable history oldest-first and return, per variable,
    the times still missing through an unhealed request-cut record."""
    gaps: dict[str, set[int]] = {}
    for manifest in manifests:
        record_variables = [
            str(entry.get("name"))
            for entry in manifest.get("variables", ())
            if isinstance(entry, Mapping)
        ]
        if _heals_gap(manifest, time_steps, rows, cols):
            for name in list(gaps):
                if not record_variables or name in record_variables:
                    # A whole-scene recompute redid every time step of the
                    # variables it published; records without variable
                    # metadata (hand-built) are treated as covering all.
                    gaps[name] = set()
            gaps = {name: times for name, times in gaps.items() if times}
            continue
        missing = _record_missing_times(manifest, time_steps)
        if not missing:
            continue
        for name in record_variables or list(variables):
            gaps.setdefault(name, set()).update(missing)
    return {name: times for name, times in gaps.items() if times}


class TimeCoverageGap(StoreError):
    """Composing this history would silently drop an accepted edit's
    effect at the request-cut revision's uncovered times.

    Typed refusal (never a silent wrong compose): the caller heals by
    recomputing FULL coverage for its publication. ``variables`` and
    ``missing_times`` are machine-readable for route-facing translation.
    """

    advice = (
        "this revision's history contains a request-cut publication whose "
        "uncovered times were never made durable; recompute full coverage "
        "(full-tile solve / bootstrap rebuild) before publishing"
    )

    def __init__(self, variables: Sequence[str], missing_times: Sequence[int]) -> None:
        self.variables = tuple(str(name) for name in variables)
        self.missing_times = sorted({int(t) for t in missing_times})
        super().__init__(
            "composing this history would keep request-cut times stale for "
            f"variables {list(self.variables)} at times {self.missing_times}; "
            "refusing to compose (recompute full coverage instead)"
        )

    def body(self) -> dict[str, Any]:
        return {
            "code": "time_coverage_gap",
            "variables": list(self.variables),
            "missing_times": self.missing_times,
            "advice": self.advice,
        }


def guard_composition(
    manifests: Sequence[Mapping[str, Any]],
    variables: Sequence[str],
    *,
    time_steps: int,
    rows: int,
    cols: int,
) -> None:
    """Refuse to compose a history with unhealed request-cut records.

    Called from :func:`solweig_gpu.server.jobs._compose_current_state`
    BEFORE the composition-cache probe (a cached gapped composition must
    not be served either), only when :func:`enabled` is True at the call
    site.
    """
    gaps = _gap_times(manifests, variables, time_steps, rows, cols)
    if gaps:
        names = [name for name in variables if name in gaps] or sorted(gaps)
        missing = sorted(set().union(*gaps.values()))
        raise TimeCoverageGap(names, missing)


def coverage_report(
    store: Store,
    workspace_id: str,
    variables: Sequence[str],
    time_steps: int,
) -> dict[str, Any]:
    """Per-variable served-coverage summary at the newest result version.

    ``missing_times`` uses the guard's own model — the times an unhealed
    request-cut record left untrustworthy — so the report can never claim
    coverage the composition would refuse. Records without the
    ``time_coverage`` block (native sparse patches, pre-flag history) are
    trusted at their manifest indices.
    """
    scenario = store.get_scenario(workspace_id)
    rows = cols = 0
    if scenario is not None:
        # Window geometry comes from the record manifests themselves; the
        # scenario only supplies the current revision for the header.
        pass
    manifests: list[Mapping[str, Any]] = []
    per_record: list[dict[str, Any]] = []
    for version in store.result_versions(workspace_id):
        record = store.get_result(workspace_id, version)
        if record is None:  # pragma: no cover - versions come from the store
            continue
        manifest = record.manifest
        manifests.append(manifest)
        window = manifest.get("window") or {}
        rows = max(rows, int(window.get("row_stop", 0)))
        cols = max(cols, int(window.get("col_stop", 0)))
        per_record.append(
            {
                "scene_version": int(version),
                "window": dict(window),
                "time_indices": [int(t) for t in manifest.get("time_indices", ())],
                "request_cut": bool(
                    isinstance(manifest.get(MANIFEST_KEY), Mapping)
                    and manifest[MANIFEST_KEY].get("complete") is False
                ),
            }
        )
    gaps = _gap_times(manifests, variables, time_steps, rows, cols)
    all_times = set(range(int(time_steps)))
    per_variable: dict[str, dict[str, Any]] = {}
    for name in variables:
        missing = sorted(gaps.get(name, set()))
        per_variable[name] = {
            "missing_times": missing,
            "covered_times": sorted(all_times - set(missing)),
        }
    # T15 introspection (exit criterion: latest-state age + supersession
    # counts): how stale the newest durable state is, and how many jobs
    # this workspace's churn superseded on the way there. Read-only, best-
    # effort — a missing stamp reads as None, never a fabricated 0.
    newest_age_s: float | None = None
    newest_created = store._read(
        "SELECT created_at FROM results WHERE scenario_id = ? "
        "ORDER BY scene_version DESC LIMIT 1",
        (workspace_id,),
    )
    if newest_created is not None and newest_created["created_at"]:
        try:
            newest_age_s = round(
                (
                    datetime.now(timezone.utc)
                    - _parse_store_stamp(str(newest_created["created_at"]))
                ).total_seconds(),
                3,
            )
        except ValueError:  # pragma: no cover - corrupt stamp: stay honest
            newest_age_s = None
    superseded = store._read(
        "SELECT COUNT(*) AS n FROM jobs "
        "WHERE scenario_id = ? AND status = 'superseded'",
        (workspace_id,),
    )
    return {
        "workspace_id": workspace_id,
        "exact_revision": (
            scenario.exact_result_version if scenario is not None else 0
        ),
        "workspace_revision": (
            scenario.scene_version if scenario is not None else 0
        ),
        "time_steps": int(time_steps),
        "complete": not gaps,
        "variables": per_variable,
        "records": per_record,
        "latest_result_age_s": newest_age_s,
        "superseded_jobs": int(superseded["n"]) if superseded is not None else 0,
    }


def initial_frame(
    report: Mapping[str, Any], requested: Sequence[int]
) -> dict[str, Any]:
    """The first ``selected_time`` SSE frame for a new stream.

    Answers exactly what a selected-time client must know before its
    first compose: which of ITS requested times the server can serve
    bitwise-correctly right now (per the coverage report's own model),
    which are still unhealed request-cut gaps, and at which revisions.
    """
    requested_times = sorted({int(t) for t in requested})
    missing: set[int] = set()
    for entry in report.get("variables", {}).values():
        missing.update(int(t) for t in entry.get("missing_times", ()))
    return {
        "workspace_id": report.get("workspace_id"),
        "requested": requested_times,
        "covered": [t for t in requested_times if t not in missing],
        "missing": [t for t in requested_times if t in missing],
        "complete": bool(report.get("complete")),
        "exact_revision": int(report.get("exact_revision", 0)),
        "workspace_revision": int(report.get("workspace_revision", 0)),
        "latest_result_age_s": report.get("latest_result_age_s"),
        "snapshot": True,
    }


class SelectedTimeDemand:
    """In-process registry of live selected-time stream demand.

    The adaptive-epoch seam: while any client holds a selected-time
    subscription the epoch window shrinks toward ``EPOCH_TICK_MIN_MS``
    (the service contract's admitted 50 ms floor), returning to the
    configured cadence the moment demand clears. Deliberately simple:
    presence-only, per-process, never durable — the epoch window is a
    latency knob, never a correctness fact.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def register(self, workspace_id: str) -> None:
        with self._lock:
            self._counts[workspace_id] = self._counts.get(workspace_id, 0) + 1

    def release(self, workspace_id: str) -> None:
        with self._lock:
            count = self._counts.get(workspace_id, 0) - 1
            if count > 0:
                self._counts[workspace_id] = count
            else:
                self._counts.pop(workspace_id, None)

    def pending(self) -> bool:
        with self._lock:
            return bool(self._counts)

    def workspaces(self) -> list[str]:
        with self._lock:
            return sorted(self._counts)
