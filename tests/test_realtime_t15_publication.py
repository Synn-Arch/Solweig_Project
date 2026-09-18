# SPDX-License-Identifier: GPL-3.0-only
"""T15 protocol tests: publication, mixed edit, liveness.

One RED witness per seam the T15 card names, plus the pins whose guards
already exist (each pin records the mutation that kills it). The
selected-time streaming feature (``SOLWEIG_RT_SELECTED_TIME_STREAMING``)
is covered in its own section — every behavior it adds is behind the
flag and the flag-off contract is pinned byte-identical.

Witness map (card order):

* dense offset time mis-scatter ................ `_apply_result_into_state`
  refuses a partial-time payload with no manifest indices instead of
  silently scattering it at dense offsets (``test_partial_time_payload_
  without_indices_refuses_dense_scatter``).
* stale exact publication ....................... the T13 OutputLease
  discipline integrated CPU-side (:mod:`realtime.publish_fence`): publish
  only after completion, refuse stale generations.
* accepted operation loss ....................... selected-time streaming
  guard — a request-cut publication may never silently drop the edit's
  effect at its uncovered times in a later composition.
* full-coverage false claim ..................... a request-cut publication
  discloses ``time_coverage.complete=false`` and the coverage surface
  never reports full coverage it does not have.
* mixed-family partial no-op wedge .............. the no-op settle +
  disclosure + re-arm cycle.
* pending epoch permanently stuck after restart . the crash-shape restart
  recovery drill.
* slow client blocking the compute thread ....... ``BroadcastHub._publish``
  serializes the SSE frame ONCE per publish, not once per subscriber.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.server import patch_codec
from solweig_gpu.server.jobs import _apply_result_into_state

# ---------------------------------------------------------------------------
# Witness 7 (built first — smallest seam): slow client blocking the compute
# thread = per-subscriber SSE serialization running on the publishing thread
# ---------------------------------------------------------------------------


def _sync_subscriptions(hub, workspace: str, count: int) -> list:
    """Subscribe ``count`` clients from OUTSIDE any event loop.

    ``Subscription._loop`` is then ``None`` and delivery takes the direct
    ``_enqueue`` path, so everything a publish does on the PUBLISHING
    thread is observable synchronously here.
    """
    return [hub.subscribe(workspace) for _ in range(count)]


def test_publish_serializes_the_frame_once_per_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED witness (slow client / compute thread): the SSE frame is
    serialized once per publish. At base ``BroadcastHub._publish`` calls
    ``format_sse`` (``json.dumps`` of the whole event) once PER
    SUBSCRIBER on the publishing thread — the epoch scheduler's daemon
    thread, i.e. the compute thread that must never spend time
    proportional to the subscriber count. The designated mutation
    (moving ``format_sse`` back inside the subscriber loop) is killed by
    exactly this count."""
    from solweig_gpu.server.realtime import broadcast as broadcast_module

    hub = broadcast_module.BroadcastHub(queue_size=8)
    _sync_subscriptions(hub, "ws_t15", 3)

    calls = {"n": 0}
    real_dumps = broadcast_module.json.dumps

    def counting_dumps(*args, **kwargs):
        calls["n"] += 1
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(broadcast_module.json, "dumps", counting_dumps)
    hub.broadcast_canonical("ws_t15", {"workspace_revision": 1, "operations": [{"big": "x" * 512}]})
    assert calls["n"] == 1, (
        "BroadcastHub._publish serialized the SSE frame "
        f"{calls['n']} times for 3 subscribers — per-subscriber "
        "serialization runs on the publishing (compute) thread"
    )


def test_publish_return_time_does_not_scale_with_subscriber_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The publishing thread's per-publish work must not grow with the
    subscriber count (the computational half of the slow-client fence;
    queue overflow itself is already drop-oldest on the subscriber's own
    loop). A big canonical payload with 1 subscriber vs 16 must cost the
    same ONE serialization each."""
    from solweig_gpu.server.realtime import broadcast as broadcast_module

    event = {"workspace_revision": 1, "blob": "y" * 4096}
    for count in (1, 16):
        hub = broadcast_module.BroadcastHub(queue_size=4)
        _sync_subscriptions(hub, "ws_t15", count)
        serialized = {"n": 0}
        real = broadcast_module.format_sse

        def counting_format(event_name, data):
            serialized["n"] += 1
            return real(event_name, data)

        monkeypatch.setattr(broadcast_module, "format_sse", counting_format)
        hub.broadcast_fast("ws_t15", event)
        assert serialized["n"] == 1, (
            f"publish with {count} subscribers serialized {serialized['n']} frames"
        )
        monkeypatch.setattr(broadcast_module, "format_sse", real)


# ---------------------------------------------------------------------------
# Witness 1: dense offset time mis-scatter
# ---------------------------------------------------------------------------

_FULL_WINDOW = {"row_start": 0, "row_stop": 2, "col_start": 0, "col_stop": 2}


def test_partial_time_payload_without_indices_refuses_dense_scatter() -> None:
    """RED witness (dense offset time mis-scatter): a payload covering a
    SELECTED time subset whose manifest carries no ``time_indices`` must
    be refused, never scattered at dense offsets. At base the fallback
    ``time_indices = list(range(patch.shape[0]))`` lands the changed-t
    plane at t=0 while the changed global t keeps its stale value —
    exactly DESIGN §18.4's ``sparse t index -> dense offset scatter``
    mutation class. The designated mutation (restoring the unconditional
    dense fallback) is killed by this test."""
    state = {"utci": np.zeros((24, 2, 2), dtype=np.float32)}
    manifest = {"window": dict(_FULL_WINDOW)}  # no time_indices at all
    patch = {"utci": np.full((1, 2, 2), 7.0, dtype=np.float32)}  # global t=12
    with pytest.raises(patch_codec.PartialTimeScatterError):
        _apply_result_into_state(state, manifest, patch, ("utci",))
    assert not (state["utci"] == 7.0).any(), (
        "the mis-scattered plane must not land anywhere when refused"
    )


def test_mismatched_indices_on_partial_payload_refuses_dense_scatter() -> None:
    """The mismatch arm of the same fallback: manifest indices that do
    not match the payload's time depth must refuse (the payload knows its
    own planes; guessing dense offsets would mis-scatter them)."""
    state = {"utci": np.zeros((24, 2, 2), dtype=np.float32)}
    manifest = {"window": dict(_FULL_WINDOW), "time_indices": [3, 9, 21]}
    patch = {"utci": np.full((2, 2, 2), 5.0, dtype=np.float32)}
    with pytest.raises(patch_codec.PartialTimeScatterError):
        _apply_result_into_state(state, manifest, patch, ("utci",))


def test_full_depth_payload_keeps_legacy_dense_prefix_semantics() -> None:
    """The fence must not regress legacy composition: a payload covering
    the site's FULL time axis may keep the dense-prefix fallback whether
    or not the manifest names indices."""
    state = {"utci": np.zeros((4, 2, 2), dtype=np.float32)}
    patch = {"utci": np.arange(16, dtype=np.float32).reshape(4, 2, 2)}
    for manifest in (
        {"window": dict(_FULL_WINDOW)},
        {"window": dict(_FULL_WINDOW), "time_indices": [9, 9, 9]},  # mismatch, full depth
    ):
        fresh = {"utci": state["utci"].copy()}
        _apply_result_into_state(fresh, manifest, patch, ("utci",))
        np.testing.assert_array_equal(fresh["utci"], patch["utci"])


def test_sparse_indices_scatter_at_global_times_exactly() -> None:
    """The positive contract (the multi-time compose gate DESIGN §18.4
    asks for): a sparse patch with explicit global indices lands each
    plane at its global t and touches nothing else."""
    state = {"utci": np.zeros((24, 2, 2), dtype=np.float32)}
    manifest = {"window": dict(_FULL_WINDOW), "time_indices": [1, 12, 23]}
    patch = {
        "utci": np.stack(
            [np.full((2, 2), t + 1.0, dtype=np.float32) for t in (1, 12, 23)]
        )
    }
    _apply_result_into_state(state, manifest, patch, ("utci",))
    for t in (1, 12, 23):
        assert (state["utci"][t] == t + 1.0).all(), f"plane {t} mis-scattered"
    untouched = [t for t in range(24) if t not in (1, 12, 23)]
    assert all((state["utci"][t] == 0.0).all() for t in untouched), (
        "scatter leaked outside the declared global times"
    )
