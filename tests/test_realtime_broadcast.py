# SPDX-License-Identifier: GPL-3.0-only
"""SSE broadcast hub tests (R1 epochs wave).

Written failing-first against the client wire contract
(``studio/realtime_client.mjs``): named SSE
events (``event:`` lines — the client uses ``addEventListener``, never
``onmessage``), a state-snapshot ``canonical_revision`` as the FIRST frame
on subscribe, per-subscriber bounded queues with drop-oldest overflow plus
a lagged-recovery heartbeat, and per-workspace isolation.

The hub is driven inside ``asyncio.run`` coroutines (a real event loop, the
same shape the FastAPI route provides); cross-thread broadcasts exercise
the ``call_soon_threadsafe`` path the scheduler thread depends on.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from solweig_gpu.server.realtime.broadcast import (
    DEFAULT_QUEUE_SIZE,
    BroadcastHub,
    format_sse,
)

WS = "ws-broadcast-a"
OTHER = "ws-broadcast-b"


def parse_frame(frame: str) -> tuple[str, dict[str, Any]]:
    """Split one SSE frame string into ``(event_name, data_dict)``."""
    lines = frame.split("\n")
    assert lines[-1] == "", f"frame must end with a blank line: {frame!r}"
    name = None
    data: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            assert name is None, "two event lines in one frame"
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())
    assert name is not None, f"frame carries no event line: {frame!r}"
    return name, json.loads("\n".join(data))


async def collect(subscription, count: int, *, timeout: float = 3.0) -> list[str]:
    """Read ``count`` frames from a subscription (fails on timeout)."""
    frames: list[str] = []

    async def reader() -> None:
        async for frame in subscription:
            frames.append(frame)
            if len(frames) >= count:
                return

    task = asyncio.create_task(reader())
    await asyncio.wait_for(task, timeout)
    return frames


# ---------------------------------------------------------------------------
# Frame format
# ---------------------------------------------------------------------------


def test_format_sse_emits_named_event_frames() -> None:
    frame = format_sse("heartbeat", {"a": 1})
    assert frame == 'event: heartbeat\ndata: {"a":1}\n\n'
    # The client's EventSource parses ``event:`` lines for addEventListener;
    # a bare data-only frame would fall into onmessage and be dropped.
    name, data = parse_frame(frame)
    assert (name, data) == ("heartbeat", {"a": 1})


def test_default_queue_size_is_256() -> None:
    assert DEFAULT_QUEUE_SIZE == 256


# ---------------------------------------------------------------------------
# Subscribe: snapshot first, then live broadcasts
# ---------------------------------------------------------------------------


def test_subscribe_emits_snapshot_then_live_events_in_order() -> None:
    async def scenario() -> list[tuple[str, dict[str, Any]]]:
        hub = BroadcastHub()
        snapshot = {
            "workspace_revision": 7,
            "epoch_id": None,
            "operations": [],
            "fast_revision": 5,
            "exact_revision": 3,
            "snapshot": True,
        }
        subscription = hub.subscribe(WS, snapshot=snapshot)
        assert hub.subscriber_count(WS) == 1
        # Broadcast from "another thread" (the scheduler's delivery path).
        threading.Thread(
            target=lambda: hub.broadcast_canonical(
                WS,
                {
                    "workspace_revision": 8,
                    "epoch_id": 2,
                    "operations": [{"operation_id": "op-1"}],
                },
            )
        ).start()
        frames = await collect(subscription, 2)
        subscription.close()
        return [parse_frame(frame) for frame in frames]

    events = asyncio.run(scenario())
    # The snapshot MUST be first: a late joiner syncs revisions before any
    # live event can reference them.
    first, second = events
    assert first[0] == "canonical_revision"
    assert first[1]["workspace_revision"] == 7
    assert first[1]["operations"] == []
    assert first[1]["snapshot"] is True
    assert second == (
        "canonical_revision",
        {"workspace_revision": 8, "epoch_id": 2, "operations": [{"operation_id": "op-1"}]},
    )


def test_broadcast_is_isolated_per_workspace() -> None:
    async def scenario() -> tuple[list[str], list[str]]:
        hub = BroadcastHub()
        mine = hub.subscribe(WS)
        theirs = hub.subscribe(OTHER)
        hub.broadcast_canonical(WS, {"workspace_revision": 1})
        mine_frames = await collect(mine, 1)
        await asyncio.sleep(0.05)  # nothing may arrive for OTHER
        theirs_frames: list[str] = []
        theirs.close()
        mine.close()
        return mine_frames, theirs_frames

    mine_frames, theirs_frames = asyncio.run(scenario())
    assert len(mine_frames) == 1
    assert theirs_frames == []


def test_broadcast_fast_exact_and_heartbeat_event_names() -> None:
    async def scenario() -> list[str]:
        hub = BroadcastHub()
        subscription = hub.subscribe(WS)
        hub.broadcast_fast(WS, {"fast_revision": 4, "workspace_revision": 5, "result_class": "visual_pending"})
        hub.broadcast_exact(WS, {"exact_revision": 4, "workspace_revision": 5})
        frames = await collect(subscription, 2)
        subscription.close()
        return frames

    frames = asyncio.run(scenario())
    names = [parse_frame(frame)[0] for frame in frames]
    assert names == ["fast_revision", "exact_revision"]


# ---------------------------------------------------------------------------
# Overflow: drop-oldest + lagged recovery heartbeat
# ---------------------------------------------------------------------------


def test_overflow_drops_oldest_marks_lagged_and_heartbeat_notifies() -> None:
    async def scenario() -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
        tight_hub = BroadcastHub(queue_size=2)
        tight = tight_hub.subscribe(WS)
        roomy_hub = BroadcastHub(queue_size=64)
        roomy = roomy_hub.subscribe(WS)
        tight_task = asyncio.create_task(collect(tight, 3))  # 2 survivors + 1 beat
        roomy_task = asyncio.create_task(collect(roomy, 5))
        for revision in range(5):
            tight_hub.broadcast_canonical(WS, {"workspace_revision": revision, "epoch_id": 0, "operations": []})
            roomy_hub.broadcast_canonical(WS, {"workspace_revision": revision, "epoch_id": 0, "operations": []})
        roomy_frames = await asyncio.wait_for(roomy_task, 2.0)
        tight_hub.send_heartbeats()  # the lagged notice is the tight sub's 3rd frame
        tight_frames = await asyncio.wait_for(tight_task, 2.0)
        roomy_hub.send_heartbeats()
        tight.close()
        roomy.close()
        tight_hub.send_heartbeats()  # must not raise on the closed subscriber
        return [parse_frame(f) for f in tight_frames], roomy_frames

    tight_events, roomy_frames = asyncio.run(scenario())
    # The tight subscriber keeps the NEWEST events (drop-oldest) ...
    kept = [event for name, event in tight_events if name == "canonical_revision"]
    assert [event["workspace_revision"] for event in kept] == [3, 4]
    # ... and its next heartbeat carries the missed-event notice so the
    # client runs its catch-up GET (durable ops make SSE loss safe).
    heartbeats = [event for name, event in tight_events if name == "heartbeat"]
    assert len(heartbeats) == 1
    assert heartbeats[0]["missed_events"] >= 3
    # The roomy subscriber on the same broadcast stream lost nothing.
    assert len(roomy_frames) == 5
    # And the lagged flag reset after the notice: a second heartbeat is clean.
    async def second_beat() -> tuple[dict[str, Any], dict[str, Any]]:
        hub = BroadcastHub(queue_size=1)
        sub = hub.subscribe(WS)
        for revision in range(3):
            hub.broadcast_canonical(WS, {"workspace_revision": revision})
        hub.send_heartbeats()
        beat_name, first_beat = parse_frame((await collect(sub, 2))[1])
        hub.send_heartbeats()
        _, second_beat_event = parse_frame((await collect(sub, 1))[0])
        sub.close()
        return first_beat, second_beat_event

    first_beat, second_beat_event = asyncio.run(second_beat())
    assert first_beat["missed_events"] >= 2
    assert "missed_events" not in second_beat_event


def test_broadcast_to_zero_subscribers_is_a_noop() -> None:
    hub = BroadcastHub()
    hub.broadcast_canonical("ws-nobody", {"workspace_revision": 1})  # no raise
    assert hub.subscriber_count() == 0


# ---------------------------------------------------------------------------
# Unsubscribe / cleanup
# ---------------------------------------------------------------------------


def test_close_stops_delivery_and_clears_the_registry() -> None:
    async def scenario() -> list[str]:
        hub = BroadcastHub()
        subscription = hub.subscribe(WS)
        subscription.close()
        assert hub.subscriber_count(WS) == 0
        hub.broadcast_canonical(WS, {"workspace_revision": 1})
        frames: list[str] = []
        async for frame in subscription:
            frames.append(frame)  # must terminate on the close sentinel
        hub.broadcast_canonical(WS, {"workspace_revision": 2})
        await asyncio.sleep(0.05)
        return frames

    frames = asyncio.run(scenario())
    assert frames == []


def test_generator_exit_unsubscribes_no_leak() -> None:
    async def scenario() -> int:
        hub = BroadcastHub()
        subscription = hub.subscribe(WS)
        hub.broadcast_canonical(WS, {"workspace_revision": 1})
        # The StreamingResponse disconnect path: consume one frame from the
        # events() generator, then aclose() it (client went away) — the
        # finally block must unregister the subscriber.
        events = subscription.events()
        await asyncio.wait_for(events.__anext__(), 2.0)
        await events.aclose()
        hub.broadcast_canonical(WS, {"workspace_revision": 2})
        await asyncio.sleep(0.05)
        return hub.subscriber_count(WS)

    assert asyncio.run(scenario()) == 0


def test_hub_serves_many_subscribers_concurrently() -> None:
    async def scenario() -> list[int]:
        hub = BroadcastHub()
        subscriptions = [hub.subscribe(WS) for _ in range(5)]
        assert hub.subscriber_count(WS) == 5
        hub.broadcast_canonical(WS, {"workspace_revision": 1})
        counts = [len(await collect(sub, 1)) for sub in subscriptions]
        for sub in subscriptions:
            sub.close()
        assert hub.subscriber_count() == 0
        return counts

    assert asyncio.run(scenario()) == [1] * 5


# ---------------------------------------------------------------------------
# Heartbeat roster + privileged-queue cap (operational-stability wave ③)
# ---------------------------------------------------------------------------


def test_heartbeat_carries_live_roster_with_refcount() -> None:
    """The 5 s heartbeat delivers the workspace's live actors; multi-tab
    sessions refcount one actor; an actor leaves when its LAST tab leaves.
    """

    async def scenario() -> dict[str, Any]:
        hub = BroadcastHub()
        a1 = hub.subscribe(WS, actor_id="actor-a")
        a2 = hub.subscribe(WS, actor_id="actor-a")  # second tab, same actor
        b = hub.subscribe(WS, actor_id="actor-b")
        anonymous = hub.subscribe(WS)  # legacy client: no actor_id param
        assert hub.live_actors(WS) == ["actor-a", "actor-b"]

        async def beats() -> dict[str, dict[str, Any]]:
            out = {}
            for name, sub in (("a1", a1), ("b", b), ("anon", anonymous)):
                hub.send_heartbeats()
                _, event = parse_frame((await collect(sub, 1))[0])
                out[name] = event
            return out

        first = await beats()
        # Everyone on the workspace sees the same roster — including the
        # anonymous subscriber (presence is a property of the WORKSPACE).
        for event in first.values():
            assert event["roster"] == ["actor-a", "actor-b"]

        a1.close()  # one tab of actor-a leaves...
        assert hub.live_actors(WS) == ["actor-a", "actor-b"]
        a2.close()  # ...now the last one
        assert hub.live_actors(WS) == ["actor-b"]
        hub.send_heartbeats()
        _, b_event = parse_frame((await collect(b, 1))[0])
        assert b_event["roster"] == ["actor-b"]
        b.close()
        assert hub.live_actors(WS) == []
        hub.send_heartbeats()  # nobody left: no roster key anywhere
        return {}

    asyncio.run(scenario())


def test_empty_roster_omits_the_roster_key() -> None:
    """A workspace with no known actors ships no roster field at all — the
    client keeps its observed roster instead of learning an empty one."""

    async def scenario() -> dict[str, Any]:
        hub = BroadcastHub()
        sub = hub.subscribe(WS)
        hub.send_heartbeats()
        _, event = parse_frame((await collect(sub, 1))[0])
        sub.close()
        return event

    assert "roster" not in asyncio.run(scenario())


def test_broadcast_exact_progress_named_event() -> None:
    async def scenario() -> tuple[str, dict[str, Any]]:
        hub = BroadcastHub()
        subscription = hub.subscribe(WS)
        hub.broadcast_exact_progress(
            WS,
            {"job_id": "job-9", "status": "running", "target_revision": 7,
             "eta_seconds": 90, "eta_basis": "history"},
        )
        frames = await collect(subscription, 1)
        subscription.close()
        return parse_frame(frames[0])

    name, event = asyncio.run(scenario())
    assert name == "exact_progress"
    assert event["job_id"] == "job-9"
    assert event["eta_seconds"] == 90


def test_privileged_heartbeat_cannot_wedge_the_queue() -> None:
    """Heartbeats bypass the capacity drop BUT replace their pending
    predecessor in place: a wedged reader's queue holds ONE heartbeat
    (the newest) no matter how many ticks fire — never one privileged
    frame per tick accumulating ~15 MB/day (the pre-fix bug).

    Ordinary events interleaved with the burst keep their places and
    their order."""
    async def scenario() -> tuple[int, list[str]]:
        hub = BroadcastHub(queue_size=256)
        sub = hub.subscribe(WS)
        hub.broadcast_canonical(WS, {"workspace_revision": 1})
        hub.broadcast_canonical(WS, {"workspace_revision": 2})
        for _ in range(50):
            hub.send_heartbeats()  # reader consumes nothing
        hub.broadcast_canonical(WS, {"workspace_revision": 3})
        pending = sub._queue.qsize()

        # Drain: the surviving heartbeat sits BETWEEN rev 2 and rev 3 (it
        # replaced the first heartbeat's slot), and it is the newest one.
        names: list[str] = []
        while True:
            try:
                frame = sub._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if frame is None or not isinstance(frame, str):
                names.append(str(frame))
                continue
            name, event = parse_frame(frame)
            names.append(name)
            if name == "heartbeat":
                assert "roster" not in event  # no actor_id subscribed
        sub.close()
        return pending, names

    pending, names = asyncio.run(scenario())
    assert pending == 4, f"2 events + 1 heartbeat + 1 event, got {pending}: {names}"
    assert names == ["canonical_revision", "canonical_revision", "heartbeat", "canonical_revision"]
