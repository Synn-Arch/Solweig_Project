# SPDX-License-Identifier: GPL-3.0-only
"""SSE broadcast hub for realtime collaboration (R1 epochs wave).

One hub per app process. The epoch scheduler thread publishes canonical
revision events; FastAPI request handlers subscribe on behalf of connected
EventSource clients (``realtime_client.mjs`` uses ``addEventListener`` with
named events — frames MUST carry an ``event:`` line).

Cross-thread delivery: each subscription captures the event loop it was
created on; publishes arriving on another thread (the scheduler daemon) go
through ``loop.call_soon_threadsafe`` so the loop's self-pipe wakes it —
a bare ``put_nowait`` from a foreign thread can leave the loop sleeping in
select past the event.

Loss model: SSE is a convenience, never a correctness plane. Every broadcast
operation is already durable in the operation log, so a slow subscriber that
overflows its bounded queue simply drops the OLDEST frames, is flagged
lagged, and receives the count of missed events on its next heartbeat. The
catch-up GET (``GET /workspaces/{id}/operations?since_server_sequence=N``)
is the recovery path the protocol guarantees; ``realtime_client.mjs``
consumes the heartbeat's ``missed_events`` count to trigger that GET.
That keeps a wedged client from bloating server memory while guaranteeing
it a recovery path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256

# Sentinel enqueued by close(); the async iterator translates it into a clean
# end-of-stream (StopAsyncIteration) instead of raising into the response.
_CLOSE = object()


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Render one SSE frame with a named event and compact JSON payload."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


class HeartbeatFrame(str):
    """A heartbeat frame, marked so a pending one can be REPLACED.

    Heartbeats are privileged (a lagged subscriber must still get its
    ``missed_events`` notice), but privilege must not mean accumulation:
    a subscriber whose event loop lives but never reads would otherwise
    bank one privileged frame per tick forever (~15 MB/day). At most ONE
    pending heartbeat per subscriber — a newer tick replaces an undelivered
    older one in place (the lag flag is re-read at build time, so nothing
    is lost).
    """


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class Subscription:
    """One connected client's view of one workspace's event stream."""

    def __init__(
        self, hub: "BroadcastHub", workspace_id: str, actor_id: str | None = None
    ) -> None:
        self._hub = hub
        self.workspace_id = workspace_id
        # Presence identity (best-effort, never authenticated): rides the
        # heartbeat roster so other clients can tell WHO is live. Anonymous
        # streams simply carry None and are counted only in subscriber_count.
        self.actor_id = actor_id
        # The loop the SSE response iterator lives on; publishes from other
        # threads ride call_soon_threadsafe onto it.
        self._loop = _running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._lagged = False
        self._missed = 0
        self._closed = False

    # -- delivery (hub-side; may be called from any thread) -----------------

    def deliver(self, item: Any, *, privileged: bool = False) -> None:
        """Land one item on the subscriber's queue, loop-safely.

        ``privileged`` frames (heartbeats, the close sentinel) bypass the
        capacity drop: a lagged subscriber MUST still get its missed-event
        notice, or it could never recover.
        """
        if self._loop is None or self._loop is _running_loop():
            self._enqueue(item, privileged=privileged)
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue, item, privileged)
        except RuntimeError:  # loop already closed (interpreter shutdown)
            # Expected at interpreter teardown; debug-level so a shutdown
            # storm of dying loops cannot spam the logs (r1-review T1).
            logger.debug(
                "delivery to a closed loop dropped for %s", self.workspace_id
            )

    def _enqueue(self, item: Any, privileged: bool = False) -> None:
        if self._closed and item is not _CLOSE:
            return
        queue = self._queue
        if isinstance(item, HeartbeatFrame):
            # Privileged-but-capped: replace any still-pending heartbeat in
            # place instead of accumulating one privileged frame per tick
            # behind a wedged reader (the lag flag rides the newest frame —
            # heartbeat_payload is consulted at build time, never queued).
            pending = queue._queue  # asyncio.Queue's underlying deque
            for index, existing in enumerate(pending):
                if isinstance(existing, HeartbeatFrame):
                    pending[index] = item
                    return
        if (
            item is not _CLOSE
            and not privileged
            and queue.qsize() >= self._hub.queue_size
        ):
            # Drop the OLDEST frame: a lagging client needs the newest state
            # far more than the oldest, and the durable log serves history.
            try:
                queue.get_nowait()
                self._missed += 1
                self._lagged = True
            except asyncio.QueueEmpty:  # pragma: no cover - raced with reader
                pass
        queue.put_nowait(item)

    def push_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Enqueue a corrective state-snapshot frame (route-side, post-subscribe).

        Closes the subscribe TOCTOU (r1-epochs-review L1): if the durable
        revision advanced between the route's scenario read and
        ``hub.subscribe``, the route re-reads the scenario and pushes the
        TRUE snapshot here — the subscriber sees revision-sync frames in
        non-decreasing order and can never strand on a stale revision.
        Privileged, like heartbeats: a corrective frame is exactly what a
        lagging client must not drop.
        """
        self.deliver(format_sse("canonical_revision", snapshot), privileged=True)

    def push_selected_time(self, frame: dict[str, Any]) -> None:
        """Enqueue the stream's first ``selected_time`` coverage frame.

        T15 (flag-gated route parameter): the selected-time stream's
        contract statement — which requested times are servable
        bitwise-correctly right now. Privileged for the same reason as
        the corrective snapshot: dropping it would leave the client
        composing coverage it was never told it lacks.
        """
        self.deliver(format_sse("selected_time", frame), privileged=True)

    def heartbeat_payload(self) -> dict[str, Any]:
        """Build this subscriber's heartbeat frame, consuming the lag flag."""
        payload: dict[str, Any] = {}
        if self._lagged:
            payload["missed_events"] = self._missed
            self._lagged = False
            self._missed = 0
        return payload

    # -- client side (the response iterator) -------------------------------

    def close(self) -> None:
        """Stop delivery and unregister from the hub (idempotent)."""
        self._hub.unsubscribe(self)

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is _CLOSE:
            # Clean end-of-stream; the generator's finally (or the route's)
            # already unregistered us.
            raise StopAsyncIteration
        return item

    async def events(self) -> AsyncIterator[str]:
        """Async generator with finally-unsubscribe (the route's iterator).

        A bare ``__aiter__``/``__anext__`` pair would leak the subscriber if
        the client disconnects mid-frame; the generator's ``finally`` runs on
        GeneratorExit — exactly what StreamingResponse triggers on
        disconnect.
        """
        try:
            while True:
                yield await self.__anext__()
        finally:
            self.close()


class BroadcastHub:
    """Per-workspace SSE fan-out, safe for the scheduler thread + N routes."""

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self.queue_size = queue_size
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Subscription]] = {}
        # Live presence: workspace -> {actor_id: subscriber refcount}. A
        # subscriber registers on subscribe() and leaves on unsubscribe()
        # (the SSE generator's finally runs on client disconnect, and a
        # dead TCP fails its next privileged heartbeat write within ~1-2
        # ticks — no TTL machinery needed). Multi-tab sessions of one actor
        # refcount; the roster lists the actor while any tab is live.
        self._live_actors: dict[str, dict[str, int]] = {}
        self._closed = False

    # -- publish (any thread; typically the epoch scheduler) ----------------

    def _publish(self, workspace_id: str, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(workspace_id, ()))
        # Serialize ONCE, not per subscriber (T15 slow-client fence): the
        # publishing thread is the epoch scheduler's daemon thread (or the
        # fast lane's) — the compute thread. Per-subscriber format_sse made
        # its serialization cost linear in the subscriber count; delivery
        # itself is already non-blocking (call_soon_threadsafe onto each
        # subscriber's own loop, bounded queue with drop-oldest on that
        # loop), so the frame bytes were the only per-subscriber work left
        # on THIS thread.
        frame = format_sse(event, data) if subscribers else None
        for subscription in subscribers:
            subscription.deliver(frame)

    def broadcast_canonical(self, workspace_id: str, event: dict[str, Any]) -> None:
        self._publish(workspace_id, "canonical_revision", event)

    def broadcast_fast(self, workspace_id: str, event: dict[str, Any]) -> None:
        self._publish(workspace_id, "fast_revision", event)

    def broadcast_exact(self, workspace_id: str, event: dict[str, Any]) -> None:
        self._publish(workspace_id, "exact_revision", event)

    def broadcast_exact_progress(self, workspace_id: str, event: dict[str, Any]) -> None:
        """Named ``exact_progress`` frame (additive job-transition telemetry).

        The exact lane's jobs are minted server-side and their ids ride no
        operation, so without this event a subscriber cannot show a
        countdown — the stream otherwise carries ``exact_revision`` only at
        completion. GET /jobs/{id} remains the source of truth; this is a
        convenience plane (same loss model as every other event).
        """
        self._publish(workspace_id, "exact_progress", event)

    def broadcast_selected_time(self, workspace_id: str, event: dict[str, Any]) -> None:
        """Named ``selected_time`` frame (T15 flag-gated coverage surface).

        Same single-serialization fence as every other event: the frame is
        serialized once on the publishing thread and delivered
        non-blockingly per subscriber.
        """
        self._publish(workspace_id, "selected_time", event)

    def send_heartbeats(self) -> int:
        """Push one heartbeat frame to every subscriber; return that count.

        The heartbeat is also the lagged-recovery notice: a subscriber whose
        queue overflowed gets ``missed_events`` here and runs its catch-up.
        Each frame additionally carries the workspace's live-actor ``roster``
        (idempotent full-state presence — no join/leave events to lose, a
        reconnect storm's worst case is a 5 s-stale roster that self-corrects).
        """
        with self._lock:
            if self._closed:
                return 0
            subscribers = [
                subscription
                for queue in self._subscribers.values()
                for subscription in queue
            ]
            rosters = {
                workspace_id: sorted(actors)
                for workspace_id, actors in self._live_actors.items()
                if actors
            }
        for subscription in subscribers:
            payload = subscription.heartbeat_payload()
            roster = rosters.get(subscription.workspace_id)
            if roster is not None:
                payload["roster"] = roster
            subscription.deliver(
                HeartbeatFrame(format_sse("heartbeat", payload)),
                privileged=True,
            )
        return len(subscribers)

    def live_actors(self, workspace_id: str) -> list[str]:
        """The workspace's live actor ids (sorted; presence truth)."""
        with self._lock:
            return sorted(self._live_actors.get(workspace_id, ()))

    # -- subscribe (the SSE route, inside its event loop) -------------------

    def subscribe(
        self,
        workspace_id: str,
        snapshot: dict[str, Any] | None = None,
        actor_id: str | None = None,
    ) -> Subscription:
        subscription = Subscription(self, workspace_id, actor_id=actor_id)
        # The snapshot goes on the queue SYNCHRONOUSLY, still inside the
        # route's event loop and before registration: no live event can ever
        # precede the late joiner's revision-sync frame.
        if snapshot is not None:
            subscription._enqueue(format_sse("canonical_revision", snapshot))
        with self._lock:
            if self._closed:
                subscription._closed = True
                subscription._queue.put_nowait(_CLOSE)
                return subscription
            self._subscribers.setdefault(workspace_id, []).append(subscription)
            if actor_id:
                actors = self._live_actors.setdefault(workspace_id, {})
                actors[actor_id] = actors.get(actor_id, 0) + 1
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            queue = self._subscribers.get(subscription.workspace_id)
            if queue is None:
                return  # workspace already gone: its actor entry went with it
            remaining = [sub for sub in queue if sub is not subscription]
            if remaining:
                self._subscribers[subscription.workspace_id] = remaining
            else:
                del self._subscribers[subscription.workspace_id]
            actor_id = subscription.actor_id
            if actor_id:
                actors = self._live_actors.get(subscription.workspace_id)
                if actors is not None and actor_id in actors:
                    actors[actor_id] -= 1
                    if actors[actor_id] <= 0:
                        del actors[actor_id]
                    if not actors:
                        del self._live_actors[subscription.workspace_id]
        # Mark closed and wake the response iterator so the stream ends
        # cleanly (idempotent: unsubscribe of an already-gone subscriber
        # returns above before touching it again).
        subscription._closed = True
        subscription.deliver(_CLOSE)

    def subscriber_count(self, workspace_id: str | None = None) -> int:
        with self._lock:
            if workspace_id is None:
                return sum(len(queue) for queue in self._subscribers.values())
            return len(self._subscribers.get(workspace_id, ()))

    def close(self) -> None:
        """End every stream (app shutdown)."""
        with self._lock:
            self._closed = True
            subscribers = [
                subscription
                for queue in self._subscribers.values()
                for subscription in queue
            ]
            self._subscribers.clear()
        for subscription in subscribers:
            subscription._closed = True
            subscription.deliver(_CLOSE)
