// SPDX-License-Identifier: GPL-3.0-only
//
// r2c fast-lane consumption tests for RealtimeClient: the result-class
// supersession fence (never render a fast result behind the last rendered
// canonical revision), forward-compatible class handling, reconnect with
// missed fast frames (fast payloads are SSE/in-memory only), missed_events
// heartbeat recovery (broadcast.py L2), and typed admission rejections with
// the idempotent-retry-429 reconciliation (r2b F7).

import test from "node:test";
import assert from "node:assert/strict";

import {
  RealtimeClient,
  RealtimeClientError,
  RealtimeAdmissionError,
} from "../realtime_client.mjs";

// ---------------------------------------------------------------------------
// Injected seams (same shape as realtime_client.test.mjs)
// ---------------------------------------------------------------------------

function makeClock(startMs = 0) {
  let nowMs = startMs;
  const tasks = new Set();
  let nextId = 1;
  const timers = {
    setTimeout(fn, ms) {
      const task = { id: nextId, fn, due: nowMs + Number(ms ?? 0) };
      nextId += 1;
      tasks.add(task);
      return task;
    },
    clearTimeout(task) {
      tasks.delete(task);
    },
  };
  async function settle() {
    for (let round = 0; round < 12; round += 1) {
      await new Promise((resolve) => setImmediate(resolve));
    }
  }
  async function advance(ms) {
    await settle();
    const target = nowMs + Number(ms);
    for (;;) {
      const due = [...tasks]
        .filter((task) => task.due <= target)
        .sort((a, b) => a.due - b.due || a.id - b.id)[0];
      if (!due) break;
      tasks.delete(due);
      nowMs = Math.max(nowMs, due.due);
      due.fn();
      await settle();
    }
    nowMs = target;
    await settle();
  }
  return { timers, advance, settle, now: () => nowMs, pendingCount: () => tasks.size };
}

class FakeEventSource {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.listeners = new Map();
    this.closed = false;
    FakeEventSource.instances.push(this);
  }
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }
  close() {
    this.closed = true;
  }
  emit(type, payload) {
    for (const listener of this.listeners.get(type) ?? []) {
      listener({
        type,
        data: typeof payload === "string" ? payload : JSON.stringify(payload ?? {}),
      });
    }
  }
}

function jsonResponse(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

function makeFetch(handler) {
  const requests = [];
  const fetchImpl = async (url, init = {}) => {
    const request = {
      url: String(url),
      method: init.method ?? "GET",
      headers: init.headers ?? {},
      body: init.body ?? null,
    };
    requests.push(request);
    const response = await handler(request, requests.length);
    if (response === null) throw new TypeError("simulated network failure");
    return response;
  };
  fetchImpl.requests = requests;
  return fetchImpl;
}

const notFoundHandler = async () => jsonResponse({ error: { code: "not_found" } }, 404);

function makeClient({
  handler = notFoundHandler,
  clock = makeClock(),
  stallTimeoutMs = 10_000,
  actorId = "actor-a",
  onStatus = null,
} = {}) {
  FakeEventSource.instances = [];
  const fetchImpl = makeFetch(handler);
  const client = new RealtimeClient({
    fetch: fetchImpl,
    eventSourceFactory: (url) => new FakeEventSource(url),
    actorId,
    crypto: null,
    timers: clock.timers,
    now: clock.now,
    stallTimeoutMs,
    ...(onStatus ? { onStatus } : {}),
  });
  return { client, clock, fetchImpl };
}

function subscribe(client, handlers = {}) {
  const applied = [];
  const subscription = client.subscribeWorkspace("ws1", {
    onOperation: (operation) => applied.push(operation.operationId),
    ...handlers,
  });
  const source = FakeEventSource.instances.at(-1);
  return { subscription, source, applied };
}

function canonicalEvent({
  workspaceRevision,
  operations = [],
  fastRevision = null,
  exactRevision = null,
  epochId = 1,
  snapshot = false,
}) {
  return {
    workspace_revision: workspaceRevision,
    epoch_id: epochId,
    ...(snapshot ? { snapshot: true } : {}),
    operations: operations.map((operation) => ({
      operation_id: operation.operationId ?? operation.operation_id,
      actor_id: operation.actorId ?? "actor-remote",
      source_family: operation.sourceFamily ?? "vegetation_geometry",
      entity_id: operation.entityId ?? null,
      verb: operation.verb ?? "add",
      compact_payload: operation.payload ?? operation.compact_payload ?? {},
      ...(operation.serverSequence !== undefined
        ? { server_sequence: operation.serverSequence }
        : {}),
    })),
    ...(fastRevision !== null ? { fast_revision: fastRevision } : {}),
    ...(exactRevision !== null ? { exact_revision: exactRevision } : {}),
  };
}

function fastEvent({ fastRevision, workspaceRevision, resultClass, exactBaseRevision = 0, provenance = null }) {
  return {
    fast_revision: fastRevision,
    workspace_revision: workspaceRevision,
    result_class: resultClass,
    exact_base_revision: exactBaseRevision,
    ...(provenance !== null ? { provenance } : {}),
  };
}

function ackFor(operationId, { serverSequence, epochId = 1, duplicate = false } = {}) {
  return { operation_id: operationId, server_sequence: serverSequence, epoch_id: epochId, duplicate };
}

async function requestsOf(fetchImpl, method) {
  return fetchImpl.requests.filter((request) => request.method === method);
}

// ---------------------------------------------------------------------------
// Result-class supersession fence
// ---------------------------------------------------------------------------

test("fast result renders for its target revision and is superseded once a newer canonical revision arrives", () => {
  const { client } = makeClient();
  const { source } = subscribe(client);

  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 2, operations: [{ operationId: "srv:op-1" }] }));
  source.emit("fast_revision", fastEvent({ fastRevision: 2, workspaceRevision: 2, resultClass: "fast_exact", exactBaseRevision: 1 }));

  let view = client.classFor("ws1");
  assert.equal(view.class, "fast_exact");
  assert.equal(view.effectiveClass, "fast_exact", "fast_exact is authoritative for its revision");
  assert.equal(view.authoritative, true);
  assert.equal(view.targetRevision, 2);
  assert.equal(view.superseded, false);

  // A newer canonical revision (its canonical frame arrives FIRST per the
  // r2b wire order) supersedes the fast result for revision 2 — the client
  // must never render the fast result behind the rendered canonical.
  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 3, operations: [{ operationId: "srv:op-2" }] }));
  view = client.classFor("ws1");
  assert.equal(view.superseded, true);
  assert.equal(view.authoritative, false, "a superseded fast result is never authoritative");
  assert.equal(view.effectiveClass, "visual_pending", "a superseded result degrades to the conservative display");
  assert.equal(view.targetRevision, 2);
});

test("an exact_revision frame supersedes a stale fast result: the exact plane only covers canonical revisions", () => {
  const { client } = makeClient();
  const { source } = subscribe(client);

  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 2, operations: [{ operationId: "srv:op-1" }] }));
  source.emit("fast_revision", fastEvent({ fastRevision: 2, workspaceRevision: 2, resultClass: "visual_pending" }));
  assert.equal(client.classFor("ws1").superseded, false);

  source.emit("exact_revision", { exact_revision: 3, workspace_revision: 3 });
  const view = client.classFor("ws1");
  assert.equal(view.class, "exact_reconciled");
  assert.equal(view.superseded, false, "an exact result covering the workspace is current");
  assert.equal(view.authoritative, true);
});

test("forward compatibility: fast_qualified (reserved) and unknown classes degrade to the conservative visual display", () => {
  const { client } = makeClient();
  const { source } = subscribe(client);

  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 1 }));
  source.emit("fast_revision", fastEvent({ fastRevision: 1, workspaceRevision: 1, resultClass: "fast_qualified" }));
  let view = client.classFor("ws1");
  assert.equal(view.class, "fast_qualified", "the raw wire class is preserved");
  assert.equal(view.effectiveClass, "visual_pending", "the reserved class displays conservatively");
  assert.equal(view.authoritative, false);

  source.emit("fast_revision", fastEvent({ fastRevision: 2, workspaceRevision: 1, resultClass: "fast_quantum_v9" }));
  view = client.classFor("ws1");
  assert.equal(view.class, "fast_quantum_v9");
  assert.equal(view.effectiveClass, "visual_pending", "an unknown future class displays conservatively");
  assert.equal(view.authoritative, false);
});

test("the triple and the fast record never regress: a late old fast frame stays superseded", () => {
  const { client } = makeClient();
  const { source } = subscribe(client);

  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 4, fastRevision: 3, exactRevision: 2 }));
  source.emit("fast_revision", fastEvent({ fastRevision: 3, workspaceRevision: 4, resultClass: "fast_exact" }));
  // A stale fast frame (older revision) must not roll anything back.
  source.emit("fast_revision", fastEvent({ fastRevision: 2, workspaceRevision: 3, resultClass: "visual_pending" }));

  assert.deepEqual(client.revisionsFor("ws1"), { workspaceRevision: 4, fastRevision: 3, exactRevision: 2 });
  const view = client.classFor("ws1");
  assert.equal(view.class, "fast_exact");
  assert.equal(view.fastRevision, 3);
  assert.equal(view.superseded, false);
});

// ---------------------------------------------------------------------------
// Snapshot frame + reconnect with missed fast frames
// ---------------------------------------------------------------------------

test("the subscribe snapshot frame (canonical, snapshot:true, empty ops) syncs the triple without applying operations", () => {
  const { client } = makeClient();
  const { source, applied } = subscribe(client);

  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 5,
    fastRevision: 3,
    exactRevision: 2,
    snapshot: true,
  }));
  assert.deepEqual(applied, [], "history belongs to the catch-up GET, never the stream");
  assert.deepEqual(client.revisionsFor("ws1"), { workspaceRevision: 5, fastRevision: 3, exactRevision: 2 });
});

test("reconnect zero-loss: missed fast frames are legitimate — the client never waits for them", async () => {
  const { client, clock, fetchImpl } = makeClient({
    handler: async (request) => {
      if (request.method === "POST") return jsonResponse({ acks: [] }, 202);
      if (request.url.includes("/operations?since_server_sequence=")) {
        // The catch-up GET recovers DURABLE operations only — fast payloads
        // are in-memory + SSE, so nothing fast rides this response.
        return jsonResponse({
          operations: [
            { operation_id: "srv:op-2", server_sequence: 2, actor_id: "actor-remote", source_family: "vegetation_geometry", entity_id: null, verb: "add", payload: {} },
          ],
          workspace_revision: 3,
          fast_revision: 1,
          exact_revision: 0,
        });
      }
      return jsonResponse({}, 404);
    },
  });
  const { source: sourceA, applied } = subscribe(client);

  sourceA.emit("canonical_revision", canonicalEvent({ workspaceRevision: 1, operations: [{ operationId: "srv:op-1" }] }));
  sourceA.emit("fast_revision", fastEvent({ fastRevision: 1, workspaceRevision: 1, resultClass: "visual_pending" }));
  assert.equal(client.classFor("ws1").effectiveClass, "visual_pending");

  sourceA.emit("error");
  await clock.advance(1000);
  const sourceB = FakeEventSource.instances[1];
  await clock.settle();

  assert.deepEqual(applied, ["srv:op-1", "srv:op-2"], "the missed durable op is recovered");
  assert.deepEqual(client.revisionsFor("ws1"), { workspaceRevision: 3, fastRevision: 1, exactRevision: 0 });

  // No fast frame ever arrives for revision 2-3: the visual placeholder for
  // revision 1 is superseded (the workspace moved on) and the subscription
  // is LIVE — visual_pending-then-no-fast-frame is a legitimate terminal
  // state until the next epoch, never a hang.
  const view = client.classFor("ws1");
  assert.equal(view.superseded, true, "the stale placeholder is fenced off");
  assert.equal(view.effectiveClass, "visual_pending");
  assert.equal(client.subscriptionState("ws1").state, "live");
  assert.equal(FakeEventSource.instances.length, 2, "no reconnect loop spun up waiting for fast frames");

  // The next epoch's fast frame renders again, unsuperseded, for its target.
  sourceB.emit("canonical_revision", canonicalEvent({ workspaceRevision: 3, operations: [] }));
  sourceB.emit("fast_revision", fastEvent({ fastRevision: 2, workspaceRevision: 3, resultClass: "fast_exact" }));
  const fresh = client.classFor("ws1");
  assert.equal(fresh.superseded, false);
  assert.equal(fresh.authoritative, true);
  assert.equal(fresh.targetRevision, 3);
});

// ---------------------------------------------------------------------------
// missed_events heartbeat recovery (broadcast.py L2)
// ---------------------------------------------------------------------------

test("a missed_events heartbeat triggers a catch-up that replays the dropped ops in order exactly once", async () => {
  const catchUpUrls = [];
  const releaseCatchUp = [];
  const statuses = [];
  const heartbeats = [];
  const { client, clock } = makeClient({
    onStatus: (status) => statuses.push(status),
    handler: async (request) => {
      if (request.url.includes("/operations?since_server_sequence=")) {
        catchUpUrls.push(request.url);
        const index = catchUpUrls.length;
        // Every catch-up GET is held in flight so the test can observe the
        // states between cycles.
        return new Promise((resolve) => {
          releaseCatchUp.push(() =>
            resolve(
              jsonResponse({
                operations:
                  index === 1
                    ? [
                        { operation_id: "srv:op-2", server_sequence: 2, actor_id: "actor-remote", source_family: "vegetation_geometry", entity_id: null, verb: "add", payload: {} },
                        { operation_id: "srv:op-3", server_sequence: 3, actor_id: "actor-remote", source_family: "vegetation_geometry", entity_id: null, verb: "add", payload: {} },
                      ]
                    : [],
                workspace_revision: 5,
              }),
            ),
          );
        });
      }
      return jsonResponse({}, 404);
    },
  });
  const { source, applied } = subscribe(client, {
    onHeartbeat: (event) => heartbeats.push(event),
  });

  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 1, operations: [{ operationId: "srv:op-1" }] }));
  assert.deepEqual(applied, ["srv:op-1"]);

  // The subscriber's queue overflowed; the next heartbeat carries the count.
  source.emit("heartbeat", { missed_events: 2 });
  assert.deepEqual(heartbeats, [{ missed_events: 2 }], "the heartbeat still reaches the app handler");
  await clock.settle();
  assert.equal(catchUpUrls.length, 1, "the heartbeat triggers exactly one catch-up GET");
  assert.equal(
    catchUpUrls[0].endsWith("since_server_sequence=0"),
    true,
    "the GET resumes from the applied watermark (op-1 carried no server_sequence)",
  );
  assert.ok(statuses.some((status) => status.phase === "catching_up" && status.missedEvents === 2));

  // Live frames during the open gap buffer instead of applying out of order;
  // a second missed_events heartbeat is buffered too — it must NOT start a
  // second CONCURRENT cycle.
  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 5, operations: [{ operationId: "srv:op-5" }] }));
  source.emit("fast_revision", fastEvent({ fastRevision: 4, workspaceRevision: 5, resultClass: "fast_exact" }));
  source.emit("heartbeat", { missed_events: 1 });
  assert.deepEqual(applied, ["srv:op-1"], "gap frames wait for the catch-up");
  assert.equal(client.classFor("ws1"), null, "nothing — fast frames included — renders during the gap");
  assert.equal(catchUpUrls.length, 1, "no concurrent cycle while the gap is open");

  releaseCatchUp[0]();
  await clock.settle();
  assert.deepEqual(applied, ["srv:op-1", "srv:op-2", "srv:op-3", "srv:op-5"]);
  assert.deepEqual(client.revisionsFor("ws1"), { workspaceRevision: 5, fastRevision: 4, exactRevision: 0 });
  assert.equal(client.classFor("ws1").superseded, false);

  // The flushed stale heartbeat re-verifies AFTER the flush (sequential, not
  // concurrent) from the advanced watermark — and the ops still applied
  // exactly once.
  assert.equal(catchUpUrls.length, 2);
  assert.ok(
    catchUpUrls[1].endsWith("since_server_sequence=3"),
    `the re-verification resumes from the advanced watermark: ${catchUpUrls[1]}`,
  );
  assert.deepEqual(applied, ["srv:op-1", "srv:op-2", "srv:op-3", "srv:op-5"]);

  // Cycle 2 owns the status surface until ITS catch-up lands: cycle 1's tail
  // must not flip the badge to "live" while the re-verification GET is
  // still in flight.
  assert.equal(
    client.subscriptionState("ws1").state,
    "catching_up",
    "no live flap while the nested recovery GET is in flight",
  );

  releaseCatchUp[1]();
  await clock.settle();
  assert.equal(client.subscriptionState("ws1").state, "live");

  // Status sequence: the final transition is catching_up -> live, once.
  const phases = statuses.map((status) => status.phase);
  const lastCatchingUp = phases.lastIndexOf("catching_up");
  assert.ok(lastCatchingUp > -1);
  assert.equal(
    phases.slice(lastCatchingUp).filter((phase) => phase === "live").length,
    1,
    `exactly one live after the last catching_up, got ${JSON.stringify(phases)}`,
  );
});

test("a heartbeat without missed_events never triggers a catch-up; a zero count neither", async () => {
  let catchUps = 0;
  const { client, clock } = makeClient({
    handler: async (request) => {
      if (request.url.includes("/operations?since_server_sequence=")) {
        catchUps += 1;
        return jsonResponse({ operations: [] });
      }
      return jsonResponse({}, 404);
    },
  });
  const { source } = subscribe(client);

  source.emit("heartbeat", {});
  source.emit("heartbeat", { missed_events: 0 });
  await clock.settle();
  assert.equal(catchUps, 0);
});

test("a failed catch-up after missed_events falls back to the reconnect cycle instead of stranding the gap", async () => {
  let failGet = true;
  const { client, clock } = makeClient({
    handler: async (request) => {
      if (request.url.includes("/operations?since_server_sequence=")) {
        if (failGet) return jsonResponse({ error: { code: "network_error" } }, 503);
        return jsonResponse({ operations: [], workspace_revision: 2 });
      }
      return jsonResponse({}, 404);
    },
  });
  const { source } = subscribe(client);

  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 1, operations: [{ operationId: "srv:op-1" }] }));
  source.emit("heartbeat", { missed_events: 3 });
  await clock.settle();
  assert.equal(client.subscriptionState("ws1").state, "reconnecting", "the failed recovery hands over to the reconnect cycle");

  // The reconnect cycle re-opens the stream and lands its own catch-up.
  failGet = false;
  await clock.advance(1000);
  await clock.settle();
  const sourceB = FakeEventSource.instances.at(-1);
  assert.ok(sourceB, "a replacement stream was opened");
  assert.equal(client.subscriptionState("ws1").state, "live");
  assert.deepEqual(client.revisionsFor("ws1").workspaceRevision, 2);
});

// ---------------------------------------------------------------------------
// Typed admission rejections (submit surface)
// ---------------------------------------------------------------------------

test("a workspace-burst 429 surfaces a typed burst rejection without blind retries", async () => {
  let posts = 0;
  const { client } = makeClient({
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      posts += 1;
      return jsonResponse(
        {
          error: {
            code: "rate_limited",
            message: "admitting 8 more operations would exceed the workspace's admitted rate",
            retry_after_ms: 250,
            advice: ["reduce stroke rate", "coalesce edits client-side"],
          },
        },
        429,
      );
    },
  });
  subscribe(client);

  const outcome = client
    .submitOperations("ws1", [{ sourceFamily: "vegetation_geometry", entityId: "tree-1", verb: "add", payload: {} }])
    .then((value) => ({ value }), (error) => ({ error }));
  await new Promise((resolve) => setImmediate(resolve));
  const { error } = await outcome;

  assert.ok(error instanceof RealtimeAdmissionError, "the rejection is typed");
  assert.ok(error instanceof RealtimeClientError);
  assert.equal(error.kind, "burst");
  assert.equal(error.retryAfterMs, 250);
  assert.deepEqual(error.advice, ["reduce stroke rate", "coalesce edits client-side"]);
  assert.equal(error.status, 429);
  assert.equal(error.code, "rate_limited");
  assert.equal(posts, 1, "an admission rejection is surfaced, never retried blind");
});

test("a server-wide 503 fast_lane_unavailable surfaces a server_saturated rejection", async () => {
  const { client } = makeClient({
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      return jsonResponse(
        {
          error: {
            code: "fast_lane_unavailable",
            message: "the fast lane's deadline queue is saturated; retry after the hinted interval",
            retry_after_ms: 2000,
          },
        },
        503,
      );
    },
  });
  subscribe(client);

  const outcome = client
    .submitOperations("ws1", [{ sourceFamily: "vegetation_geometry", entityId: "tree-1", verb: "add", payload: {} }])
    .then((value) => ({ value }), (error) => ({ error }));
  await new Promise((resolve) => setImmediate(resolve));
  const { error } = await outcome;

  assert.ok(error instanceof RealtimeAdmissionError);
  assert.equal(error.kind, "server_saturated");
  assert.equal(error.retryAfterMs, 2000);
  assert.equal(error.status, 503);
});

test("an idempotent retry that 429s reconciles via GET first: an already-durable batch resolves, not rejects (r2b F7)", async () => {
  let postCalls = 0;
  let catchUps = 0;
  const stalls = [];
  const { client, clock } = makeClient({
    onStall: (detail) => stalls.push(detail),
    handler: async (request) => {
      if (request.method === "POST") {
        postCalls += 1;
        if (postCalls === 1) return null; // response lost AFTER durable acceptance
        return jsonResponse(
          { error: { code: "rate_limited", message: "burst", retry_after_ms: 100, advice: [] } },
          429,
        );
      }
      if (request.url.includes("/operations?since_server_sequence=")) {
        catchUps += 1;
        return jsonResponse({
          operations: [
            { operation_id: "actor-a:op-1", server_sequence: 42, actor_id: "actor-a", source_family: "vegetation_geometry", entity_id: "tree-1", verb: "add", payload: {} },
          ],
        });
      }
      return jsonResponse({}, 404);
    },
  });
  subscribe(client);

  const acksPromise = client.submitOperations("ws1", [
    { sourceFamily: "vegetation_geometry", entityId: "tree-1", verb: "add", payload: {} },
  ]);
  await clock.advance(1000); // backoff -> retry lands the 429 -> reconcile GET
  const acks = await acksPromise;

  assert.equal(catchUps, 1, "the 429 on a retry reconciles through the catch-up GET");
  assert.equal(acks.length, 1);
  assert.equal(acks[0].operationId, "actor-a:op-1");
  assert.equal(acks[0].serverSequence, 42, "the ack resolves with the durable server_sequence");
  assert.equal(stalls.length, 0);
});

test("an idempotent retry that 429s with a NOT-durable batch surfaces the typed rejection", async () => {
  let postCalls = 0;
  const { client, clock } = makeClient({
    handler: async (request) => {
      if (request.method === "POST") {
        postCalls += 1;
        if (postCalls === 1) return null; // lost, nothing durable
        return jsonResponse(
          { error: { code: "rate_limited", message: "burst", retry_after_ms: 100, advice: ["slow down"] } },
          429,
        );
      }
      if (request.url.includes("/operations?since_server_sequence=")) {
        return jsonResponse({ operations: [] }); // the batch never landed
      }
      return jsonResponse({}, 404);
    },
  });
  subscribe(client);

  const outcome = client
    .submitOperations("ws1", [{ sourceFamily: "vegetation_geometry", entityId: "tree-1", verb: "add", payload: {} }])
    .then((value) => ({ value }), (error) => ({ error }));
  await clock.advance(1000);
  const { error } = await outcome;

  assert.ok(error instanceof RealtimeAdmissionError, "the unverifiable rejection surfaces");
  assert.equal(error.kind, "burst");
  assert.equal(postCalls, 2);
});
