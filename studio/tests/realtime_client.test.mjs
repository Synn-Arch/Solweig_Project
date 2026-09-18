// SPDX-License-Identifier: GPL-3.0-only
//
// RealtimeClient tests (additive module, not yet adopted by the studio UI)
// against injected fetch / EventSource / timer / clock seams: operation
// identity, idempotent submit retries, the revision triple, ack resolution
// via canonical_revision, the stall fence, zero-loss reconnect catch-up, and
// result-class transitions with age tracking.

import test from "node:test";
import assert from "node:assert/strict";

import { RealtimeClient } from "../realtime_client.mjs";

// ---------------------------------------------------------------------------
// Injected seams
// ---------------------------------------------------------------------------

/** Delay-aware manual clock: tasks fire in (due, id) order as time advances. */
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
    // Let already-running microtask chains (e.g. a POST that is failing right
    // now) schedule their backoff timers before filtering for due tasks.
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

/** EventSource stand-in: the factory seam the app replaces with the real one. */
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
    // No closed guard: the close test deliberately emits after close() to
    // prove the CLIENT ignores a dead source's events.
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

/** fetch stand-in; handler returning null simulates a network failure. */
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
  crypto = null,
  onStatus = null,
  onStall = null,
} = {}) {
  FakeEventSource.instances = [];
  const fetchImpl = makeFetch(handler);
  const client = new RealtimeClient({
    fetch: fetchImpl,
    eventSourceFactory: (url) => new FakeEventSource(url),
    actorId,
    crypto,
    timers: clock.timers,
    now: clock.now,
    stallTimeoutMs,
    ...(onStatus ? { onStatus } : {}),
    ...(onStall ? { onStall } : {}),
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

function designOp(overrides = {}) {
  return {
    sourceFamily: "vegetation_geometry",
    entityId: "tree-1",
    verb: "add",
    payload: { height_m: 12 },
    ...overrides,
  };
}

function canonicalEvent({ workspaceRevision, operations, fastRevision = null, exactRevision = null, epochId = 1 }) {
  return {
    workspace_revision: workspaceRevision,
    epoch_id: epochId,
    operations: operations.map((operation) => ({
      operation_id: operation.operationId ?? operation.operation_id,
      actor_id: operation.actorId ?? "actor-remote",
      source_family: operation.sourceFamily ?? "vegetation_geometry",
      entity_id: operation.entityId ?? null,
      verb: operation.verb ?? "add",
      compact_payload: operation.payload ?? operation.compact_payload ?? {},
      ...(operation.serverSequence !== undefined ? { server_sequence: operation.serverSequence } : {}),
    })),
    ...(fastRevision !== null ? { fast_revision: fastRevision } : {}),
    ...(exactRevision !== null ? { exact_revision: exactRevision } : {}),
  };
}

function ackFor(operationId, { serverSequence, epochId = 1, duplicate = false } = {}) {
  return { operation_id: operationId, server_sequence: serverSequence, epoch_id: epochId, duplicate };
}

async function posts(fetchImpl) {
  return fetchImpl.requests.filter((request) => request.method === "POST");
}

// ---------------------------------------------------------------------------
// Operation identity
// ---------------------------------------------------------------------------

test("operation identity: per-actor monotonic client_sequence and derived operation ids", async () => {
  let ackSequence = 0;
  const { client, clock, fetchImpl } = makeClient({
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      const body = JSON.parse(request.body);
      return jsonResponse({
        acks: body.operations.map((operation) =>
          ackFor(operation.operation_id, { serverSequence: (ackSequence += 1) }),
        ),
      });
    },
  });
  const { source } = subscribe(client);

  const first = client.submitOperations("ws1", [designOp(), designOp()], { baseRevision: 3 });
  await clock.settle();
  const second = client.submitOperations("ws1", [
    designOp({ operationId: "custom-op-id" }),
    designOp(),
  ]);
  await clock.settle();

  const submitted = await posts(fetchImpl);
  assert.equal(submitted.length, 2);
  const firstBody = JSON.parse(submitted[0].body);
  assert.equal(firstBody.actor_id, "actor-a");
  assert.deepEqual(firstBody.operations[0], {
    operation_id: "actor-a:op-1",
    client_sequence: 1,
    base_revision: 3,
    source_family: "vegetation_geometry",
    entity_id: "tree-1",
    verb: "add",
    payload: { height_m: 12 },
  });
  assert.equal(firstBody.operations[1].operation_id, "actor-a:op-2");
  assert.equal(firstBody.operations[1].client_sequence, 2);

  const secondBody = JSON.parse(submitted[1].body);
  // The overridden id is honored; the counter still advances monotonically.
  assert.equal(secondBody.operations[0].operation_id, "custom-op-id");
  assert.equal(secondBody.operations[0].client_sequence, 3);
  assert.equal(secondBody.operations[1].operation_id, "actor-a:op-4");
  assert.equal(secondBody.operations[1].client_sequence, 4);

  // baseRevision is advisory: it rides the body but never gates the submit.
  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 9,
    operations: [
      { operationId: "actor-a:op-1" },
      { operationId: "actor-a:op-2" },
      { operationId: "custom-op-id" },
      { operationId: "actor-a:op-4" },
    ],
  }));
  const [firstAcks, secondAcks] = await Promise.all([first, second]);
  assert.equal(firstAcks[0].operationId, "actor-a:op-1");
  assert.deepEqual(
    [firstAcks[1].serverSequence, secondAcks[0].serverSequence, secondAcks[1].serverSequence],
    [2, 3, 4],
  );
});

test("actorId defaults to a generated uuid and can be injected", () => {
  const deterministic = makeClient({
    crypto: { randomUUID: () => "11111111-2222-4333-8444-555555555555" },
    actorId: null,
  });
  assert.equal(deterministic.client.actorId, "11111111-2222-4333-8444-555555555555");
});

// ---------------------------------------------------------------------------
// Submit retries and duplicate resolution
// ---------------------------------------------------------------------------

test("network failure re-POSTs the identical idempotent body; a duplicate ack resolves with the original server_sequence", async () => {
  const stalls = [];
  let call = 0;
  const { client, clock, fetchImpl } = makeClient({
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      call += 1;
      if (call === 1) return null; // first attempt dies on the network
      // The lost first POST was durably recorded: the retry answers duplicate
      // with the ORIGINAL server_sequence and epoch.
      return jsonResponse({ acks: [ackFor("actor-a:op-1", { serverSequence: 42, epochId: 3, duplicate: true })] });
    },
    onStall: (detail) => stalls.push(detail),
  });
  const { source } = subscribe(client);

  const acksPromise = client.submitOperations("ws1", [designOp()]);
  await clock.settle();
  assert.equal((await posts(fetchImpl)).length, 1, "no retry before the backoff fires");
  assert.equal(stalls.length, 0);

  await clock.advance(1000); // 1s backoff -> retry lands, server answers duplicate
  const submitted = await posts(fetchImpl);
  assert.equal(submitted.length, 2);
  assert.equal(submitted[0].headers["Idempotency-Key"], submitted[1].headers["Idempotency-Key"]);
  assert.equal(submitted[0].body, submitted[1].body, "retry must repeat the identical batch");

  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 1,
    operations: [{ operationId: "actor-a:op-1" }],
  }));
  const acks = await acksPromise;
  assert.deepEqual(acks, [
    { operationId: "actor-a:op-1", serverSequence: 42, epochId: 3, duplicate: true },
  ]);
  assert.equal(stalls.length, 0);
});

test("submit gives up after three backoff retries and rejects with the transport error", async () => {
  const { client, clock, fetchImpl } = makeClient({
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      return null;
    },
  });
  subscribe(client);

  const outcome = client
    .submitOperations("ws1", [designOp()])
    .then((value) => ({ value }), (error) => ({ error }));
  await clock.advance(7000); // 1s + 2s + 4s backoffs
  const { error } = await outcome;

  assert.ok(error, "the submit must reject after exhausting retries");
  assert.equal(error.code, "network_error");
  assert.equal((await posts(fetchImpl)).length, 4, "initial attempt + three retries");
});

// ---------------------------------------------------------------------------
// Revision triple tracking
// ---------------------------------------------------------------------------

test("canonical/fast/exact events advance the revision triple and never regress it", async () => {
  const { client, clock } = makeClient();
  const { source } = subscribe(client);

  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 1,
    fastRevision: 1,
    exactRevision: 0,
    operations: [{ operationId: "srv:op-1" }],
  }));
  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 2,
    fastRevision: 2,
    exactRevision: 1,
    operations: [{ operationId: "srv:op-2" }],
  }));
  // A delayed/duplicate older event must not roll the triple back.
  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 1,
    fastRevision: 1,
    exactRevision: 0,
    operations: [{ operationId: "srv:op-1" }],
  }));
  assert.deepEqual(client.revisionsFor("ws1"), {
    workspaceRevision: 2,
    fastRevision: 2,
    exactRevision: 1,
  });

  source.emit("fast_revision", { fast_revision: 3, workspace_revision: 2, result_class: "fast_qualified", exact_base_revision: 1 });
  source.emit("exact_revision", { exact_revision: 2, workspace_revision: 2 });
  await clock.settle();
  assert.deepEqual(client.revisionsFor("ws1"), {
    workspaceRevision: 2,
    fastRevision: 3,
    exactRevision: 2,
  });
  assert.equal(client.revisionsFor("missing"), null);
});

// ---------------------------------------------------------------------------
// Ack resolution via canonical_revision
// ---------------------------------------------------------------------------

test("submit promises resolve only when the op appears in canonical_revision, exactly once", async () => {
  const { client, clock } = makeClient({
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      const body = JSON.parse(request.body);
      return jsonResponse({
        acks: body.operations.map((operation) =>
          ackFor(operation.operation_id, { serverSequence: 7, epochId: 2 }),
        ),
      });
    },
  });
  const { source, applied } = subscribe(client);

  const acksPromise = client.submitOperations("ws1", [designOp()]);
  await clock.settle();

  const early = await Promise.race([
    acksPromise.then(() => "resolved"),
    new Promise((resolve) => setImmediate(() => resolve("pending"))),
  ]);
  assert.equal(early, "pending", "an HTTP ack alone must not resolve the submit");

  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 1,
    operations: [{ operationId: "actor-a:op-1" }],
  }));
  // Duplicate canonical carriage of the same op must not re-apply it.
  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 2,
    operations: [{ operationId: "actor-a:op-1" }],
  }));
  const acks = await acksPromise;
  assert.deepEqual(acks, [
    { operationId: "actor-a:op-1", serverSequence: 7, epochId: 2, duplicate: false },
  ]);
  assert.deepEqual(applied, ["actor-a:op-1"]);
});

// ---------------------------------------------------------------------------
// Stall fence
// ---------------------------------------------------------------------------

test("acks still unresolved past the deadline surface through onStall", async () => {
  const stalls = [];
  const { client, clock } = makeClient({
    stallTimeoutMs: 5000,
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      const body = JSON.parse(request.body);
      return jsonResponse({
        acks: body.operations.map((operation) =>
          ackFor(operation.operation_id, { serverSequence: 7, epochId: 2 }),
        ),
      });
    },
    onStall: (detail) => stalls.push(detail),
  });
  const { source } = subscribe(client);

  const acksPromise = client.submitOperations("ws1", [designOp()]);
  await clock.settle();
  await clock.advance(4999);
  assert.equal(stalls.length, 0, "no stall before the deadline");

  await clock.advance(1);
  assert.equal(stalls.length, 1);
  assert.deepEqual(stalls[0], {
    workspaceId: "ws1",
    operationId: "actor-a:op-1",
    serverSequence: 7,
    ageMs: 5000,
  });

  // The fence surfaces; the promise itself still resolves when the op lands.
  source.emit("canonical_revision", canonicalEvent({
    workspaceRevision: 1,
    operations: [{ operationId: "actor-a:op-1" }],
  }));
  assert.equal((await acksPromise)[0].serverSequence, 7);
  await clock.advance(10_000);
  assert.equal(stalls.length, 1, "a resolved ack never re-stalls");
});

// ---------------------------------------------------------------------------
// Reconnect with zero-loss catch-up
// ---------------------------------------------------------------------------

test("reconnect closes the event gap: catch-up applies missed ops, buffered live events resume in order", async () => {
  const catchUpUrls = [];
  let releaseCatchUp = null;
  const statuses = [];
  const { client, clock } = makeClient({
    onStatus: (status) => statuses.push(status),
    handler: async (request) => {
      if (request.method === "POST") return jsonResponse({ acks: [] }, 202);
      if (request.url.includes("/operations?since_server_sequence=")) {
        catchUpUrls.push(request.url);
        return new Promise((resolve) => {
          releaseCatchUp = () =>
            resolve(
              jsonResponse({
                operations: [
                  { operation_id: "srv:op-3", server_sequence: 3, actor_id: "actor-remote", source_family: "vegetation_geometry", entity_id: null, verb: "add", payload: { height_m: 9 } },
                  { operation_id: "srv:op-4", server_sequence: 4, actor_id: "actor-remote", source_family: "landcover_surface", entity_id: null, verb: "paint", payload: { chunk: "c1" } },
                ],
              }),
            );
        });
      }
      return jsonResponse({}, 404);
    },
  });
  const { source: sourceA, applied } = subscribe(client);

  // Live events 1-2 arrive before the drop.
  sourceA.emit("canonical_revision", canonicalEvent({ workspaceRevision: 1, operations: [{ operationId: "srv:op-1" }] }));
  sourceA.emit("canonical_revision", canonicalEvent({ workspaceRevision: 2, operations: [{ operationId: "srv:op-2" }] }));
  assert.deepEqual(applied, ["srv:op-1", "srv:op-2"]);

  sourceA.emit("error");
  await clock.settle();
  assert.ok(sourceA.closed, "the broken source is closed");
  assert.ok(statuses.some((status) => status.phase === "reconnecting" && status.attempt === 1));
  assert.deepEqual(applied, ["srv:op-1", "srv:op-2"]);

  await clock.advance(1000); // 1s reconnect backoff
  const sourceB = FakeEventSource.instances[1];
  assert.ok(sourceB, "a replacement source was created");
  assert.ok(!sourceB.closed);
  assert.equal(catchUpUrls.length, 1);
  assert.ok(
    catchUpUrls[0].endsWith("/api/v1/workspaces/ws1/operations?since_server_sequence=0"),
    `unexpected catch-up url ${catchUpUrls[0]}`,
  );

  // A live event arriving while the gap is open must be buffered, not applied.
  sourceB.emit("canonical_revision", canonicalEvent({ workspaceRevision: 5, operations: [{ operationId: "srv:op-5" }] }));
  assert.deepEqual(applied, ["srv:op-1", "srv:op-2"], "gap events apply only after catch-up");

  releaseCatchUp();
  await clock.settle();
  assert.deepEqual(applied, ["srv:op-1", "srv:op-2", "srv:op-3", "srv:op-4", "srv:op-5"]);

  // Live streaming resumes without a second catch-up.
  sourceB.emit("canonical_revision", canonicalEvent({ workspaceRevision: 6, operations: [{ operationId: "srv:op-6" }] }));
  await clock.settle();
  assert.deepEqual(applied, ["srv:op-1", "srv:op-2", "srv:op-3", "srv:op-4", "srv:op-5", "srv:op-6"]);
  assert.equal(catchUpUrls.length, 1);
  assert.equal(FakeEventSource.instances.length, 2);
  assert.equal(statuses.at(-1).phase, "live");
  assert.deepEqual(client.revisionsFor("ws1"), { workspaceRevision: 6, fastRevision: 0, exactRevision: 0 });
  // Ops 3-4 carried server sequences; the next catch-up would resume from 4.
  assert.equal(client.subscriptionState("ws1").lastServerSequence, 4);
});

// ---------------------------------------------------------------------------
// Result-class state machine
// ---------------------------------------------------------------------------

test("result-class transitions track the fast classes and age since the last exact_revision event", async () => {
  const { client, clock } = makeClient();
  const { source } = subscribe(client);

  assert.equal(client.classFor("ws1"), null, "no class before any class-bearing event");

  source.emit("fast_revision", { fast_revision: 1, workspace_revision: 1, result_class: "visual_pending", exact_base_revision: 0, provenance: { model: "pending" } });
  let view = client.classFor("ws1");
  assert.equal(view.class, "visual_pending");
  assert.equal(view.fastRevision, 1);
  assert.equal(view.exactBaseRevision, 0);
  assert.equal(view.ageMs, 0);

  await clock.advance(500);
  assert.equal(client.classFor("ws1").ageMs, 500);

  source.emit("fast_revision", { fast_revision: 2, workspace_revision: 2, result_class: "fast_qualified", exact_base_revision: 0, provenance: { model: "response-kernel" } });
  view = client.classFor("ws1");
  assert.equal(view.class, "fast_qualified");
  assert.equal(view.fastRevision, 2);
  assert.equal(view.ageMs, 500);

  source.emit("exact_revision", { exact_revision: 2, workspace_revision: 2 });
  view = client.classFor("ws1");
  assert.equal(view.class, "exact_reconciled");
  assert.equal(view.exactRevision, 2);
  assert.equal(view.ageMs, 0);

  // A new edit moves the class off exact again; age restarts from the exact event.
  source.emit("fast_revision", { fast_revision: 3, workspace_revision: 3, result_class: "fast_exact", exact_base_revision: 2, provenance: { model: "cache" } });
  view = client.classFor("ws1");
  assert.equal(view.class, "fast_exact");
  assert.equal(view.exactBaseRevision, 2);
  assert.equal(view.ageMs, 0);
  await clock.advance(250);
  assert.equal(client.classFor("ws1").ageMs, 250);
  assert.equal(client.classFor("missing"), null);
});

// ---------------------------------------------------------------------------
// Heartbeat and close
// ---------------------------------------------------------------------------

test("heartbeat events are bookkept and close() silences the subscription", async () => {
  const heartbeats = [];
  const { client, clock } = makeClient();
  const { source, applied, subscription } = subscribe(client, {
    onHeartbeat: (event) => heartbeats.push(event),
  });

  source.emit("heartbeat", { server_time: "t0" });
  await clock.settle();
  assert.deepEqual(heartbeats, [{ server_time: "t0" }]);
  assert.equal(client.subscriptionState("ws1").lastHeartbeatAt, clock.now());
  assert.equal(subscription.state, "live");

  subscription.close();
  assert.ok(source.closed);
  assert.equal(subscription.state, "closed");
  assert.equal(client.subscriptionState("ws1").state, "closed");

  source.emit("canonical_revision", canonicalEvent({ workspaceRevision: 3, operations: [{ operationId: "srv:op-9" }] }));
  await clock.settle();
  assert.deepEqual(applied, [], "a closed subscription applies nothing");
});

// ---------------------------------------------------------------------------
// Presence identity on the subscribe URL
// ---------------------------------------------------------------------------

test("the SSE subscribe URL carries the client's actor_id", () => {
  // The server's heartbeat roster only lists actors that registered one at
  // subscribe time; a URL without ?actor_id= would silently demote every
  // client's live presence tier to "+N earlier" forever.
  const { client } = makeClient({ actorId: "actor-a" });
  const { source } = subscribe(client);
  const url = new URL(source.url, "http://example.test");
  assert.equal(url.pathname, "/api/v1/workspaces/ws1/events");
  assert.equal(url.searchParams.get("actor_id"), "actor-a");
});
