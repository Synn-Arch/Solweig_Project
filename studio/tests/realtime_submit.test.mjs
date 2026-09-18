// SPDX-License-Identifier: GPL-3.0-only
//
// Submit-plane tests for the realtime collaboration client against the
// FROZEN server vocabulary (routes_realtime.py): operation envelopes minted
// by `op_builder.mjs` (design gestures -> realtime operations), the exact
// POST /api/v1/workspaces/{id}/operations body, canonical-fence ack
// resolution, duplicate/idempotent-retry resolution, admission rejections
// (429 fast_lane_overloaded / 503 fast_lane_unavailable — surfaced typed,
// never auto-retried), the transient 1s/2s/4s submit ladder, the typed 409
// operation_id_reused, and the advisory base_divergent ack flag.
//
// The fake plane below is an in-memory ledger speaking the real route's ack
// grammar ({accepted: [...]} with the merged-client `acks` alias, plus the
// revision triple) and enforcing the frozen family/verb vocabularies on
// every POST, so each submit test doubles as a wire-shape test.

import test from "node:test";
import assert from "node:assert/strict";

import {
  RealtimeClient,
  RealtimeClientError,
  RealtimeAdmissionError,
} from "../realtime_client.mjs";
import {
  nextClientSequence,
  mintOperationId,
  treeOperation,
  frameFromGeometry,
  frameFromSpans,
  landcoverPaintOperation,
  solveFrameFromSamples,
  uvToWorld,
  worldToUv,
  worldTree,
  meteorologySetOperation,
  parameterSetOperation,
  viewSelectOperation,
  dateOperation,
} from "../op_builder.mjs";
import { apiTree } from "./contract_harness.mjs";

// ---------------------------------------------------------------------------
// Injected seams (same shape as realtime_client.test.mjs)
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

function makeClient({
  handler,
  clock = makeClock(),
  actorId = "actor-a",
  stallTimeoutMs = 10_000,
  onStall = null,
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
    ...(onStall ? { onStall } : {}),
  });
  return { client, clock, fetchImpl };
}

function subscribe(client, handlers = {}) {
  const applied = [];
  client.subscribeWorkspace("ws1", {
    onOperation: (operation) => applied.push(operation),
    ...handlers,
  });
  const source = FakeEventSource.instances.at(-1);
  return { source, applied };
}

/**
 * Submit one op_builder envelope through the client. The client's input
 * vocabulary is camelCase (`operationId`, `baseRevision`) while the builder
 * speaks the frozen wire vocabulary, and the WIRE client_sequence is always
 * assigned by the client's own per-connection counter (the envelope's
 * minted operation_id rides verbatim) — this is the adapter seam the studio
 * UI performs for every gesture.
 */
function submitEnvelope(client, workspaceId, envelope) {
  return client.submitOperations(workspaceId, [
    {
      operationId: envelope.operation_id,
      baseRevision: envelope.base_revision,
      sourceFamily: envelope.source_family,
      entityId: envelope.entity_id,
      verb: envelope.verb,
      payload: envelope.payload,
    },
  ]);
}

// ---------------------------------------------------------------------------
// Fake realtime plane: the frozen server vocabulary over an in-memory ledger
// ---------------------------------------------------------------------------

const SOURCE_FAMILIES = new Set([
  "building_geometry",
  "vegetation_geometry",
  "landcover_surface",
  "meteorological_forcing",
  "model_receptor_parameters",
  "selected_date_time",
  "output_view",
]);

const OPERATION_VERBS = new Set([
  "add",
  "replace",
  "move",
  "delete",
  "paint",
  "set",
  "select",
]);

const OPERATIONS_PATH = /\/api\/v1\/workspaces\/[^/]+\/operations$/;

/**
 * In-memory POST /operations ledger mirroring routes_realtime.py: frozen
 * family/verb/payload shape validation, idempotent duplicate replays by
 * operation fingerprint, a typed 409 `operation_id_reused` when a known id
 * arrives with a mutated payload, the advisory `base_divergent` ack flag,
 * and the acceptance body's `accepted` array (plus the merged-client `acks`
 * alias) with the revision triple. `loseFirstResponses` drops the HTTP
 * response AFTER the durable accept — the lost-ack retry scenario.
 */
class FakeRealtimePlane {
  constructor({ workspaceRevision = 0, epochId = 3, loseFirstResponses = 0 } = {}) {
    this.workspaceRevision = workspaceRevision;
    this.epochId = epochId;
    this.loseResponses = loseFirstResponses;
    this.serverSequence = 0;
    this.durable = new Map(); // operation_id -> ledger record
    this.posts = []; // {path, headers, body (raw string), parsed}
    this.ackBodies = []; // accepted[] per completed POST (server-side evidence)
  }

  /** The identity the store deduplicates on: the operation sans transport. */
  static fingerprint(operation) {
    return JSON.stringify({
      source_family: operation.source_family,
      entity_id: operation.entity_id,
      verb: operation.verb,
      payload: operation.payload,
    });
  }

  #json(body, status = 200) {
    return jsonResponse(body, status);
  }

  #shapeError(parsed) {
    const operations = parsed?.operations;
    if (
      !Array.isArray(operations) ||
      operations.length < 1 ||
      operations.length > 128 ||
      typeof parsed?.actor_id !== "string" ||
      parsed.actor_id === ""
    ) {
      return { code: "invalid_batch", message: "POST body is {actor_id, operations:[1..128]}" };
    }
    for (const [index, operation] of operations.entries()) {
      if (!SOURCE_FAMILIES.has(operation?.source_family)) {
        return { code: "unknown_source_family", message: `operations[${index}].source_family` };
      }
      if (!OPERATION_VERBS.has(operation?.verb)) {
        return { code: "unknown_operation_verb", message: `operations[${index}].verb` };
      }
      if (
        !operation?.payload ||
        typeof operation.payload !== "object" ||
        Array.isArray(operation.payload) ||
        Object.keys(operation.payload).length === 0
      ) {
        return { code: "invalid_operation_payload", message: `operations[${index}].payload` };
      }
    }
    return null;
  }

  /** Accept one batch against the ledger; returns the fetch response (or null). */
  acceptBatch(parsed) {
    const invalid = this.#shapeError(parsed);
    if (invalid) {
      return this.#json({ error: { code: invalid.code, message: invalid.message } }, 400);
    }
    // Conflict check BEFORE any append: a reused id with a mutated payload
    // refuses the whole batch, exactly like the store's fingerprint conflict.
    for (const operation of parsed.operations) {
      const existing = this.durable.get(operation.operation_id);
      if (
        existing &&
        FakeRealtimePlane.fingerprint(existing.request) !== FakeRealtimePlane.fingerprint(operation)
      ) {
        return this.#json(
          {
            error: {
              code: "operation_id_reused",
              message: `operation ${operation.operation_id} was accepted with a different payload`,
              operation_id: operation.operation_id,
            },
          },
          409,
        );
      }
    }
    const accepted = [];
    for (const operation of parsed.operations) {
      const existing = this.durable.get(operation.operation_id);
      if (existing) {
        accepted.push({
          operation_id: existing.operation_id,
          server_sequence: existing.server_sequence,
          epoch_id: existing.epoch_id,
          duplicate: true,
        });
        continue;
      }
      const record = {
        operation_id: operation.operation_id,
        server_sequence: (this.serverSequence += 1),
        epoch_id: this.epochId,
        actor_id: parsed.actor_id,
        source_family: operation.source_family,
        entity_id: operation.entity_id,
        verb: operation.verb,
        payload: operation.payload,
        request: operation,
      };
      this.durable.set(record.operation_id, record);
      accepted.push({
        operation_id: record.operation_id,
        server_sequence: record.server_sequence,
        epoch_id: record.epoch_id,
        duplicate: false,
      });
    }
    // Advisory only, never a rejection (routes_realtime.py): a carried
    // base_revision that no longer matches the published workspace_revision.
    for (const [index, operation] of parsed.operations.entries()) {
      const base = operation.base_revision;
      if (base !== null && base !== undefined && Number(base) !== this.workspaceRevision) {
        accepted[index].base_divergent = true;
      }
    }
    this.ackBodies.push(accepted);
    if (this.loseResponses > 0) {
      this.loseResponses -= 1;
      return null; // response lost AFTER the durable accept
    }
    return this.#json({
      workspace_id: "ws1",
      actor_id: parsed.actor_id,
      accepted,
      acks: accepted, // the merged client reads `acks ?? operations`
      workspace_revision: this.workspaceRevision,
      fast_revision: 0,
      exact_revision: 0,
      exact_base_revision: 0,
    });
  }

  fetchImpl = async (url, init = {}) => {
    const method = init.method ?? "GET";
    const path = new URL(String(url), "http://server.test").pathname;
    if (method === "POST" && OPERATIONS_PATH.test(path)) {
      const parsed = JSON.parse(init.body);
      this.posts.push({ path, headers: init.headers ?? {}, body: init.body, parsed });
      return this.acceptBatch(parsed);
    }
    return this.#json({ error: { code: "not_found", message: path } }, 404);
  };

  /** Canonical SSE carriage for durable operations (bumps the revision). */
  emitCanonical(source, operationIds) {
    const event = {
      workspace_revision: (this.workspaceRevision += 1),
      epoch_id: this.epochId,
      operations: operationIds.map((operationId) => {
        const record = this.durable.get(operationId);
        assert.ok(record, `emitCanonical: ${operationId} is not durable`);
        return {
          operation_id: record.operation_id,
          actor_id: record.actor_id,
          source_family: record.source_family,
          entity_id: record.entity_id,
          verb: record.verb,
          compact_payload: record.payload,
          server_sequence: record.server_sequence,
        };
      }),
    };
    source.emit("canonical_revision", event);
  }
}

/** A client wired to the fake plane (handler may script failures around it). */
function makePlaneClient(plane, { actorId, handler = null, ...rest } = {}) {
  return makeClient({
    actorId,
    handler: handler ?? ((request, count) => plane.fetchImpl(request.url, {
      method: request.method,
      headers: request.headers,
      body: request.body,
    })),
    ...rest,
  });
}

const PAINT_WINDOW = { row_start: 2, row_stop: 6, col_start: 3, col_stop: 7 };

// ---------------------------------------------------------------------------
// op_builder: gesture -> operation envelope (pure module)
// ---------------------------------------------------------------------------

test("op_builder: every family builder stamps the frozen envelope — family, verb, entity identity, family payload shape", () => {
  const actor = "actor-shapes";
  // The adapter schema the app actually submits (`worldTree` output): world
  // metres, radius, trunk_ratio — the vegetation adapter's exact-lane
  // validation rejects the legacy u/v/canopy_diameter_m spelling.
  const tree = {
    tree_id: "tree-7",
    x_m: 622194.7,
    y_m: 3354294.3,
    height_m: 18,
    canopy_radius_m: 5.5,
    trunk_ratio: 0.25,
  };
  const envelopes = [
    {
      built: treeOperation(actor, "add", tree, 11),
      source_family: "vegetation_geometry",
      verb: "add",
      entity_id: "tree-7",
      payload: tree,
    },
    {
      built: treeOperation(actor, "delete", { tree_id: "tree-7" }, 11),
      source_family: "vegetation_geometry",
      verb: "delete",
      entity_id: "tree-7",
      payload: { tree_id: "tree-7" },
    },
    {
      built: landcoverPaintOperation(actor, PAINT_WINDOW, 3, 11),
      source_family: "landcover_surface",
      verb: "paint",
      entity_id: null,
      payload: { window: PAINT_WINDOW, class: 3 },
    },
    {
      built: meteorologySetOperation(actor, "Ta", 12, 31.5, 11),
      source_family: "meteorological_forcing",
      verb: "set",
      entity_id: "Ta",
      payload: { values: { Ta: 31.5 }, time_index: 12 },
    },
    {
      built: parameterSetOperation(actor, "transVeg", 0.04, 11),
      source_family: "model_receptor_parameters",
      verb: "set",
      entity_id: "transVeg",
      payload: { values: { transVeg: 0.04 } },
    },
    {
      built: viewSelectOperation(actor, "utci", 11),
      source_family: "output_view",
      verb: "select",
      entity_id: "utci",
      payload: { values: { output: "utci" } },
    },
    {
      built: dateOperation(actor, "2026-07-21T15:00:00+02:00", 11),
      source_family: "selected_date_time",
      verb: "set",
      entity_id: null,
      payload: { values: { date: "2026-07-21T15:00:00+02:00" } },
    },
  ];

  for (const { built, source_family, verb, entity_id, payload } of envelopes) {
    assert.ok(SOURCE_FAMILIES.has(source_family), `${source_family} is frozen vocabulary`);
    assert.ok(OPERATION_VERBS.has(verb), `${verb} is frozen vocabulary`);
    assert.deepEqual(
      Object.keys(built).sort(),
      [
        "base_revision",
        "client_sequence",
        "entity_id",
        "operation_id",
        "payload",
        "source_family",
        "verb",
      ],
      "the envelope carries exactly the seven contract fields",
    );
    assert.equal(built.source_family, source_family);
    assert.equal(built.verb, verb);
    assert.equal(built.entity_id, entity_id);
    assert.deepEqual(built.payload, payload);
    // The wire refuses empty payloads (invalid_operation_payload): every
    // gesture family must compose a non-empty JSON object.
    assert.equal(typeof built.payload, "object");
    assert.ok(Object.keys(built.payload).length > 0);
  }
});

test("op_builder frame: uvToWorld/worldToUv round-trip in the minus-y convention", () => {
  const frame = frameFromGeometry({
    originX: 622000,
    originY: 3355000,
    cols: 500,
    rows: 500,
    pixelSize: 2,
  });
  const { x_m, y_m } = uvToWorld(0.25, 0.75, frame);
  assert.equal(x_m, 622000 + 0.25 * 1000);
  assert.equal(y_m, 3355000 - 0.75 * 1000); // v grows south, world y grows north
  const back = worldToUv(x_m, y_m, frame);
  assert.ok(Math.abs(back.u - 0.25) < 1e-12);
  assert.ok(Math.abs(back.v - 0.75) < 1e-12);
});

test("op_builder frame: solveFrameFromSamples solves the TRANSFORM convention (positive spanY) and round-trips", () => {
  const frame = frameFromGeometry({
    originX: 1000,
    originY: 2000,
    cols: 600,
    rows: 400,
    pixelSize: 2.5,
  });
  // Two observed trees: the client knows (u, v); the funnelled ops carried
  // true world metres. Distinct on BOTH axes.
  const samples = [0.2, 0.8].map((u) => {
    const world = uvToWorld(u, u + 0.1, frame);
    return { u, v: u + 0.1, x_m: world.x_m, y_m: world.y_m };
  });
  const solved = solveFrameFromSamples(samples);
  assert.ok(solved, "two distinct samples solve the frame");
  assert.ok(Math.abs(solved.originX - frame.originX) < 1e-6);
  assert.ok(Math.abs(solved.originY - frame.originY) < 1e-6);
  assert.ok(Math.abs(solved.spanX - frame.spanX) < 1e-6);
  // Positive spanY is the whole point: solving the raw slope form hands
  // back a negative span that worldToUv then flips twice (mirrored v).
  assert.ok(Math.abs(solved.spanY - frame.spanY) < 1e-6, `spanY ${solved.spanY}`);
  const back = worldToUv(samples[0].x_m, samples[0].y_m, solved);
  assert.ok(Math.abs(back.u - samples[0].u) < 1e-9);
  assert.ok(Math.abs(back.v - samples[0].v) < 1e-9);
});

test("op_builder frame: a degenerate sample pair is skipped, not fatal", () => {
  const frame = frameFromSpans({ originX: 0, originY: 0, spanX: 1000, spanY: 1000 });
  const good = uvToWorld(0.3, 0.4, frame);
  const other = uvToWorld(0.6, 0.9, frame);
  const solved = solveFrameFromSamples([
    { u: 0.3, v: 0.4, x_m: good.x_m, y_m: good.y_m },
    { u: 0.3, v: 0.4, x_m: good.x_m, y_m: good.y_m }, // duplicate observation
    { u: 0.6, v: 0.9, x_m: other.x_m, y_m: other.y_m },
  ]);
  assert.ok(solved, "the duplicate pair is skipped; the distinct pair still solves");
});

test("op_builder worldTree carries the adapter schema (world metres, radius, trunk default)", () => {
  const frame = frameFromSpans({ originX: 10, originY: 500, spanX: 200, spanY: 100 });
  const world = worldTree({ u: 0.5, v: 0.25, heightM: 12, canopyDiameterM: 7 }, frame);
  assert.deepEqual(world, {
    x_m: 110,
    y_m: 475,
    height_m: 12,
    canopy_radius_m: 3.5,
    trunk_ratio: 0.25,
  });
});

test("op_builder: operation ids are minted from a per-actor monotonic sequence starting at 1", () => {
  assert.equal(mintOperationId("actor-x", 4), "actor-x:op-4");
  assert.equal(mintOperationId("solo", 1), "solo:op-1");

  // A fresh actor starts at 1 and steps by exactly one.
  assert.equal(nextClientSequence("actor-fresh"), 1);
  assert.equal(nextClientSequence("actor-fresh"), 2);
  assert.equal(nextClientSequence("actor-fresh"), 3);

  // Actors count independently: no gesture of one actor moves another's ids.
  assert.equal(nextClientSequence("actor-fresh-b"), 1);
  assert.equal(nextClientSequence("actor-fresh"), 4);

  const actor = "actor-seq";
  const first = treeOperation(actor, "add", apiTree("tree-1"), 0);
  const second = dateOperation(actor, "2026-07-21T15:00:00+02:00", 0);
  const third = viewSelectOperation(actor, "utci", 0);
  assert.equal(second.client_sequence, first.client_sequence + 1);
  assert.equal(third.client_sequence, second.client_sequence + 1);
  for (const envelope of [first, second, third]) {
    assert.equal(envelope.operation_id, mintOperationId(actor, envelope.client_sequence));
    assert.match(envelope.operation_id, /^actor-seq:op-\d+$/);
  }
});

test("op_builder: base_revision passes through untouched (advisory metadata, never minted locally)", () => {
  assert.equal(treeOperation("actor-pass", "add", apiTree("tree-1"), 42).base_revision, 42);
  assert.equal(meteorologySetOperation("actor-pass", "Ta", 3, 30, null).base_revision, null);
  assert.equal(dateOperation("actor-pass", "2026-07-21T15:00:00+02:00", null).base_revision, null);
});

// ---------------------------------------------------------------------------
// Submit happy path through the frozen wire
// ---------------------------------------------------------------------------

test("submit happy path: the POST body is the frozen shape and the promise resolves only at the canonical fence", async () => {
  const actor = "actor-happy";
  const plane = new FakeRealtimePlane({ epochId: 5 });
  const { client, clock, fetchImpl } = makePlaneClient(plane, { actorId: actor });
  const { source, applied } = subscribe(client);

  const envelope = treeOperation(actor, "add", apiTree("tree-7"), 7);
  const acksPromise = submitEnvelope(client, "ws1", envelope);
  await clock.settle();

  // The POST body: {actor_id, operations:[one op]} with the envelope's minted
  // identity carried verbatim (client_sequence is the CLIENT's connection
  // counter — 1 here, in lockstep with the builder's fresh actor).
  const posts = fetchImpl.requests.filter((request) => request.method === "POST");
  assert.equal(posts.length, 1);
  assert.deepEqual(JSON.parse(posts[0].body), {
    actor_id: actor,
    operations: [
      {
        operation_id: `${actor}:op-1`,
        client_sequence: 1,
        base_revision: 7,
        source_family: "vegetation_geometry",
        entity_id: "tree-7",
        verb: "add",
        payload: apiTree("tree-7"),
      },
    ],
  });
  assert.equal(typeof posts[0].headers["Idempotency-Key"], "string");

  // The HTTP ack alone must not resolve the submit (zero-loss client fence).
  const early = await Promise.race([
    acksPromise.then(
      () => "resolved",
      () => "rejected",
    ),
    new Promise((resolve) => setImmediate(() => resolve("pending"))),
  ]);
  assert.equal(early, "pending");

  plane.emitCanonical(source, [`${actor}:op-1`]);
  const acks = await acksPromise;
  assert.equal(acks.length, 1);
  assert.equal(acks[0].operationId, `${actor}:op-1`);
  assert.equal(acks[0].serverSequence, 1);
  assert.equal(acks[0].epochId, 5);
  assert.equal(acks[0].duplicate, false);
  assert.deepEqual(applied.map((operation) => operation.operationId), [`${actor}:op-1`]);
  assert.equal(applied[0].entityId, "tree-7");
  assert.equal(applied[0].verb, "add");
  assert.equal(client.revisionsFor("ws1").workspaceRevision, 1);
});

// ---------------------------------------------------------------------------
// Duplicate resolution (idempotent replay)
// ---------------------------------------------------------------------------

test("a lost ack re-POSTs identically; the duplicate ack resolves ONE effect with the original server_sequence", async () => {
  const actor = "actor-dup";
  const plane = new FakeRealtimePlane({ epochId: 2, loseFirstResponses: 1 });
  const { client, clock } = makePlaneClient(plane, { actorId: actor });
  const { source, applied } = subscribe(client);

  // base 0 matches the plane's published revision so the duplicate path is
  // observed WITHOUT the (separately tested) base_divergent advisory.
  const envelope = landcoverPaintOperation(actor, PAINT_WINDOW, 3, 0);
  const acksPromise = submitEnvelope(client, "ws1", envelope);
  await clock.settle();
  assert.equal(plane.posts.length, 1, "no retry before the backoff fires");

  await clock.advance(1000); // 1s backoff -> the retry lands on the ledger
  assert.equal(plane.posts.length, 2);
  assert.equal(plane.posts[0].body, plane.posts[1].body, "the retry repeats the identical batch");
  // Ledger-level single resolution: one durable record, second ack duplicate.
  assert.equal(plane.durable.size, 1);
  assert.deepEqual(plane.ackBodies[1], [
    { operation_id: `${actor}:op-1`, server_sequence: 1, epoch_id: 2, duplicate: true },
  ]);

  plane.emitCanonical(source, [`${actor}:op-1`]);
  const acks = await acksPromise;
  assert.deepEqual(
    [acks[0].operationId, acks[0].serverSequence, acks[0].epochId, acks[0].duplicate],
    [`${actor}:op-1`, 1, 2, true],
  );
  assert.deepEqual(applied.map((operation) => operation.operationId), [`${actor}:op-1`]);
  assert.equal(client.subscriptionState("ws1").appliedOperationCount, 1);
});

test("resubmitting an already-applied operation_id draws a duplicate ack and produces no second effect", async () => {
  const actor = "actor-resub";
  const plane = new FakeRealtimePlane({ epochId: 4 });
  const { client, clock } = makePlaneClient(plane, { actorId: actor });
  const { source, applied } = subscribe(client);

  const envelope = treeOperation(actor, "add", apiTree("tree-1"), 0);
  const first = submitEnvelope(client, "ws1", envelope);
  await clock.settle();
  plane.emitCanonical(source, [envelope.operation_id]);
  await first;

  // The same operation_id submitted again (a re-gesture replaying a minted
  // id): the ledger answers duplicate, and the canonical re-carriage applies
  // nothing — exactly-once apply is keyed on operation_id.
  const second = submitEnvelope(client, "ws1", envelope);
  second.catch(() => {}); // fence guard: never settles spuriously
  await clock.settle();
  assert.equal(plane.posts.length, 2);
  assert.equal(plane.ackBodies[1][0].duplicate, true);
  assert.equal(plane.ackBodies[1][0].server_sequence, 1, "the original sequence is restated");

  plane.emitCanonical(source, [envelope.operation_id]);
  await clock.settle();
  assert.deepEqual(
    applied.map((operation) => operation.operationId),
    [envelope.operation_id],
    "the operation applies exactly once regardless of re-carriage",
  );
  assert.equal(plane.durable.size, 1);
  assert.equal(client.subscriptionState("ws1").appliedOperationCount, 1);
  // The second submit stays fenced at the canonical fence (the operation's
  // first carriage already resolved): one pending ack, no double resolution.
  assert.equal(client.subscriptionState("ws1").pendingAckCount, 1);
});

// ---------------------------------------------------------------------------
// Admission rejections: typed, surfaced, NEVER auto-retried
// ---------------------------------------------------------------------------

test("a 429 fast_lane_overloaded batch-size rejection surfaces typed with retry advice and never auto-retries", async () => {
  const actor = "actor-burst";
  const { client, clock, fetchImpl } = makeClient({
    actorId: actor,
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      return jsonResponse(
        {
          error: {
            code: "fast_lane_overloaded",
            message:
              "a 129-operation batch exceeds the admitted envelope of 128 operations per epoch; " +
              "reduce the brush size or submit rate",
            max_operations_per_epoch: 128,
            retry_after_ms: 250,
            advice: ["reduce brush size or stroke rate", "coarsen the preview window"],
          },
        },
        429,
      );
    },
  });
  subscribe(client);

  const envelope = meteorologySetOperation(actor, "Ta", 12, 31.5, 6);
  const outcome = submitEnvelope(client, "ws1", envelope).then(
    (value) => ({ value }),
    (error) => ({ error }),
  );
  await new Promise((resolve) => setImmediate(resolve));
  const { error } = await outcome;

  // The client's kind vocabulary maps every workspace-side 429 (batch size
  // and burst alike) to "burst"; the server-wide 503 is "server_saturated".
  assert.ok(error instanceof RealtimeAdmissionError, "the rejection is typed");
  assert.ok(error instanceof RealtimeClientError);
  assert.equal(error.kind, "burst");
  assert.equal(error.retryAfterMs, 250);
  assert.deepEqual(error.advice, ["reduce brush size or stroke rate", "coarsen the preview window"]);
  assert.equal(error.status, 429);
  assert.equal(error.code, "fast_lane_overloaded");

  // An admission rejection is the caller's to handle: even with the whole
  // submit backoff budget elapsed, no second POST may leave the client.
  await clock.advance(7000);
  assert.equal(
    fetchImpl.requests.filter((request) => request.method === "POST").length,
    1,
    "admission errors are surfaced, never auto-retried",
  );
});

test("an IP-gate 429 with a header-only Retry-After hint carries it into the typed rejection", async () => {
  const actor = "actor-ip-gate";
  // The transport-level IP rate limiter (server middleware) answers 429 with
  // the wait in the Retry-After HEADER only — its JSON envelope has no retry
  // field — so the client must mine the header or the studio's held-row
  // countdown falls back to a hint shorter than the server's own window.
  const { client } = makeClient({
    actorId: actor,
    handler: async (request) => {
      if (request.method !== "POST") return jsonResponse({}, 404);
      return {
        ok: false,
        status: 429,
        headers: { get: (name) => (name === "retry-after" ? "4" : null) },
        json: async () => ({
          error: {
            code: "rate_limited",
            message: "more than 600 requests per minute from this client; retry after 4s",
          },
        }),
      };
    },
  });
  subscribe(client);

  const envelope = parameterSetOperation(actor, "transVeg", 0.04, 6);
  const outcome = submitEnvelope(client, "ws1", envelope).then(
    (value) => ({ value }),
    (error) => ({ error }),
  );
  await new Promise((resolve) => setImmediate(resolve));
  const { error } = await outcome;

  assert.ok(error instanceof RealtimeAdmissionError, "the rejection is typed");
  assert.equal(error.kind, "burst");
  assert.equal(error.status, 429);
  assert.equal(error.code, "rate_limited");
  assert.equal(error.details.retry_after_ms, 4000, "the header hint lands in details");
  assert.equal(error.retryAfterMs, 4000, "the admission rejection carries the server's hint");
});

test("a 503 fast_lane_unavailable capacity rejection surfaces typed server_saturated and never auto-retries", async () => {
  const actor = "actor-sat";
  const { client, clock, fetchImpl } = makeClient({
    actorId: actor,
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

  const envelope = parameterSetOperation(actor, "transVeg", 0.04, 6);
  const outcome = submitEnvelope(client, "ws1", envelope).then(
    (value) => ({ value }),
    (error) => ({ error }),
  );
  await new Promise((resolve) => setImmediate(resolve));
  const { error } = await outcome;

  assert.ok(error instanceof RealtimeAdmissionError);
  assert.equal(error.kind, "server_saturated");
  assert.equal(error.retryAfterMs, 2000);
  assert.deepEqual(error.advice, []);
  assert.equal(error.status, 503);
  assert.equal(error.code, "fast_lane_unavailable");

  await clock.advance(7000);
  assert.equal(
    fetchImpl.requests.filter((request) => request.method === "POST").length,
    1,
    "capacity shedding is surfaced, never auto-retried",
  );
});

// ---------------------------------------------------------------------------
// Transient failures: the idempotent 1s/2s/4s submit ladder
// ---------------------------------------------------------------------------

test("transient network failures re-POST the byte-identical body on the 1s/2s/4s ladder; the last retry lands", async () => {
  const actor = "actor-ladder";
  const plane = new FakeRealtimePlane({ epochId: 8 });
  let attempt = 0;
  const { client, clock, fetchImpl } = makePlaneClient(plane, {
    actorId: actor,
    // Three attempts die on the network BEFORE reaching the server (nothing
    // durable); the fourth reaches the plane and is accepted. Count POST
    // attempts, not raw request indices — the subscribe-time catch-up GET
    // (late-join scene rebuild) also rides this fetch and must not consume
    // the kill window.
    handler: (request) => {
      const postAttempts = fetchImpl.requests.filter((r) => r.method === "POST").length;
      if (request.method === "POST" && postAttempts <= 3) return null;
      return plane.fetchImpl(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
    },
  });
  const { source } = subscribe(client);

  const envelope = viewSelectOperation(actor, "utci", 4);
  const acksPromise = submitEnvelope(client, "ws1", envelope);

  const postCount = () => fetchImpl.requests.filter((request) => request.method === "POST").length;
  await clock.settle();
  assert.equal(postCount(), 1, "the initial attempt");
  await clock.advance(999);
  assert.equal(postCount(), 1, "no retry before the 1s backoff");
  await clock.advance(1);
  assert.equal(postCount(), 2, "the 1s backoff fires");
  await clock.advance(1999);
  assert.equal(postCount(), 2, "no retry before the 2s backoff");
  await clock.advance(1);
  assert.equal(postCount(), 3, "the 2s backoff fires");
  await clock.advance(3999);
  assert.equal(postCount(), 3, "no retry before the 4s backoff");
  await clock.advance(1);
  assert.equal(postCount(), 4, "the 4s backoff fires and the batch lands");

  const posts = fetchImpl.requests.filter((request) => request.method === "POST");
  for (const post of posts) {
    assert.equal(post.body, posts[0].body, "every retry is byte-identical (operation_id included)");
    assert.equal(post.headers["Idempotency-Key"], posts[0].headers["Idempotency-Key"]);
  }
  assert.equal(plane.durable.size, 1, "nothing was durable until the server actually saw the batch");

  plane.emitCanonical(source, [envelope.operation_id]);
  const acks = await acksPromise;
  assert.deepEqual(
    [acks[0].operationId, acks[0].serverSequence, acks[0].duplicate],
    [envelope.operation_id, 1, false],
  );
});

// ---------------------------------------------------------------------------
// Typed 409: operation id reuse with a mutated payload
// ---------------------------------------------------------------------------

test("a 409 operation_id_reused (mutated payload under a minted id) rejects typed with no retry", async () => {
  const actor = "actor-409";
  const plane = new FakeRealtimePlane({ epochId: 9 });
  const { client, clock, fetchImpl } = makePlaneClient(plane, { actorId: actor });
  const { source } = subscribe(client);

  const envelope = treeOperation(actor, "add", apiTree("tree-1"), 2);
  const first = submitEnvelope(client, "ws1", envelope);
  await clock.settle();
  plane.emitCanonical(source, [envelope.operation_id]);
  await first;

  // The same minted id re-submitted with a DIFFERENT payload: the store's
  // fingerprint conflict — a hard error, not an idempotent replay.
  const mutated = submitEnvelope(client, "ws1", {
    ...envelope,
    payload: { height_m: 99 },
  });
  const { error } = await mutated.then(
    (value) => ({ value }),
    (rejection) => ({ error: rejection }),
  );

  assert.ok(error instanceof RealtimeClientError, "the conflict is a typed client error");
  assert.ok(!(error instanceof RealtimeAdmissionError), "a 409 is not an admission rejection");
  assert.equal(error.code, "operation_id_reused");
  assert.equal(error.status, 409);
  assert.equal(error.details.operation_id, envelope.operation_id);

  // A conflict is deterministic: retrying it cannot succeed, so none happens.
  await clock.advance(7000);
  assert.equal(
    fetchImpl.requests.filter((request) => request.method === "POST").length,
    2,
    "exactly the original accept and the conflicting replay — no retries",
  );
  assert.equal(plane.durable.size, 1, "the ledger still holds exactly the original operation");
});

// ---------------------------------------------------------------------------
// base_divergent: advisory, never a rejection
// ---------------------------------------------------------------------------

test("a divergent base_revision acks base_divergent:true and still resolves through the canonical fence", async () => {
  const actor = "actor-divergent";
  const stalls = [];
  // The workspace has moved to revision 12 (other actors landed ops); this
  // actor's gesture was composed against revision 5.
  const plane = new FakeRealtimePlane({ workspaceRevision: 12, epochId: 6 });
  const { client, clock, fetchImpl } = makePlaneClient(plane, {
    actorId: actor,
    onStall: (detail) => stalls.push(detail),
  });
  const { source } = subscribe(client);

  const envelope = dateOperation(actor, "2026-07-21T15:00:00+02:00", 5);
  const acksPromise = submitEnvelope(client, "ws1", envelope);
  await clock.settle();

  // Server-side evidence: the acceptance carries the advisory flag.
  assert.equal(plane.posts.length, 1, "a divergent base is ONE POST — never retried");
  assert.equal(plane.ackBodies[0][0].base_divergent, true);

  // Advisory means accepted: the submit resolves at the canonical fence like
  // any other op, with the server's authoritative sequence and epoch. (The
  // advisory flag itself currently rides the HTTP ack only — the resolved
  // ack object carries {operationId, serverSequence, epochId, duplicate}.)
  plane.emitCanonical(source, [envelope.operation_id]);
  const acks = await acksPromise;
  assert.equal(acks.length, 1);
  assert.equal(acks[0].operationId, envelope.operation_id);
  assert.equal(acks[0].serverSequence, 1);
  assert.equal(acks[0].epochId, 6);
  assert.equal(acks[0].duplicate, false);
  assert.equal(stalls.length, 0, "a divergent base never stalls or rejects");
  assert.equal(
    fetchImpl.requests.filter((request) => request.method === "POST").length,
    1,
  );
});
