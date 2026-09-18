// SPDX-License-Identifier: GPL-3.0-only
//
// ExactSession gate tests (UI-001..UI-004) against an in-memory contract
// server: canned scenario/edit/job/result responses driven by a manual clock.

import test from "node:test";
import assert from "node:assert/strict";

import { ApiClient } from "../api_client.mjs";
import { ExactSession, defaultConflictPolicy } from "../exact_session.mjs";
import snapshot from "../assets/capabilities_snapshot.json" with { type: "json" };
import {
  COLS,
  FakeContractServer,
  ROWS,
  SCENARIO,
  TIME_STEPS,
  addEdit,
  apiTree,
  makeFakeClock,
} from "./contract_harness.mjs";

function makeSession(server, callbacks = {}, clock = makeFakeClock()) {
  const events = [];
  const session = new ExactSession({
    client: new ApiClient({ fetch: server.fetchImpl, sessionId: "sess-test" }),
    callbacks: {
      onStatus: (status) => events.push({ type: "status", ...status }),
      onBaseline: (result) => events.push({ type: "baseline", version: result.manifest.scene_version }),
      onPatchApplied: (result) =>
        events.push({
          type: "applied",
          version: result.manifest.scene_version,
          window: result.window,
          metrics: result.manifest.metrics,
          impactPlan: result.impactPlan ?? null,
        }),
      onScope: (scope) => events.push({ type: "scope", ...scope }),
      onValidation: (error) => events.push({ type: "validation", ...error }),
      onAuthoritative: (state) => events.push({ type: "authoritative", ...state }),
      ...callbacks,
    },
    pollIntervalMs: 5,
    timers: clock.timers,
    conflictPolicy: defaultConflictPolicy,
  });
  return { session, events, clock };
}

async function connect(server, clock) {
  const harness = makeSession(server, {}, clock ?? makeFakeClock());
  const baseline = await harness.session.connect({ siteId: "campus-1km-v1" });
  return { ...harness, baseline };
}

// ---------------------------------------------------------------------------
// UI-001 + happy path
// ---------------------------------------------------------------------------

test("one committed interaction sends exactly one POST /edits through the full poll cycle", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  assert.equal(server.editPosts().length, 0);
  await session.commitEdits({ edits: addEdit("tree-a"), requested: { timeIndices: [12], variables: ["utci"] }, label: "add" });
  await clock.settle();

  // UI-001: exactly one edit POST for the committed interaction.
  assert.equal(server.editPosts().length, 1);
  const post = server.editPosts()[0];
  assert.equal(JSON.parse(post.body).base_scene_version, 0);
  assert.ok(post.headers["Idempotency-Key"]);
  assert.equal(post.headers["If-Match"], '"scene-version-0"');

  assert.equal(session.sceneVersion, 1);
  assert.equal(session.awaitingVersion, 1);
  assert.ok(events.some((event) => event.type === "status" && event.phase === "queued"));

  await clock.tick(1); // queued
  await clock.tick(1); // running
  assert.ok(events.some((event) => event.type === "status" && event.phase === "running"));
  await clock.tick(1); // complete -> manifest -> payload -> apply

  const applied = events.filter((event) => event.type === "applied");
  assert.equal(applied.length, 1);
  assert.equal(applied[0].version, 1);
  assert.deepEqual(applied[0].window, { rowStart: 2, rowStop: 6, colStart: 3, colStop: 7 });
  assert.equal(applied[0].metrics.mean_utci_delta_c, -0.84);
  assert.equal(session.exactResultVersion, 1);
  assert.equal(session.awaitingVersion, null);
  assert.equal(session.isExactCurrent(), true);
  assert.equal(server.editPosts().length, 1, "polling must never issue extra edits");
});

test("consecutive commits chain base versions without self-inflicted conflicts", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "add a" });
  await session.commitEdits({ edits: addEdit("tree-b"), label: "add b" });
  await clock.settle();

  const posts = server.editPosts();
  assert.equal(posts.length, 2);
  assert.equal(JSON.parse(posts[0].body).base_scene_version, 0);
  assert.equal(JSON.parse(posts[1].body).base_scene_version, 1);
  assert.ok(posts[0].headers["Idempotency-Key"] !== posts[1].headers["Idempotency-Key"]);
  assert.equal(session.sceneVersion, 2);
  assert.equal(session.awaitingVersion, 2);
});

// ---------------------------------------------------------------------------
// UI-003 stale rejection
// ---------------------------------------------------------------------------

test("results for an older scene version are dropped silently (stale rejection)", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  // The server publishes a stale manifest (v1) while the client awaits v2.
  server.onManifest = (version, record) =>
    version === 2 ? { ...record.manifest, scene_version: 1 } : null;

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await session.commitEdits({ edits: addEdit("tree-b"), label: "b" });
  await clock.settle();
  await clock.tick(6); // drain both pollers

  const applied = events.filter((event) => event.type === "applied");
  assert.equal(applied.length, 0, "stale manifests must never be applied");
  assert.ok(
    !events.some((event) => event.type === "status" && event.phase === "failed"),
    "stale results are dropped, not surfaced as failures",
  );
  assert.equal(session.sceneVersion, 2, "design state untouched");
});

test("out-of-order poll completions are ignored once a newer job owns the result", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await session.commitEdits({ edits: addEdit("tree-b"), label: "b" });
  await clock.settle();

  // The v1 poller's timer still fires after the v2 commit took over; its
  // completion must be dropped by the poll-generation guard.
  await clock.tick(8);

  const applied = events.filter((event) => event.type === "applied");
  assert.ok(applied.every((event) => event.version === 2), "only the newest job's result applies");
  assert.equal(applied.length, 1);
});

test("superseded jobs are dropped silently", async () => {
  const server = new FakeContractServer({ jobScript: ["superseded"] });
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await clock.tick(2);

  assert.ok(events.some((event) => event.type === "status" && event.phase === "superseded"));
  assert.equal(events.filter((event) => event.type === "applied").length, 0);
  assert.equal(session.lastError, null);
  assert.ok(
    !events.some((event) => event.type === "status" && event.phase === "failed"),
  );
});

// ---------------------------------------------------------------------------
// UI-004 failure preserves design state + retry
// ---------------------------------------------------------------------------

test("failed jobs preserve state and retry re-requests the analysis", async () => {
  const server = new FakeContractServer({ jobScript: ["failed"] });
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);
  session.provideRecompute = () => ({
    tree: apiTree("tree-a"),
    requested: { timeIndices: [12], variables: ["utci"] },
  });

  const eventIndexAtCommit = events.length;
  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await clock.tick(2);

  const failed = events.find(
    (event, index) => index >= eventIndexAtCommit && event.type === "status" && event.phase === "failed",
  );
  assert.ok(failed, "failure must be reported");
  assert.equal(failed.error.code, "job_failed");
  assert.equal(failed.error.message, "worker exploded");
  // The edit itself was accepted; versions survive the failure.
  assert.equal(session.sceneVersion, 1);
  assert.equal(session.lastAttempt.accepted, true);
  assert.ok(
    !events.slice(eventIndexAtCommit).some((event) => event.type === "authoritative"),
    "a failed job must not rewrite design state",
  );

  // Retry affordance: the failed job cannot be rerun, so the session asks the
  // app for a recompute request (no-op update edit).
  server.jobScript = ["complete"];
  const postsBefore = server.editPosts().length;
  await session.retryLast();
  await clock.settle();
  await clock.tick(3);

  assert.equal(server.editPosts().length, postsBefore + 1);
  const retryPost = JSON.parse(server.editPosts().at(-1).body);
  assert.equal(retryPost.edits[0].operation, "update");
  assert.equal(retryPost.edits[0].tree.tree_id, "tree-a");
  const applied = events.filter((event) => event.type === "applied");
  assert.equal(applied.length, 1);
});

test("network failure during POST keeps the attempt retryable and idempotent", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);
  server.loseResponseEdits = 1; // server applies the edit, the response is lost

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();

  assert.ok(events.some((event) => event.type === "status" && event.phase === "failed"));
  assert.equal(session.lastAttempt.accepted, false);
  assert.equal(server.appliedEdits, 1);

  // Retry re-sends the SAME idempotency key; the server replays the stored
  // response instead of applying the edit a second time.
  await session.retryLast();
  await clock.settle();

  assert.equal(server.appliedEdits, 1, "idempotent replay must not re-apply");
  assert.equal(server.editPosts().length, 2);
  assert.equal(session.sceneVersion, 1);
  await clock.tick(4);
  assert.equal(events.filter((event) => event.type === "applied").length, 1);
});

test("a rate-limited POST bounces once off the server's hint and never double-applies", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  // First POST draws the transport gate's 429; the hint rides the
  // Retry-After header exactly like the server middleware sends it (the
  // JSON envelope has no retry field).
  let gated = true;
  server.onEdit = () => {
    if (!gated) return null;
    gated = false;
    return new Response(
      JSON.stringify({
        error: { code: "rate_limited", message: "more than 600 requests per minute" },
      }),
      { status: 429, headers: { "content-type": "application/json", "Retry-After": "1" } },
    );
  };

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  assert.equal(server.editPosts().length, 1);
  assert.equal(session.sceneVersion, 0, "the 429'd POST applied nothing");
  assert.ok(
    !events.some((event) => event.type === "status" && event.phase === "failed"),
    "a single 429 must not banner",
  );

  await clock.tick(1); // fire the Retry-After wait -> the retry POSTs again
  await clock.settle();

  const posts = server.editPosts();
  assert.equal(posts.length, 2, "exactly one automatic retry");
  assert.equal(
    posts[0].headers["Idempotency-Key"],
    posts[1].headers["Idempotency-Key"],
    "same key: the retry is a replay and can never double-apply",
  );
  assert.equal(server.appliedEdits, 1);
  assert.equal(session.sceneVersion, 1);
  await clock.tick(4); // poll cycle to applied
  assert.equal(events.filter((event) => event.type === "applied").length, 1);
});

test("a second rate_limited surfaces the failure instead of retry-looping", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);
  // No Retry-After at all: the retry uses the 2 s fallback and its own 429
  // is terminal for the attempt.
  server.onEdit = () =>
    new Response(
      JSON.stringify({ error: { code: "rate_limited", message: "still gated" } }),
      { status: 429, headers: { "content-type": "application/json" } },
    );

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await clock.tick(1); // fire the fallback wait -> the retry 429s too
  await clock.settle();

  assert.equal(server.editPosts().length, 2, "exactly one bounce, then surfaced");
  assert.ok(events.some((event) => event.type === "status" && event.phase === "failed"));
  assert.equal(session.lastAttempt.accepted, false);
  assert.equal(session.sceneVersion, 0);
  assert.equal(clock.pendingCount(), 0, "no further retries are scheduled");
});

test("invalid tree geometry surfaces the field error without losing state", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);
  server.onEdit = () =>
    new Response(
      JSON.stringify({
        error: {
          code: "invalid_tree_geometry",
          message: "canopy_diameter_m must be between 1 and 30",
          field: "edits[0].tree.canopy_diameter_m",
        },
      }),
      { status: 422, headers: { "content-type": "application/json" } },
    );

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();

  const validation = events.find((event) => event.type === "validation");
  assert.ok(validation);
  assert.equal(validation.field, "edits[0].tree.canopy_diameter_m");
  assert.equal(session.sceneVersion, 0);
  assert.equal(server.jobs.size, 0);
  assert.ok(!events.some((event) => event.type === "status" && event.phase === "failed"));

  // The next valid commit still works from the same base version.
  server.onEdit = null;
  await session.commitEdits({ edits: addEdit("tree-b"), label: "b" });
  await clock.settle();
  assert.equal(session.sceneVersion, 1);
});

// ---------------------------------------------------------------------------
// 409 scene_version_conflict recovery
// ---------------------------------------------------------------------------

test("409 conflict reloads authoritative state and reapplies an add once", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  let conflictOnce = true;
  server.onEdit = (srv) => {
    if (!conflictOnce) return null;
    conflictOnce = false;
    // Another client moved the scenario ahead to v3 without our tree.
    srv.sceneVersion = 3;
    srv.trees = [apiTree("tree-z")];
    return new Response(
      JSON.stringify({
        error: {
          code: "scene_version_conflict",
          message: "The scenario changed after the client's base version.",
          current_scene_version: 3,
          scenario_url: `/api/v1/scenarios/${SCENARIO}`,
        },
      }),
      { status: 409, headers: { "content-type": "application/json" } },
    );
  };

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await clock.settle();

  assert.ok(events.some((event) => event.type === "status" && event.phase === "conflict"));
  const adopt = events.find(
    (event) => event.type === "authoritative" && event.reason === "adopt" && event.sceneVersion === 3,
  );
  assert.ok(adopt, "authoritative version v3 adopted after reload");

  const posts = server.editPosts();
  assert.equal(posts.length, 2, "conflict recovery re-submits exactly once");
  const retry = JSON.parse(posts[1].body);
  assert.equal(retry.base_scene_version, 3);
  assert.notEqual(
    posts[0].headers["Idempotency-Key"],
    posts[1].headers["Idempotency-Key"],
    "the recovery POST must use a fresh idempotency key",
  );

  await clock.tick(4);
  const applied = events.filter((event) => event.type === "applied");
  assert.equal(applied.length, 1);
  assert.equal(applied[0].version, 4);
});

test("409 conflict discards an edit the server state can no longer accept", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  server.onEdit = (srv) => {
    srv.sceneVersion = 2;
    srv.trees = []; // the tree we wanted to delete is already gone
    return new Response(
      JSON.stringify({
        error: {
          code: "scene_version_conflict",
          message: "stale",
          current_scene_version: 2,
          scenario_url: `/api/v1/scenarios/${SCENARIO}`,
        },
      }),
      { status: 409, headers: { "content-type": "application/json" } },
    );
  };

  await session.commitEdits({
    edits: [{ operation: "delete", tree_id: "tree-gone" }],
    label: "delete",
  });
  await clock.settle();
  await clock.settle();

  assert.ok(events.some((event) => event.type === "status" && event.phase === "conflict_discarded"));
  assert.equal(server.editPosts().length, 1, "discarded conflicts are not re-sent");
  assert.equal(session.sceneVersion, 2);
});

// ---------------------------------------------------------------------------
// Scope disclosure + reset + recompute
// ---------------------------------------------------------------------------

test("connect surfaces the model scope (UI-005 data path)", async () => {
  const server = new FakeContractServer();
  const baseline = await server.seedBaseline();
  const { session, events } = await connect(server);

  const scope = events.find((event) => event.type === "scope");
  assert.ok(scope);
  assert.equal(scope.modelVersion, "solweig-gpu-2.0.0+incremental.1");
  assert.equal(scope.siteCacheVersion, "campus-1km-v1:cache-1");
  assert.deepEqual(scope.limitations, ["Tree-induced local wind-field changes are not recomputed."]);

  const baselineEvent = events.find((event) => event.type === "baseline");
  assert.equal(baselineEvent.version, 0);
  assert.equal(baseline.manifest.time_indices.length, TIME_STEPS);
  assert.equal(session.grid.rows, ROWS);
  assert.equal(session.grid.cols, COLS);
});

test("requestRecompute sends a no-op update edit with the requested hour", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, clock } = await connect(server);
  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();

  await session.requestRecompute({
    tree: apiTree("tree-a"),
    requested: { timeIndices: [15], variables: ["utci"] },
    label: "hour change",
  });
  await clock.settle();

  const post = JSON.parse(server.editPosts().at(-1).body);
  assert.equal(post.edits.length, 1);
  assert.equal(post.edits[0].operation, "update");
  assert.equal(post.edits[0].tree.tree_id, "tree-a");
  assert.deepEqual(post.requested_result, { time_indices: [15], variables: ["utci"] });
});

test("recompute jobs carry no source nodes so the impact panel cannot go stale", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  // A real edit announces its source nodes (UEDIT-009 classification input)…
  await session.commitEdits({
    edits: addEdit("tree-a"),
    label: "a",
    sourceNodes: ["vegetation_dsm"],
  });
  await clock.settle();
  const editQueued = events.find(
    (event) => event.type === "status" && event.phase === "queued" && event.sourceNodes,
  );
  assert.deepEqual(editQueued.sourceNodes, ["vegetation_dsm"]);

  // …while a time-only recompute (rerun / retry) writes no source node: its
  // statuses carry sourceNodes null, which the UI must render as a fresh
  // recompute-only panel rather than the previous edit's classification.
  await session.requestRecompute({
    tree: apiTree("tree-a"),
    requested: { timeIndices: [15], variables: ["utci"] },
    label: "hour change",
  });
  await clock.settle();
  const recomputeQueued = events
    .filter((event) => event.type === "status" && event.phase === "queued")
    .at(-1);
  assert.equal(recomputeQueued.sourceNodes, null);
});

test("reset is a versioned mutation that restores the exact baseline", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);
  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await clock.tick(3);

  const result = await session.reset();
  assert.equal(result.manifest.scene_version, 2);
  assert.equal(session.sceneVersion, 2);
  assert.equal(session.exactResultVersion, 2);
  assert.equal(session.awaitingVersion, null);

  const resetPost = server.calls.find((call) => call.path.endsWith("/reset"));
  assert.ok(resetPost);
  assert.equal(resetPost.headers["If-Match"], '"scene-version-1"');
  assert.ok(resetPost.headers["Idempotency-Key"]);
  assert.ok(events.some((event) => event.type === "baseline" && event.version === 2));
  assert.ok(events.some((event) => event.type === "authoritative" && event.reason === "reset"));
});

// ---------------------------------------------------------------------------
// M-3: edits committed while connecting are held and replayed
// ---------------------------------------------------------------------------

test("edits committed while connecting replay after the baseline loads", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  let releaseScenario;
  server.scenarioGate = new Promise((resolve) => {
    releaseScenario = resolve;
  });

  const clock = makeFakeClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle();

  // Not ready yet: the edit must be held, never posted against the
  // half-connected session.
  const committed = harness.session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  const recompute = harness.session.requestRecompute({
    tree: apiTree("tree-b"),
    requested: { timeIndices: [12] },
  });
  await clock.settle();
  assert.equal(server.editPosts().length, 0, "no edits may post before ready");
  assert.equal(harness.session.sceneVersion, 0);

  releaseScenario();
  await connecting;
  await committed;
  await recompute;
  await clock.settle();

  const posts = server.editPosts();
  assert.equal(posts.length, 2, "backlog replays in order after connect");
  assert.equal(JSON.parse(posts[0].body).edits[0].tree.tree_id, "tree-a");
  assert.equal(JSON.parse(posts[1].body).edits[0].tree.tree_id, "tree-b");
  assert.equal(JSON.parse(posts[0].body).base_scene_version, 0);
  assert.equal(JSON.parse(posts[1].body).base_scene_version, 1);
  // Held attempts get fresh idempotency keys minted at replay time.
  assert.notEqual(posts[0].headers["Idempotency-Key"], posts[1].headers["Idempotency-Key"]);
});

test("a permanent connect failure (4xx) releases held edits without retrying", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  let releaseScenario;
  server.scenarioGate = new Promise((resolve) => {
    releaseScenario = resolve;
  });
  // A 4xx from the scenario endpoint is permanent (e.g. unknown site): unlike
  // a transport drop it must surface immediately, without reconnect backoff.
  const upstream = server.fetchImpl;
  server.fetchImpl = async (url, init = {}) => {
    const path = new URL(url, "http://server.test").pathname;
    if (path === `/api/v1/scenarios/${SCENARIO}` && (init.method ?? "GET") === "GET") {
      await server.scenarioGate;
      return new Response(
        JSON.stringify({ error: { code: "site_not_found", message: "unknown site" } }),
        { status: 404, headers: { "content-type": "application/json" } },
      );
    }
    return upstream(url, init);
  };

  const clock = makeFakeClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle();

  const committed = harness.session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  releaseScenario();
  await assert.rejects(connecting, /unknown site/);
  assert.equal(await committed, null, "held edit resolves null, never posted");
  assert.equal(server.editPosts().length, 0);
  assert.equal(clock.pendingCount(), 0, "a permanent connect failure must not schedule backoff");
});

// ---------------------------------------------------------------------------
// R-1: transient connect failures reconnect with bounded backoff; the backlog
// replays on the attempt that lands
// ---------------------------------------------------------------------------

/** Fake clock that records every scheduled delay, for backoff assertions. */
function makeRecordingClock(base = makeFakeClock()) {
  const delays = [];
  return {
    delays,
    timers: {
      setTimeout(fn, ms) {
        delays.push(ms);
        return base.timers.setTimeout(fn, ms);
      },
      clearTimeout: (id) => base.timers.clearTimeout(id),
    },
    tick: (count = 1) => base.tick(count),
    settle: () => base.settle(),
    pendingCount: () => base.pendingCount(),
  };
}

function armScenarioGate(server) {
  let settle;
  server.scenarioGate = new Promise((resolve, reject) => {
    settle = { resolve, reject };
  });
  return settle;
}

test("transient connect failures reconnect with backoff and the backlog drains in order", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();

  const clock = makeRecordingClock();
  const harness = makeSession(server, {}, clock);
  let gate = armScenarioGate(server);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle(); // attempt 1 hangs fetching the scenario

  // Edits committed while connecting queue in the backlog (M-3).
  const first = harness.session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  const second = harness.session.commitEdits({ edits: addEdit("tree-b"), label: "b" });
  await clock.settle();
  assert.equal(server.editPosts().length, 0);

  // Attempt 1 fails with a transport error; connect must schedule a retry
  // instead of rejecting and dropping the backlog.
  gate.reject(new Error("ECONNREFUSED"));
  await clock.settle();
  assert.ok(
    harness.events.some((event) => event.type === "status" && event.phase === "reconnecting"),
    "a transient connect failure announces a reconnect",
  );
  assert.equal(server.editPosts().length, 0, "held edits wait for the retry");

  // Attempt 2 fails as well.
  gate = armScenarioGate(server);
  await clock.tick(1); // fire the 1 s backoff
  await clock.settle();
  gate.reject(new Error("ECONNREFUSED"));
  await clock.settle();

  // Attempt 3 lands: the backlog must drain in order against the baseline.
  gate = armScenarioGate(server);
  await clock.tick(1); // fire the 2 s backoff
  gate.resolve();
  const baseline = await connecting;
  assert.equal(baseline.manifest.scene_version, 0);
  await first;
  await second;
  await clock.settle();

  const posts = server.editPosts();
  assert.equal(posts.length, 2, "the backlog replays in order after reconnect");
  assert.equal(JSON.parse(posts[0].body).edits[0].tree.tree_id, "tree-a");
  assert.equal(JSON.parse(posts[1].body).edits[0].tree.tree_id, "tree-b");
  assert.equal(JSON.parse(posts[0].body).base_scene_version, 0);
  assert.equal(JSON.parse(posts[1].body).base_scene_version, 1);
  // The two backoff waits come first; the trailing zero-delay timers are the
  // replayed edits' job polls.
  assert.deepEqual(
    clock.delays.slice(0, 2),
    [1000, 2000],
    "exponential backoff 1 s then 2 s",
  );
});

test("connect retries are bounded: exhaustion releases held edits and rejects", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();

  const clock = makeRecordingClock();
  const harness = makeSession(server, {}, clock);
  server.scenarioGate = Promise.reject(new Error("server went away"));
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  // Attach the rejection expectation up front: the final attempt rejects
  // inside the fake clock's settle rounds, before a late await could.
  const rejected = assert.rejects(connecting, /server went away/);

  const committed = harness.session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  // Transport-class failures ride the cold-boot wake ladder: exponential
  // start, then a capped cadence, stopping at the ~90 s cumulative budget.
  await clock.tick(30); // fire every scheduled wake backoff
  await rejected;

  assert.equal(await committed, null, "held edit resolves null, never posted");
  assert.equal(server.editPosts().length, 0);
  const scenarioCreates = server.calls.filter(
    (call) => call.method === "POST" && call.path === "/api/v1/scenarios",
  );
  // The scenario is minted ONCE: a retry rejoins it instead of re-POSTing
  // create, so a strained transport never pays for extra scenarios (each
  // with its own baseline wait).
  assert.equal(scenarioCreates.length, 1, "the minted scenario is reused across retries");
  assert.deepEqual(clock.delays.slice(0, 3), [1000, 2000, 4000], "exponential backoff start");
  assert.ok(
    clock.delays.length > 3 && clock.delays.slice(3).every((ms) => ms === 5000),
    "wake retries continue at the capped cadence",
  );
  const totalWaitMs = clock.delays.reduce((sum, ms) => sum + ms, 0);
  assert.ok(
    totalWaitMs >= 90000 && totalWaitMs < 96000,
    `waking stops at the ~90 s budget (waited ${totalWaitMs} ms)`,
  );
  assert.equal(clock.pendingCount(), 0, "no further retries are scheduled");
});

test("a transient failure while awaiting the baseline job backs off instead of failing connect", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  // Cold-site shape: create schedules a baseline job (status_url), and the
  // job's first status GET draws a transient transport failure — one drop
  // must back off and keep booting, not kill the connect (self-429 guard).
  server.jobs.set("job_baseline", {
    job_id: "job_baseline",
    scenario_id: SCENARIO,
    target_scene_version: 0,
    script: ["running", "complete"],
    step: 0,
  });
  const upstream = server.fetchImpl;
  server.fetchImpl = async (url, init = {}) => {
    const response = await upstream(url, init);
    if (
      (init.method ?? "GET") === "POST" &&
      new URL(url, "http://server.test").pathname === "/api/v1/scenarios"
    ) {
      const body = await response.json();
      body.job_id = "job_baseline";
      body.status_url = "/api/v1/jobs/job_baseline";
      return new Response(JSON.stringify(body), {
        status: 201,
        headers: { "content-type": "application/json" },
      });
    }
    return response;
  };
  server.jobGetFailures = 1;

  const clock = makeRecordingClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle(); // mint + capabilities land; the baseline GET fails
  await clock.tick(2); // backoff wait, then the running->complete poll wait

  const baseline = await connecting;
  assert.equal(baseline.manifest.scene_version, 0, "connect lands on the baseline");
  const jobGets = server.calls.filter((call) => call.path === "/api/v1/jobs/job_baseline");
  assert.equal(jobGets.length, 3, "one failed GET, then running, then complete");
  // The first wait is the doubled backoff (pollIntervalMs * 2), not the
  // plain cadence — the ladder fired before the next poll.
  assert.deepEqual(clock.delays.slice(0, 2), [10, 5], "backoff ladder then normal cadence");
  assert.ok(
    !harness.events.some((event) => event.type === "status" && event.phase === "failed"),
  );
});

// ---------------------------------------------------------------------------
// R-2: reset during the connecting window drops the held backlog
// ---------------------------------------------------------------------------

test("reset while connecting drops held edits so nothing replays onto the baseline", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const gate = armScenarioGate(server);

  const clock = makeFakeClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle();

  const committed = harness.session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  assert.equal(server.editPosts().length, 0);

  // Reset during the connecting window: held edits are local-only, so the
  // reset's whole job is dropping them.
  const resetResult = await harness.session.reset({ label: "reset" });
  assert.equal(resetResult, null);
  assert.ok(
    harness.events.some((event) => event.type === "status" && event.phase === "held_edits_discarded"),
    "the dropped backlog is surfaced, not silent",
  );

  gate.resolve();
  await connecting; // connect still completes on the pristine baseline
  assert.equal(await committed, null, "the held edit was dropped by the reset");
  await clock.settle();

  assert.equal(server.editPosts().length, 0, "no stale edit may replay after the reset");
  assert.ok(
    !server.calls.some((call) => call.path.endsWith("/reset")),
    "no server reset is needed before the session is ready",
  );
  assert.ok(harness.events.some((event) => event.type === "status" && event.phase === "ready"));
  assert.equal(harness.session.sceneVersion, 0);
});

test("reset during the baseline download lets connect finish cleanly", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  let releasePayload;
  let gatedOnce = false;
  server.payloadGateFor = (version) => {
    if (version === 0 && !gatedOnce) {
      gatedOnce = true;
      return new Promise((resolve) => {
        releasePayload = resolve;
      });
    }
    return null;
  };

  const clock = makeFakeClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle(); // connect hangs downloading the baseline payload

  const committed = harness.session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await harness.session.reset({ label: "reset" }); // mid-download reset

  releasePayload();
  const baseline = await connecting; // must resolve, not reject
  assert.equal(baseline.manifest.scene_version, 0);
  assert.equal(await committed, null, "the held edit was dropped by the reset");
  await clock.settle();

  assert.equal(server.editPosts().length, 0, "no stale edit may replay after the reset");
  assert.ok(
    !server.calls.some((call) => call.path.endsWith("/reset")),
    "no server reset is needed before the session is ready",
  );
  assert.ok(harness.events.some((event) => event.type === "status" && event.phase === "ready"));
  assert.equal(harness.session.sceneVersion, 0);
  assert.equal(harness.session.awaitingVersion, null);
});

// ---------------------------------------------------------------------------
// M-4: reset invalidates in-flight polls; late results never apply
// ---------------------------------------------------------------------------

test("reset invalidates in-flight polls so nothing applies afterwards", async () => {
  const server = new FakeContractServer({ jobScript: ["complete"] });
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  const jobGetsBeforeReset = server.calls.filter((call) =>
    /\/api\/v1\/jobs\//.test(call.path),
  ).length;

  await session.reset();
  await clock.tick(8); // drain anything still scheduled from the old poll

  assert.equal(
    events.filter((event) => event.type === "applied").length,
    0,
    "a pre-reset job must never announce an applied result",
  );
  assert.ok(
    !events.some((event) => event.type === "status" && event.phase === "applied"),
    "no spurious 'Exact result applied' announcement after reset",
  );
  assert.ok(events.some((event) => event.type === "baseline" && event.version === 2));
  assert.equal(
    server.calls.filter((call) => /\/api\/v1\/jobs\//.test(call.path)).length,
    jobGetsBeforeReset,
    "the invalidated poll stops consuming job requests",
  );
});

test("a payload that resolves after reset is dropped, not applied", async () => {
  const server = new FakeContractServer({ jobScript: ["complete"] });
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  let releasePayload;
  server.payloadGateFor = (version) =>
    version === 1
      ? new Promise((resolve) => {
          releasePayload = resolve;
        })
      : null;

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await clock.tick(1); // job completes; result fetch blocks on the payload gate

  await session.reset(); // adopts the reset baseline while v1's payload hangs
  await clock.settle();

  releasePayload();
  await clock.settle();
  await clock.settle();

  assert.equal(
    events.filter((event) => event.type === "applied").length,
    0,
    "the pre-reset result must be dropped even though its payload arrived",
  );
  assert.equal(session.sceneVersion, 2);
  assert.equal(session.awaitingVersion, null);
});

// ---------------------------------------------------------------------------
// L-4: transient GET /jobs failures back off instead of killing the poll
// ---------------------------------------------------------------------------

test("transient job-poll failures retry with backoff and still apply", async () => {
  const server = new FakeContractServer({ jobScript: ["complete"] });
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);
  server.jobGetFailures = 2; // two ECONNRESETs, then a clean complete

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await clock.tick(8); // initial attempt + two backoff retries + completion

  assert.ok(
    !events.some((event) => event.type === "status" && event.phase === "failed"),
    "transient poll errors must not surface as failures",
  );
  const applied = events.filter((event) => event.type === "applied");
  assert.equal(applied.length, 1);
  assert.equal(applied[0].version, 1);
  assert.equal(session.lastError, null);
});

test("persistent job-poll failures eventually fail with the retry affordance", async () => {
  const server = new FakeContractServer({ jobScript: ["complete"] });
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);
  server.jobGetFailures = 99; // every GET /jobs drops

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await clock.tick(12); // exhaust the backoff budget

  assert.ok(events.some((event) => event.type === "status" && event.phase === "failed"));
  assert.equal(session.sceneVersion, 1, "the accepted edit is preserved");
  assert.ok(session.lastAttempt.accepted);
});

// ---------------------------------------------------------------------------
// Universal edit transport (u-d4): family edits ride /edits/universal with
// the same job queue, idempotency, conflict recovery, and poll machinery.
// ---------------------------------------------------------------------------

const MET_ITEM = {
  adapter: "meteorological_forcing",
  operation: "update_time_row",
  values: {
    air_temperature: 31.5,
    humidity: 14,
    radiation: 800,
    wind_speed: 2.4,
    pressure: 1013,
    uhii: 0,
  },
  target: null,
  time_index: 12,
  old_values: null,
};

const EXECUTED_PLAN = {
  schema_version: 1,
  status: "executed",
  mode: "local",
  scene_revision: 1,
  job_id: "job_1",
  nodes: [
    { node: "meteorology", stage: "changed", why: "edited family input" },
    {
      node: "radiation",
      stage: "recomputed",
      why: "downstream of meteorology",
      spatial_scope: "window",
    },
    { node: "tmrt", stage: "reused", why: "patch cache hit" },
  ],
  routing: { mode: "windowed", write_windows: 1, transport_window: "local" },
  estimates: { work_units: 12 },
  realized: { patch_count: 1 },
};

test("a universal family edit posts to /edits/universal and relays the executed plan", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.impactPlanFor = (sceneVersion) => (sceneVersion === 1 ? EXECUTED_PLAN : null);
  const { session, events, clock } = await connect(server);

  await session.commitEdits({
    edits: [MET_ITEM],
    transport: "universal-edits-v1",
    sourceNodes: ["meteorology"],
    requested: { timeIndices: [12], variables: ["utci"] },
    label: "met hour 12",
  });
  await clock.settle();
  await clock.tick(3); // queued -> running -> complete

  // Exactly one universal POST, and the tree endpoint stayed untouched.
  assert.equal(server.universalEditPosts().length, 1);
  assert.equal(server.editPosts().length, 0);
  const post = server.universalEditPosts()[0];
  const sent = JSON.parse(post.body);
  assert.equal(sent.base_scene_version, 0);
  assert.deepEqual(sent.edits, [MET_ITEM]);
  assert.deepEqual(sent.requested_result, { time_indices: [12], variables: ["utci"] });
  assert.equal(post.headers["If-Match"], '"scene-version-0"');
  assert.ok(post.headers["Idempotency-Key"]);

  const applied = events.filter((event) => event.type === "applied");
  assert.equal(applied.length, 1);
  assert.equal(applied[0].version, 1);
  assert.deepEqual(applied[0].impactPlan, EXECUTED_PLAN);
  const status = events.find((event) => event.type === "status" && event.phase === "applied");
  assert.deepEqual(status.impactPlan, EXECUTED_PLAN);
  assert.equal(session.exactResultVersion, 1);
});

test("a dropped universal response replays idempotently without a second mutation", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, clock } = await connect(server);
  server.loseResponseEdits = 1;

  await session.commitEdits({
    edits: [MET_ITEM],
    transport: "universal-edits-v1",
    label: "met",
  });
  await clock.settle();
  assert.equal(session.lastAttempt.accepted, false, "the drop left the attempt retryable");

  await session.retryLast();
  await clock.settle();
  await clock.tick(3);

  const posts = server.universalEditPosts();
  assert.equal(posts.length, 2, "the retry re-POSTs with the same key");
  assert.equal(
    posts[0].headers["Idempotency-Key"],
    posts[1].headers["Idempotency-Key"],
    "the replay must reuse the idempotency key",
  );
  assert.equal(server.universalEdits.length, 1, "the server applied the batch once");
  assert.equal(session.sceneVersion, 1);
});

test("409 conflict on a universal edit adopts authoritative state and reapplies once", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  let conflictOnce = true;
  server.onUniversalEdit = (srv) => {
    if (!conflictOnce) return null;
    conflictOnce = false;
    srv.sceneVersion = 3; // another client moved the scenario ahead
    return new Response(
      JSON.stringify({
        error: {
          code: "scene_version_conflict",
          message: "The scenario changed after the client's base version.",
          current_scene_version: 3,
          scenario_url: `/api/v1/scenarios/${SCENARIO}`,
        },
      }),
      { status: 409, headers: { "content-type": "application/json" } },
    );
  };

  await session.commitEdits({
    edits: [MET_ITEM],
    transport: "universal-edits-v1",
    label: "met",
  });
  await clock.settle();
  await clock.settle();
  await clock.tick(4);

  const posts = server.universalEditPosts();
  assert.equal(posts.length, 2, "conflict recovery re-submits exactly once");
  assert.equal(JSON.parse(posts[1].body).base_scene_version, 3);
  assert.notEqual(posts[0].headers["Idempotency-Key"], posts[1].headers["Idempotency-Key"]);
  const applied = events.filter((event) => event.type === "applied");
  assert.equal(applied.length, 1);
  assert.equal(applied[0].version, 4);
});

test("the default conflict policy defers universal items to the server's fences", () => {
  // A family edit carries no client-side state to consult: the reapply
  // happens once and the server's registry validators rule on it (a
  // duplicate building add is a typed edit_rejected, not silent duplication).
  assert.equal(
    defaultConflictPolicy({
      operation: "add",
      treeId: null,
      serverTreeIds: new Set(),
      item: MET_ITEM,
    }),
    true,
  );
  // Tree policy is unchanged.
  assert.equal(
    defaultConflictPolicy({
      operation: "delete",
      treeId: "tree-gone",
      serverTreeIds: new Set(),
      item: { operation: "delete", tree_id: "tree-gone" },
    }),
    false,
  );
});

// ---------------------------------------------------------------------------
// Shared world: join-by-default through the capability document
// ---------------------------------------------------------------------------

test("a visitor without an explicit scenario joins the capabilities' shared world", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  // The server publishes its one shared workspace; the harness scenario id
  // doubles as that workspace so the canned routes answer the join.
  server.capabilitiesDocument = {
    ...snapshot,
    default_workspace_id: SCENARIO,
  };

  const harness = makeSession(server);
  const baseline = await harness.session.connect({ siteId: "campus-1km-v1" });

  assert.equal(baseline.manifest.scene_version, 0);
  assert.equal(harness.session.scenarioId, SCENARIO);
  const scenarioCreates = server.calls.filter(
    (call) => call.method === "POST" && call.path === "/api/v1/scenarios",
  );
  assert.equal(scenarioCreates.length, 0, "joining the shared world never mints a scenario");
});

test("an older server without the shared-world pointer still mints a private scenario", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();

  const harness = makeSession(server);
  const baseline = await harness.session.connect({ siteId: "campus-1km-v1" });

  assert.equal(baseline.manifest.scene_version, 0);
  const scenarioCreates = server.calls.filter(
    (call) => call.method === "POST" && call.path === "/api/v1/scenarios",
  );
  assert.equal(scenarioCreates.length, 1, "back-compat: no pointer means a fresh mint");
});

test("an explicit scenario id joins that workspace without consulting the pointer", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.capabilitiesDocument = {
    ...snapshot,
    default_workspace_id: "scn_someone_elses",
  };

  const harness = makeSession(server);
  const baseline = await harness.session.connect({
    siteId: "campus-1km-v1",
    scenarioId: SCENARIO,
  });

  assert.equal(baseline.manifest.scene_version, 0);
  assert.equal(harness.session.scenarioId, SCENARIO);
  assert.equal(
    server.calls.filter((call) => call.method === "POST" && call.path === "/api/v1/scenarios")
      .length,
    0,
  );
});

test("the shared-world join waits out a boot-time baseline job before reading the baseline", async () => {
  const server = new FakeContractServer();
  // Cold shape: the shared world exists but its baseline job is still
  // running (a site without stored baseline outputs). The join must wait
  // for the scenario's active job and then read the published baseline.
  server.jobs.set("job_baseline", {
    job_id: "job_baseline",
    scenario_id: SCENARIO,
    target_scene_version: 0,
    script: ["running", "complete"],
    step: 0,
  });
  const upstream = server.fetchImpl;
  server.fetchImpl = async (url, init = {}) => {
    const response = await upstream(url, init);
    const path = new URL(url, "http://server.test").pathname;
    if (path === "/api/v1/jobs/job_baseline") {
      const body = await response.json();
      if (body.status === "complete") await server.seedBaseline();
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    return response;
  };
  server.capabilitiesDocument = {
    ...snapshot,
    default_workspace_id: SCENARIO,
  };

  const clock = makeFakeClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle();
  await clock.tick(2); // running -> complete polls; the job seeds the baseline

  const baseline = await connecting;
  assert.equal(baseline.manifest.scene_version, 0);
  assert.equal(harness.session.scenarioId, SCENARIO);
});

// ---------------------------------------------------------------------------
// Cold-boot wake: transport-class connect failures retry ~90 s, 4xx does not
// ---------------------------------------------------------------------------

test("transport-class connect failures keep waking the server past the short ladder", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  // Cold machine (scale-to-zero wake): the first five requests drop.
  let failuresLeft = 5;
  const upstream = server.fetchImpl;
  server.fetchImpl = async (url, init = {}) => {
    if (failuresLeft > 0) {
      failuresLeft -= 1;
      throw new TypeError("simulated cold boot: connection refused");
    }
    return upstream(url, init);
  };

  const clock = makeRecordingClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle();
  for (let round = 0; round < 10; round += 1) await clock.tick(1);

  const baseline = await connecting;
  assert.equal(baseline.manifest.scene_version, 0, "the connect lands once the machine is up");
  const reconnecting = harness.events.filter(
    (event) => event.type === "status" && event.phase === "reconnecting",
  );
  assert.equal(reconnecting.length, 5, "one waking announcement per failed attempt");
  assert.ok(
    reconnecting.every((event) => event.waking === true),
    "cold-boot retries announce themselves as waking, not as retry counts",
  );
  assert.deepEqual(
    clock.delays.slice(0, 5),
    [1000, 2000, 4000, 5000, 5000],
    "exponential start, then the capped wake cadence",
  );
});

test("a transport failure fetching the capability document wakes instead of failing connect", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  let capabilityFailures = 1;
  const upstream = server.fetchImpl;
  server.fetchImpl = async (url, init = {}) => {
    const path = new URL(url, "http://server.test").pathname;
    if (path === "/api/v1/capabilities" && capabilityFailures > 0) {
      capabilityFailures -= 1;
      throw new TypeError("simulated cold boot: connection refused");
    }
    return upstream(url, init);
  };

  const clock = makeRecordingClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle();
  await clock.tick(1); // fire the wake backoff

  const baseline = await connecting;
  assert.equal(baseline.manifest.scene_version, 0);
  assert.ok(
    harness.events.some(
      (event) => event.type === "status" && event.phase === "reconnecting" && event.waking,
    ),
    "a transport-class document fetch rides the wake ladder, not a hard failure",
  );
});

// ---------------------------------------------------------------------------
// Full-recompute honesty (S3b): a full-tile running job says so on the wire,
// so the exact status line can promise minutes instead of implying seconds
// ---------------------------------------------------------------------------

test("a full-tile running job reports its mode on the running status", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  // The solver routed a FULL recompute (window below threshold / cold cache):
  // the job body carries mode "full" while it runs. Set it before the poll
  // script reaches its running step.
  server.jobs.get("job_1").mode = "full";
  await clock.tick(1); // queued poll
  await clock.tick(1); // running poll

  const running = events.find((event) => event.type === "status" && event.phase === "running");
  assert.equal(running.mode, "full", "the mode rides the running status for honest copy");

  // Control: an ordinary windowed/local job carries the server's own mode.
  await session.commitEdits({ edits: addEdit("tree-b"), label: "b" });
  await clock.settle();
  await clock.tick(1); // queued poll
  await clock.tick(1); // running poll
  const second = events
    .filter((event) => event.type === "status" && event.phase === "running")
    .at(-1);
  assert.equal(second.mode, "local");
});

// ---------------------------------------------------------------------------
// Remediation: create-key replay across retries + 429 vs the wake budget
// ---------------------------------------------------------------------------

test("a create whose response is lost replays with the same idempotency key on retry", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.loseResponseCreates = 1; // server applies the create, the response is lost

  const clock = makeRecordingClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle(); // create POST lands; the lost response rides the wake ladder
  await clock.tick(1); // 1 s backoff -> the retry re-POSTs the create

  const baseline = await connecting;
  assert.equal(baseline.manifest.scene_version, 0);
  const creates = server.calls.filter(
    (call) => call.method === "POST" && call.path === "/api/v1/scenarios",
  );
  assert.equal(creates.length, 2, "exactly one replay");
  assert.equal(
    creates[0].headers["Idempotency-Key"],
    creates[1].headers["Idempotency-Key"],
    "the retry is a replay: one idempotency key across connect retries",
  );
});

test("a rate-limited connect falls back to the wake ladder once the short ladder is spent", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  // The shared transport bucket 429s the scenario GET four times (admission
  // control under load), then recovers.
  let gated = 4;
  const upstream = server.fetchImpl;
  server.fetchImpl = async (url, init = {}) => {
    const path = new URL(url, "http://server.test").pathname;
    if (
      gated > 0 &&
      path === `/api/v1/scenarios/${SCENARIO}` &&
      (init.method ?? "GET") === "GET"
    ) {
      gated -= 1;
      return new Response(
        JSON.stringify({ error: { code: "rate_limited", message: "gated" } }),
        { status: 429, headers: { "content-type": "application/json" } },
      );
    }
    return upstream(url, init);
  };

  const clock = makeRecordingClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  await clock.settle(); // mint + first 429 -> short ladder
  for (let round = 0; round < 6; round += 1) await clock.tick(1);

  const baseline = await connecting;
  assert.equal(baseline.manifest.scene_version, 0, "connect lands after the gate lifts");
  assert.deepEqual(
    clock.delays.slice(0, 4),
    [1000, 2000, 4000, 5000],
    "429 keeps the short cadence, then falls back to the capped wake cadence",
  );
  assert.ok(
    harness.events.some(
      (event) => event.type === "status" && event.phase === "reconnecting" && event.waking,
    ),
    "the fallback announces itself as waking, not as a dead end",
  );
  const creates = server.calls.filter(
    (call) => call.method === "POST" && call.path === "/api/v1/scenarios",
  );
  assert.equal(creates.length, 1, "retries rejoin the minted scenario");
});

test("an interleaved 429 does not reset the wake budget", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  // Attempt 1 drops the transport (cold boot); every later scenario GET is
  // rate-limited forever. The 429 pauses must not hand back a fresh 90 s.
  let transportDropped = false;
  const upstream = server.fetchImpl;
  server.fetchImpl = async (url, init = {}) => {
    const path = new URL(url, "http://server.test").pathname;
    if (path === `/api/v1/scenarios/${SCENARIO}` && (init.method ?? "GET") === "GET") {
      if (!transportDropped) {
        transportDropped = true;
        throw new TypeError("simulated cold boot: connection refused");
      }
      return new Response(
        JSON.stringify({ error: { code: "rate_limited", message: "gated" } }),
        { status: 429, headers: { "content-type": "application/json" } },
      );
    }
    return upstream(url, init);
  };

  const clock = makeRecordingClock();
  const harness = makeSession(server, {}, clock);
  const connecting = harness.session.connect({ siteId: "campus-1km-v1" });
  const rejected = assert.rejects(connecting, /gated/);
  await clock.settle();
  for (let round = 0; round < 30; round += 1) await clock.tick(1);
  await rejected;

  const totalWaitMs = clock.delays.reduce((sum, ms) => sum + ms, 0);
  // Bound: the wake budget (checked at entry, so at most one capped wait
  // over) plus the short ladder's 1+2+4 s of admission-control pauses.
  assert.ok(
    totalWaitMs >= 90000 && totalWaitMs <= 97000,
    `the interleaved 429s neither reset nor inflate the budget (waited ${totalWaitMs} ms)`,
  );
  assert.equal(clock.pendingCount(), 0, "no further retries are scheduled");
});
