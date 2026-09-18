// SPDX-License-Identifier: GPL-3.0-only
//
// Honest-ETA consumption (server a83ac9f): QUEUED and RUNNING job bodies
// carry eta_seconds (int | null) and eta_basis ("history" | "static" | null).
// null means honestly unknown — the studio must NEVER invent a number. These
// gates pin both halves of the wave:
//
//   * ExactSession threads the fields through the polled status events in
//     camelCase (same convention as mode), unknown staying null.
//   * The app's pure status-text builders render the exact user-facing
//     strings: an ETA clause only when the server sent one, never a guess.
//
// No live server is involved; the in-memory contract harness serves the new
// job-body fields.

import test from "node:test";
import assert from "node:assert/strict";

import { ApiClient } from "../api_client.mjs";
import { ExactSession, defaultConflictPolicy } from "../exact_session.mjs";
import { FakeContractServer, addEdit, makeFakeClock } from "./contract_harness.mjs";

// app.mjs boots its DOM shell only in a browser (the `typeof document` guard
// keeps the module node-importable); its pure status-text builders are the
// seam these tests assert. Loaded lazily so a missing export fails the
// individual render assertions instead of unloading the whole file.
let appModule = null;
async function app() {
  appModule ??= await import("../app.mjs");
  return appModule;
}

function makeSession(server) {
  const events = [];
  const clock = makeFakeClock();
  const session = new ExactSession({
    client: new ApiClient({ fetch: server.fetchImpl, sessionId: "sess-eta" }),
    callbacks: {
      onStatus: (status) => events.push({ type: "status", ...status }),
      onBaseline: () => {},
      onPatchApplied: () => {},
      onScope: () => {},
      onValidation: () => {},
      onAuthoritative: () => {},
    },
    pollIntervalMs: 5,
    timers: clock.timers,
    conflictPolicy: defaultConflictPolicy,
  });
  return { session, events, clock };
}

async function connect(server) {
  const harness = makeSession(server);
  await harness.session.connect({ siteId: "campus-1km-v1" });
  return harness;
}

// ---------------------------------------------------------------------------
// ExactSession: the server's eta fields ride the polled status events
// ---------------------------------------------------------------------------

test("a running job's history-based eta rides the running status", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  // The server learned this scenario's own p80 for the current mode.
  server.jobs.get("job_1").etaSeconds = 240;
  server.jobs.get("job_1").etaBasis = "history";
  await clock.tick(1); // queued poll
  await clock.tick(1); // running poll

  const running = events.find((event) => event.type === "status" && event.phase === "running");
  assert.equal(running.etaSeconds, 240, "eta_seconds maps to etaSeconds on the status event");
  assert.equal(running.etaBasis, "history", "eta_basis maps to etaBasis on the status event");
});

test("a running job without an eta stays honestly null — never a client-side guess", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  // Ordinary job: the mode is not known yet, so the server sends (null, null).
  await clock.tick(1); // queued poll
  await clock.tick(1); // running poll

  const running = events.find((event) => event.type === "status" && event.phase === "running");
  assert.equal(running.etaSeconds, null, "unknown stays null on the status event");
  assert.equal(running.etaBasis, null);
});

test("a queued reconcile job's full eta rides the queued status; ordinary queued stays null", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session, events, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  // Exact-lane reconcile job: queued bodies carry the full estimate.
  server.jobs.get("job_1").etaSeconds = 480;
  server.jobs.get("job_1").etaBasis = "static";
  await clock.tick(1); // queued poll carries the eta
  const queued = events.filter((event) => event.type === "status" && event.phase === "queued").at(-1);
  assert.equal(queued.etaSeconds, 480, "the queued poll status carries the reconcile eta");
  assert.equal(queued.etaBasis, "static");

  // Control: an ordinary queued job (no eta on the body) stays (null, null).
  await session.commitEdits({ edits: addEdit("tree-b"), label: "b" });
  await clock.settle();
  await clock.tick(1); // job_2 queued poll
  const ordinary = events.filter((event) => event.type === "status" && event.phase === "queued").at(-1);
  assert.equal(ordinary.etaSeconds, null, "ordinary queued jobs carry no eta");
  assert.equal(ordinary.etaBasis, null);
});

// ---------------------------------------------------------------------------
// Rendered strings: ETA clause only when the server sent one
// ---------------------------------------------------------------------------

test("formatEtaWait rounds sensibly: nearest 10 s below 90 s, minutes above", async () => {
  const { formatEtaWait } = await app();
  assert.equal(formatEtaWait(75), "about 80 s");
  assert.equal(formatEtaWait(44), "about 40 s");
  assert.equal(formatEtaWait(89), "about 90 s");
  assert.equal(formatEtaWait(90), "about 2 min");
  assert.equal(formatEtaWait(240), "about 4 min");
  assert.equal(formatEtaWait(480), "about 8 min");
});

test("running without an eta keeps the honest copy — NO eta clause", async () => {
  const { exactRunningStatusText } = await app();
  const text = exactRunningStatusText({
    mode: "local",
    stage: "time_loop",
    progress: { completed_time_steps: 9, total_time_steps: 24 },
    etaSeconds: null,
    etaBasis: null,
  });
  assert.equal(text, "Updating the map · 9/24 steps");
  assert.doesNotMatch(text, /past runs/, "null eta must never read as an estimate");
  assert.doesNotMatch(text, /typically/, "null eta must never read as a fallback guess");
  assert.doesNotMatch(text, /time_loop/, "server stage names stay out of visible copy");
});

test("running before the first step says starting up, never a fake count", async () => {
  const { exactRunningStatusText } = await app();
  assert.equal(
    exactRunningStatusText({ mode: "local", stage: "time_loop", progress: null }),
    "Updating the map · starting up",
  );
});

test("running with a history eta appends the past-runs clause", async () => {
  const { exactRunningStatusText } = await app();
  const text = exactRunningStatusText({
    mode: "local",
    stage: "time_loop",
    progress: { completed_time_steps: 9, total_time_steps: 24 },
    etaSeconds: 240,
    etaBasis: "history",
  });
  assert.equal(text, "Updating the map · 9/24 steps · about 4 min left (from past runs)");
});

test("running with a short history eta shows seconds, not a padded minute count", async () => {
  const { exactRunningStatusText } = await app();
  const text = exactRunningStatusText({
    mode: "local",
    stage: "time_loop",
    progress: { completed_time_steps: 9, total_time_steps: 24 },
    etaSeconds: 75,
    etaBasis: "history",
  });
  assert.equal(text, "Updating the map · 9/24 steps · about 80 s left (from past runs)");
});

test("running with a static eta discloses the fallback basis", async () => {
  const { exactRunningStatusText } = await app();
  const text = exactRunningStatusText({
    mode: "local",
    stage: "time_loop",
    progress: { completed_time_steps: 9, total_time_steps: 24 },
    etaSeconds: 480,
    etaBasis: "static",
  });
  assert.equal(text, "Updating the map · 9/24 steps · typically about 8 min");
});

test("full recompute without an eta keeps its honest phrasing — NO eta clause", async () => {
  const { exactRunningStatusText } = await app();
  const text = exactRunningStatusText({ mode: "full", etaSeconds: null, etaBasis: null });
  assert.equal(text, "Recalculating the whole map — a few minutes");
  assert.doesNotMatch(text, /past runs|typically/, "null eta must not grow a clause");
});

test("full recompute with an eta collapses to ONE duration clause", async () => {
  const { exactRunningStatusText } = await app();
  const text = exactRunningStatusText({
    mode: "full",
    etaSeconds: 480,
    etaBasis: "static",
  });
  assert.equal(text, "Recalculating the whole map — typically about 8 min");
  assert.equal(
    (text.match(/ — /g) ?? []).length,
    1,
    "the server's estimate replaces the generic promise — never two stacked dashes",
  );
});

test("full recompute with a history-based eta names the past-runs basis once", async () => {
  const { exactRunningStatusText } = await app();
  const text = exactRunningStatusText({
    mode: "full",
    etaSeconds: 470,
    etaBasis: "history",
  });
  assert.equal(text, "Recalculating the whole map — about 8 min left (from past runs)");
});

test("queued without an eta keeps the current copy unchanged", async () => {
  const { exactQueuedStatusText } = await app();
  assert.equal(
    exactQueuedStatusText({ queuePosition: 1, sceneVersion: 3, etaSeconds: null, etaBasis: null }),
    "Queued — position 1",
  );
  assert.equal(
    exactQueuedStatusText({ queuePosition: null, sceneVersion: 3, etaSeconds: null, etaBasis: null }),
    "Queued",
  );
});

test("queued with the reconcile eta names the queued full recompute", async () => {
  const { exactQueuedStatusText } = await app();
  const text = exactQueuedStatusText({
    queuePosition: 1,
    sceneVersion: 3,
    etaSeconds: 480,
    etaBasis: "static",
  });
  assert.equal(text, "Waiting its turn — typically about 8 min");
});

test("queued reconcile eta with history basis discloses the past-runs basis", async () => {
  const { exactQueuedStatusText } = await app();
  const text = exactQueuedStatusText({
    queuePosition: 1,
    sceneVersion: 3,
    etaSeconds: 470,
    etaBasis: "history",
  });
  assert.equal(
    text,
    "Waiting its turn — about 8 min (from past runs)",
  );
});

test("queued line keeps the scene version out of the visible copy", async () => {
  const { exactQueuedStatusText } = await app();
  // The builder still accepts sceneVersion (tooltips may spend it), but the
  // visible line renders only the wait or the queue position.
  const text = exactQueuedStatusText({
    queuePosition: 2,
    sceneVersion: 5,
    etaSeconds: null,
    etaBasis: null,
  });
  assert.equal(text, "Queued — position 2");
  assert.doesNotMatch(text, /scene v/, "revision numbers stay out of visible copy");
});
