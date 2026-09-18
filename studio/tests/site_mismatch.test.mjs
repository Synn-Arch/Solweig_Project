// SPDX-License-Identifier: GPL-3.0-only
//
// 409 site_identity_mismatch (P7 extension): the scenario is pinned to a site
// identity the deployment no longer serves. The session must fail visibly
// with pinned vs current identity, refuse further mutations (no zombie
// retries), and refuse resets — recovery is a fresh scenario. In-memory
// contract harness only; no live server.

import test from "node:test";
import assert from "node:assert/strict";

import { ApiClient, ApiClientError } from "../api_client.mjs";
import { ExactSession, defaultConflictPolicy } from "../exact_session.mjs";
import { FakeContractServer, addEdit, makeFakeClock } from "./contract_harness.mjs";

const PINNED = {
  rows: 512,
  cols: 512,
  pixel_size_m: 2,
  origin_x_m: 410000,
  origin_y_m: 4590000,
};
const CURRENT = { ...PINNED, pixel_size_m: 1, cols: 1024 };

function mismatchResponse() {
  // Wire shape: ApiError.details are flattened into the error object.
  return new Response(
    JSON.stringify({
      error: {
        code: "site_identity_mismatch",
        message:
          "the scenario's pinned site identity does not match the live site " +
          "(deployment geometry changed); start a new scenario",
        pinned_identity: PINNED,
        current_identity: CURRENT,
      },
    }),
    { status: 409, headers: { "content-type": "application/json" } },
  );
}

function makeSession(server, callbacks = {}) {
  const events = [];
  const clock = makeFakeClock();
  const session = new ExactSession({
    client: new ApiClient({ fetch: server.fetchImpl, sessionId: "sess-test" }),
    callbacks: {
      onStatus: (status) => events.push({ type: "status", ...status }),
      onBaseline: () => {},
      onAuthoritative: () => {},
      ...callbacks,
    },
    pollIntervalMs: 5,
    timers: clock.timers,
    conflictPolicy: defaultConflictPolicy,
  });
  return { session, events, clock };
}

async function connect(server, callbacks = {}) {
  const harness = makeSession(server, callbacks);
  await harness.session.connect({ siteId: "campus-1km-v1" });
  return harness;
}

test("a 409 site_identity_mismatch records both identities and blocks commits", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.onEdit = () => mismatchResponse();

  const mismatches = [];
  const { session, events, clock } = await connect(server, {
    onSiteMismatch: (detail) => mismatches.push(detail),
  });
  assert.equal(session.siteMismatch, null);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();

  // Visible failure with pinned vs current — not a generic retryable error.
  assert.equal(session.siteMismatch.code ?? "site_identity_mismatch", "site_identity_mismatch");
  assert.deepEqual(session.siteMismatch.pinnedIdentity, PINNED);
  assert.deepEqual(session.siteMismatch.currentIdentity, CURRENT);
  assert.equal(mismatches.length, 1);
  assert.deepEqual(mismatches[0].pinnedIdentity, PINNED);
  assert.deepEqual(mismatches[0].currentIdentity, CURRENT);
  assert.match(mismatches[0].message, /pinned site identity/);
  // The status channel carries the same fact for the connection pill.
  assert.ok(
    events.some((event) => event.type === "status" && event.phase === "site_mismatch"),
  );
  // Not classified as an ordinary failure (design state is preserved).
  assert.ok(
    !events.some((event) => event.type === "status" && event.phase === "failed"),
  );
  // No poller is left waiting for a result that can never arrive.
  assert.equal(session.awaitingVersion, null);
});

test("blocked commits resolve without issuing another POST", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.onEdit = () => mismatchResponse();
  const { session, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "first" });
  await clock.settle();
  const postsAfterFirst = server.editPosts().length;

  const second = await session.commitEdits({ edits: addEdit("tree-b"), label: "second" });
  await clock.settle();

  assert.equal(second, null, "the blocked commit resolves as a no-op, not a rejection");
  assert.equal(server.editPosts().length, postsAfterFirst, "no zombie POST after the 409");
});

test("reset is refused while the scenario is pinned to a stale identity", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.onEdit = () => mismatchResponse();
  const { session, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();

  const resetsBefore = server.calls.filter((call) => call.path.endsWith("/reset")).length;
  await assert.rejects(session.reset({ label: "reset" }), (error) => {
    assert.ok(error instanceof ApiClientError);
    assert.equal(error.code, "site_identity_mismatch");
    return true;
  });
  const resetsAfter = server.calls.filter((call) => call.path.endsWith("/reset")).length;
  assert.equal(resetsAfter, resetsBefore, "reset must not reach the server once blocked");
});

test("retry under a site mismatch re-POSTs nothing (no zombie retry)", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.onEdit = () => mismatchResponse();
  const mismatches = [];
  const { session, clock } = await connect(server, {
    onSiteMismatch: (detail) => mismatches.push(detail),
  });

  // The refused commit is the only POST; its 409 raises the banner and leaves
  // the attempt unacknowledged — exactly the state retryLast re-sends.
  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  const postsAfterRefusal = server.editPosts().length;
  assert.equal(postsAfterRefusal, 1);
  assert.equal(session.lastAttempt.accepted, false);
  const bannersAfterRefusal = mismatches.length;

  const retried = await session.retryLast();
  await clock.settle();

  // Blocked exactly like commitEdits: resolves as a no-op, zero /edits POSTs,
  // and the mismatch channel re-asserts the blocked state.
  assert.equal(retried, null);
  assert.equal(server.editPosts().length, postsAfterRefusal, "retry must not re-POST under a mismatch");
  assert.equal(mismatches.length, bannersAfterRefusal + 1);
});

test("a reset that 409s with a site mismatch records it and blocks further mutations", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const mismatches = [];
  const { session, clock } = await connect(server, {
    onSiteMismatch: (detail) => mismatches.push(detail),
  });
  // The mismatch is unknown until the reset POST itself is refused.
  server.onReset = () => mismatchResponse();

  await assert.rejects(session.reset({ label: "reset" }), (error) => {
    assert.equal(error.code, "site_identity_mismatch");
    return true;
  });
  await clock.settle();

  // The session learned the mismatch from the refused reset: banner rendered…
  assert.deepEqual(session.siteMismatch.pinnedIdentity, PINNED);
  assert.deepEqual(session.siteMismatch.currentIdentity, CURRENT);
  assert.equal(mismatches.length, 1);
  // …and every later mutation now fails fast locally instead of POSTing.
  const postsBefore = server.editPosts().length;
  const resetsBefore = server.calls.filter((call) => call.path.endsWith("/reset")).length;
  assert.equal(await session.commitEdits({ edits: addEdit("tree-a"), label: "a" }), null);
  await assert.rejects(session.reset({ label: "reset again" }), ApiClientError);
  await clock.settle();
  assert.equal(server.editPosts().length, postsBefore, "no zombie edit POST after the refused reset");
  assert.equal(
    server.calls.filter((call) => call.path.endsWith("/reset")).length,
    resetsBefore,
    "no zombie reset POST after the refused reset",
  );
});

test("view operations stay available — they never mutate the pinned scene", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.onEdit = () => mismatchResponse();
  const { session, clock } = await connect(server);

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();

  const body = await session.createView({ operation: "legend", layer: "utci" });
  assert.equal(body.operation, "select_layer"); // canned harness answer
  assert.equal(body.job_enqueued, false);
});

test("a fresh session clears the mismatch flag on connect", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.onEdit = () => mismatchResponse();
  const first = await connect(server);
  await first.session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await first.clock.settle();
  assert.ok(first.session.siteMismatch);

  // Recovery = new scenario (the UI's reconnect affordance).
  const second = await connect(server);
  assert.equal(second.session.siteMismatch, null);
});

test("retry works again once a fresh scenario clears the mismatch", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.onEdit = () => mismatchResponse();
  const mismatches = [];
  const { session, clock } = await connect(server, {
    onSiteMismatch: (detail) => mismatches.push(detail),
  });

  await session.commitEdits({ edits: addEdit("tree-a"), label: "a" });
  await clock.settle();
  await session.retryLast(); // blocked — no POST under the mismatch
  const postsWhenBlocked = server.editPosts().length;
  assert.equal(postsWhenBlocked, 1);

  // Recovery: a fresh scenario re-pins the site identity…
  server.onEdit = null;
  await session.connect({ siteId: "campus-1km-v1" });
  const bannersAfterReconnect = mismatches.length;

  // …so the retry affordance is live again: the never-acknowledged attempt
  // re-sends with its original idempotency key.
  await session.retryLast();
  await clock.settle();

  assert.equal(server.editPosts().length, postsWhenBlocked + 1, "retry POSTs again after recovery");
  assert.equal(mismatches.length, bannersAfterReconnect, "no spurious banners after recovery");
});
