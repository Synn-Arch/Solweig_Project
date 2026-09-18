// SPDX-License-Identifier: GPL-3.0-only
//
// View-only operations (UEDIT-007): ApiClient.createView request shape and
// typed error envelopes through an injectable fetch, plus ExactSession
// plumbing (connected-gate, exact-version default, zero-job flags) against
// the in-memory contract harness. No live server is involved.

import test from "node:test";
import assert from "node:assert/strict";

import { ApiClient, ApiClientError } from "../api_client.mjs";
import { ExactSession, defaultConflictPolicy } from "../exact_session.mjs";
import { viewResultModel } from "../capabilities.mjs";
import { COLS, ROWS, FakeContractServer, makeFakeClock } from "./contract_harness.mjs";

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// ApiClient.createView — request shape against a recording fake fetch

test("createView POSTs the contract body with snake_case fields", async () => {
  const calls = [];
  const client = new ApiClient({
    baseUrl: "http://server.test",
    fetch: async (url, init = {}) => {
      calls.push({ url: String(url), body: JSON.parse(init.body) });
      return jsonResponse(200, {
        operation: "compare",
        scene_version: 3,
        job_enqueued: false,
        zero_scientific_jobs: true,
      });
    },
  });
  await client.createView("scn 1", {
    operation: "compare",
    layer: "utci",
    timeIndex: 12,
    compareLayer: "shadow",
    sceneVersion: 3,
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://server.test/api/v1/scenarios/scn%201/views");
  assert.deepEqual(calls[0].body, {
    operation: "compare",
    layer: "utci",
    time_index: 12,
    compare_layer: "shadow",
    scene_version: 3,
  });
});

test("createView omits compare_layer and scene_version when not given", async () => {
  const calls = [];
  const client = new ApiClient({
    baseUrl: "http://server.test",
    fetch: async (url, init = {}) => {
      calls.push(JSON.parse(init.body));
      return jsonResponse(200, {});
    },
  });
  await client.createView("scn_test_01", { operation: "legend", layer: "utci" });
  assert.deepEqual(calls[0], { operation: "legend", layer: "utci", time_index: 0 });
});

test("typed view refusals surface code/status/details", async () => {
  const client = new ApiClient({
    baseUrl: "http://server.test",
    fetch: async () =>
      // Wire shape matches the server's envelope(): ApiError.details are
      // flattened into the error object (models.py), not nested.
      jsonResponse(409, {
        error: {
          code: "view_not_available",
          message: "layer not published at this scene version",
          published_layers: ["utci", "tmrt"],
        },
      }),
  });
  await assert.rejects(
    client.createView("scn_test_01", { operation: "select_layer", layer: "wind" }),
    (error) => {
      assert.ok(error instanceof ApiClientError);
      assert.equal(error.code, "view_not_available");
      assert.equal(error.status, 409);
      assert.deepEqual(error.details.published_layers, ["utci", "tmrt"]);
      return true;
    },
  );
});

test("cached_time refusal keeps the cached index list in details", async () => {
  const client = new ApiClient({
    baseUrl: "http://server.test",
    fetch: async () =>
      jsonResponse(409, {
        error: {
          code: "view_not_available",
          message: "time index not cached",
          cached_time_indices: [6, 12, 18],
        },
      }),
  });
  await assert.rejects(
    client.createView("scn_test_01", {
      operation: "cached_time",
      layer: "utci",
      timeIndex: 3,
    }),
    (error) => {
      assert.deepEqual(error.details.cached_time_indices, [6, 12, 18]);
      return true;
    },
  );
});

// ---------------------------------------------------------------------------
// ExactSession.createView — gate, defaults, queue bypass

async function connectHarness(server) {
  const clock = makeFakeClock();
  const session = new ExactSession({
    client: new ApiClient({ fetch: server.fetchImpl, sessionId: "sess-test" }),
    callbacks: { onStatus: () => {}, onBaseline: () => {}, onAuthoritative: () => {} },
    pollIntervalMs: 5,
    timers: clock.timers,
    conflictPolicy: defaultConflictPolicy,
  });
  await session.connect({ siteId: "campus-1km-v1" });
  return { session, clock };
}

test("views answer from the published result without a solver job and bypass the edit queue", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  const { session } = await connectHarness(server);

  const body = await session.createView({ operation: "legend", layer: "utci", timeIndex: 12 });

  const viewCall = server.calls.find((call) => call.path.endsWith("/views"));
  assert.ok(viewCall, "one POST /views must be issued");
  const parsed = JSON.parse(viewCall.body);
  assert.equal(parsed.operation, "legend");
  assert.equal(parsed.layer, "utci");
  assert.equal(parsed.time_index, 12);
  // Defaults to the session's applied exact version (the published result).
  assert.equal(parsed.scene_version, session.exactResultVersion);
  // Zero-job affordance quotes the server's own flags.
  assert.equal(body.job_enqueued, false);
  assert.equal(body.zero_scientific_jobs, true);
  assert.match(viewResultModel(body).zeroJob, /0 solver jobs/);
  // A view never mutates the scene: no edit POST, no version change.
  assert.equal(server.editPosts().length, 0);
  assert.equal(session.sceneVersion, 0);
  assert.deepEqual(body.layer.shape, [3, ROWS, COLS]);
});

test("view refusals propagate as typed errors with published layers", async () => {
  const server = new FakeContractServer();
  await server.seedBaseline();
  server.viewHandler = () =>
    jsonResponse(409, {
      error: {
        code: "view_not_available",
        message: "layer not published",
        published_layers: ["utci"],
      },
    });
  const { session } = await connectHarness(server);

  await assert.rejects(
    session.createView({ operation: "select_layer", layer: "wind" }),
    (error) => {
      assert.ok(error instanceof ApiClientError);
      assert.equal(error.code, "view_not_available");
      assert.deepEqual(error.details.published_layers, ["utci"]);
      return true;
    },
  );
});

test("createView refuses when the session is not connected", async () => {
  const session = new ExactSession({
    client: new ApiClient({ fetch: async () => jsonResponse(200, {}) }),
    callbacks: {},
  });
  await assert.rejects(session.createView({ operation: "legend", layer: "utci" }), (error) => {
    assert.ok(error instanceof ApiClientError);
    assert.equal(error.code, "not_connected");
    return true;
  });
});
