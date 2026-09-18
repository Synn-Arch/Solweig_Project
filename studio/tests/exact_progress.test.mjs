// SPDX-License-Identifier: GPL-3.0-only
//
// exact_progress + roster view gates (operational-stability waves ① + ③,
// client half): the pure builders app.mjs renders from the server's new
// SSE events. No server, no DOM — the countdown contract is pinned here:
//
//   * exactProgressState — terminal retires ONLY that job's countdown; the
//     eta deadline is monotone (identical re-emitted etas never push the
//     deadline out); unknown etas render presence without inventing a time.
//   * exactProgressClause — decreasing remaining time; past the deadline it
//     NEVER goes negative or vanishes, it says "taking longer than usual".
//   * rosterView — server-truth live actors first, observed-but-gone actors
//     collapse into "+N earlier" (never silently forgotten, never fake-live).

import test from "node:test";
import assert from "node:assert/strict";

import {
  exactProgressState,
  exactProgressClause,
  rosterView,
} from "../app.mjs";

const NOW = 1_000_000;

function event(overrides = {}) {
  return {
    job_id: "job-1",
    status: "running",
    target_revision: 7,
    eta_seconds: 90,
    eta_basis: "history",
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// exactProgressState
// ---------------------------------------------------------------------------

test("queued without eta: presence without an invented deadline", () => {
  const state = exactProgressState(null, event({ status: "queued", eta_seconds: null }), NOW);
  assert.equal(state.jobId, "job-1");
  assert.equal(state.deadlineMs, null);
  assert.equal(state.etaBasis, "history");
});

test("eta arrival sets the deadline at now + eta", () => {
  const state = exactProgressState(null, event({ eta_seconds: 90 }), NOW);
  assert.equal(state.deadlineMs, NOW + 90_000);
});

test("identical re-emitted eta keeps the original deadline (monotone)", () => {
  const first = exactProgressState(null, event({ eta_seconds: 90 }), NOW);
  const again = exactProgressState(first, event({ eta_seconds: 90 }), NOW + 5_000);
  assert.equal(again.deadlineMs, NOW + 90_000, "re-emit must not push the deadline");
});

test("a genuinely revised eta recomputes the deadline", () => {
  const first = exactProgressState(null, event({ eta_seconds: 90 }), NOW);
  const revised = exactProgressState(first, event({ eta_seconds: 200 }), NOW + 5_000);
  assert.equal(revised.deadlineMs, NOW + 5_000 + 200_000);
});

test("terminal status retires that job's view", () => {
  for (const status of ["complete", "failed", "superseded", "cancelled"]) {
    const live = exactProgressState(null, event(), NOW);
    assert.equal(exactProgressState(live, event({ status }), NOW + 1_000), null, status);
  }
});

test("another job's terminal event never clears a live countdown", () => {
  const live = exactProgressState(null, event({ job_id: "job-1" }), NOW);
  const after = exactProgressState(live, event({ job_id: "job-2", status: "complete" }), NOW);
  assert.equal(after.jobId, "job-1");
  assert.equal(after.deadlineMs, NOW + 90_000);
});

test("malformed events pass the previous state through untouched", () => {
  const live = exactProgressState(null, event(), NOW);
  assert.equal(exactProgressState(live, {}, NOW), live);
  assert.equal(exactProgressState(live, { job_id: null }, NOW), live);
  assert.equal(exactProgressState(null, {}, NOW), null);
});

test("queue position rides through when the server sends one", () => {
  const state = exactProgressState(null, event({ status: "queued", eta_seconds: null, queue_position: 2 }), NOW);
  assert.equal(state.queuePosition, 2);
  const none = exactProgressState(null, event({ status: "queued", eta_seconds: null }), NOW);
  assert.equal(none.queuePosition, null);
});

// ---------------------------------------------------------------------------
// exactProgressClause
// ---------------------------------------------------------------------------

test("no progress or no deadline renders no clause", () => {
  assert.equal(exactProgressClause(null, NOW), null);
  assert.equal(exactProgressClause({ deadlineMs: null }, NOW), null);
});

test("remaining time reads as a decreasing countdown", () => {
  const progress = exactProgressState(null, event({ eta_seconds: 90 }), NOW);
  assert.equal(exactProgressClause(progress, NOW + 30_000), "about 60 s left (from past runs)");
});

test("static basis says 'typically' without claiming history", () => {
  const progress = exactProgressState(null, event({ eta_seconds: 120, eta_basis: "static" }), NOW);
  assert.equal(exactProgressClause(progress, NOW), "about 2 min left");
});

test("past the deadline: honest 'taking longer', never negative", () => {
  const history = exactProgressState(null, event({ eta_seconds: 90, eta_basis: "history" }), NOW);
  assert.equal(exactProgressClause(history, NOW + 91_000), "taking longer than past runs");
  const fallback = exactProgressState(null, event({ eta_seconds: 90, eta_basis: "static" }), NOW);
  assert.equal(exactProgressClause(fallback, NOW + 300_000), "taking longer than usual");
});

// ---------------------------------------------------------------------------
// rosterView
// ---------------------------------------------------------------------------

test("live roster leads; observed-but-gone actors become '+N earlier'", () => {
  const view = rosterView(["editor-b", "editor-a"], ["editor-a"]);
  assert.deepEqual(view.live, ["editor-a"]);
  assert.deepEqual(view.earlier, ["editor-b"]);
});

test("everyone live: no earlier tier", () => {
  const view = rosterView(["editor-a", "editor-b"], ["editor-b", "editor-a"]);
  assert.deepEqual(view.live, ["editor-b", "editor-a"]);
  assert.deepEqual(view.earlier, []);
});

test("empty/absent rosters degrade to empty views, never a throw", () => {
  assert.deepEqual(rosterView([], []), { live: [], earlier: [] });
  assert.deepEqual(rosterView(null, null), { live: [], earlier: [] });
  assert.deepEqual(rosterView(["editor-a"], null), { live: [], earlier: ["editor-a"] });
});
