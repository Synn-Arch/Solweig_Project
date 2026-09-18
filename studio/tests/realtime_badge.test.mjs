// SPDX-License-Identifier: GPL-3.0-only
//
// realtime_badge.mjs tests: the pure revision-triple + result-class display
// state consumed by the studio's collab pill, plus the DOM glue (fake
// elements; no browser needed).

import test from "node:test";
import assert from "node:assert/strict";

import {
  realtimeBadgeState,
  applyRealtimeBadge,
  collabSubscriptionAlive,
  owedDurationText,
} from "../realtime_badge.mjs";

const revisions = { workspaceRevision: 7, fastRevision: 7, exactRevision: 3 };

function classView(overrides = {}) {
  return {
    class: "visual_pending",
    effectiveClass: "visual_pending",
    authoritative: false,
    superseded: false,
    targetRevision: 7,
    fastRevision: 7,
    exactRevision: 3,
    exactBaseRevision: 0,
    ageMs: 120,
    ...overrides,
  };
}

test("no subscription state renders nothing (badge stays hidden)", () => {
  assert.equal(realtimeBadgeState(null, null, "idle"), null);
  assert.equal(realtimeBadgeState(undefined, null, "idle"), null);
});

test("a visual_pending fast result displays the conservative placeholder label", () => {
  const badge = realtimeBadgeState(revisions, classView(), "live");
  assert.equal(badge.state, "visual_pending");
  assert.equal(badge.text, "Preview", "the visible word carries no revision number");
  assert.match(badge.title, /workspace 7/, "the revision triple rides the title/aria description");
  assert.match(badge.title, /fast 7/);
  assert.match(badge.title, /exact 3/);
  assert.match(
    badge.title,
    /precise map pending/i,
    "visual_pending means the precise map has not landed yet",
  );
});

test("a fast_exact result displays as authoritative for its revision", () => {
  const badge = realtimeBadgeState(
    revisions,
    classView({
      class: "fast_exact",
      effectiveClass: "fast_exact",
      authoritative: true,
      exactBaseRevision: 3,
    }),
    "live",
  );
  assert.equal(badge.state, "fast_exact");
  assert.equal(badge.text, "Precise");
  assert.doesNotMatch(badge.title, /precise map pending/i);
});

test("a superseded fast result is fenced: stale state, never authoritative wording", () => {
  const badge = realtimeBadgeState(
    revisions,
    classView({ superseded: true, effectiveClass: "visual_pending", targetRevision: 6 }),
    "live",
  );
  assert.equal(badge.state, "stale");
  assert.match(badge.title, /superseded/i);
});

test("exact_reconciled displays the reconciled state", () => {
  const badge = realtimeBadgeState(
    revisions,
    classView({
      class: "exact_reconciled",
      effectiveClass: "exact_reconciled",
      authoritative: true,
      ageMs: 0,
    }),
    "live",
  );
  assert.equal(badge.state, "exact_reconciled");
  assert.equal(badge.text, "Precise");
});

test("connection phases override the class display: catching up, reconnecting, failed", () => {
  assert.equal(realtimeBadgeState(revisions, classView(), "catching_up").state, "catching_up");
  assert.equal(realtimeBadgeState(revisions, classView(), "reconnecting").state, "reconnecting");
  assert.equal(realtimeBadgeState(revisions, classView(), "failed").state, "failed");
});

test("a live subscription with no revisions yet renders an awaiting state, not a fake revision", () => {
  const badge = realtimeBadgeState(
    { workspaceRevision: 0, fastRevision: 0, exactRevision: 0 },
    null,
    "live",
  );
  assert.equal(badge.state, "awaiting");
  assert.doesNotMatch(badge.text, /R0/);
});

test("applyRealtimeBadge writes state, text, and unhides; a null badge hides the pill", () => {
  const pill = { dataset: {}, hidden: true };
  const text = { textContent: "" };

  applyRealtimeBadge(pill, text, realtimeBadgeState(revisions, classView(), "live"));
  assert.equal(pill.dataset.state, "visual_pending");
  assert.equal(pill.hidden, false);
  assert.notEqual(text.textContent, "");

  applyRealtimeBadge(pill, text, null);
  assert.equal(pill.hidden, true, "an unknown workspace hides the pill");
});

test("collabSubscriptionAlive: only a live-ish subscription suppresses resubscription (F1)", () => {
  assert.equal(collabSubscriptionAlive(null), false, "no subscription is not alive");
  assert.equal(collabSubscriptionAlive(undefined), false);
  // Self-healing states own their own recovery — a session-ready during them
  // must NOT tear the subscription down.
  assert.equal(collabSubscriptionAlive({ state: "live" }), true);
  assert.equal(collabSubscriptionAlive({ state: "connecting" }), true);
  assert.equal(collabSubscriptionAlive({ state: "buffering" }), true);
  assert.equal(collabSubscriptionAlive({ state: "reconnecting" }), true);
  assert.equal(collabSubscriptionAlive({ state: "catching_up" }), true);
  // Terminal states never recover on their own — a later session-ready must
  // resubscribe.
  assert.equal(collabSubscriptionAlive({ state: "failed" }), false);
  assert.equal(collabSubscriptionAlive({ state: "closed" }), false);
});

// --- owedDurationText: duration copy for the workspace-owed state --------
// No job → no ETA fields to echo; the only honest duration is the session's
// measured history, and a single sample must not read as a habit.

test("owedDurationText returns null without a usable measurement (never invents)", () => {
  assert.equal(owedDurationText(), null);
  assert.equal(owedDurationText({}), null);
  assert.equal(owedDurationText({ lastExactDurationMs: null }), null);
  assert.equal(owedDurationText({ lastExactDurationMs: undefined }), null);
  assert.equal(owedDurationText({ lastExactDurationMs: -1000 }), null);
  assert.equal(owedDurationText({ lastExactDurationMs: 0 }), null);
  assert.equal(owedDurationText({ lastExactDurationMs: Number.NaN }), null);
  assert.equal(owedDurationText({ lastExactDurationMs: Number.POSITIVE_INFINITY }), null);
  // measuredCount alone never rescues a missing measurement.
  assert.equal(owedDurationText({ measuredCount: 5 }), null);
});

test("owedDurationText: a single sample is a measurement, not a habit", () => {
  assert.equal(owedDurationText({ lastExactDurationMs: 28_000 }), "last update took about 30 s");
  assert.equal(
    owedDurationText({ lastExactDurationMs: 28_000, measuredCount: 1 }),
    "last update took about 30 s",
    "count 1 is still just the last update",
  );
});

test("owedDurationText: two or more measured updates may claim 'usually'", () => {
  assert.equal(
    owedDurationText({ lastExactDurationMs: 28_000, measuredCount: 2 }),
    "usually about 30 s",
  );
  assert.equal(
    owedDurationText({ lastExactDurationMs: 28_000, measuredCount: 7 }),
    "usually about 30 s",
  );
});

test("owedDurationText rounds like formatEtaWait: nearest 10 s below 90 s, minutes at/above", () => {
  assert.equal(owedDurationText({ lastExactDurationMs: 12_000, measuredCount: 2 }), "usually about 10 s");
  assert.equal(owedDurationText({ lastExactDurationMs: 89_000, measuredCount: 2 }), "usually about 90 s");
  // 90 s crosses into minutes: round(90/60) = 2.
  assert.equal(owedDurationText({ lastExactDurationMs: 90_000, measuredCount: 2 }), "usually about 2 min");
  assert.equal(owedDurationText({ lastExactDurationMs: 150_000, measuredCount: 2 }), "usually about 3 min");
});

test("owedDurationText caps the display at 10 min", () => {
  // Exactly 10 min is still "10 min"; anything above reads "10+ min".
  assert.equal(owedDurationText({ lastExactDurationMs: 600_000, measuredCount: 2 }), "usually about 10 min");
  assert.equal(owedDurationText({ lastExactDurationMs: 601_000, measuredCount: 2 }), "usually about 10+ min");
  assert.equal(owedDurationText({ lastExactDurationMs: 42 * 60_000 }), "last update took about 10+ min");
});
