// SPDX-License-Identifier: GPL-3.0-only
//
// Catch-up copy gates: a visitor joining an already-settled shared world (or
// reconnecting) renders from the workspace's revision triple, because no
// LOCAL exact job ever ran. These tests pin the pure copy builders app.mjs
// renders on the catch-up path:
//
//   * catchUpExactStatusText  — the #jobStatusDetail line (never the boot
//     default "No exact job yet" once the world's truth is known).
//   * catchUpPipelineLamps    — honest five-stage bar states for replayed,
//     already-receipted operations (Stamp must not claim "receipt pending").
//   * catchUpBadgeView        — pristine-world badge truth: W0 with a landed
//     baseline IS precise ("Precise"), never "Preview".
//   * exactBadgeEcho          — the #exactBadge echo never says "Connecting"
//     for a world whose revisions are already known.
//   * offlineRailNoteHidden / bootTransportPhase — connected sessions never
//     render the OFFLINE word (rail note hidden; boot transport copy is
//     "Connecting", not "Offline").
//   * railOwedVisibility      — the owed chip is fully hidden at delta 0.
//   * verifyOwedTexts         — ONE source for the owed lamp/detail strings
//     the live path (renderVerifyOwed) and the catch-up path must agree on.
//
// All builders are pure and live outside app.mjs's DOM guard; no server is
// involved.

import test from "node:test";
import assert from "node:assert/strict";

// app.mjs boots its DOM shell only in a browser (the `typeof document` guard
// keeps the module node-importable); its pure copy builders are the seam.
let appModule = null;
async function app() {
  appModule ??= await import("../app.mjs");
  return appModule;
}

// ---------------------------------------------------------------------------
// #jobStatusDetail — workspace-level exact state instead of "No exact job yet"
// ---------------------------------------------------------------------------

test("catch-up into a settled world mirrors markVerifySettled's line", async () => {
  const { catchUpExactStatusText } = await app();
  assert.equal(
    catchUpExactStatusText({ workspaceRevision: 7, fastRevision: 7, exactRevision: 7 }),
    "Precise map · ready",
  );
});

test("catch-up into an owed world reuses the verify-owed detail line", async () => {
  const { catchUpExactStatusText } = await app();
  assert.equal(
    catchUpExactStatusText({ workspaceRevision: 7, fastRevision: 7, exactRevision: 5 }),
    "Preview · precise map pending",
  );
});

test("a pristine world (W0) whose baseline landed is Precise map · ready", async () => {
  const { catchUpExactStatusText } = await app();
  assert.equal(
    catchUpExactStatusText(
      { workspaceRevision: 0, fastRevision: 0, exactRevision: 0 },
      { hasBaselineExact: true },
    ),
    "Precise map · ready",
  );
});

test("a pristine world without a landed baseline keeps whatever line it had", async () => {
  const { catchUpExactStatusText } = await app();
  assert.equal(
    catchUpExactStatusText(
      { workspaceRevision: 0, fastRevision: 0, exactRevision: 0 },
      { hasBaselineExact: false },
    ),
    null,
  );
});

// ---------------------------------------------------------------------------
// Five-stage pipeline — honest lamps for replayed, already-receipted operations
// ---------------------------------------------------------------------------

test("settled catch-up: verify done at the exact revision, earlier lamps receipted/applied/fast", async () => {
  const { catchUpPipelineLamps } = await app();
  assert.deepEqual(catchUpPipelineLamps({ workspaceRevision: 7, fastRevision: 7, exactRevision: 7 }), {
    focus: "verify",
    verifyOwed: false,
    smalls: {
      stamp: "receipted · R7",
      share: "applied · R7",
      paint: "fast · R7",
      verify: "precise · R7",
    },
  });
});

test("owed catch-up: paint done, verify owed with the renderVerifyOwed lamp text", async () => {
  const { catchUpPipelineLamps, verifyOwedTexts } = await app();
  const lamps = catchUpPipelineLamps({ workspaceRevision: 7, fastRevision: 6, exactRevision: 5 });
  assert.equal(lamps.focus, "paint");
  assert.equal(lamps.verifyOwed, true);
  assert.equal(
    lamps.smalls.verify,
    "Preview shown — the precise map is being prepared.",
  );
  // The catch-up lamp must be the SAME string the live owed path renders.
  assert.equal(
    lamps.smalls.verify,
    verifyOwedTexts({ workspaceRevision: 7, fastRevision: 6, exactRevision: 5 }).lamp,
  );
  assert.equal(lamps.smalls.stamp, "receipted · R7");
  assert.equal(lamps.smalls.share, "applied · R7");
  assert.equal(lamps.smalls.paint, "fast · R6");
});

test("owed catch-up with no fast frames yet settles at Paint with no paint small", async () => {
  const { catchUpPipelineLamps } = await app();
  // W>0, F=0: the world holds edits but the fast lane has published nothing —
  // the bar settles at Paint without a paint lamp text (never a stale or
  // invented F), and Verify still carries the owed line.
  const lamps = catchUpPipelineLamps({ workspaceRevision: 7, fastRevision: 0, exactRevision: 5 });
  assert.equal(lamps.focus, "paint");
  assert.equal(lamps.verifyOwed, true);
  assert.equal("paint" in lamps.smalls, false, "no fast revision known — no paint small");
  assert.equal(lamps.smalls.verify, "Preview shown — the precise map is being prepared.");
});

test("pristine worlds leave the bar alone (no receipts to claim)", async () => {
  const { catchUpPipelineLamps } = await app();
  assert.equal(catchUpPipelineLamps({ workspaceRevision: 0, fastRevision: 0, exactRevision: 0 }), null);
  assert.equal(catchUpPipelineLamps(null), null);
});

// ---------------------------------------------------------------------------
// Result-class badge — pristine truth and the echo's connecting flicker
// ---------------------------------------------------------------------------

test("pristine world with a baseline displays Precise, not Preview", async () => {
  const { catchUpBadgeView } = await app();
  assert.deepEqual(
    catchUpBadgeView(
      { workspaceRevision: 0, exactRevision: 0 },
      { hasBaselineExact: true },
    ),
    { word: "Precise", glyph: "■", state: "exact_reconciled" },
  );
});

test("non-pristine and baseline-less worlds get no badge override", async () => {
  const { catchUpBadgeView } = await app();
  assert.equal(
    catchUpBadgeView({ workspaceRevision: 7, exactRevision: 5 }, { hasBaselineExact: true }),
    null,
  );
  assert.equal(
    catchUpBadgeView({ workspaceRevision: 0, exactRevision: 0 }, { hasBaselineExact: false }),
    null,
  );
});

test("the exact-badge echo never claims Connecting once revisions are known", async () => {
  const { exactBadgeEcho } = await app();
  // Caught up into an owed world with no class-bearing frame yet: the pixels
  // are a preview — "Connecting" (the observed flicker) overstates transport.
  assert.deepEqual(
    exactBadgeEcho({ rawClass: null, superseded: false, revisionsKnown: true }),
    { text: "Preview", state: "preview" },
  );
  // Truly nothing known yet (fresh subscribe, empty snapshot): connecting is
  // the honest echo.
  assert.deepEqual(
    exactBadgeEcho({ rawClass: null, superseded: false, revisionsKnown: false }),
    { text: "Connecting", state: "connecting" },
  );
  // The pristine baseline-exact catch-up echoes Exact like any settled truth.
  assert.deepEqual(
    exactBadgeEcho({ rawClass: null, superseded: false, revisionsKnown: false, catchUpExact: true }),
    { text: "Exact", state: "exact" },
  );
  // Live classes keep the existing echo vocabulary.
  assert.deepEqual(
    exactBadgeEcho({ rawClass: "exact_reconciled", superseded: false, revisionsKnown: true }),
    { text: "Exact", state: "exact" },
  );
  assert.deepEqual(
    exactBadgeEcho({ rawClass: "fast_qualified", superseded: false, revisionsKnown: true }),
    { text: "Estimate", state: "preview" },
  );
});

// ---------------------------------------------------------------------------
// Rail honesty — the OFFLINE word never renders while connected
// ---------------------------------------------------------------------------

test("the rail offline note hides for connecting and connected sessions", async () => {
  const { offlineRailNoteHidden } = await app();
  assert.equal(offlineRailNoteHidden("connecting"), true);
  assert.equal(offlineRailNoteHidden("connected"), true);
  assert.equal(offlineRailNoteHidden("failed"), false, "a failed connect IS the offline preview");
  assert.equal(offlineRailNoteHidden("local"), false, "offline mode owns the note");
});

test("connected-mode boot transport copy says Connecting, never Offline", async () => {
  const { bootTransportPhase } = await app();
  assert.equal(bootTransportPhase(true), "connecting");
  assert.equal(bootTransportPhase(false), "offline");
});

test("the owed chip hides entirely at delta 0 (no 'updating +0')", async () => {
  const { railOwedVisibility } = await app();
  assert.deepEqual(railOwedVisibility({ workspaceRevision: 7, exactRevision: 7 }), {
    hidden: true,
    text: null,
  });
  assert.deepEqual(railOwedVisibility(null), { hidden: true, text: null });
  assert.deepEqual(railOwedVisibility({ workspaceRevision: 7, exactRevision: 5 }), {
    hidden: false,
    text: "updating +2",
  });
});

// ---------------------------------------------------------------------------
// One source of truth for the owed strings (live path == catch-up path)
// ---------------------------------------------------------------------------

test("verifyOwedTexts carries the exact live-path strings", async () => {
  const { verifyOwedTexts } = await app();
  assert.deepEqual(verifyOwedTexts({ workspaceRevision: 7, fastRevision: 6, exactRevision: 5 }), {
    lamp: "Preview shown — the precise map is being prepared.",
    detail: "Preview · precise map pending",
  });
});
