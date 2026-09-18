// SPDX-License-Identifier: GPL-3.0-only
//
// Wave-3 realtime UX copy/data gates: the pure builders app.mjs renders on
// its realtime surfaces, all pinned here so the DOM only wires never invents:
//
//   * ledgerStageDots     — the per-row five-stage micro strip (Stamp →
//     Share → Paint → Verify; Sketch is done the moment a row exists).
//     Verify stays owed-style hollow: the client never observes a ROW-level
//     exact settle, so the strip must not claim one.
//   * receiptQuoteText    — the single quotable string a receipt click copies
//     ("{operation_id} · seq {n} · epoch {id?}").
//   * ladderLegendLine    — the one-line degradation-ladder key the result
//     class badge tooltip carries where the symbols are used.
//   * transportPillLabel  — "Reconnecting · attempt N of M" when the realtime
//     client's bounded reconnect budget carries counts, the plain label when
//     they are unknown (null-safe).
//   * withCapacityReassurance — the rate-limit cooldown clause, append-once
//     so idempotent re-announces never double it.
//   * treeForLegacyRecompute — the legacy recompute anchor gate: a tree the
//     exact funnel can actually address, or null when every tree is
//     realtime-native (the POST would 404 — the caller must skip it).
//
// All builders are pure and live outside app.mjs's DOM guard; no server, no
// DOM, no clock is involved.

import test from "node:test";
import assert from "node:assert/strict";

// app.mjs boots its DOM shell only in a browser (the `typeof document` guard
// keeps the module node-importable); its pure copy builders are the seam.
// Loaded lazily so a missing export fails the individual assertions instead
// of unloading the whole file.
let appModule = null;
async function app() {
  appModule ??= await import("../app.mjs");
  return appModule;
}

// ---------------------------------------------------------------------------
// ledgerStageDots — per-row five-stage strip
// ---------------------------------------------------------------------------

test("a sending row shows stamp active, everything after it owed", async () => {
  const { ledgerStageDots } = await app();
  const dots = ledgerStageDots({ state: "sending" });
  assert.deepEqual(
    dots.map(({ stage, state }) => [stage, state]),
    [
      ["sketch", "done"],
      ["stamp", "active"],
      ["share", "owed"],
      ["paint", "owed"],
      ["verify", "owed"],
    ],
  );
});

test("an applied row settles stamp and share, paint waits for its revision", async () => {
  const { ledgerStageDots } = await app();
  const dots = ledgerStageDots({ state: "applied", paintedRevision: undefined });
  const byStage = Object.fromEntries(dots.map(({ stage, state }) => [stage, state]));
  assert.equal(byStage.stamp, "done", "the ack fence settles stamp");
  assert.equal(byStage.share, "done", "the ack fence settles share");
  assert.equal(byStage.paint, "owed", "applied but not yet painted");
  assert.equal(byStage.verify, "owed", "verify stays hollow");
});

test("a painted row (paintedRevision set) marks paint done — verify still owed", async () => {
  const { ledgerStageDots } = await app();
  const byStage = Object.fromEntries(
    ledgerStageDots({ state: "applied", paintedRevision: 7 }).map(({ stage, state }) => [
      stage,
      state,
    ]),
  );
  assert.equal(byStage.paint, "done");
  // Row-level exact settle is a server fact the client never observes per
  // operation: the strip must keep the owed-style hollow dot forever.
  assert.equal(byStage.verify, "owed");
});

test("a held row shows stamp owed — nothing is in flight while held", async () => {
  const { ledgerStageDots } = await app();
  const byStage = Object.fromEntries(
    ledgerStageDots({ state: "held", heldKind: "burst" }).map(({ stage, state }) => [
      stage,
      state,
    ]),
  );
  assert.equal(byStage.stamp, "owed");
  assert.equal(byStage.share, "owed");
});

test("a refused row fails at stamp and never implies later stages", async () => {
  const { ledgerStageDots } = await app();
  const byStage = Object.fromEntries(
    ledgerStageDots({ state: "refused" }).map(({ stage, state }) => [stage, state]),
  );
  assert.equal(byStage.stamp, "failed");
  assert.equal(byStage.share, "owed");
  assert.equal(byStage.paint, "owed");
});

test("a null-ish row degrades to all-owed stages instead of throwing", async () => {
  const { ledgerStageDots } = await app();
  const dots = ledgerStageDots({});
  assert.equal(dots.length, 5);
  assert.ok(dots.every(({ state }) => state === "owed" || state === "done"));
});

// ---------------------------------------------------------------------------
// receiptQuoteText — the click-to-copy payload
// ---------------------------------------------------------------------------

test("the full receipt quotes operation id, seq, and epoch in one line", async () => {
  const { receiptQuoteText } = await app();
  assert.equal(
    receiptQuoteText({ operationId: "studio-ab12:op-3", serverSequence: 12, epochId: "e7" }),
    "studio-ab12:op-3 · seq 12 · epoch e7",
  );
});

test("the receipt omits the epoch segment when there is none", async () => {
  const { receiptQuoteText } = await app();
  assert.equal(
    receiptQuoteText({ operationId: "studio-ab12:op-3", serverSequence: 12, epochId: null }),
    "studio-ab12:op-3 · seq 12",
  );
  // Pre-ack rows (no server sequence yet) still copy the one durable id.
  assert.equal(
    receiptQuoteText({ operationId: "studio-ab12:op-3", serverSequence: null, epochId: null }),
    "studio-ab12:op-3",
  );
});

// ---------------------------------------------------------------------------
// ladderLegendLine — the badge tooltip's symbol key
// ---------------------------------------------------------------------------

test("the ladder legend line names all four symbols in ladder order", async () => {
  const { ladderLegendLine } = await app();
  assert.equal(
    ladderLegendLine(),
    "Picture quality — ■ Precise result  ● Fast precise  ◐ Close estimate  ◌ Preview",
  );
});

// ---------------------------------------------------------------------------
// transportPillLabel — reconnect budget honesty
// ---------------------------------------------------------------------------

test("reconnecting with known counts renders the bounded attempt line", async () => {
  const { transportPillLabel } = await app();
  assert.equal(
    transportPillLabel("reconnecting", { attempt: 2, maxAttempts: 5 }),
    "Reconnecting · attempt 2 of 5",
  );
  assert.equal(
    transportPillLabel("reconnecting", { attempt: 1, maxAttempts: 3 }),
    "Reconnecting · attempt 1 of 3",
  );
});

test("reconnecting without counts stays the plain label — null-safe", async () => {
  const { transportPillLabel } = await app();
  assert.equal(transportPillLabel("reconnecting"), "Reconnecting");
  assert.equal(transportPillLabel("reconnecting", {}), "Reconnecting");
  assert.equal(
    transportPillLabel("reconnecting", { attempt: null, maxAttempts: null }),
    "Reconnecting",
  );
  assert.equal(transportPillLabel("reconnecting", { attempt: 0, maxAttempts: null }), "Reconnecting");
});

test("counts never leak into the other phases", async () => {
  const { transportPillLabel } = await app();
  assert.equal(transportPillLabel("live", { attempt: 2, maxAttempts: 5 }), "Live");
  assert.equal(transportPillLabel("catching_up", { attempt: 2, maxAttempts: 5 }), "Catching up");
  assert.equal(transportPillLabel("closed", { attempt: 2, maxAttempts: 5 }), "Offline");
  assert.equal(transportPillLabel("nonsense-phase", { attempt: 2 }), "Offline");
});

// ---------------------------------------------------------------------------
// withCapacityReassurance — append-once cooldown clause
// ---------------------------------------------------------------------------

test("the rate-limit reassurance clause appends exactly once", async () => {
  const { withCapacityReassurance, RATE_LIMIT_REASSURANCE } = await app();
  const once = withCapacityReassurance("Scenario rate limit reached — next edit in 2 s.");
  assert.equal(
    once,
    `Scenario rate limit reached — next edit in 2 s. ${RATE_LIMIT_REASSURANCE}`,
  );
  assert.equal(withCapacityReassurance(once), once, "a second pass is a no-op");
  assert.equal(
    once.split(RATE_LIMIT_REASSURANCE).length - 1,
    1,
    "the clause appears exactly once",
  );
});

// ---------------------------------------------------------------------------
// treeForLegacyRecompute — the legacy recompute anchor gate
// ---------------------------------------------------------------------------

test("the anchor prefers the selected tree only when the funnel can address it", async () => {
  const { treeForLegacyRecompute } = await app();
  const legacyA = { id: "t1", realtimeNative: false };
  const legacyB = { id: "t2", realtimeNative: false };
  const native = { id: "t3", realtimeNative: true };
  assert.equal(treeForLegacyRecompute([legacyA, legacyB, native], "t2"), legacyB);
  // A realtime-native SELECTION cannot anchor: fall to a legacy tree.
  assert.equal(treeForLegacyRecompute([legacyA, native], "t3"), legacyA);
  assert.equal(treeForLegacyRecompute([legacyA, native], null), legacyA);
});

test("an all-realtime-native scene yields NO anchor — the caller must skip the POST", async () => {
  const { treeForLegacyRecompute } = await app();
  const nativeA = { id: "t1", realtimeNative: true };
  const nativeB = { id: "t2", realtimeNative: true };
  assert.equal(treeForLegacyRecompute([nativeA, nativeB], "t1"), null);
  assert.equal(treeForLegacyRecompute([nativeA], null), null);
  assert.equal(treeForLegacyRecompute([], null), null);
});

// ---------------------------------------------------------------------------
// exactHourVerified / solarTimeSkipAnnounce — the slider skip disclosure
// ---------------------------------------------------------------------------

test("an hour is verified only when its exact plane is covered AND the exact lane caught up", async () => {
  const { exactHourVerified } = await app();
  const covered = new Set([9, 12]);
  assert.equal(
    exactHourVerified({ displayedHour: 9, coveredHours: covered, exactRevision: 4, workspaceRevision: 4 }),
    true,
    "covered hour, exact lane current",
  );
  assert.equal(
    exactHourVerified({ displayedHour: 9, coveredHours: covered, exactRevision: 3, workspaceRevision: 4 }),
    false,
    "exact lane behind the workspace — the covered plane may be stale",
  );
  assert.equal(
    exactHourVerified({ displayedHour: 8, coveredHours: covered, exactRevision: 4, workspaceRevision: 4 }),
    false,
    "hour never covered by the newest exact patch",
  );
  // Null-safe: no revisions known yet means nothing can be claimed verified.
  assert.equal(exactHourVerified({ displayedHour: 9, coveredHours: null }), false);
  assert.equal(exactHourVerified({}), false);
});

test("the skip announce claims pending or verified — never a refresh that isn't coming", async () => {
  const { solarTimeSkipAnnounce } = await app();
  assert.equal(
    solarTimeSkipAnnounce("09:00", true),
    "Solar time changed to 09:00 — this hour's precise picture is already verified.",
  );
  assert.equal(
    solarTimeSkipAnnounce("14:00", false),
    "Solar time changed to 14:00 — this hour's precise picture is being prepared.",
  );
});
