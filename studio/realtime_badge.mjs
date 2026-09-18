// SPDX-License-Identifier: GPL-3.0-only
//
// Collab badge for the studio topbar: renders the realtime client's
// revision triple + result-class view (r2c fast-lane consumption contract,
// see README "Fast-lane consumption contract").
//
// `realtimeBadgeState` is pure and node-testable; `applyRealtimeBadge` is the
// tiny DOM glue. The badge is display-only telemetry — it never gates edits
// (the studio's edit plane is still the exact session; the collaborative
// operation plane is adopted separately). Class words follow the canonical
// D2 glossary (interaction_simulations.md §5 D2): the split-pill lamp on the
// scene header (`renderClassBadge` in app.mjs) owns word+glyph display; this
// rail lamp keeps only the revision triple + phase.

const CLASS_LABELS = Object.freeze({
  fast_exact: "Precise",
  fast_qualified: "Close estimate",
  exact_reconciled: "Precise",
  visual_pending: "Preview",
});

const PHASE_STATES = Object.freeze({
  catching_up: "catching_up",
  reconnecting: "reconnecting",
  buffering: "buffering",
  failed: "failed",
});

/**
 * Compute the badge display state.
 *
 * @param {object|null} revisions `{workspaceRevision, fastRevision,
 *        exactRevision}` from `client.revisionsFor` (null when unknown).
 * @param {object|null} classView `client.classFor` output (null before any
 *        class-bearing event).
 * @param {string} phase subscription/status phase (`live`, `catching_up`,
 *        `reconnecting`, `buffering`, `failed`, ...). Each self-healing
 *        phase keeps its own state — buffering is never conflated with
 *        reconnecting.
 * @returns {object|null} `{state, text, title}` or null when there is no
 *          workspace to show (the pill stays hidden). Unknown wire classes
 *          display conservatively as Preview (effectiveClass rule).
 */
export function realtimeBadgeState(revisions, classView, phase = "live") {
  if (!revisions) return null;
  const phaseState = PHASE_STATES[phase];
  if (phaseState) {
    const phaseText =
      phaseState === "catching_up"
        ? "Collab · catching up"
        : phaseState === "failed"
          ? "Collab · failed"
          : phaseState === "buffering"
            ? "Collab · buffering"
            : "Collab · reconnecting";
    return {
      state: phaseState,
      text: phaseText,
      title: "Collaborative stream is recovering",
    };
  }

  if (!revisions.workspaceRevision && !revisions.fastRevision && !revisions.exactRevision) {
    return {
      state: "awaiting",
      text: "Collab · live",
      title: "Subscribed; no revisions published yet",
    };
  }

  const effective =
    classView && CLASS_LABELS[classView.effectiveClass]
      ? classView.effectiveClass
      : "visual_pending"; // unknown / forward-compat classes display conservatively
  const stale = classView?.superseded === true;
  const state = stale ? "stale" : effective;
  const label = CLASS_LABELS[effective];
  // The visible word carries NO revision number — it names the picture's
  // quality, nothing else. The revision rides the title/aria description.
  const text = `${label}${stale ? " (stale)" : ""}`;
  const title =
    `${label}` +
    ` · workspace ${revisions.workspaceRevision} · fast ${revisions.fastRevision} · exact ${revisions.exactRevision}` +
    (effective === "visual_pending" && !stale ? " · precise map pending" : "") +
    (stale
      ? ` · fast result for R${classView.targetRevision} superseded by a newer revision`
      : "");
  return { state, text, title };
}

/**
 * Duration copy for the workspace-owed state. A server job carries honest
 * ETA fields we can echo, but the workspace-owed state has no job — the only
 * duration we may disclose there is MEASURED history: how long the last
 * completed exact update took on this session. Never invents a number.
 *
 * @param {object} [options]
 * @param {number} options.lastExactDurationMs measured wall time of the last
 *        completed exact update (only a finite positive number is usable).
 * @param {number} [options.measuredCount] exact completions measured this
 *        session. >= 2 may claim a habit ("usually"); a single sample is a
 *        measurement, not a distribution, so it reads as "last update took".
 * @returns {string|null} e.g. "usually about 30 s" or "last update took
 *          about 2 min"; null when there is no usable measurement (caller
 *          keeps its current line).
 */
export function owedDurationText({ lastExactDurationMs, measuredCount = 0 } = {}) {
  if (!Number.isFinite(lastExactDurationMs) || lastExactDurationMs <= 0) return null;
  const seconds = lastExactDurationMs / 1000;
  const lead = measuredCount >= 2 ? "usually " : "last update took ";
  // Same rounding as `formatEtaWait` in app.mjs (reimplemented here so this
  // module stays dependency-free): nearest 10 s below 90 s, else minutes.
  let wait;
  if (seconds > 600) {
    wait = "about 10+ min"; // cap: an owed exact never promises longer than 10 min
  } else if (seconds < 90) {
    wait = `about ${Math.round(seconds / 10) * 10} s`;
  } else {
    wait = `about ${Math.max(1, Math.round(seconds / 60))} min`;
  }
  return lead + wait;
}

/** DOM glue: write a badge state into the pill (fake elements work in tests). */
export function applyRealtimeBadge(pill, textElement, badge) {
  if (!pill) return;
  if (!badge) {
    pill.hidden = true;
    return;
  }
  pill.dataset.state = badge.state;
  pill.hidden = false;
  if (textElement) textElement.textContent = badge.text;
  if (pill.title !== undefined) pill.title = badge.title;
}

/**
 * Whether an existing collab subscription still owns its own recovery
 * (r2c-review F1): only a TERMINAL state (`failed` after the reconnect
 * budget is exhausted, `closed`) justifies tearing it down and
 * resubscribing on a later session-ready — self-healing states
 * (reconnecting, catching_up, buffering) must be left alone.
 */
export function collabSubscriptionAlive(subscription) {
  const state = subscription?.state;
  return typeof state === "string" && state !== "" && state !== "failed" && state !== "closed";
}
