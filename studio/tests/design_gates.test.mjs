// SPDX-License-Identifier: GPL-3.0-only
//
// Gate tests for the design-tool mapping layer (UI-002 debounce, contract
// tree mapping, patch application math, colorization).

import test from "node:test";
import assert from "node:assert/strict";

import {
  TREE_PRESETS,
  computeExactMetrics,
  exactBadgeForConnection,
  makeTree,
  treeToApiObject,
  apiObjectToTree,
  editItemForChange,
  selectTimeIndex,
  nearestTimeIndexPosition,
  gridWindowToRect,
  createPropertyEditor,
} from "../model.mjs";
import { applyExactPatch, cubeTimePlane } from "../solver_kernel.mjs";
import { colorizeValues, colorizePreviewValues } from "../renderer.mjs";

// ---------------------------------------------------------------------------
// Manual clock
// ---------------------------------------------------------------------------

function makeFakeTimers() {
  const tasks = [];
  let now = 0;
  let nextId = 1;
  return {
    setTimeout(fn, delay) {
      const task = { id: nextId, fn, at: now + Number(delay) || now };
      nextId += 1;
      tasks.push(task);
      return task.id;
    },
    clearTimeout(id) {
      const index = tasks.findIndex((task) => task.id === id);
      if (index >= 0) tasks.splice(index, 1);
    },
    advance(ms) {
      const target = now + ms;
      for (;;) {
        tasks.sort((a, b) => a.at - b.at || a.id - b.id);
        const next = tasks[0];
        if (!next || next.at > target) break;
        now = next.at;
        tasks.shift();
        next.fn();
      }
      now = target;
    },
    now: () => now,
    pendingCount: () => tasks.length,
  };
}

function makeRecordingEditor({ previewDelayMs = 150, commitDelayMs = 480 } = {}) {
  const timers = makeFakeTimers();
  const previews = [];
  const commits = [];
  const editor = createPropertyEditor({
    preview: (oldTree, newTree) => previews.push({ at: timers.now(), oldTree, newTree }),
    commit: (oldTree, newTree) => commits.push({ at: timers.now(), oldTree, newTree }),
    previewDelayMs,
    commitDelayMs,
    setTimeoutFn: timers.setTimeout,
    clearTimeoutFn: timers.clearTimeout,
  });
  return { timers, editor, previews, commits };
}

function withHeight(tree, heightM) {
  return { ...tree, heightM };
}

// ---------------------------------------------------------------------------
// UI-002: property debounce
// ---------------------------------------------------------------------------

test("rapid slider input coalesces to one preview and one committed edit", () => {
  const { timers, editor, previews, commits } = makeRecordingEditor();
  const tree = makeTree("broad", 0.5, 0.4, 2);

  editor.input(withHeight(tree, 10)); // t=0: debounce interval opens here
  timers.advance(30);
  editor.input(withHeight(tree, 11));
  timers.advance(50);
  editor.input(withHeight(tree, 12));
  timers.advance(40);
  editor.input(withHeight(tree, 13)); // t=120: latest value
  assert.equal(previews.length, 0, "no preview before the trailing edge");
  assert.equal(commits.length, 0, "no commit before the trailing window closes");

  timers.advance(600); // preview at 270, commit at 600

  assert.equal(previews.length, 1, "rapid input produces exactly one preview");
  assert.equal(previews[0].oldTree.heightM, 10, "preview keeps the interval-start state");
  assert.equal(previews[0].newTree.heightM, 13, "preview uses the latest value");

  assert.equal(commits.length, 1, "rapid input commits exactly one replacement (UI-001/002)");
  assert.equal(commits[0].oldTree.heightM, 10, "commit keeps the interval-start tree (spec)");
  assert.equal(commits[0].newTree.heightM, 13, "the final slider value wins");
});

test("the commit lands inside the spec's 400-600 ms trailing window", () => {
  const { timers, editor, previews, commits } = makeRecordingEditor();
  const tree = makeTree("broad", 0.5, 0.4, 3);

  editor.input(withHeight(tree, 10));
  timers.advance(200);
  editor.input(withHeight(tree, 14)); // restart the trailing edge
  const lastInputAt = timers.now();

  timers.advance(399);
  assert.equal(commits.length, 0, "commit must not fire before 400 ms of quiet");
  timers.advance(201);
  assert.equal(commits.length, 1);
  const elapsed = commits[0].at - lastInputAt;
  assert.ok(
    elapsed >= 400 && elapsed <= 600,
    `commit fired ${elapsed} ms after the last input; spec requires 400-600 ms`,
  );
  assert.ok(previews.length >= 1, "a preview happens while waiting");
  assert.equal(timers.pendingCount(), 0, "both timers are settled after commit");
});

test("separate quiet gaps produce separate commits with correct start states", () => {
  const { timers, editor, commits } = makeRecordingEditor();
  const tree = makeTree("broad", 0.5, 0.4, 4);

  editor.input(withHeight(tree, 10));
  timers.advance(600); // round 1 commits
  editor.input(withHeight(tree, 20)); // round 2 opens from round 1's committed tree
  timers.advance(80);
  editor.input(withHeight(tree, 25));
  timers.advance(600);

  assert.equal(commits.length, 2);
  assert.equal(commits[0].oldTree.heightM, 10);
  assert.equal(commits[0].newTree.heightM, 10, "single input commits the same tree");
  assert.equal(commits[1].oldTree.heightM, 20, "round 2 opens from round 1's committed state");
  assert.equal(commits[1].newTree.heightM, 25);
});

test("cancel drops pending preview and commit", () => {
  const { timers, editor, previews, commits } = makeRecordingEditor();
  const tree = makeTree("broad", 0.5, 0.4, 5);

  editor.input(withHeight(tree, 10));
  timers.advance(100);
  editor.cancel();
  timers.advance(2000);

  assert.equal(previews.length, 0);
  assert.equal(commits.length, 0);
  assert.equal(timers.pendingCount(), 0);
});

test("a pointer drag mid-debounce rebases the pending commit onto the new position", () => {
  const { timers, editor, previews, commits } = makeRecordingEditor();
  const tree = makeTree("broad", 0.5, 0.4, 6);

  editor.input(withHeight(tree, 14)); // property debounce round opens at u=0.5
  timers.advance(100);
  // The user drops the same tree at a new position before the 480 ms commit.
  const dragged = { ...withHeight(tree, 14), u: 0.8, v: 0.62 };
  editor.rebase(dragged);
  timers.advance(600);

  assert.equal(commits.length, 1);
  assert.equal(commits[0].newTree.u, 0.8, "the pending property commit carries the dragged position");
  assert.equal(commits[0].newTree.v, 0.62);
  assert.equal(commits[0].newTree.heightM, 14, "the latest property value still wins");
  assert.equal(commits[0].oldTree.u, 0.5, "the interval-start tree is kept for the old region");
  assert.equal(previews.length, 1);

  // Rebase without a pending round is a no-op (no phantom commit later).
  editor.rebase({ ...tree, u: 0.99 });
  timers.advance(1000);
  assert.equal(commits.length, 1);
});

// ---------------------------------------------------------------------------
// Contract tree mapping
// ---------------------------------------------------------------------------

test("treeToApiObject emits strict contract fields only", () => {
  const tree = makeTree("evergreen", 0.25, 0.75, 7);
  const apiTree = treeToApiObject(tree);

  assert.deepEqual(Object.keys(apiTree).sort(), [
    "canopy_diameter_m",
    "component_type",
    "height_m",
    "metadata",
    "phenology",
    "transmissivity",
    "tree_id",
    "trunk_ratio",
    "u",
    "v",
  ]);
  assert.equal(apiTree.tree_id, tree.id);
  assert.equal(apiTree.u, 0.25);
  assert.equal(apiTree.v, 0.75);
  assert.equal(apiTree.height_m, tree.heightM);
  assert.equal(apiTree.canopy_diameter_m, tree.canopyDiameterM);
  assert.equal(apiTree.component_type, "broad_canopy");
  // Presentation-only attributes ride in metadata, not the strict schema.
  assert.equal(apiTree.metadata.label, tree.label);
  assert.equal(apiTree.metadata.preset, "evergreen");
  assert.equal(treeToApiObject(null), null);
});

test("apiObjectToTree restores geometry and presentation metadata", () => {
  const original = makeTree("shade", 0.6, 0.3, 9);
  const restored = apiObjectToTree(treeToApiObject(original), 9);
  assert.equal(restored.id, original.id);
  assert.equal(restored.label, original.label);
  assert.equal(restored.preset, "shade");
  assert.equal(restored.u, original.u);
  assert.equal(restored.v, original.v);
  assert.equal(restored.heightM, original.heightM);
  assert.equal(restored.canopyDiameterM, original.canopyDiameterM);

  // Server-side tree without our metadata still maps with sane defaults.
  const foreign = apiObjectToTree(
    {
      tree_id: "t-srv",
      component_type: "broad_canopy",
      u: 0.1,
      v: 0.2,
      height_m: 15,
      canopy_diameter_m: 8,
      trunk_ratio: 0.25,
      transmissivity: 0.02,
      phenology: "deciduous",
    },
    3,
  );
  assert.equal(foreign.id, "t-srv");
  assert.equal(foreign.preset, "broad");
  assert.ok(foreign.label.endsWith("03"));
  assert.ok(TREE_PRESETS[foreign.preset], "fallback preset must exist");
});

test("editItemForChange shapes add, update, move, and delete items", () => {
  const tree = makeTree("broad", 0.5, 0.5, 2);
  const moved = { ...tree, u: 0.7, v: 0.6 };

  assert.deepEqual(editItemForChange("add", null, tree), {
    operation: "add",
    tree: treeToApiObject(tree),
  });
  assert.deepEqual(editItemForChange("update", tree, moved), {
    operation: "update",
    tree: treeToApiObject(moved),
  });
  assert.deepEqual(editItemForChange("move", tree, moved), {
    operation: "move",
    tree: treeToApiObject(moved),
  });
  assert.deepEqual(editItemForChange("delete", tree, null), {
    operation: "delete",
    tree_id: tree.id,
  });
});

// ---------------------------------------------------------------------------
// Time-index and window math
// ---------------------------------------------------------------------------

test("selectTimeIndex returns the value while nearestTimeIndexPosition returns the slot", () => {
  const timeIndices = [8, 12, 16];
  assert.equal(selectTimeIndex(timeIndices, 13), 12);
  assert.equal(nearestTimeIndexPosition(timeIndices, 13), 1);
  assert.equal(selectTimeIndex(timeIndices, 17), 16);
  assert.equal(nearestTimeIndexPosition(timeIndices, 17), 2);
  assert.equal(selectTimeIndex([], 12), null);
  assert.equal(nearestTimeIndexPosition([], 12), -1);
});

test("gridWindowToRect maps half-open grid windows to site UV", () => {
  const rect = gridWindowToRect({ rowStart: 10, rowStop: 30, colStart: 20, colStop: 60 }, 100, 200);
  assert.deepEqual(rect, { minU: 0.1, maxU: 0.3, minV: 0.1, maxV: 0.3 });
});

test("computeExactMetrics uses the decoded camelCase window, never the raw manifest one", () => {
  // Raw server manifest: snake_case window, metrics WITHOUT window_fraction —
  // reading rowStart off this object yields NaN (the M-2 regression).
  const manifest = {
    scene_version: 7,
    metrics: {
      mean_utci_delta_c: -1.5,
      peak_utci_delta_c: -4.25,
      improved_area_m2: 9123,
      duration_ms: 1400,
    },
    window: { row_start: 4, row_stop: 10, col_start: 2, col_stop: 8 },
  };
  const decodedWindow = { rowStart: 4, rowStop: 10, colStart: 2, colStop: 8 };

  const derived = computeExactMetrics(manifest, decodedWindow, { rows: 50, cols: 40 });
  assert.equal(derived.windowCells, 36);
  assert.equal(derived.totalCells, 2000);
  assert.equal(derived.fraction, 36 / 2000);
  assert.equal(derived.sceneVersion, 7);
  assert.equal(derived.meanDeltaC, -1.5);
  assert.equal(derived.durationMs, 1400);
  assert.deepEqual(derived.window, decodedWindow);

  // A different decoded window wins over the manifest's copy (they can
  // legitimately differ when the patch window is re-derived client-side).
  const shifted = computeExactMetrics(manifest, { rowStart: 0, rowStop: 5, colStart: 0, colStop: 4 }, { rows: 50, cols: 40 });
  assert.equal(shifted.windowCells, 20);

  // Missing decoded window: no NaN, callers skip window-derived outputs.
  const bare = computeExactMetrics(manifest, null, { rows: 50, cols: 40 });
  assert.equal(bare.windowCells, null);
  assert.equal(bare.fraction, null);
  assert.equal(bare.window, null);

  // window_fraction, when present, is still honored.
  const withFraction = computeExactMetrics(
    { scene_version: 8, metrics: { window_fraction: 0.5 } },
    decodedWindow,
    { rows: 50, cols: 40 },
  );
  assert.equal(withFraction.fraction, 0.5);
});

test("exact badge claims Exact only after a verified result, never while connecting (R-4)", () => {
  // Connecting: nothing has been verified yet — the badge must stay neutral,
  // not claim "Exact" before the first server result lands.
  const connecting = exactBadgeForConnection("connecting");
  assert.equal(connecting.text, "Connecting");
  assert.equal(connecting.state, "connecting");
  assert.notEqual(connecting.state, "exact");

  // "Exact" only once connected, i.e. after the verified baseline applied.
  assert.deepEqual(exactBadgeForConnection("connected"), { text: "Exact", state: "exact" });

  // Offline and failed-connect states stay honest about being a preview.
  assert.deepEqual(exactBadgeForConnection("local"), { text: "Preview", state: "preview" });
  assert.deepEqual(exactBadgeForConnection("failed"), { text: "Preview", state: "preview" });
});

// ---------------------------------------------------------------------------
// Exact patch application
// ---------------------------------------------------------------------------

function makeCube(timeSteps, rows, cols, fill = 0) {
  return new Float32Array(timeSteps * rows * cols).fill(fill);
}

test("applyExactPatch writes the window plane at the nearest covered hour", () => {
  const timeSteps = 24;
  const rows = 10;
  const cols = 12;
  const cube = makeCube(timeSteps, rows, cols, 20);
  const window = { rowStart: 2, rowStop: 5, colStart: 3, colStop: 7 };
  const values = new Float32Array(3 * 12).map((_, index) => 40 + index);

  const written = applyExactPatch(cube, values, {
    window,
    timeIndices: [12],
    hour: 12,
    timeSteps,
    gridWidth: cols,
    gridHeight: rows,
  });

  assert.equal(written, 12);
  const plane = cubeTimePlane(cube, 12, cols, rows);
  assert.equal(plane[2 * cols + 3], 40); // window origin
  assert.equal(plane[4 * cols + 6], values[2 * 4 + 3]); // last window cell
  assert.equal(plane[0], 20, "cells outside the window are untouched");
  assert.equal(plane[5 * cols + 3], 20);
  // The displayed-hour guard uses the nearest slot of a multi-hour patch.
  const multi = applyExactPatch(cube, values, {
    window,
    timeIndices: [8, 14],
    hour: 15,
    timeSteps,
    gridWidth: cols,
    gridHeight: rows,
  });
  assert.equal(multi, 14, "|14-15| < |8-15| picks slot 1");
  assert.equal(cubeTimePlane(cube, 14, cols, rows)[2 * cols + 3], values[12]);
});

test("applyExactPatch rejects uncovered hours and out-of-grid windows", () => {
  const cube = makeCube(24, 8, 8);
  const values = new Float32Array(16);

  assert.equal(
    applyExactPatch(cube, values, {
      window: { rowStart: 0, rowStop: 4, colStart: 0, colStop: 4 },
      timeIndices: [],
      hour: 12,
      timeSteps: 24,
      gridWidth: 8,
      gridHeight: 8,
    }),
    null,
    "an empty time index list covers no hour",
  );
  assert.equal(
    applyExactPatch(cube, values, {
      window: { rowStart: 0, rowStop: 4, colStart: 0, colStop: 4 },
      timeIndices: [99],
      hour: 12,
      timeSteps: 24,
      gridWidth: 8,
      gridHeight: 8,
    }),
    null,
    "a time index beyond the cube is refused",
  );
  assert.throws(() =>
    applyExactPatch(cube, values, {
      window: { rowStart: 6, rowStop: 12, colStart: 0, colStop: 4 },
      timeIndices: [12],
      hour: 12,
      timeSteps: 24,
      gridWidth: 8,
      gridHeight: 8,
    }),
    /outside the site grid/,
  );
});

// ---------------------------------------------------------------------------
// Colorization (renderer seam)
// ---------------------------------------------------------------------------

test("colorizeValues maps UTCI values to clamped RGBA bytes", () => {
  const pixels = colorizeValues(Float32Array.from([20, 30, 37, 44, 60]), {
    minimum: 30,
    maximum: 44,
    alpha: 200,
  });
  assert.ok(pixels instanceof Uint8Array);
  assert.equal(pixels.length, 5 * 4);
  assert.equal(pixels[3], 200, "alpha is honored");

  const colorAt = (index) => [pixels[index * 4], pixels[index * 4 + 1], pixels[index * 4 + 2]];
  // The ramp's first stop paints everything at or below the minimum.
  assert.deepEqual(colorAt(0), [41, 153, 146], "values below the minimum clamp to the cold stop");
  assert.deepEqual(colorAt(1), [41, 153, 146]);
  // The last stop paints everything at or above the maximum.
  assert.deepEqual(colorAt(3), [221, 72, 62], "values above the maximum clamp to the hot stop");
  assert.deepEqual(colorAt(4), [221, 72, 62]);

  // A mid-scale value lands strictly between the endpoint colors.
  const mid = colorAt(2);
  assert.notDeepEqual(mid, colorAt(1));
  assert.notDeepEqual(mid, colorAt(3));
  assert.ok(mid[0] > 41 && mid[0] < 221, "red channel rises through the scale");

  // Non-finite cells decode as transparent nodata, not a wrong color.
  const nodata = colorizeValues(Float32Array.from([Number.NaN]), { minimum: 30, maximum: 44 });
  assert.deepEqual([...nodata], [0, 0, 0, 0]);
});

test("colorizePreviewValues is a calmer ramp than the exact texture's", () => {
  // The preview kernel's layer composites ON TOP of the exact texture, which
  // read as MORE saturated than the server-exact picture it previews — the
  // preview→exact swap jarred. The preview ramp must keep the exact ramp's
  // hue ORDER at lower chroma: every anchor lighter (closer to white) with a
  // smaller channel spread, so the sketch never out-shouts the result.
  const exact = colorizeValues(Float32Array.from([30, 37, 44]), { minimum: 30, maximum: 44 });
  const preview = colorizePreviewValues(Float32Array.from([30, 37, 44]), {
    minimum: 30,
    maximum: 44,
  });
  const colorAt = (pixels, index) => [
    pixels[index * 4],
    pixels[index * 4 + 1],
    pixels[index * 4 + 2],
  ];
  for (const index of [0, 1, 2]) {
    const a = colorAt(exact, index);
    const b = colorAt(preview, index);
    assert.ok(
      b.every((channel, i) => channel >= a[i]),
      `preview anchor ${index} must be lighter, not darker (${a} -> ${b})`,
    );
    assert.ok(
      Math.max(...b) - Math.min(...b) < Math.max(...a) - Math.min(...a),
      `preview anchor ${index} must be less saturated (${a} -> ${b})`,
    );
  }
  // Hue order survives the softening: cold end green/blue-dominant, hot end
  // red-dominant — the legend's direction still reads on the sketch.
  const cold = colorAt(preview, 0);
  const hot = colorAt(preview, 2);
  assert.ok(cold[1] > cold[0], "cold preview anchor stays green-dominant");
  assert.ok(hot[0] > hot[1] && hot[0] > hot[2], "hot preview anchor stays red-dominant");
  // The two ramps stay distinguishable (a calmer copy, not a new palette).
  assert.notDeepEqual(colorAt(preview, 1), colorAt(exact, 1));
});

test("colorizePreviewValues keeps the exact ramp's nodata and clamp contracts", () => {
  const nodata = colorizePreviewValues(Float32Array.from([Number.NaN]), {
    minimum: 30,
    maximum: 44,
  });
  assert.deepEqual([...nodata], [0, 0, 0, 0]);
  // Out-of-range values clamp to the ramp's end anchors, exactly like the
  // exact path — a calmer palette is not a looser contract.
  const clamped = colorizePreviewValues(Float32Array.from([10, 60, 20, 60]), {
    minimum: 30,
    maximum: 44,
  });
  const cold = [clamped[0], clamped[1], clamped[2]];
  const hot = [clamped[4], clamped[5], clamped[6]];
  assert.deepEqual(cold, [clamped[8], clamped[9], clamped[10]], "below-minimum clamps to one anchor");
  assert.deepEqual(hot, [clamped[12], clamped[13], clamped[14]], "above-maximum clamps to one anchor");
  assert.notDeepEqual(cold, hot, "the two clamp anchors are distinct colors");
});
