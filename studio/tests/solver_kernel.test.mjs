// SPDX-License-Identifier: GPL-3.0-only

import test from "node:test";
import assert from "node:assert/strict";

import { makeTree } from "../model.mjs";
import {
  applyPatch,
  computeIncrementalPatch,
  decodeAnalysisFixture,
} from "../solver_kernel.mjs";

const scene = {
  widthMeters: 100,
  heightMeters: 100,
  gridWidth: 10,
  gridHeight: 10,
};

function fixture() {
  return {
    gridWidth: 10,
    gridHeight: 10,
    utci: Array.from({ length: 100 }, () => 40),
    buildingMask: Array.from({ length: 100 }, () => 0),
  };
}

test("decode fixture preserves nulls as NaN", () => {
  const source = fixture();
  source.utci[3] = null;
  source.buildingMask[3] = 1;
  const decoded = decodeAnalysisFixture(source);
  assert.ok(Number.isNaN(decoded.baseline[3]));
  assert.equal(decoded.buildingMask[3], 1);
});

test("incremental kernel returns only requested patch", () => {
  const decoded = decodeAnalysisFixture(fixture());
  const tree = makeTree("shade", 0.5, 0.5, 1);
  const window = { rowStart: 2, rowStop: 8, colStart: 1, colStop: 7 };
  const result = computeIncrementalPatch({
    baseline: decoded.baseline,
    buildingMask: decoded.buildingMask,
    trees: [tree],
    window,
    scene,
    hour: 12,
  });
  assert.equal(result.patch.length, 36);
  assert.equal(result.metrics.affectedCells, 36);
  assert.ok(result.metrics.peakDelta < 0);
});

test("applying patch leaves cells outside dirty window unchanged", () => {
  const decoded = decodeAnalysisFixture(fixture());
  const target = decoded.baseline.slice();
  const tree = makeTree("broad", 0.45, 0.45, 1);
  const window = { rowStart: 3, rowStop: 7, colStart: 3, colStop: 7 };
  const result = computeIncrementalPatch({
    baseline: decoded.baseline,
    buildingMask: decoded.buildingMask,
    trees: [tree],
    window,
    scene,
    hour: 12,
  });
  applyPatch(target, result.patch, window, scene.gridWidth);

  for (let row = 0; row < scene.gridHeight; row += 1) {
    for (let col = 0; col < scene.gridWidth; col += 1) {
      const inside = row >= 3 && row < 7 && col >= 3 && col < 7;
      const index = row * scene.gridWidth + col;
      if (!inside) assert.equal(target[index], 40);
    }
  }
});

test("deleting all trees restores baseline when dirty window is recomputed", () => {
  const decoded = decodeAnalysisFixture(fixture());
  const window = { rowStart: 0, rowStop: 10, colStart: 0, colStop: 10 };
  const result = computeIncrementalPatch({
    baseline: decoded.baseline,
    buildingMask: decoded.buildingMask,
    trees: [],
    window,
    scene,
    hour: 12,
  });
  assert.ok(result.patch.every((value) => value === 40));
  assert.equal(result.metrics.meanDelta, 0);
});
