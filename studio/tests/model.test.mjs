// SPDX-License-Identifier: GPL-3.0-only

import test from "node:test";
import assert from "node:assert/strict";

import {
  BUILDING_PRESETS,
  TREE_PRESETS,
  accumulateDirtyRect,
  apiValuesToBuilding,
  applyHomography,
  buildingValuesToApi,
  canvasToUv,
  cloneBuilding,
  dirtyRectForBuildingEdit,
  dirtyRectForEdit,
  gridWindowArea,
  influenceRectForBuilding,
  isCurrentWorkerMessage,
  makeBuilding,
  makeTree,
  rectToGridWindow,
  shadowVectorMeters,
  uvToCanvas,
} from "../model.mjs";

const scene = {
  widthMeters: 1000,
  heightMeters: 1000,
  homography: [[900, -202, 238], [0, 744, 106], [0, -0.28, 1]],
};
scene.inverseHomography = invert3x3(scene.homography);

function invert3x3(matrix) {
  const [a, b, c] = matrix;
  const determinant =
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0]);
  const inverseDeterminant = 1 / determinant;
  return [
    [
      (b[1] * c[2] - b[2] * c[1]) * inverseDeterminant,
      (a[2] * c[1] - a[1] * c[2]) * inverseDeterminant,
      (a[1] * b[2] - a[2] * b[1]) * inverseDeterminant,
    ],
    [
      (b[2] * c[0] - b[0] * c[2]) * inverseDeterminant,
      (a[0] * c[2] - a[2] * c[0]) * inverseDeterminant,
      (a[2] * b[0] - a[0] * b[2]) * inverseDeterminant,
    ],
    [
      (b[0] * c[1] - b[1] * c[0]) * inverseDeterminant,
      (a[1] * c[0] - a[0] * c[1]) * inverseDeterminant,
      (a[0] * b[1] - a[1] * b[0]) * inverseDeterminant,
    ],
  ];
}

test("homography round trip preserves UV coordinates", () => {
  const screen = uvToCanvas(0.42, 0.61, scene);
  const uv = canvasToUv(screen.x, screen.y, scene);
  assert.ok(Math.abs(uv.u - 0.42) < 1e-9);
  assert.ok(Math.abs(uv.v - 0.61) < 1e-9);
});

test("applyHomography rejects points at infinity", () => {
  assert.throws(
    () => applyHomography([[1, 0, 0], [0, 1, 0], [0, 0, 0]], 1, 1),
    /infinity/,
  );
});

test("noon sun to the south casts tree shadow north", () => {
  const tree = makeTree("shade", 0.5, 0.5, 1);
  const vector = shadowVectorMeters(tree, 12);
  assert.ok(Math.abs(vector.eastM) < 1e-8);
  assert.ok(vector.northM > 0);
});

test("move invalidation includes both old and new influence regions", () => {
  const oldTree = makeTree("shade", 0.15, 0.2, 1);
  const newTree = { ...oldTree, u: 0.8, v: 0.75 };
  const dirty = dirtyRectForEdit(oldTree, newTree, 12, scene, {
    lowestSkyPatchAltitudeDeg: 45,
    safetyMarginM: 0,
  });
  assert.ok(dirty.minU < 0.15);
  assert.ok(dirty.maxU > 0.8);
  assert.ok(dirty.minV < 0.2);
  assert.ok(dirty.maxV > 0.75);
});

test("grid windows are aligned and bounded", () => {
  const window = rectToGridWindow(
    { minU: 0.101, minV: 0.209, maxU: 0.509, maxV: 0.812 },
    128,
    128,
    4,
  );
  assert.equal(window.colStart % 4, 0);
  assert.equal(window.rowStart % 4, 0);
  assert.equal(window.colStop % 4, 0);
  assert.equal(window.rowStop % 4, 0);
  assert.ok(gridWindowArea(window) > 0);
});


test("stale worker messages are rejected by job and scene revision", () => {
  assert.equal(isCurrentWorkerMessage({ jobId: 8, revision: 4 }, 8, 4), true);
  assert.equal(isCurrentWorkerMessage({ jobId: 7, revision: 4 }, 8, 4), false);
  assert.equal(isCurrentWorkerMessage({ jobId: 8, revision: 3 }, 8, 4), false);
});


test("unapplied dirty regions accumulate until a current result is accepted", () => {
  const first = { minU: 0.05, minV: 0.05, maxU: 0.15, maxV: 0.15 };
  const second = { minU: 0.65, minV: 0.65, maxU: 0.75, maxV: 0.75 };
  const accumulated = accumulateDirtyRect(first, second, 0.9);
  assert.deepEqual(accumulated, {
    minU: 0.05,
    minV: 0.05,
    maxU: 0.75,
    maxV: 0.75,
  });
  assert.deepEqual(accumulateDirtyRect(first, second, 0.3), {
    minU: 0,
    minV: 0,
    maxU: 1,
    maxV: 1,
  });
});


test("every preset declares finite trunk_ratio and transmissivity attributes", () => {
  // makeTree()/treeToApiObject() pass these through WITHOUT local default
  // restatements (the schema default lives in the capability document, the
  // contract default on the server). This invariant is what makes that safe:
  // a preset missing either value would surface as NaN in a contract item.
  for (const [name, preset] of Object.entries(TREE_PRESETS)) {
    assert.ok(
      Number.isFinite(preset.trunkRatio),
      `${name} must declare a finite trunkRatio`,
    );
    assert.ok(
      Number.isFinite(preset.transmissivity),
      `${name} must declare a finite transmissivity`,
    );
    const tree = makeTree(name, 0.5, 0.5, 1);
    assert.ok(Number.isFinite(tree.trunkRatio));
    assert.ok(Number.isFinite(tree.transmissivity));
  }
});


test("makeBuilding clamps placement and mints distinct ids within the same millisecond", () => {
  const clamped = makeBuilding("pavilion", -0.2, 1.4, 7, scene);
  assert.equal(clamped.u, 0);
  assert.equal(clamped.v, 1);
  assert.equal(clamped.label, "Pavilion 07");
  // Same ms + same ordinal: only the random tail separates the ids (the
  // millisecond stamp alone would collide across actors — see makeTree).
  const first = makeBuilding("pavilion", 0.5, 0.5, 7, scene);
  const second = makeBuilding("pavilion", 0.5, 0.5, 7, scene);
  assert.ok(first.id.startsWith("building-"));
  assert.notEqual(first.id, second.id);
  assert.throws(() => makeBuilding("castle", 0.5, 0.5, 1, scene), /Unknown building preset/);
});

test("building footprint corners land at world-meter coordinates around the placement", () => {
  // block is 16 x 10 m centered on u=v=0.5 -> world (500 m, 500 m) in the
  // 1000 x 1000 m scene; world y grows north while v grows downward.
  const building = makeBuilding("block", 0.5, 0.5, 1, scene);
  assert.deepEqual(building.footprintM, [
    [492, 495],
    [508, 495],
    [508, 505],
    [492, 505],
  ]);
  assert.equal(building.heightM, 15);
});

test("every building preset satisfies the building_geometry contract shape", () => {
  for (const [name, preset] of Object.entries(BUILDING_PRESETS)) {
    assert.ok(
      Number.isFinite(preset.widthM) && preset.widthM > 0,
      `${name} must declare a finite positive widthM`,
    );
    assert.ok(
      Number.isFinite(preset.depthM) && preset.depthM > 0,
      `${name} must declare a finite positive depthM`,
    );
    // height_m is constrained to (0, 1000] on the wire; presets are the
    // only local source of heights, so they must already comply.
    assert.ok(
      preset.heightM > 0 && preset.heightM <= 1000,
      `${name} heightM must be in (0, 1000]`,
    );
    const building = makeBuilding(name, 0.5, 0.5, 1, scene);
    assert.equal(building.footprintM.length, 4);
    // Unclosed ring: the first vertex is not repeated at the end.
    assert.notDeepEqual(building.footprintM.at(-1), building.footprintM[0]);
  }
});

test("building influence rect contains the footprint and the shadow corridor", () => {
  const building = makeBuilding("tower", 0.5, 0.5, 1, scene);
  const shadow = shadowVectorMeters(building, 12);
  assert.ok(Math.abs(shadow.eastM) < 1e-8);
  assert.ok(shadow.northM > 0); // noon shadow falls north, i.e. toward lower v
  const rect = influenceRectForBuilding(building, 12, scene, {
    lowestSkyPatchAltitudeDeg: 45,
    safetyMarginM: 0,
  });
  // Footprint coverage: 12 x 12 m centered on (500 m, 500 m).
  assert.ok(rect.minU <= 494 / 1000 && rect.maxU >= 506 / 1000);
  assert.ok(rect.minV <= 494 / 1000 && rect.maxV >= 506 / 1000);
  // The rect reaches past the footprint on the shadow side (v below the
  // footprint's own minimum).
  assert.ok(rect.minV < 494 / 1000);
  assert.ok(rect.maxV > 0.5);
});

test("building shadow corridor extends from the far footprint corner", () => {
  const building = makeBuilding("tower", 0.5, 0.5, 1, scene); // 12 x 12 m, h = 40
  const shadow = shadowVectorMeters(building, 12); // length = 40 / tan(66 deg)
  const rect = influenceRectForBuilding(building, 12, scene, {
    // 80 deg keeps the SVF ring (~7 m) smaller than the corridor (~18 m),
    // so the corridor alone drives the shadow-side bound.
    lowestSkyPatchAltitudeDeg: 80,
    safetyMarginM: 0,
  });
  const farCornerYM = 506; // north edge of the footprint
  assert.ok(
    Math.abs(rect.minV - (1 - (farCornerYM + shadow.lengthM) / 1000)) < 1e-9,
  );
});

test("building influence rects clamp to the site at edge placements", () => {
  const rect = influenceRectForBuilding(makeBuilding("tower", 0, 1, 1, scene), 12, scene);
  assert.equal(rect.minU, 0);
  assert.equal(rect.maxV, 1);
  assert.ok(rect.maxU > 0);
  assert.ok(rect.minV < 1);
});

test("building move invalidation includes both old and new influence regions", () => {
  const oldBuilding = makeBuilding("block", 0.15, 0.2, 1, scene);
  // A real move re-derives the footprint at the new placement — only the
  // random id tail differs, which the dirty rect never inspects.
  const newBuilding = makeBuilding("block", 0.8, 0.75, 1, scene);
  const options = { lowestSkyPatchAltitudeDeg: 45, safetyMarginM: 0 };
  const dirty = dirtyRectForBuildingEdit(oldBuilding, newBuilding, 12, scene, options);
  assert.ok(dirty.minU < 0.15);
  assert.ok(dirty.maxU > 0.8);
  assert.ok(dirty.minV < 0.2);
  assert.ok(dirty.maxV > 0.75);
  // One-sided edits (add/delete) and the both-null guard mirror the tree
  // dirtyRectForEdit contract.
  assert.ok(dirtyRectForBuildingEdit(null, newBuilding, 12, scene, options));
  assert.ok(dirtyRectForBuildingEdit(oldBuilding, null, 12, scene, options));
  assert.throws(
    () => dirtyRectForBuildingEdit(null, null, 12, scene, options),
    /required/,
  );
});

test("building wire values carry geometry only, verbatim", () => {
  const building = makeBuilding("pavilion", 0.3173, 0.5241, 3, scene);
  const values = buildingValuesToApi(building);
  // Identity rides as entity_id on the operation — a building_id inside
  // values would be stripped or refused as a re-aimed id.
  assert.deepEqual(Object.keys(values).sort(), ["footprint_m", "height_m"]);
  assert.deepEqual(values.footprint_m, building.footprintM);
  assert.equal(values.height_m, building.heightM);
  // Fractional world coordinates pass through with no rounding.
  assert.ok(!Number.isInteger(values.footprint_m[0][0]));
});

test("wire values adopt back into a local building (round trip)", () => {
  const building = makeBuilding("tower", 0.4, 0.6, 2, scene);
  const adopted = apiValuesToBuilding(buildingValuesToApi(building), building);
  assert.deepEqual(adopted.footprintM, building.footprintM);
  assert.equal(adopted.heightM, building.heightM);
  // Presentation and identity stay local through adoption.
  assert.equal(adopted.id, building.id);
  assert.equal(adopted.label, building.label);
  assert.equal(adopted.preset, building.preset);
  // A remote reshape replaces geometry and re-measures the dimensions from
  // the ring so stale widthM/depthM cannot survive beside new vertices.
  const edited = apiValuesToBuilding(
    { footprint_m: [[10, 10], [30, 10], [30, 20], [10, 20]], height_m: 22 },
    building,
  );
  assert.equal(edited.heightM, 22);
  assert.deepEqual(edited.footprintM[0], [10, 10]);
  assert.equal(edited.widthM, 20);
  assert.equal(edited.depthM, 10);
});

test("cloneBuilding deep-copies the footprint ring", () => {
  const building = makeBuilding("pavilion", 0.5, 0.5, 1, scene);
  const copy = cloneBuilding(building);
  assert.deepEqual(copy, building);
  assert.notEqual(copy.footprintM, building.footprintM);
  // Corner arrays must not alias: editing one clone's ring leaves the
  // original untouched.
  copy.footprintM[0][0] = -999;
  assert.notEqual(building.footprintM[0][0], -999);
  assert.equal(cloneBuilding(null), null);
});
