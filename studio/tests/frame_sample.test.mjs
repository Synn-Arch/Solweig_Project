// SPDX-License-Identifier: GPL-3.0-only
//
// Frame-calibration correspondence gates. A remote edit's wire position may
// only pair with the local object's position when the edit did NOT relocate
// that object — a moved op would sample the pre-edit local corner against
// the post-edit wire corner, and one bogus sample locks a permanently wrong
// local↔wire frame. Pure gates: no DOM, no server, no clock.

import test from "node:test";
import assert from "node:assert/strict";

import { makeBuilding, worldToUv as sceneWorldToUv } from "../model.mjs";
import { frameFromSpans, frameResidualOk, worldTree } from "../op_builder.mjs";
import {
  FRAME_SAMPLE_TOLERANCE_M,
  buildingFootprintThroughFrame,
  buildingFrameSample,
  vegetationFrameSample,
} from "../app.mjs";

// The span-local default frame every degraded (manifest-less) world starts
// on — the same hypothesis on every client, so an unmoved remote op's wire
// values read back to the local position exactly.
const SCENE = { widthMeters: 500, heightMeters: 500 };
const FRAME = frameFromSpans({ spanX: SCENE.widthMeters, spanY: SCENE.heightMeters });

/** A moved copy of a building (same ring shape, new UV center). */
function movedTo(building, u, v) {
  const xM = u * SCENE.widthMeters;
  const yM = (1 - v) * SCENE.heightMeters;
  const halfWidthM = building.widthM / 2;
  const halfDepthM = building.depthM / 2;
  return {
    ...building,
    u,
    v,
    footprintM: [
      [xM - halfWidthM, yM - halfDepthM],
      [xM + halfWidthM, yM - halfDepthM],
      [xM + halfWidthM, yM + halfDepthM],
      [xM - halfWidthM, yM + halfDepthM],
    ],
  };
}

// ---------------------------------------------------------------------------
// buildingFrameSample
// ---------------------------------------------------------------------------

test("a height-only building replace (ring unchanged) yields a calibration sample", () => {
  const local = makeBuilding("pavilion", 0.4, 0.6, 1, SCENE);
  const wire = buildingFootprintThroughFrame(local.footprintM, SCENE, FRAME);
  const sample = buildingFrameSample(local, wire, FRAME, SCENE);
  assert.ok(sample, "unmoved replace must sample");
  const cornerUv = sceneWorldToUv(local.footprintM[0][0], local.footprintM[0][1], SCENE);
  assert.equal(sample.u, cornerUv.u);
  assert.equal(sample.v, cornerUv.v);
  assert.equal(sample.x_m, wire[0][0]);
  assert.equal(sample.y_m, wire[0][1]);
});

test("a building move is refused — wire corner 0 no longer describes local corner 0", () => {
  const local = makeBuilding("pavilion", 0.4, 0.6, 1, SCENE);
  const moved = movedTo(local, 0.55, 0.45);
  const wire = buildingFootprintThroughFrame(moved.footprintM, SCENE, FRAME);
  assert.equal(buildingFrameSample(local, wire, FRAME, SCENE), null);
});

test("a building resize that moves the ring is refused, a vertex-count change is refused", () => {
  const local = makeBuilding("pavilion", 0.4, 0.6, 1, SCENE);
  const grown = {
    ...local,
    footprintM: local.footprintM.map(([xM, yM]) => [xM * 1.5, yM * 1.5]),
  };
  const wire = buildingFootprintThroughFrame(grown.footprintM, SCENE, FRAME);
  assert.equal(buildingFrameSample(local, wire, FRAME, SCENE), null);
  assert.equal(
    buildingFrameSample(local, [wire[0], wire[1], wire[2]], FRAME, SCENE),
    null,
  );
});

test("a malformed wire footprint (missing corner, non-finite) is refused", () => {
  const local = makeBuilding("pavilion", 0.4, 0.6, 1, SCENE);
  assert.equal(buildingFrameSample(local, [], FRAME, SCENE), null);
  assert.equal(buildingFrameSample(local, [["x", 1], [2, 3]], FRAME, SCENE), null);
  assert.equal(
    buildingFrameSample(local, [[NaN, NaN], ...local.footprintM.slice(1)], FRAME, SCENE),
    null,
  );
});

// ---------------------------------------------------------------------------
// vegetationFrameSample
// ---------------------------------------------------------------------------

test("a property-only tree replace (position unchanged) yields a calibration sample", () => {
  const local = { tree_id: "tree_07", u: 0.3, v: 0.25 };
  const values = worldTree({ ...local, heightM: 14, canopyDiameterM: 7 }, FRAME);
  const sample = vegetationFrameSample(local, values, FRAME, SCENE);
  assert.ok(sample, "unmoved replace must sample");
  assert.equal(sample.u, 0.3);
  assert.equal(sample.v, 0.25);
  assert.equal(sample.x_m, values.x_m);
  assert.equal(sample.y_m, values.y_m);
});

test("a nudged tree is refused — wire metres describe the post-nudge position", () => {
  const local = { tree_id: "tree_07", u: 0.3, v: 0.25 };
  const moved = { ...local, u: local.u + 0.01, v: local.v - 0.01 }; // 5 m ≫ 0.5 m tolerance
  const values = worldTree(moved, FRAME);
  assert.equal(vegetationFrameSample(local, values, FRAME, SCENE), null);
});

test("a sub-tolerance drift still samples, values without world metres are refused", () => {
  const local = { tree_id: "tree_07", u: 0.3, v: 0.25 };
  // One sub-tolerance drift (0.4 m on a 500 m scene, tolerance 0.5 m).
  const drifted = {
    ...local,
    u: local.u + (0.8 * FRAME_SAMPLE_TOLERANCE_M) / SCENE.widthMeters,
  };
  const values = worldTree(drifted, FRAME);
  assert.ok(vegetationFrameSample(local, values, FRAME, SCENE), "at-tolerance drift must sample");

  assert.equal(vegetationFrameSample(local, { u: 0.3, v: 0.25 }, FRAME, SCENE), null);
  assert.equal(vegetationFrameSample(local, null, FRAME, SCENE), null);
});

// ---------------------------------------------------------------------------
// frameResidualOk (the lock gate)
// ---------------------------------------------------------------------------

test("frameResidualOk accepts a consistent sample set and rejects any violator", () => {
  const good = [
    { u: 0.1, v: 0.2, x_m: 50, y_m: -100 },
    { u: 0.8, v: 0.9, x_m: 400, y_m: -450 },
  ];
  assert.equal(frameResidualOk(good, FRAME), true);
  assert.equal(
    frameResidualOk([...good, { u: 0.5, v: 0.5, x_m: 250, y_m: -100 }], FRAME),
    false,
    "one inconsistent sample must veto the lock",
  );
});

test("frameResidualOk refuses empty samples, a missing frame, and non-samples", () => {
  assert.equal(frameResidualOk([], FRAME), false);
  assert.equal(frameResidualOk(null, FRAME), false);
  assert.equal(frameResidualOk([{ u: 0, v: 0, x_m: 0, y_m: 0 }], null), false);
});
