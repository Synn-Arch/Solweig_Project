// SPDX-License-Identifier: GPL-3.0-only
//
// The overlay renderer is DOM-bound (SceneRenderer needs a canvas 2D
// context), so the building pass is tested at its pure geometry seams:
// buildingExtrusionPolygons is the pseudo-3d decomposition #drawBuilding
// paints, and buildingShadowPolygon is the displaced ring #drawBuildingShadow
// fills. If those two hold, the draw methods only ever stroke/fill what they
// return — no canvas required to pin the geometry.

import test from "node:test";
import assert from "node:assert/strict";

import { buildingExtrusionPolygons, buildingShadowPolygon } from "../renderer.mjs";

// Screen-axis rectangle, the shape makeBuilding's axis-aligned footprints
// project to (the homography is near-affine).
const RECT = [
  { x: 100, y: 40 },
  { x: 160, y: 40 },
  { x: 160, y: 88 },
  { x: 100, y: 88 },
];

test("extrusion offsets the base straight down by the wall height", () => {
  const { roof, base } = buildingExtrusionPolygons(RECT, 24);
  assert.deepEqual(roof, RECT);
  assert.deepEqual(base, RECT.map((point) => ({ x: point.x, y: point.y + 24 })));
});

test("extrusion keeps only walls with area — vertical edges drop out", () => {
  const { walls } = buildingExtrusionPolygons(RECT, 24);
  // Straight-down extrusion of a screen-axis rectangle: exactly the two
  // horizontal edges become paintable quads; the vertical edges project to
  // zero-width slivers and must not reach the canvas.
  assert.equal(walls.length, 2);
});

test("each wall connects its roof edge to the displaced base edge", () => {
  const { walls } = buildingExtrusionPolygons(RECT, 24);
  // The first surviving wall hangs off the top edge RECT[0] → RECT[1].
  assert.deepEqual(walls[0], [
    RECT[0],
    RECT[1],
    { x: RECT[1].x, y: RECT[1].y + 24 },
    { x: RECT[0].x, y: RECT[0].y + 24 },
  ]);
});

test("a zero wall height extrudes nothing but still reports roof and base", () => {
  const { roof, base, walls } = buildingExtrusionPolygons(RECT, 0);
  assert.deepEqual(roof, RECT);
  assert.deepEqual(base, RECT);
  assert.equal(walls.length, 0);
});

test("the shadow ring shifts every corner by the same canvas offset", () => {
  const polygon = buildingShadowPolygon(RECT, { x: 30, y: 12 });
  assert.deepEqual(polygon, RECT.map((point) => ({ x: point.x + 30, y: point.y + 12 })));
});
