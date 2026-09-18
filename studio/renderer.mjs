// SPDX-License-Identifier: GPL-3.0-only

import {
  buildingToWorld,
  shadowVectorMeters,
  treeToWorld,
  uvToCanvas,
  worldToUv,
} from "./model.mjs";
import { applyPatch } from "./solver_kernel.mjs";

const HEAT_STOPS = [
  [0.0, [41, 153, 146]],
  [0.24, [67, 185, 157]],
  [0.45, [137, 198, 111]],
  [0.64, [225, 205, 91]],
  [0.82, [237, 145, 73]],
  [1.0, [221, 72, 62]],
];

// Preview-only ramp (the local sketch kernel's layer): the preview texture
// composites ON TOP of the exact layer, which read as MORE saturated than the
// server-exact picture it previews — the preview→exact swap jarred. The
// preview keeps HEAT_STOPS' hue order and stop positions at lower chroma
// (each anchor mixed ~35% toward white — light-theme friendly), so the sketch
// reads as a softer cousin of the exact palette, never a louder rival. The
// exact texture path below stays untouched.
const PREVIEW_HEAT_STOPS = [
  [0.0, [116, 189, 184]],
  [0.24, [133, 210, 191]],
  [0.45, [178, 218, 161]],
  [0.64, [236, 223, 148]],
  [0.82, [241, 168, 141]],
  [1.0, [234, 118, 112]],
];

function rampRgba(value, stops, minimum, maximum, alpha) {
  if (!Number.isFinite(value)) return [0, 0, 0, 0];
  const t = Math.min(1, Math.max(0, (value - minimum) / (maximum - minimum)));
  let lower = stops[0];
  let upper = stops.at(-1);
  for (let index = 1; index < stops.length; index += 1) {
    if (t <= stops[index][0]) {
      lower = stops[index - 1];
      upper = stops[index];
      break;
    }
  }
  const local = (t - lower[0]) / Math.max(upper[0] - lower[0], 1e-6);
  return [
    Math.round(lower[1][0] + (upper[1][0] - lower[1][0]) * local),
    Math.round(lower[1][1] + (upper[1][1] - lower[1][1]) * local),
    Math.round(lower[1][2] + (upper[1][2] - lower[1][2]) * local),
    alpha,
  ];
}

export function temperatureToRgba(value, minimum = 30, maximum = 44, alpha = 150) {
  return rampRgba(value, HEAT_STOPS, minimum, maximum, alpha);
}

/** Preview variant of {@link temperatureToRgba} (PREVIEW_HEAT_STOPS ramp). */
export function previewTemperatureToRgba(value, minimum = 30, maximum = 44, alpha = 160) {
  return rampRgba(value, PREVIEW_HEAT_STOPS, minimum, maximum, alpha);
}

/** Colorize a flat float32 plane into a row-major RGBA8 buffer (pure). */
export function colorizeValues(values, { minimum = 30, maximum = 44, alpha = 154 } = {}) {
  const pixels = new Uint8Array(values.length * 4);
  for (let index = 0; index < values.length; index += 1) {
    const [red, green, blue, pixelAlpha] = temperatureToRgba(
      values[index],
      minimum,
      maximum,
      alpha,
    );
    const offset = index * 4;
    pixels[offset] = red;
    pixels[offset + 1] = green;
    pixels[offset + 2] = blue;
    pixels[offset + 3] = pixelAlpha;
  }
  return pixels;
}

/**
 * Colorize a PREVIEW plane (the sketch kernel's values) with the calmer
 * preview ramp — same contract as {@link colorizeValues} (clamps, nodata
 * transparency, row-major RGBA8), different palette.
 */
export function colorizePreviewValues(values, { minimum = 30, maximum = 44, alpha = 160 } = {}) {
  const pixels = new Uint8Array(values.length * 4);
  for (let index = 0; index < values.length; index += 1) {
    const [red, green, blue, pixelAlpha] = previewTemperatureToRgba(
      values[index],
      minimum,
      maximum,
      alpha,
    );
    const offset = index * 4;
    pixels[offset] = red;
    pixels[offset + 1] = green;
    pixels[offset + 2] = blue;
    pixels[offset + 3] = pixelAlpha;
  }
  return pixels;
}

// ---------------------------------------------------------------------------
// Realtime result-class canvas furniture (ux_redesign): design_plan.md §2.3
// trio hues for canvas-side class ink, §7 motion moments 2 (fast-lane ROI
// flash, one-shot) and the collaborator reconciliation pulse
// (interaction_simulations.md I-31 `.pulse-remote`), plus the per-timestep
// freshness data hook (I-05 / decision D8, `#timelineFreshness`).
// Chrome-side animation stays in CSS; only what must composite over the
// orthophoto lives here.
// ---------------------------------------------------------------------------

/** Trio hues (design_plan.md §2.3): Viridian / Ochre / Slate; stale is Slate. */
export const RESULT_CLASS_COLORS = Object.freeze({
  fast_exact: "#0E6E4E",
  exact_reconciled: "#0E6E4E",
  fast_qualified: "#96590A",
  visual_pending: "#5F6A66",
  stale: "#5F6A66",
});

/**
 * Degradation-ladder rank (interaction_simulations.md I-20): higher is more
 * trustworthy. Used to gate the ROI flash — upward transitions animate,
 * downgrades log only (the ledger owns downgrade narration).
 */
export const RESULT_CLASS_LADDER = Object.freeze({
  visual_pending: 0,
  fast_qualified: 1,
  fast_exact: 2,
  exact_reconciled: 3,
});

const CLASS_KIND_ALIASES = Object.freeze({
  exact: "exact_reconciled",
  stale: "visual_pending",
});

/** Conservative canonical kind for a wire class (unknown → `visual_pending`). */
function resolveClassKind(kind) {
  const canonical = CLASS_KIND_ALIASES[kind] ?? kind;
  return RESULT_CLASS_COLORS[canonical] ? canonical : "visual_pending";
}

function numericOrNull(value) {
  if (value === null || value === undefined || typeof value === "boolean") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function hexToRgb(hex) {
  const value = String(hex).replace("#", "");
  if (!/^[0-9a-fA-F]{6}$/.test(value) && !/^[0-9a-fA-F]{3}$/.test(value)) return null;
  const full = value.length === 3 ? [...value].map((c) => c + c).join("") : value;
  const numeric = Number.parseInt(full, 16);
  return [(numeric >> 16) & 255, (numeric >> 8) & 255, numeric & 255];
}

/** Linear white→`to` mix used by the one-shot flash profile (§7 moment 2). */
function mixHex(from, to, amount) {
  const start = hexToRgb(from);
  const end = hexToRgb(to);
  if (!start || !end) return to;
  const t = Math.min(1, Math.max(0, amount));
  return `rgb(${Math.round(start[0] + (end[0] - start[0]) * t)}, ${Math.round(
    start[1] + (end[1] - start[1]) * t,
  )}, ${Math.round(start[2] + (end[2] - start[2]) * t)})`;
}

function withAlpha(color, alpha) {
  const rgb = hexToRgb(color);
  if (!rgb) return `rgba(43, 73, 204, ${alpha})`;
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

let canvasMotionReduced = false;

/**
 * Reduced-motion flag the app can set from
 * `matchMedia("(prefers-reduced-motion: reduce)")` (or a settings toggle):
 * canvas-side motion moments (ROI flash sweep, pulse expansion) become
 * static steps (design_plan.md §7 reduced-motion column).
 */
export function setReducedMotion(reduced) {
  canvasMotionReduced = reduced === true;
}

/** Read back the flag set via {@link setReducedMotion}. */
export function isReducedMotion() {
  return canvasMotionReduced;
}

/**
 * Per-timestep freshness dots for `#timelineFreshness` (pure data hook;
 * interaction_simulations.md I-05, decision D8). The DOM/CSS side renders
 * the dots; this computes what each timestep's picture is made of.
 *
 * Each frame record describes one timestep of the published cube:
 * - `index` — timestep index (echoed through; defaults to array position)
 * - `exactRevision` — revision the exact plane for this timestep was
 *   verified at (`null`/absent = never computed)
 * - `fastRevision` — revision of the newest fast-lane paint that covered
 *   this timestep (`null` = none)
 * - `receivedAt` — epoch-ms timestamp of the basis frame currently shown
 *   (`null` = unknown → `ageMs` is null)
 *
 * Basis per timestep:
 * - `exact` — the exact plane is verified for the reference (workspace)
 *   revision → filled dot
 * - `fast` — a fast-lane paint covers the reference revision (exact still
 *   owed) → half dot in the redesign vocabulary
 * - `cached` — anything older (stale-but-honest exact, superseded fast) or
 *   never computed; `ageMs === null` marks never-computed (hollow dot)
 *
 * @param {Array<object>|object|null} frames array of per-timestep records,
 *        or a sparse map keyed by timestep index (keys sorted numerically).
 * @param {object} [options]
 * @param {number} [options.now=Date.now()] clock for `ageMs` (injectable
 *        for tests).
 * @param {number} [options.workspaceRevision] current canonical revision the
 *        dots are judged against; omitted → each timestep's own newest
 *        revision is the reference (no `cached`-by-divergence then).
 * @returns {Array<{index: number, basis: "exact"|"fast"|"cached", ageMs: number|null}>}
 */
export function timeBasisDots(frames, { now = Date.now(), workspaceRevision = null } = {}) {
  const asOf = numericOrNull(now) ?? Date.now();
  const entries = normalizeTimeFrames(frames);
  return entries.map(({ index, frame }) => {
    const exactRevision = numericOrNull(frame?.exactRevision);
    const fastRevision = numericOrNull(frame?.fastRevision);
    const reference =
      numericOrNull(workspaceRevision) ??
      (exactRevision !== null || fastRevision !== null
        ? Math.max(exactRevision ?? -1, fastRevision ?? -1)
        : null);
    let basis = "cached";
    if (reference !== null) {
      if (exactRevision !== null && exactRevision >= reference) basis = "exact";
      else if (fastRevision !== null && fastRevision >= reference) basis = "fast";
    }
    const receivedAt = numericOrNull(frame?.receivedAt);
    return {
      index,
      basis,
      ageMs: receivedAt === null ? null : Math.max(0, asOf - receivedAt),
    };
  });
}

function normalizeTimeFrames(frames) {
  if (Array.isArray(frames)) {
    return frames.map((frame, position) => ({
      index: numericOrNull(frame?.index) ?? position,
      frame,
    }));
  }
  if (frames && typeof frames === "object") {
    return Object.entries(frames)
      .map(([key, frame]) => ({ key: Number(key), frame }))
      .filter(({ key }) => Number.isFinite(key))
      .sort((a, b) => a.key - b.key)
      .map(({ key, frame }) => ({ index: key, frame }));
  }
  return [];
}

function createProgram(gl, vertexSource, fragmentSource) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`WebGL shader compilation failed: ${message}`);
    }
    return shader;
  };

  const program = gl.createProgram();
  const vertex = compile(gl.VERTEX_SHADER, vertexSource);
  const fragment = compile(gl.FRAGMENT_SHADER, fragmentSource);
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`WebGL program linking failed: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

function canvasPointToClip(point, width, height) {
  return [point[0] / width * 2 - 1, 1 - point[1] / height * 2];
}

// ---------------------------------------------------------------------------
// Building canvas geometry (pure seams for the building draw pass)
// ---------------------------------------------------------------------------
//
// SceneRenderer is DOM-bound, so the building pass keeps its geometry in
// node-testable functions: #drawBuilding paints exactly what
// buildingExtrusionPolygons decomposes, and #drawBuildingShadow fills
// exactly what buildingShadowPolygon displaces.

/** Flat neutral roof/wall ramps per building preset (light-theme friendly). */
const BUILDING_PALETTES = Object.freeze({
  pavilion: Object.freeze(["#e8dcc8", "#cbb994", "#a8987a"]),
  block: Object.freeze(["#e3c9b4", "#c29a83", "#8f6b58"]),
  tower: Object.freeze(["#dfe3e6", "#b9c0c6", "#848d94"]),
});

/** |Shoelace| area of a closed polygon given in canvas points. */
function polygonAreaPx(points) {
  let doubled = 0;
  for (let index = 0; index < points.length; index += 1) {
    const a = points[index];
    const b = points[(index + 1) % points.length];
    doubled += a.x * b.y - b.x * a.y;
  }
  return Math.abs(doubled) / 2;
}

/** Vertex-average centroid of a canvas-point ring (gradient anchor). */
function centroidOf(points) {
  let x = 0;
  let y = 0;
  for (const point of points) {
    x += point.x;
    y += point.y;
  }
  return { x: x / points.length, y: y / points.length };
}

/**
 * Decompose a footprint quad (canvas space) into the pseudo-3d pieces
 * #drawBuilding paints: `roof` (the input ring), `base` (the ring displaced
 * straight down by `wallHeightPx` — vertical, because the scene's
 * perspective idiom scales with v but never skews), and the `walls`
 * bridging each roof edge to its base edge. Edges parallel to the
 * extrusion project to zero area and are dropped, so a screen-axis
 * rectangle yields exactly two paintable walls.
 */
export function buildingExtrusionPolygons(corners, wallHeightPx) {
  const roof = corners.map((point) => ({ x: point.x, y: point.y }));
  const base = corners.map((point) => ({ x: point.x, y: point.y + wallHeightPx }));
  const walls = [];
  for (let index = 0; index < roof.length; index += 1) {
    const next = (index + 1) % roof.length;
    const wall = [roof[index], roof[next], base[next], base[index]];
    if (polygonAreaPx(wall) > 1e-6) walls.push(wall);
  }
  return { roof, base, walls };
}

/**
 * Displace a footprint ring (canvas space) by one shadow offset: every
 * corner moves by the same vector — the flat-ground assumption the tree
 * shadows already make (the homography is near-affine across a footprint,
 * so per-corner offsets would differ by sub-pixel amounts).
 */
export function buildingShadowPolygon(cornerPoints, offset) {
  return cornerPoints.map((point) => ({
    x: point.x + offset.x,
    y: point.y + offset.y,
  }));
}

export class SceneRenderer {
  // Transient canvas-side motion (design_plan.md §7): active one-shot effects
  // and the repaint machinery that animates them between the app's own
  // renderOverlay calls. #lastOverlayArgs lets the renderer replay the app's
  // last overlay draw so effects composite over (not instead of) the scene.
  #effects = [];
  #effectsFrame = null;
  #effectsTimeout = null;
  #lastOverlayArgs = null;
  #lastFlashRank = null;

  constructor(glCanvas, overlayCanvas, scene) {
    this.glCanvas = glCanvas;
    this.overlayCanvas = overlayCanvas;
    this.scene = scene;
    this.gl = glCanvas.getContext("webgl", {
      alpha: false,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    this.overlay = overlayCanvas.getContext("2d");
    this.baseTexture = null;
    this.heatTexture = null;
    this.heatValues = null;
    this.previewTexture = null;
    this.previewPixels = null;
    this.fallbackImage = null;

    if (this.gl) {
      this.#initializeWebGl();
    }
  }

  #initializeWebGl() {
    const gl = this.gl;
    this.program = createProgram(
      gl,
      `
        attribute vec2 a_position;
        attribute vec2 a_texCoord;
        varying vec2 v_texCoord;
        void main() {
          gl_Position = vec4(a_position, 0.0, 1.0);
          v_texCoord = a_texCoord;
        }
      `,
      `
        precision mediump float;
        uniform sampler2D u_texture;
        uniform float u_opacity;
        varying vec2 v_texCoord;
        void main() {
          vec4 sampleColor = texture2D(u_texture, v_texCoord);
          gl_FragColor = vec4(sampleColor.rgb, sampleColor.a * u_opacity);
        }
      `,
    );
    this.positionLocation = gl.getAttribLocation(this.program, "a_position");
    this.texCoordLocation = gl.getAttribLocation(this.program, "a_texCoord");
    this.opacityLocation = gl.getUniformLocation(this.program, "u_opacity");
    this.positionBuffer = gl.createBuffer();
    this.texCoordBuffer = gl.createBuffer();
    gl.useProgram(this.program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.04, 0.055, 0.056, 1);
  }

  async setBaseImage(url) {
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    await image.decode();
    this.fallbackImage = image;
    if (this.gl) {
      this.baseTexture = this.#createTextureFromImage(image);
    }
  }

  #createTextureFromImage(image) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  updateHeatmap(values, gridWidth, gridHeight, { alpha = 154 } = {}) {
    const pixels = colorizeValues(values, { alpha });
    // Persist the grid shape here: updateHeatRegion only receives a window,
    // and sites are not guaranteed to be square.
    this.heatGrid = { width: gridWidth, height: gridHeight };

    if (!this.gl) {
      this.heatPixels = { pixels, gridWidth, gridHeight };
      this.heatValues = Float32Array.from(values);
      return;
    }

    const gl = this.gl;
    if (!this.heatTexture) this.heatTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.heatTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gridWidth,
      gridHeight,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.heatValues = Float32Array.from(values);
  }

  /**
   * Upload only the changed sub-rectangle of the exact layer
   * (frontend_spec.md: "use texSubImage2D rather than replacing the full
   * texture"). Falls back to a full re-colorize when WebGL is unavailable.
   *
   * @param {Float32Array} regionValues row-major patch for the window
   * @param {object} window half-open {rowStart, rowStop, colStart, colStop}
   */
  updateHeatRegion(regionValues, window, { alpha = 154 } = {}) {
    const width = window.colStop - window.colStart;
    const height = window.rowStop - window.rowStart;
    if (regionValues.length !== width * height) {
      throw new Error("region values do not match the update window");
    }
    if (!this.heatValues || this.heatValues.length === 0) {
      throw new Error("updateHeatRegion requires an existing heatmap");
    }
    const gridWidth = this.heatGrid?.width;
    const gridHeight = this.heatGrid?.height;
    if (!gridWidth || !gridHeight || gridWidth * gridHeight !== this.heatValues.length) {
      throw new Error("updateHeatRegion requires an updateHeatmap-established grid shape");
    }
    // Keep the CPU mirror authoritative so the 2D fallback stays consistent.
    applyPatch(this.heatValues, regionValues, window, gridWidth);

    if (!this.gl) {
      this.heatPixels = {
        pixels: colorizeValues(this.heatValues, { alpha }),
        gridWidth,
        gridHeight,
      };
      return;
    }

    const gl = this.gl;
    const pixels = colorizeValues(regionValues, { alpha });
    gl.bindTexture(gl.TEXTURE_2D, this.heatTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      window.colStart,
      window.rowStart,
      width,
      height,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
  }

  /**
   * Preview (non-scientific) overlay layer, drawn only while the exact result
   * is stale (frontend_spec.md rendering layer 4). Lives on the small local
   * fixture grid, independent of the exact layer's server grid, and colorizes
   * with the CALMER preview ramp (PREVIEW_HEAT_STOPS) — the exact layer above
   * must never be out-shouted by the sketch previewing it.
   */
  updatePreviewHeatmap(values, gridWidth, gridHeight, { alpha = 160 } = {}) {
    const pixels = colorizePreviewValues(values, { alpha });
    if (!this.gl) {
      this.previewPixels = { pixels, gridWidth, gridHeight };
      return;
    }
    const gl = this.gl;
    if (!this.previewTexture) this.previewTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.previewTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gridWidth,
      gridHeight,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  clearPreviewHeatmap() {
    this.previewPixels = null;
    if (this.gl && this.previewTexture) {
      this.gl.deleteTexture(this.previewTexture);
    }
    this.previewTexture = null;
  }

  #drawTexture(texture, positions, texCoords, opacity = 1) {
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.bindTexture(gl.TEXTURE_2D, texture);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.positionLocation);
    gl.vertexAttribPointer(this.positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.texCoordLocation);
    gl.vertexAttribPointer(this.texCoordLocation, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1f(this.opacityLocation, opacity);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  renderBase({
    showHeat = true,
    heatOpacity = 0.62,
    showPreview = false,
    previewOpacity = 0.5,
  } = {}) {
    if (!this.gl) {
      const context = this.glCanvas.getContext("2d");
      context.clearRect(0, 0, this.glCanvas.width, this.glCanvas.height);
      if (this.fallbackImage) {
        context.drawImage(this.fallbackImage, 0, 0, this.glCanvas.width, this.glCanvas.height);
      }
      context.globalAlpha = heatOpacity;
      if (showHeat && this.heatPixels) this.#drawPixelsFallback(context, this.heatPixels);
      context.globalAlpha = previewOpacity;
      if (showPreview && this.previewPixels) {
        this.#drawPixelsFallback(context, this.previewPixels);
      }
      context.globalAlpha = 1;
      return;
    }

    const gl = this.gl;
    gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (this.baseTexture) {
      this.#drawTexture(
        this.baseTexture,
        [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        1,
      );
    }

    const heatLayers = [];
    if (showHeat && this.heatTexture) {
      heatLayers.push([this.heatTexture, heatOpacity]);
    }
    // Layer 4 of the spec's rendering order: the preview delta texture is
    // drawn only while the exact result is stale.
    if (showPreview && this.previewTexture) {
      heatLayers.push([this.previewTexture, previewOpacity]);
    }
    if (heatLayers.length > 0) {
      const [topLeft, topRight, bottomRight, bottomLeft] = this.scene.mapQuad;
      const tl = canvasPointToClip(topLeft, this.glCanvas.width, this.glCanvas.height);
      const tr = canvasPointToClip(topRight, this.glCanvas.width, this.glCanvas.height);
      const br = canvasPointToClip(bottomRight, this.glCanvas.width, this.glCanvas.height);
      const bl = canvasPointToClip(bottomLeft, this.glCanvas.width, this.glCanvas.height);
      for (const [texture, opacity] of heatLayers) {
        this.#drawTexture(
          texture,
          [...bl, ...br, ...tl, ...tl, ...br, ...tr],
          [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
          opacity,
        );
      }
    }
  }

  #drawPixelsFallback(context, { pixels, gridWidth, gridHeight }) {
    const imageData = new ImageData(
      new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, pixels.byteLength),
      gridWidth,
      gridHeight,
    );
    const offscreen = document.createElement("canvas");
    offscreen.width = gridWidth;
    offscreen.height = gridHeight;
    offscreen.getContext("2d").putImageData(imageData, 0, 0);
    const [topLeft, , bottomRight] = this.scene.mapQuad;
    context.drawImage(
      offscreen,
      topLeft[0],
      topLeft[1],
      bottomRight[0] - topLeft[0],
      bottomRight[1] - topLeft[1],
    );
  }

  renderOverlay({
    trees,
    selectedTreeId,
    dirtyRect,
    hour,
    showShadows = true,
    showDirtyRegion = true,
    ghostTree = null,
    showPreviewMarker = false,
    buildings = [],
    selectedBuildingId = null,
  }) {
    // Remember the raw argument object so transient effects can replay this
    // exact draw between the app's own render calls (see #kickEffects).
    this.#lastOverlayArgs =
      arguments.length > 0 && arguments[0] !== null && typeof arguments[0] === "object"
        ? arguments[0]
        : {};
    const context = this.overlay;
    context.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

    if (showPreviewMarker) this.#drawPreviewMarker(context);
    if (showDirtyRegion && dirtyRect) {
      this.#drawDirtyRegion(context, dirtyRect);
    }

    // Shadows are ground paint: BOTH families' shadows go down before any
    // solid geometry so neither can tint a roof or a crown. Solids then go
    // buildings-before-trees — a crown drawn under a neighboring wall reads
    // as felled timber, while a crown over a roof reads as depth.
    if (showShadows) {
      for (const building of buildings) this.#drawBuildingShadow(context, building, hour);
      for (const tree of trees) this.#drawTreeShadow(context, tree, hour);
      if (ghostTree) this.#drawTreeShadow(context, ghostTree, hour, 0.45);
    }

    for (const building of buildings) {
      this.#drawBuilding(context, building, building.id === selectedBuildingId, 1);
    }
    for (const tree of trees) {
      this.#drawTree(context, tree, tree.id === selectedTreeId, 1);
    }
    if (ghostTree) this.#drawTree(context, ghostTree, false, 0.58);
    this.#drawActiveEffects(context);
  }

  /**
   * One-shot fast-lane flash of the refresh-window casing
   * (design_plan.md §7 moment 2): the dirty-ROI casing sweeps
   * white → class color 80% → white over ~200 ms in the CURRENT class
   * color. Under reduced motion there is no sweep — the casing steps to
   * the class color for a brief static window instead.
   *
   * Ladder rule (interaction_simulations.md I-20): upward transitions
   * animate; a DOWNGRADE never flashes (downgrades are narrated by the
   * ledger on chrome surfaces) — it just updates the remembered rank.
   *
   * @param {object} rect dirty-rect shape `{minU, minV, maxU, maxV}` (same
   *        geometry `renderOverlay`'s `dirtyRect` uses).
   * @param {string} [classKind="fast_exact"] wire result class; unknown
   *        classes resolve conservatively to the `visual_pending` hue.
   * @param {object} [options]
   * @param {number} [options.durationMs=200] flash lifetime.
   * @param {boolean} [options.force=false] flash even on a ladder downgrade.
   * @returns {boolean} whether a flash was scheduled.
   */
  flashRegion(rect, classKind = "fast_exact", { durationMs = 200, force = false } = {}) {
    if (!rect) return false;
    const kind = resolveClassKind(classKind);
    const rank = RESULT_CLASS_LADDER[kind] ?? 0;
    const downgrade = this.#lastFlashRank !== null && rank < this.#lastFlashRank;
    this.#lastFlashRank = rank;
    if (downgrade && !force) return false;
    const lifetime = isReducedMotion() ? Math.max(durationMs, 1200) : Math.max(16, durationMs);
    this.#effects.push({
      kind: "roi-flash",
      rect,
      color: RESULT_CLASS_COLORS[kind],
      start: Date.now(),
      durationMs: lifetime,
      staticProgress: 0.5,
    });
    this.#kickEffects();
    return true;
  }

  /**
   * Collaborator reconciliation pulse (`.pulse-remote`,
   * interaction_simulations.md I-31): a brief highlight ring at the
   * remote-touched entity's position, expanding and fading once. Under
   * reduced motion the ring appears statically for the duration (no
   * expansion).
   *
   * @param {object} position entity anchor: `{u, v}` (tree-like scene UV)
   *        or world meters `{xM, yM}` / `{eastM, northM}`; an optional
   *        `canopyDiameterM` sizes the ring.
   * @param {object} [options]
   * @param {number} [options.durationMs=900] pulse lifetime.
   * @param {string} [options.color="#2B49CC"] ring color as `#rrggbb`
   *        (defaults to Ultramarine — the instrument-speaking hue; pass the
   *        actor's assigned presence hue to attribute the pulse).
   * @returns {boolean} whether a pulse was scheduled.
   */
  pulseRemote(position, { durationMs = 900, color = "#2B49CC" } = {}) {
    let uv = null;
    if (position && numericOrNull(position.u) !== null && numericOrNull(position.v) !== null) {
      uv = { u: Number(position.u), v: Number(position.v) };
    } else if (
      position &&
      (numericOrNull(position.xM) !== null || numericOrNull(position.eastM) !== null)
    ) {
      const worldUv = worldToUv(
        Number(position.xM ?? position.eastM),
        Number(position.yM ?? position.northM),
        this.scene,
      );
      uv = worldUv && Number.isFinite(worldUv.u) && Number.isFinite(worldUv.v) ? worldUv : null;
    }
    if (!uv) return false;
    const canopy = numericOrNull(position?.canopyDiameterM);
    this.#effects.push({
      kind: "remote-pulse",
      uv,
      baseRadiusPx: Math.max(20, canopy !== null ? canopy * 1.1 : 0),
      color,
      start: Date.now(),
      durationMs: Math.max(16, durationMs),
      staticProgress: 0.35,
    });
    this.#kickEffects();
    return true;
  }

  /** Draw every unexpired effect at its current (or static) progress. */
  #drawActiveEffects(context, nowMs = Date.now()) {
    for (const effect of this.#effects) {
      const progress = isReducedMotion()
        ? effect.staticProgress
        : Math.min(1, Math.max(0, (nowMs - effect.start) / effect.durationMs));
      if (effect.kind === "roi-flash") this.#drawRoiFlash(context, effect, progress);
      else this.#drawRemotePulse(context, effect, progress);
    }
  }

  #drawRoiFlash(context, effect, progress) {
    const rect = effect.rect;
    const corners = [
      uvToCanvas(rect.minU, rect.minV, this.scene),
      uvToCanvas(rect.maxU, rect.minV, this.scene),
      uvToCanvas(rect.maxU, rect.maxV, this.scene),
      uvToCanvas(rect.minU, rect.maxV, this.scene),
    ];
    const sweep = progress < 0.5 ? progress * 2 : (1 - progress) * 2;
    const color = isReducedMotion()
      ? effect.color
      : mixHex("#ffffff", effect.color, 0.8 * sweep);
    context.save();
    context.beginPath();
    context.moveTo(corners[0].x, corners[0].y);
    for (const corner of corners.slice(1)) context.lineTo(corner.x, corner.y);
    context.closePath();
    context.setLineDash([8, 7]);
    context.lineJoin = "round";
    // Cartographic casing (design_plan.md §6): a white halo under the
    // class-colored stroke keeps the flash legible on any orthophoto.
    context.lineWidth = 5;
    context.strokeStyle = "rgba(255, 255, 255, 0.85)";
    context.stroke();
    context.lineWidth = 2.5;
    context.strokeStyle = color;
    context.stroke();
    context.setLineDash([]);
    context.restore();
  }

  #drawRemotePulse(context, effect, progress) {
    const anchor = uvToCanvas(effect.uv.u, effect.uv.v, this.scene);
    const eased = 1 - (1 - progress) ** 2;
    const radius = effect.baseRadiusPx + 26 * eased;
    const alpha = isReducedMotion() ? 0.85 : 0.9 * (1 - progress);
    context.save();
    context.lineWidth = 4.5;
    context.strokeStyle = `rgba(255, 255, 255, ${Math.min(0.9, alpha + 0.1)})`;
    context.beginPath();
    context.arc(anchor.x, anchor.y, radius, 0, Math.PI * 2);
    context.stroke();
    context.lineWidth = 2.5;
    context.strokeStyle = withAlpha(effect.color, Math.max(0, alpha));
    context.stroke();
    context.restore();
  }

  /** Drop effects whose lifetime has elapsed. */
  #pruneEffects(nowMs = Date.now()) {
    this.#effects = this.#effects.filter((effect) => nowMs - effect.start < effect.durationMs);
  }

  /** Repaint the overlay (app's last draw + effects), or effects alone. */
  #repaintOverlayForEffects() {
    if (!this.overlay) return;
    if (this.#lastOverlayArgs) {
      this.renderOverlay(this.#lastOverlayArgs);
      return;
    }
    this.overlay.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    this.#drawActiveEffects(this.overlay);
  }

  /**
   * Paint the new effect immediately, then keep the overlay animating until
   * every effect expires. Full-motion mode re-renders the app's last overlay
   * args per animation frame; reduced motion (or a host without rAF, e.g.
   * node imports) steps statically and schedules one cleanup redraw at the
   * soonest expiry — no continuous repaint.
   */
  #kickEffects() {
    this.#repaintOverlayForEffects();
    if (this.#effectsFrame !== null || this.#effectsTimeout !== null) return;
    if (isReducedMotion() || typeof requestAnimationFrame === "undefined") {
      const soonestStop = Math.min(
        ...this.#effects.map((effect) => effect.start + effect.durationMs),
      );
      this.#effectsTimeout = setTimeout(() => {
        this.#effectsTimeout = null;
        this.#pruneEffects();
        this.#repaintOverlayForEffects();
        if (this.#effects.length > 0) this.#kickEffects();
      }, Math.max(0, soonestStop - Date.now() + 1));
      return;
    }
    const tick = () => {
      this.#effectsFrame = null;
      this.#pruneEffects();
      this.#repaintOverlayForEffects();
      if (this.#effects.length > 0) this.#effectsFrame = requestAnimationFrame(tick);
    };
    this.#effectsFrame = requestAnimationFrame(tick);
  }

  /**
   * On-canvas marker while the heat overlay mixes the non-scientific browser
   * preview with the exact layer (frontend_spec.md rendering layer 4).
   */
  #drawPreviewMarker(context) {
    const anchor = uvToCanvas(0, 0, this.scene);
    context.save();
    context.fillStyle = "rgba(11, 19, 16, 0.88)";
    context.beginPath();
    context.roundRect(anchor.x + 8, anchor.y + 8, 158, 19, 5);
    context.fill();
    context.fillStyle = "#ffd479";
    context.font = "700 10px Inter, sans-serif";
    context.fillText("PREVIEW · EXACT OWED", anchor.x + 15, anchor.y + 21);
    context.restore();
  }

  #drawDirtyRegion(context, rect) {
    const corners = [
      uvToCanvas(rect.minU, rect.minV, this.scene),
      uvToCanvas(rect.maxU, rect.minV, this.scene),
      uvToCanvas(rect.maxU, rect.maxV, this.scene),
      uvToCanvas(rect.minU, rect.maxV, this.scene),
    ];
    context.save();
    context.beginPath();
    context.moveTo(corners[0].x, corners[0].y);
    for (const corner of corners.slice(1)) context.lineTo(corner.x, corner.y);
    context.closePath();
    context.fillStyle = "rgba(91, 231, 174, 0.065)";
    context.fill();
    context.setLineDash([8, 7]);
    context.lineWidth = 1.5;
    context.strokeStyle = "rgba(122, 239, 191, 0.86)";
    context.shadowColor = "rgba(76, 215, 159, 0.45)";
    context.shadowBlur = 10;
    context.stroke();
    context.setLineDash([]);

    const labelX = corners[0].x + 9;
    const labelY = corners[0].y + 18;
    context.shadowBlur = 0;
    context.fillStyle = "rgba(11, 19, 16, 0.88)";
    context.roundRect(labelX - 4, labelY - 11, 100, 19, 5);
    context.fill();
    context.fillStyle = "#9af1cc";
    context.font = "700 10px Inter, sans-serif";
    context.fillText("REFRESH WINDOW", labelX + 3, labelY + 2);
    context.restore();
  }

  #drawTreeShadow(context, tree, hour, opacity = 1) {
    const origin = treeToWorld(tree, this.scene);
    const vector = shadowVectorMeters(tree, hour);
    const endpointUv = worldToUv(
      origin.xM + vector.eastM,
      origin.yM + vector.northM,
      this.scene,
    );
    const start = uvToCanvas(tree.u, tree.v, this.scene);
    const end = uvToCanvas(endpointUv.u, endpointUv.v, this.scene);
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = Math.max(10, Math.hypot(dx, dy));
    const angle = Math.atan2(dy, dx);
    const localScale = 0.52 + tree.v * 0.66;
    const width = Math.max(8, tree.canopyDiameterM * 1.2 * localScale);

    context.save();
    context.translate(start.x, start.y + 3);
    context.rotate(angle);
    context.filter = "blur(5px)";
    const gradient = context.createLinearGradient(0, 0, length, 0);
    gradient.addColorStop(0, `rgba(2, 5, 5, ${0.34 * opacity})`);
    gradient.addColorStop(0.55, `rgba(2, 5, 5, ${0.20 * opacity})`);
    gradient.addColorStop(1, "rgba(2, 5, 5, 0)");
    context.fillStyle = gradient;
    context.beginPath();
    context.ellipse(length * 0.48, 0, length * 0.52, width, 0, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  #drawTree(context, tree, selected, opacity) {
    const base = uvToCanvas(tree.u, tree.v, this.scene);
    const perspective = 0.58 + tree.v * 0.62;
    const canopyRadius = Math.max(11, tree.canopyDiameterM * 1.18 * perspective);
    const visualHeight = Math.max(18, tree.heightM * 1.25 * perspective);
    const crownY = base.y - visualHeight * 0.62;

    context.save();
    context.globalAlpha = opacity;

    if (selected) {
      context.beginPath();
      context.ellipse(base.x, base.y + 3, canopyRadius * 1.28, canopyRadius * 0.46, 0, 0, Math.PI * 2);
      context.fillStyle = "rgba(91, 235, 176, 0.11)";
      context.fill();
      context.lineWidth = 1.7;
      context.strokeStyle = "rgba(128, 244, 198, 0.95)";
      context.setLineDash([5, 4]);
      context.stroke();
      context.setLineDash([]);
    }

    const trunkGradient = context.createLinearGradient(base.x - 4, crownY, base.x + 5, base.y);
    trunkGradient.addColorStop(0, "#a77a51");
    trunkGradient.addColorStop(0.6, "#6d4a31");
    trunkGradient.addColorStop(1, "#3f2c22");
    context.fillStyle = trunkGradient;
    context.beginPath();
    context.moveTo(base.x - 3.3 * perspective, base.y);
    context.lineTo(base.x - 2.0 * perspective, crownY + canopyRadius * 0.24);
    context.lineTo(base.x + 2.4 * perspective, crownY + canopyRadius * 0.24);
    context.lineTo(base.x + 3.8 * perspective, base.y);
    context.closePath();
    context.fill();

    const palette = tree.preset === "evergreen"
      ? ["#9aca9d", "#3d805e", "#1d503d"]
      : tree.preset === "broad"
        ? ["#b2dc96", "#5a9f62", "#245c3e"]
        : ["#a5d894", "#4f915c", "#22543a"];

    const crownGradient = context.createRadialGradient(
      base.x - canopyRadius * 0.28,
      crownY - canopyRadius * 0.34,
      canopyRadius * 0.08,
      base.x,
      crownY,
      canopyRadius * 1.2,
    );
    crownGradient.addColorStop(0, palette[0]);
    crownGradient.addColorStop(0.5, palette[1]);
    crownGradient.addColorStop(1, palette[2]);

    context.shadowColor = "rgba(0, 0, 0, 0.35)";
    context.shadowBlur = 8;
    context.shadowOffsetY = 5;
    context.fillStyle = crownGradient;
    context.beginPath();
    if (tree.crown === "conical") {
      context.moveTo(base.x, crownY - canopyRadius * 1.15);
      context.bezierCurveTo(
        base.x - canopyRadius * 0.86,
        crownY - canopyRadius * 0.18,
        base.x - canopyRadius * 0.75,
        crownY + canopyRadius * 0.62,
        base.x,
        crownY + canopyRadius * 0.72,
      );
      context.bezierCurveTo(
        base.x + canopyRadius * 0.75,
        crownY + canopyRadius * 0.62,
        base.x + canopyRadius * 0.86,
        crownY - canopyRadius * 0.18,
        base.x,
        crownY - canopyRadius * 1.15,
      );
    } else {
      context.ellipse(
        base.x,
        crownY,
        canopyRadius * (tree.crown === "broad" ? 1.18 : 1),
        canopyRadius * 0.86,
        0,
        0,
        Math.PI * 2,
      );
    }
    context.fill();

    context.shadowBlur = 0;
    context.globalAlpha = opacity * 0.38;
    context.fillStyle = "#d9efb4";
    context.beginPath();
    context.ellipse(
      base.x - canopyRadius * 0.28,
      crownY - canopyRadius * 0.27,
      canopyRadius * 0.24,
      canopyRadius * 0.13,
      -0.35,
      0,
      Math.PI * 2,
    );
    context.fill();

    if (selected) {
      context.globalAlpha = 1;
      context.font = "700 10px Inter, sans-serif";
      const labelWidth = Math.max(94, context.measureText(tree.label).width + 18);
      context.fillStyle = "rgba(12, 17, 16, 0.92)";
      context.roundRect(base.x - labelWidth / 2, crownY - canopyRadius - 27, labelWidth, 22, 7);
      context.fill();
      context.strokeStyle = "rgba(122, 239, 191, 0.34)";
      context.stroke();
      context.fillStyle = "#eaf6f0";
      context.fillText(tree.label, base.x - labelWidth / 2 + 9, crownY - canopyRadius - 12);
    }
    context.restore();
  }

  #drawBuildingShadow(context, building, hour, opacity = 1) {
    // shadowVectorMeters reads heightM only, so a building passes as-is (the
    // model's influence rect already leans on that).
    const vector = shadowVectorMeters(building, hour);
    const center = buildingToWorld(building, this.scene);
    const endUv = worldToUv(center.xM + vector.eastM, center.yM + vector.northM, this.scene);
    const start = uvToCanvas(building.u, building.v, this.scene);
    const end = uvToCanvas(endUv.u, endUv.v, this.scene);
    const corners = building.footprintM.map(([xM, yM]) => {
      const cornerUv = worldToUv(xM, yM, this.scene);
      return uvToCanvas(cornerUv.u, cornerUv.v, this.scene);
    });
    // One canvas-space offset for the whole ring, measured at the building's
    // center the same way #drawTreeShadow measures its single-stroke vector.
    const polygon = buildingShadowPolygon(corners, {
      x: end.x - start.x,
      y: end.y - start.y,
    });

    context.save();
    context.filter = "blur(5px)";
    // Fade along the throw (footprint centroid → displaced centroid) in the
    // same 0.34/0.20 tones the tree shadows use, so both families read as
    // one shadow layer on the orthophoto.
    const near = centroidOf(corners);
    const far = centroidOf(polygon);
    const gradient = context.createLinearGradient(near.x, near.y, far.x, far.y);
    gradient.addColorStop(0, `rgba(2, 5, 5, ${0.34 * opacity})`);
    gradient.addColorStop(0.55, `rgba(2, 5, 5, ${0.20 * opacity})`);
    gradient.addColorStop(1, "rgba(2, 5, 5, 0)");
    context.fillStyle = gradient;
    context.beginPath();
    context.moveTo(polygon[0].x, polygon[0].y);
    for (const corner of polygon.slice(1)) context.lineTo(corner.x, corner.y);
    context.closePath();
    context.fill();
    context.restore();
  }

  #drawBuilding(context, building, selected, opacity) {
    // Same perspective regime as #drawTree's visualHeight (scales with v, the
    // near-bottom-of-site magnification) so a 15 m block and a 15 m tree
    // agree; the 10 px floor keeps a 4 m pavilion from collapsing into a
    // hairline slab at the top of the site.
    const perspective = 0.58 + building.v * 0.62;
    const wallHeightPx = Math.max(10, building.heightM * 1.25 * perspective);
    const corners = building.footprintM.map(([xM, yM]) => {
      const cornerUv = worldToUv(xM, yM, this.scene);
      return uvToCanvas(cornerUv.u, cornerUv.v, this.scene);
    });
    const { roof, base, walls } = buildingExtrusionPolygons(corners, wallHeightPx);
    const palette = BUILDING_PALETTES[building.preset] ?? BUILDING_PALETTES.pavilion;

    const trace = (points) => {
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      for (const point of points.slice(1)) context.lineTo(point.x, point.y);
      context.closePath();
    };

    context.save();
    context.globalAlpha = opacity;

    // Base under-fill first: it closes the silhouette so a skewed quad never
    // shows orthophoto between the near wall and the roof's far edge.
    trace(base);
    context.fillStyle = palette[2];
    context.fill();

    // Walls share one vertical ramp so the visible quads shade as a
    // continuous facade rather than two tones meeting at a corner.
    const roofTopY = Math.min(...roof.map((point) => point.y));
    const roofBottomY = Math.max(...roof.map((point) => point.y));
    const wallGradient = context.createLinearGradient(0, roofTopY, 0, roofBottomY + wallHeightPx);
    wallGradient.addColorStop(0, palette[1]);
    wallGradient.addColorStop(1, palette[2]);
    context.fillStyle = wallGradient;
    for (const wall of walls) {
      trace(wall);
      context.fill();
    }

    // Roof last: painter's occlusion — it covers the far wall bands, so only
    // viewer-facing sides read without any z-ordering.
    const roofGradient = context.createLinearGradient(0, roofTopY, 0, roofBottomY);
    roofGradient.addColorStop(0, palette[0]);
    roofGradient.addColorStop(1, palette[1]);
    context.fillStyle = roofGradient;
    trace(roof);
    context.fill();
    context.lineWidth = 1;
    context.strokeStyle = "rgba(21, 26, 24, 0.28)";
    context.stroke();

    if (selected) {
      // Selection in the tree vocabulary (dashed Viridian + soft fill) but on
      // the ground ring — the displaced base is where the walls meet earth;
      // painted AFTER the extrusion so the near wall cannot bite a
      // half-stroke out of the dashes.
      trace(base);
      context.fillStyle = "rgba(91, 235, 176, 0.11)";
      context.fill();
      context.lineWidth = 1.7;
      context.strokeStyle = "rgba(128, 244, 198, 0.95)";
      context.setLineDash([5, 4]);
      context.stroke();
      context.setLineDash([]);

      context.globalAlpha = 1;
      context.font = "700 10px Inter, sans-serif";
      const label = `${building.heightM} m`;
      const labelWidth = Math.max(60, context.measureText(label).width + 18);
      const chipX = centroidOf(roof).x - labelWidth / 2;
      const chipY = roofTopY - 30;
      context.fillStyle = "rgba(12, 17, 16, 0.92)";
      context.roundRect(chipX, chipY, labelWidth, 22, 7);
      context.fill();
      context.strokeStyle = "rgba(122, 239, 191, 0.34)";
      context.stroke();
      context.fillStyle = "#eaf6f0";
      context.fillText(label, chipX + 9, chipY + 15);
    }

    context.restore();
  }
}
