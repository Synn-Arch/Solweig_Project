#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Build non-georeferenced demo assets from a SOLWEIG test tile.

This script is not part of the scientific solver. It creates a presentation
asset and a downsampled UTCI fixture for the browser prototype.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageEnhance, ImageFilter
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter, shift


OUTPUT_WIDTH = 1400
OUTPUT_HEIGHT = 900
MAP_QUAD = np.array(
    [[238.0, 106.0], [1162.0, 106.0], [1364.0, 850.0], [36.0, 850.0]],
    dtype=np.float32,
)


def _read(path: Path, band: int = 1) -> tuple[np.ndarray, rasterio.DatasetReader]:
    dataset = rasterio.open(path)
    return dataset.read(band).astype(np.float32), dataset


def _normalize(array: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    finite = array[np.isfinite(array)]
    minimum, maximum = np.percentile(finite, [low, high])
    return np.clip((array - minimum) / max(maximum - minimum, 1e-6), 0.0, 1.0)


def _hillshade(dem: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(dem)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(43.0)
    shade = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    return _normalize(shade, 1.0, 99.0)


def _palette_landcover(landcover: np.ndarray, shade: np.ndarray) -> np.ndarray:
    palette = {
        1: np.array([87, 91, 94], dtype=np.float32),      # asphalt
        2: np.array([156, 151, 142], dtype=np.float32),  # roofs, overwritten later
        4: np.array([76, 109, 76], dtype=np.float32),    # legacy vegetation code
        5: np.array([103, 128, 82], dtype=np.float32),   # grass
        6: np.array([139, 126, 101], dtype=np.float32),  # bare soil
        7: np.array([64, 105, 124], dtype=np.float32),   # water
    }
    rgb = np.zeros((*landcover.shape, 3), dtype=np.float32)
    for code, color in palette.items():
        rgb[landcover == code] = color
    rgb[np.all(rgb == 0, axis=-1)] = np.array([112, 116, 106], dtype=np.float32)
    lighting = 0.72 + 0.38 * shade[..., None]
    return np.clip(rgb * lighting, 0, 255)


def _resize(array: np.ndarray, size: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    return cv2.resize(array, (size, size), interpolation=interpolation)


def _render_topdown(
    building_dsm: np.ndarray,
    dem: np.ndarray,
    trees: np.ndarray,
    landcover: np.ndarray,
    shadow: np.ndarray,
    *,
    size: int = 1050,
) -> Image.Image:
    rng = np.random.default_rng(21)
    shade = _hillshade(dem)
    ground = _palette_landcover(landcover.astype(np.int16), shade)

    # Add low-amplitude multi-scale texture so the result reads as a map render,
    # not a flat categorical raster.
    noise = rng.normal(0.0, 1.0, dem.shape)
    noise = gaussian_filter(noise, sigma=1.4)
    noise = _normalize(noise, 1.0, 99.0) - 0.5
    ground += noise[..., None] * 16.0

    building_height = np.clip(building_dsm - dem, 0.0, None)
    building_mask = building_height >= 2.0
    tree_mask = trees >= 1.5

    # Solar shadow output uses 0 for shade and 1 for sunlit in daytime.
    ground *= (0.79 + 0.21 * np.clip(shadow, 0.0, 1.0))[..., None]

    top = _resize(ground, size).astype(np.float32)
    height = _resize(building_height, size, cv2.INTER_LINEAR)
    mask = _resize(building_mask.astype(np.uint8), size, cv2.INTER_NEAREST).astype(bool)

    # Faux extrusion. Each height slice is shifted toward the lower-right to
    # expose a darker wall face beneath the roof plane.
    wall_layer = np.zeros_like(top)
    wall_alpha = np.zeros((size, size), dtype=np.float32)
    max_step = 20
    for step in range(1, max_step + 1):
        threshold_m = step * 1.1
        slice_mask = height > threshold_m
        shifted = shift(
            slice_mask.astype(np.float32),
            shift=(step * 0.32, step * 0.48),
            order=0,
            mode="constant",
            cval=0.0,
        ) > 0.5
        exposed = shifted & ~mask
        if not exposed.any():
            continue
        wall_layer[exposed] = np.array([83, 82, 80], dtype=np.float32) + step * 0.7
        wall_alpha[exposed] = np.maximum(wall_alpha[exposed], 0.93)

    top = top * (1.0 - wall_alpha[..., None]) + wall_layer * wall_alpha[..., None]

    roof_noise = _resize(gaussian_filter(rng.normal(size=dem.shape), 1.0), size)
    roof_noise = _normalize(roof_noise, 1.0, 99.0)
    roof_color = np.stack(
        [166 + 24 * roof_noise, 163 + 20 * roof_noise, 157 + 18 * roof_noise],
        axis=-1,
    )
    roof_light = np.clip(0.84 + _resize(shade, size)[..., None] * 0.24, 0.8, 1.1)
    roof_color *= roof_light
    top[mask] = roof_color[mask]

    # Existing tree canopy: blurred mass plus high-frequency crown highlights.
    tree_height = _resize(trees, size, cv2.INTER_LINEAR)
    tree_binary = _resize(tree_mask.astype(np.uint8), size, cv2.INTER_NEAREST).astype(bool)
    canopy_alpha = gaussian_filter(tree_binary.astype(np.float32), sigma=1.4)
    canopy_alpha = np.clip(canopy_alpha * 0.94, 0.0, 0.92)
    crown = _normalize(tree_height, 0.5, 99.5)
    crown_noise = _resize(gaussian_filter(rng.normal(size=dem.shape), 0.8), size)
    crown_noise = _normalize(crown_noise, 1.0, 99.0)
    canopy = np.stack(
        [52 + 26 * crown_noise, 88 + 62 * crown + 24 * crown_noise, 52 + 28 * crown_noise],
        axis=-1,
    )
    top = top * (1.0 - canopy_alpha[..., None]) + canopy * canopy_alpha[..., None]

    image = Image.fromarray(np.clip(top, 0, 255).astype(np.uint8), mode="RGB")
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=45, threshold=3))
    image = ImageEnhance.Contrast(image).enhance(1.06)
    return image


def _warp_scene(topdown: Image.Image) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    source = np.array(
        [
            [0.0, 0.0],
            [topdown.width - 1.0, 0.0],
            [topdown.width - 1.0, topdown.height - 1.0],
            [0.0, topdown.height - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, MAP_QUAD)
    rgba = cv2.cvtColor(np.asarray(topdown), cv2.COLOR_RGB2RGBA)
    warped = cv2.warpPerspective(
        rgba,
        homography,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # A dark studio background and soft map shadow support the UI's realistic,
    # model-viewer aesthetic.
    yy, xx = np.mgrid[0:OUTPUT_HEIGHT, 0:OUTPUT_WIDTH]
    radial = np.sqrt(
        ((xx - OUTPUT_WIDTH * 0.5) / OUTPUT_WIDTH) ** 2
        + ((yy - OUTPUT_HEIGHT * 0.55) / OUTPUT_HEIGHT) ** 2
    )
    background = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 4), dtype=np.uint8)
    background[..., 0] = np.clip(23 - radial * 16, 7, 23).astype(np.uint8)
    background[..., 1] = np.clip(29 - radial * 17, 9, 29).astype(np.uint8)
    background[..., 2] = np.clip(30 - radial * 17, 10, 30).astype(np.uint8)
    background[..., 3] = 255

    alpha = warped[..., 3:4].astype(np.float32) / 255.0
    composed = (
        background[..., :3].astype(np.float32) * (1.0 - alpha)
        + warped[..., :3].astype(np.float32) * alpha
    )

    # Add a narrow outline around the site boundary.
    outline = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.uint8)
    cv2.polylines(outline, [MAP_QUAD.astype(np.int32)], True, 170, 2, cv2.LINE_AA)
    composed = np.clip(composed + outline[..., None] * np.array([0.05, 0.09, 0.08]), 0, 255)

    unit_source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    unit_h = cv2.getPerspectiveTransform(unit_source, MAP_QUAD)
    inverse_h = np.linalg.inv(unit_h)
    return Image.fromarray(composed.astype(np.uint8), mode="RGB"), unit_h, inverse_h


def _downsample_fixture(
    utci: np.ndarray,
    building_dsm: np.ndarray,
    dem: np.ndarray,
    *,
    size: int = 128,
) -> tuple[list[float | None], list[int]]:
    valid = np.isfinite(utci)
    filled = np.where(valid, utci, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    sum_resized = cv2.resize(filled, (size, size), interpolation=cv2.INTER_AREA)
    weight_resized = cv2.resize(weight, (size, size), interpolation=cv2.INTER_AREA)
    averaged = np.divide(
        sum_resized,
        weight_resized,
        out=np.zeros_like(sum_resized),
        where=weight_resized > 0.05,
    )

    building = cv2.resize(
        ((building_dsm - dem) >= 2.0).astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
    mask = building > 0.33
    averaged[mask] = np.nan

    values: list[float | None] = []
    for value in averaged.ravel():
        values.append(None if not np.isfinite(value) else round(float(value), 2))
    return values, mask.astype(np.uint8).ravel().tolist()


def build(args: argparse.Namespace) -> None:
    building, building_ds = _read(args.building)
    dem, _ = _read(args.dem)
    trees, _ = _read(args.trees)
    utci, _ = _read(args.utci, band=args.hour_band)
    shadow, _ = _read(args.shadow, band=args.hour_band)

    with rasterio.open(args.landcover) as landcover_ds:
        window = from_bounds(*building_ds.bounds, transform=landcover_ds.transform)
        landcover = landcover_ds.read(
            1,
            window=window,
            out_shape=building.shape,
            resampling=Resampling.nearest,
            boundless=True,
        ).astype(np.float32)

    topdown = _render_topdown(building, dem, trees, landcover, shadow)
    scene, homography, inverse_h = _warp_scene(topdown)
    args.output.mkdir(parents=True, exist_ok=True)
    scene.save(args.output / "site_base.webp", "WEBP", quality=91, method=6)

    values, building_mask = _downsample_fixture(utci, building, dem)
    finite = np.asarray([value for value in values if value is not None], dtype=np.float32)
    metadata = {
        "schemaVersion": 1,
        "scene": {
            "name": "SOLWEIG 1 km² teaching tile",
            "widthMeters": 1000,
            "heightMeters": 1000,
            "pixelSizeMeters": 2,
            "canvasWidth": OUTPUT_WIDTH,
            "canvasHeight": OUTPUT_HEIGHT,
            "mapQuad": MAP_QUAD.round(4).tolist(),
            "homography": homography.round(9).tolist(),
            "inverseHomography": inverse_h.round(9).tolist(),
        },
        "analysis": {
            "gridWidth": 128,
            "gridHeight": 128,
            "hourBand": args.hour_band,
            "label": "12:00 local · baseline UTCI",
            "minimum": round(float(finite.min()), 2),
            "maximum": round(float(finite.max()), 2),
            "mean": round(float(finite.mean()), 2),
            "utci": values,
            "buildingMask": building_mask,
        },
        "provenance": {
            "note": (
                "Non-georeferenced, downsampled visualization derived from "
                "the repository test tile."
            ),
            "scientificUse": False,
        },
    }
    (args.output / "baseline.json").write_text(
        json.dumps(metadata, separators=(",", ":")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", type=Path, required=True)
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--landcover", type=Path, required=True)
    parser.add_argument("--utci", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--hour-band", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())
