# SPDX-License-Identifier: GPL-3.0-only
"""Window-typed raster I/O for the incremental SOLWEIG design tool.

All reads/writes go through :class:`~solweig_gpu.incremental.geometry.RasterWindow`
half-open pixel ranges, so a window computed by the invalidation geometry is
the single source of truth for what is read from and written to disk. Writing
a window into an existing raster never touches pixels outside the window
(outside-window invariance is enforced structurally, not by convention).
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window as RasterioWindow

from .geometry import RasterWindow


def _to_rasterio_window(window: RasterWindow) -> RasterioWindow:
    """Convert a half-open RasterWindow to a rasterio Window."""
    return RasterioWindow(
        col_off=window.col_start,
        row_off=window.row_start,
        width=window.col_stop - window.col_start,
        height=window.row_stop - window.row_start,
    )


def read_window(
    path,
    window: RasterWindow,
    *,
    bands: int | tuple[int, ...] = 1,
) -> np.ndarray:
    """Read a half-open window from a raster as ``float32``.

    Windows extending past the raster extent are clamped to the raster
    extent (the incremental halo may overhang the domain edge; rasterio
    itself would silently truncate to the intersection, so clamping is
    explicit and total here).

    Parameters
    ----------
    path:
        Raster path (str or Path).
    window:
        Half-open pixel window, clamped to the raster extent before reading.
    bands:
        Band index or tuple of band indices.

    Returns
    -------
    ``numpy.ndarray`` of shape ``(rows, cols)`` for a scalar ``bands`` and
    ``(bands, rows, cols)`` for a tuple of bands (rasterio semantics).
    """
    with rasterio.open(path) as src:
        window = window.clamp(rows=src.height, cols=src.width)
        data = src.read(bands, window=_to_rasterio_window(window))
    return data.astype(np.float32, copy=False)


def read_window_like(
    path,
    window: RasterWindow,
    reference: np.ndarray,
) -> np.ndarray:
    """Read ``window`` and validate it matches ``reference`` spatial shape.

    The incremental worker pairs each full-domain tensor with a windowed
    tensor; this helper fails fast when the raster grid and the in-memory
    window disagree.
    """
    data = read_window(path, window)
    if data.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"window read {data.shape[-2:]} from {path} does not match "
            f"reference shape {reference.shape[-2:]}"
        )
    return data


def write_window(
    path,
    window: RasterWindow,
    data: np.ndarray,
    *,
    band: int = 1,
) -> None:
    """Write a 2-D array into ``window`` of an existing raster.

    Only the pixels inside the half-open window are modified. The array
    shape must equal the window shape exactly (no silent clamping on write:
    a halo that overhangs the domain must be cropped by the caller before
    publication).
    """
    expected = (window.row_stop - window.row_start, window.col_stop - window.col_start)
    if data.ndim != 2 or tuple(data.shape) != expected:
        raise ValueError(
            f"data shape {tuple(data.shape[-2:])} does not match window shape {expected}"
        )
    with rasterio.open(path, "r+") as dst:
        dst.write(
            data.astype(np.float32, copy=False),
            band,
            window=_to_rasterio_window(window),
        )


def copy_window(
    src_path,
    dst_path,
    window: RasterWindow,
    *,
    band: int = 1,
) -> None:
    """Copy one band of ``window`` from src to dst raster (bitwise values)."""
    with rasterio.open(src_path) as src:
        data = src.read(band, window=_to_rasterio_window(window))
    with rasterio.open(dst_path, "r+") as dst:
        dst.write(data, band, window=_to_rasterio_window(window))


def window_bounds_world(
    path,
    window: RasterWindow,
) -> tuple[float, float, float, float]:
    """World (left, bottom, right, top) bounds of a window in ``path``."""
    with rasterio.open(path) as src:
        return src.window_bounds(_to_rasterio_window(window))
