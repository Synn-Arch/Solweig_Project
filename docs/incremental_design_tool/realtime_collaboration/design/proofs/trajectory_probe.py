# SPDX-License-Identifier: GPL-3.0-only
"""Per-step trajectory probe of shadow() at one cell for one patch."""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/tmp/r4_proof")
from corridor_bruteforce import PATCHES, build_main_tile, scene_of  # noqa: E402
from solweig_gpu.incremental.geometry import TreeSpec  # noqa: E402


def shadow_probe(amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale, probe):
    """Copy of shadow() (shadow.py:166-322) recording per-step state at ``probe``."""
    degrees = torch.pi / 180.0
    if azimuth == 0.0:
        azimuth = 1e-12
    azimuth = azimuth * degrees
    altitude = altitude * degrees
    dx = torch.tensor(0.0)
    dy = torch.tensor(0.0)
    dz = torch.tensor(0.0)
    sizex, sizey = a.shape
    temp = torch.zeros((sizex, sizey))
    tempvegdem = torch.zeros((sizex, sizey))
    tempvegdem2 = torch.zeros((sizex, sizey))
    sh = torch.zeros((sizex, sizey))
    vbshvegsh = torch.zeros((sizex, sizey))
    f = a.clone()
    bushplant = (bush > 1.0).float()
    vegsh = torch.zeros((sizex, sizey)) + bushplant

    pibyfour = torch.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = torch.sin(azimuth)
    cosazimuth = torch.cos(azimuth)
    tanazimuth = torch.tan(azimuth)
    signsinazimuth = torch.sign(sinazimuth)
    signcosazimuth = torch.sign(cosazimuth)
    dssin = torch.abs((1.0 / sinazimuth))
    dscos = torch.abs((1.0 / cosazimuth))
    tanaltitudebyscale = torch.tan(altitude) / scale

    index = 1.0
    trace = []
    pr, pc = probe
    while bool(amaxvalue >= dz) and bool(torch.abs(dx) < sizex) and bool(torch.abs(dy) < sizey):
        if bool(pibyfour <= azimuth < threetimespibyfour) or bool(
            fivetimespibyfour <= azimuth < seventimespibyfour
        ):
            dy = signsinazimuth * index
            dx = -1.0 * signcosazimuth * torch.abs(torch.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * torch.abs(torch.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos
        dz = ds * index * tanaltitudebyscale

        tempvegdem.zero_()
        tempvegdem2.zero_()
        temp.zero_()
        absdx = torch.abs(dx)
        absdy = torch.abs(dy)
        xc1 = int((dx + absdx) / 2.0)
        xc2 = int(sizex + (dx - absdx) / 2.0)
        yc1 = int((dy + absdy) / 2.0)
        yc2 = int(sizey + (dy - absdy) / 2.0)
        xp1 = int(-((dx - absdx) / 2.0))
        xp2 = int(sizex - (dx + absdx) / 2.0)
        yp1 = int(-((dy - absdy) / 2.0))
        yp2 = int(sizey - (dy + absdy) / 2.0)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = torch.max(f, temp)
        sh[f > a] = 1.0
        sh[f <= a] = 0.0

        fabovea = tempvegdem > a
        gabovea = tempvegdem2 > a
        vegsh2 = fabovea.float() - gabovea.float()

        vegsh = torch.max(vegsh, vegsh2)
        vegsh[(vegsh * sh > 0.0)] = 0.0

        vbshvegsh = vegsh + vbshvegsh

        if index == 1.0:
            firstvegdem = tempvegdem - temp
            firstvegdem[firstvegdem <= 0.0] = 1000.0
            vegsh[firstvegdem < dz] = 1.0
            vegsh = vegsh * (vegdem2 > a).float()
            vbshvegsh.zero_()

        trace.append(
            {
                "i": int(index),
                "dx": int(dx),
                "dy": int(dy),
                "dz": float(dz),
                "read": (pr + int(dx), pc + int(dy)),
                "veg_at_read": float(vegdem[pr + int(dx), pc + int(dy)])
                if 0 <= pr + int(dx) < sizex and 0 <= pc + int(dy) < sizey
                else None,
                "veg2_at_read": float(vegdem2[pr + int(dx), pc + int(dy)])
                if 0 <= pr + int(dx) < sizex and 0 <= pc + int(dy) < sizey
                else None,
                "a_at_read": float(a[pr + int(dx), pc + int(dy)])
                if 0 <= pr + int(dx) < sizex and 0 <= pc + int(dy) < sizey
                else None,
                "vegsh": float(vegsh[pr, pc]),
                "sh": float(sh[pr, pc]),
                "f": float(f[pr, pc]),
                "vegsh2": float(vegsh2[pr, pc]),
            }
        )
        index += 1.0

    sh = 1.0 - sh
    vbshvegsh[vbshvegsh > 0.0] = 1.0
    vbshvegsh = vbshvegsh - vegsh
    vegsh[vegsh > 0.0] = 1.0
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh
    return float(vegsh[pr, pc]), trace


if __name__ == "__main__":
    cache, grid, layer = build_main_tile()
    rng = np.random.default_rng(20260904)

    def pos(row, col):
        return (col + 0.5) * cache.pixel_size_m, (cache.rows - row - 0.5) * cache.pixel_size_m

    def open_pos():
        while True:
            r, c = int(rng.integers(2, 56)), int(rng.integers(2, 56))
            if not (4 <= r < 12 and 4 <= c < 12) and not (40 <= r < 50 and 38 <= c < 48):
                return r, c

    for i in range(4):
        r, c = open_pos()
        layer.add_tree(TreeSpec(f"t{i}", *pos(r, c), float(rng.uniform(3, 11)), float(rng.uniform(1.5, 4))))
    layer.add_tree(TreeSpec("corner_ne", *pos(2, 57), 9.0, 3.0))
    layer.add_tree(TreeSpec("corner_sw", *pos(57, 2), 6.0, 2.5))
    layer.add_tree(TreeSpec("overlap", 30.5 * cache.pixel_size_m, 20.0 * cache.pixel_size_m, 4.0, 3.0))
    layer.add_tree(TreeSpec("tall", *pos(30, 30), 18.0, 3.0))
    scene = scene_of(cache, layer)

    altitude, azimuth, _ring = PATCHES[34]
    print(f"patch 34: alt={float(altitude)} az={float(azimuth)}")
    print(f"a[t]={float(scene.a[56, 34])} vegdsm[t]={float(scene.vegdsm[56, 34])}")
    for amp in (12.0, 18.0):
        bit, trace = shadow_probe(
            torch.tensor(amp), scene.a, scene.vegdsm, scene.vegdsm2, scene.bush,
            azimuth, altitude, 1.0 / cache.pixel_size_m, (56, 34),
        )
        print(f"--- amp={amp}: final cube bit={bit}")
        for t in trace:
            print(
                f"  i={t['i']:2d} read={t['read']} veg={t['veg_at_read']} "
                f"a={t['a_at_read']} dz={t['dz']:.3f} vegsh2={t['vegsh2']:+.0f} "
                f"vegsh={t['vegsh']:.0f} sh={t['sh']:.0f} f={t['f']:.2f}"
            )
