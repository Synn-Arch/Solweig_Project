import os
import rasterio
from rasterio.windows import Window

SRC = "Input_rasters"
DST = "Input_subset"

col_off, row_off = 1369, 2525     # ← from find_dense.py
size = 500                        # 500 px x 2 m = 1 km across

os.makedirs(DST, exist_ok=True)

# stop early if the window runs off the edge of the raster —
# otherwise you get files that claim one size and hold another
with rasterio.open(f"{SRC}/Building_DSM.tif") as src:
    if col_off + size > src.width or row_off + size > src.height:
        raise SystemExit(
            f"Window runs past the edge ({src.width} x {src.height} px). "
            "Reduce size, or move col_off / row_off."
        )

win = Window(col_off, row_off, size, size)

for name in ["Building_DSM.tif", "DEM.tif", "Trees.tif"]:
    with rasterio.open(f"{SRC}/{name}") as src:
        data = src.read(1, window=win)
        prof = src.profile.copy()
        prof.update(width=size, height=size,
                    transform=src.window_transform(win))
    with rasterio.open(f"{DST}/{name}", "w", **prof) as dst:
        dst.write(data, 1)
    print(name, data.shape,
          "min", round(float(data.min()), 1),
          "max", round(float(data.max()), 1))