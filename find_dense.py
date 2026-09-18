import rasterio
import numpy as np

FOLDER = "Input_rasters"          # ← your sample data folder
N = 400                           # read everything down to 400 x 400
B = 20                            # then score it in 20 x 20 blocks

with rasterio.open(f"{FOLDER}/Building_DSM.tif") as s:
    dsm = s.read(1, out_shape=(N, N)).astype("float32")
    W, H = s.width, s.height
with rasterio.open(f"{FOLDER}/DEM.tif") as s:
    dem = s.read(1, out_shape=(N, N)).astype("float32")

height = dsm - dem
blocks = height.reshape(B, N // B, B, N // B).mean(axis=(1, 3))
r, c = np.unravel_index(blocks.argmax(), blocks.shape)

print("densest block, mean built height:", round(float(blocks.max()), 1), "m")
print("col_off =", int(c * W / B))
print("row_off =", int(r * H / B))