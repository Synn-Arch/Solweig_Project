import os
import rasterio

FOLDER = "Input_rasters"          # ← your unzipped sample folder
NAMES  = ["Building_DSM.tif", "DEM.tif", "Trees.tif"]

profiles = {}

for name in NAMES:
    path = os.path.join(FOLDER, name)
    with rasterio.open(path) as src:
        profiles[name] = (src.crs, src.transform, src.width, src.height)
        print(name)
        print("   size        ", src.width, "x", src.height, "px")
        print("   CRS         ", src.crs)
        print("   resolution  ", src.res, "(map units per pixel)")
        print("   bands       ", src.count)
        print("   bounds      ", [round(b, 1) for b in src.bounds])
        print()

# same CRS, same pixel grid, same origin, same size?
unique = set(profiles.values())
print("ALIGNED" if len(unique) == 1 else "NOT ALIGNED — the model will refuse this")

# how tall is the tallest building? read coarsely, this is only a sanity check
with rasterio.open(os.path.join(FOLDER, "Building_DSM.tif")) as s:
    dsm = s.read(1, out_shape=(300, 300))
with rasterio.open(os.path.join(FOLDER, "DEM.tif")) as s:
    dem = s.read(1, out_shape=(300, 300))

height = dsm - dem
print("tallest building  ~", round(float(height.max()), 1), "m")
print("mean built height ~", round(float(height[height > 1].mean()), 1), "m")