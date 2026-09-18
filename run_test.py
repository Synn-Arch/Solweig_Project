import os
from solweig_gpu import thermal_comfort

def main():
    thermal_comfort(
        base_path=os.path.abspath("Input_subset"),
        selected_date_str="2020-08-13",       # must match the met file
        building_dsm_filename="Building_DSM.tif",
        dem_filename="DEM.tif",
        trees_filename="Trees.tif",
        landcover_filename=None,
        tile_size=600,                        # larger than 500 → one tile
        overlap=20,
        use_own_met=True,
        own_met_file=os.path.abspath("ownmet_Forcing_data.txt"),
        save_tmrt=True,
        save_svf=True,
        save_shadow=True,
    )

if __name__ == "__main__":
    main()