from pystac_client import Client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

def fetch_and_map_friction():
    print("1. Loading API Vault...")
    load_dotenv()
    
    print("2. Connecting to Planetary Computer (ESA WorldCover)...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    bbox = [6.8, 50.4, 7.2, 50.7] # Ahr Valley
    
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox,
        datetime="2021-01-01/2021-12-31"
    )
    
    items = list(search.items())
    if not items:
        raise Exception("Could not find ESA WorldCover data for 2021.")
        
    item = items[0]
    map_url = item.assets["map"].href

    print("3. Streaming Land Cover classifications (Memory-Safe + Manual Slice)...")
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_CACHEMAX=512 
    )
    
    with env:
        with rasterio.open(map_url) as src:
            proj_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
            window = from_bounds(*proj_bbox, transform=src.transform)
            
            # Download full res, manually slice to bypass server smoothing
            raw_data = src.read(1, window=window)
            land_cover_data = raw_data[::3, ::3]
            
            out_h, out_w = land_cover_data.shape
            new_transform = src.window_transform(window) * rasterio.Affine.scale(
                window.width / out_w,
                window.height / out_h
            )
            
            profile = src.profile
            profile.update({
                'height': out_h,
                'width': out_w,
                'transform': new_transform,
                'dtype': 'float32'
            })

    print("4. Translating Land Cover into Physical Friction (Manning's n)...")
    friction_map = np.zeros_like(land_cover_data, dtype=np.float32)
    
    friction_map[land_cover_data == 10] = 0.100  # Trees/Forests 
    friction_map[land_cover_data == 20] = 0.050  # Shrubland
    friction_map[land_cover_data == 30] = 0.035  # Grassland
    friction_map[land_cover_data == 40] = 0.040  # Cropland
    friction_map[land_cover_data == 50] = 0.060  # Built-up/Cities
    friction_map[land_cover_data == 60] = 0.030  # Bare/Sparse Vegetation
    friction_map[land_cover_data == 80] = 0.025  # Rivers
    friction_map[friction_map == 0] = 0.035      # Fallback 

    print("5. Saving Master Friction Map to /data folder...")
    os.makedirs('data', exist_ok=True)
    with rasterio.open('data/mannings_n_ahr_valley.tif', 'w', **profile) as dst:
        dst.write(friction_map, 1)

    print("6. Rendering Friction Dashboard...")
    plt.figure(figsize=(10, 8))
    plt.imshow(friction_map, cmap='hot_r', vmin=0.025, vmax=0.100)
    plt.colorbar(label="Manning's Roughness Coefficient (n)")
    plt.title("Ahr Valley - Friction Map (ESA WorldCover 2021)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    fetch_and_map_friction()