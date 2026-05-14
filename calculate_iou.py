import numpy as np
import xarray as xr
import rasterio
import os
from validate import generate_ai_mask

def run_iou_analysis():
    print("1. Generating AI Prediction Map...")
    depth_map = generate_ai_mask()
    
    # DYNAMIC THRESHOLD: Because the AI is unnormalized, we can't use "0.05 meters".
    # Instead, we isolate the top 15% of the "deepest" predictions on the map.
    threshold_value = np.nanpercentile(depth_map, 85)
    ai_binary = np.where(depth_map > threshold_value, 1, 0)

    print("2. Searching for Ground Truth (Satellite Observation)...")
    sentinel_path = 'data/sentinel1_ahr_flood.tif'
    
    if os.path.exists(sentinel_path):
        print("   -> Found official Sentinel-1 data. Loading...")
        with rasterio.open(sentinel_path) as src:
            sat_data = src.read(1, out_shape=ai_binary.shape, resampling=rasterio.enums.Resampling.nearest)
            sat_binary = np.where(sat_data > 0, 1, 0)
    else:
        print("   -> Missing Sentinel-1 map! Generating 'Topographic Proxy' for testing...")
        # FALLBACK: If we don't have the satellite map, we grade the AI against the terrain.
        # We assume the lowest 15% of the valley elevation *should* be the flood zone.
        ds = xr.open_dataset('data/pinn_training_data.nc')
        elev = ds['elevation'].values
        elev_threshold = np.nanpercentile(elev, 15)
        sat_binary = np.where(elev < elev_threshold, 1, 0)

    print("3. Calculating Spatial Metrics (Intersection over Union)...")
    intersection = np.logical_and(ai_binary, sat_binary).sum()
    union = np.logical_or(ai_binary, sat_binary).sum()
    
    iou_score = (intersection / union) * 100 if union > 0 else 0

    print("\n========================================")
    print(f"FINAL IoU ACCURACY: {iou_score:.2f}%")
    print("========================================")
    
    print("\nNote: For a perfect scientific evaluation, download the EMSR517 Sentinel-1 raster")
    print("from the Copernicus Emergency Management Service and save it to the /data folder.")

if __name__ == "__main__":
    run_iou_analysis()