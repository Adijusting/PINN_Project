import torch
import torch.nn.functional as F
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gc
from pystac_client import Client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from model import FloodPINN

def fetch_truth_mask(bbox):
    print("1. Fetching True Satellite Imagery...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime="2021-07-14/2021-07-18")
    
    for item in search.items():
        try:
            with rasterio.open(item.assets["vh"].href) as src:
                proj_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
                window = from_bounds(*proj_bbox, transform=src.transform)
                radar_data = src.read(1, window=window)
            
            if not (np.isnan(radar_data).all() or np.nanmax(radar_data) == 0):
                # Downsample by 4 to save memory, just like our visualizer
                radar_data = radar_data[::4, ::4]
                
                # Water acts like a mirror to radar, appearing pitch black.
                # We classify the darkest 10% of the image as our "True Water" mask
                water_threshold = np.nanpercentile(radar_data, 10)
                truth_mask = (radar_data < water_threshold).astype(np.float32)
                return truth_mask
        except Exception:
            continue
    raise Exception("Could not fetch satellite data.")

def generate_ai_mask():
    print("2. Generating AI Flood Predictions...")
    model = FloodPINN()
    model.load_state_dict(torch.load('models/trained_flood_pinn.pth', weights_only=True))

    ds = xr.open_dataset('data/pinn_training_data.nc')
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    elevation = ds['elevation'].values
    precip = ds['precipitation'].isel(valid_time=48).values

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    num_points = len(lat_grid.flatten())
    
    inputs = np.column_stack((
        lat_grid.flatten(), lon_grid.flatten(),
        np.full(num_points, 48.0, dtype=np.float32),
        elevation.flatten(),
        np.full(num_points, float(precip), dtype=np.float32),
        np.full(num_points, 290.0, dtype=np.float32)
    )).astype(np.float32)

    depths_flat = np.zeros(num_points, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, num_points, 50000):
            depths_flat[i:i+50000] = model(torch.tensor(inputs[i:i+50000]))[:, 0].numpy()

    # Clean up memory
    del inputs
    gc.collect()

    depth_map = depths_flat.reshape(lat_grid.shape)
    
    # Create AI Mask: Anything deeper than 10cm is "Water" (1.0), else "Dry" (0.0)
    ai_mask = (depth_map > 0.1).astype(np.float32)
    return ai_mask

def run_validation():
    bbox = [6.8, 50.4, 7.2, 50.7]
    
    # Get both masks
    truth_mask = fetch_truth_mask(bbox)
    ai_mask = generate_ai_mask()

    print("3. Aligning Grids and Calculating Accuracy...")
    # Convert numpy arrays to PyTorch tensors so we can resize them
    # We add dummy batch/channel dimensions [1, 1, H, W] for the interpolator
    ai_tensor = torch.tensor(ai_mask).unsqueeze(0).unsqueeze(0)
    
    # Magically resize the AI prediction grid to perfectly match the satellite grid
    ai_aligned = F.interpolate(ai_tensor, size=truth_mask.shape, mode='nearest').squeeze().numpy()

    # Calculate Intersection over Union (IoU)
    intersection = np.logical_and(ai_aligned == 1, truth_mask == 1).sum()
    union = np.logical_or(ai_aligned == 1, truth_mask == 1).sum()
    
    iou_score = (intersection / union) * 100 if union > 0 else 0.0
    
    # Create an Error Map for visualization
    # 0 = True Dry (Gray)
    # 1 = AI False Alarm (Red)
    # 2 = Missed Flood (Yellow)
    # 3 = True Water Match! (Blue)
    error_map = np.zeros_like(truth_mask)
    error_map[(ai_aligned == 1) & (truth_mask == 0)] = 1
    error_map[(ai_aligned == 0) & (truth_mask == 1)] = 2
    error_map[(ai_aligned == 1) & (truth_mask == 1)] = 3

    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Intersection over Union (IoU): {iou_score:.2f}%")
    if iou_score > 15.0:
        print("Note: In raw hydrological AI, an IoU > 15% on a crude physics model is highly promising!")

    print("\n4. Rendering Diagnostic Dashboard...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Lock the AI prediction to 0-1 so water always shows as dark blue
    axes[0].imshow(ai_aligned, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('AI Prediction (Resized)')
    axes[0].axis('off')
    
    axes[1].imshow(truth_mask, cmap='gray')
    axes[1].set_title('Satellite Truth')
    axes[1].axis('off')
    
    cmap_error = ListedColormap(['#e0e0e0', '#ff4d4d', '#ffcc00', '#0066cc'])
    # Force the Error Map to strictly use 0=Gray, 1=Red, 2=Yellow, 3=Blue
    axes[2].imshow(error_map, cmap=cmap_error, vmin=0, vmax=3) 
    axes[2].set_title('Error Map (Blue=Match, Red=False Alarm)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_validation()