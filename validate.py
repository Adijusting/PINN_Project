import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from model import PINN

def generate_ai_mask():
    print("1. Loading Saved AI Brain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PINN().to(device)
    model.load_state_dict(torch.load('models/pinn_ahr_valley.pth', map_location=device))
    model.eval()

    print("2. Loading Environment Grid...")
    ds = xr.open_dataset('data/pinn_training_data.nc')
    
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    elevation = ds['elevation'].values
    precip_1d = ds['precipitation'].values
    friction = ds['friction'].values
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    num_points = lon_grid.size

    time_step = 48.0
    precip_val = precip_1d[-1] if len(precip_1d) > 0 else 0.0

    print("3. Assembling Memory-Safe Validation Inputs...")
    # THE FIX: Cast the huge arrays to float32 BEFORE stacking them!
    lat_flat = lat_grid.flatten().astype(np.float32)
    lon_flat = lon_grid.flatten().astype(np.float32)
    elev_flat = elevation.flatten().astype(np.float32)
    friction_flat = friction.flatten().astype(np.float32)
    
    # 7-Variable Input Stack (Must match the 7 inputs in model.py!)
    inputs_np = np.column_stack((
        lat_flat, 
        lon_flat,
        np.full(num_points, time_step, dtype=np.float32),         # Time
        elev_flat,                                                # Topography
        np.full(num_points, float(precip_val), dtype=np.float32), # Rain
        np.full(num_points, 290.0, dtype=np.float32),             # Init Depth
        friction_flat                                             # Manning's Friction
    ))

    inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32).to(device)

    print("4. AI is Predicting the Flood Map (in safe chunks)...")
    depth_list = []
    chunk_size = 50000  # Safe chunk size for CPU inference
    
    with torch.no_grad():
        for i in range(0, len(inputs_tensor), chunk_size):
            # Grab a chunk of the map
            batch = inputs_tensor[i : i + chunk_size]
            
            # Predict the water depth for this chunk
            batch_preds = model(batch)
            
            # Save the depth (h) predictions
            depth_list.append(batch_preds[:, 0].cpu().numpy())

    # Stitch all the predicted chunks back into one massive array
    depth = np.concatenate(depth_list)

    # Reshape the flat predictions back into a 2D map image
    depth_map = depth.reshape(lat_grid.shape)
    return depth_map

def run_validation():
    depth_map = generate_ai_mask()
    
    # --- THE X-RAY ---
    print(f"\n---> DEBUG X-RAY <---")
    print(f"Min Depth Predicted: {np.nanmin(depth_map):.6f}")
    print(f"Max Depth Predicted: {np.nanmax(depth_map):.6f}")
    print(f"Are there NaNs in the prediction? {np.isnan(depth_map).any()}\n")
    # -----------------
    
    # Lower threshold: Show any water deeper than 1 centimeter!
    flood_mask = np.where(depth_map > 0.01, depth_map, np.nan)

    print("5. Rendering Final Flood Simulation (Microscope Mode)...")
    plt.figure(figsize=(12, 8))
    
    # We plot the RAW depth map, and tell the colorbar to perfectly wrap 
    # around the AI's microscopic variance so the hidden patterns pop out!
    plt.imshow(depth_map, cmap='Blues', origin='lower', 
               vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))
    
    plt.colorbar(label='Raw AI Output Variance')
    plt.title('AI-Predicted Fluid Dynamics - Ahr Valley\n(Microscope Mode)')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_validation()