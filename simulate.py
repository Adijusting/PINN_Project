import torch
import xarray as xr
import numpy as np
import gc
import matplotlib.pyplot as plt
from model import FloodPINN

def visualize_flood():
    print("1. Loading trained PINN...")
    model = FloodPINN()
    model.load_state_dict(torch.load('models/trained_flood_pinn.pth', weights_only=True))
    model.eval() 

    print("2. Loading Master Terrain & Weather Data...")
    ds = xr.open_dataset('data/pinn_training_data.nc')
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    elevation = ds['elevation'].values
    
    target_hour = 48.0
    precip = ds['precipitation'].isel(valid_time=int(target_hour)).values
    
    print(f"3. Building grid for Hour {target_hour}...")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    z_flat = elevation.flatten()
    
    num_points = len(lat_flat)
    print(f"   Total pixels to process: {num_points:,}")
    
    t_flat = np.full(num_points, target_hour)
    p_flat = np.full(num_points, float(precip))
    temp_flat = np.full(num_points, 290.0) 

    inputs = np.column_stack((lat_flat, lon_flat, t_flat, z_flat, p_flat, temp_flat)).astype(np.float32)

    print("4. Calculating fluid dynamics (Memory-Safe Batching)...")
    # THE FIX: Create an empty array to hold the answers
    depths_flat = np.zeros(num_points, dtype=np.float32)
    
    # Process the map in chunks of 50,000 pixels to save RAM
    batch_size = 50000 
    
    with torch.no_grad():
        for i in range(0, num_points, batch_size):
            # Slice out a chunk of the map
            batch_inputs = torch.tensor(inputs[i:i+batch_size], dtype=torch.float32)
            
            # Predict just that chunk
            batch_predictions = model(batch_inputs)
            
            # Save the depth predictions back into our flat array
            depths_flat[i:i+batch_size] = batch_predictions[:, 0].numpy()
            
            # Print progress so you know it hasn't frozen!
            processed = min(i + batch_size, num_points)
            print(f"   Predicted {processed:,} / {num_points:,} pixels...")
            
    print("--- MEMORY CLEANUP ---")
    # Delete the massive input array since we have our answers now
    del inputs 
    del lat_flat, lon_flat, z_flat, t_flat, p_flat, temp_flat
    gc.collect()        

    print("5. Rendering Final Map...")
    # Fold the flat predictions back into the 2D map shape
    depth_map = depths_flat.reshape(lat_grid.shape)
    print(f"Maximum predicted depth: {depth_map.max():.4f}")
    print(f"Minimum predicted depth: {depth_map.min():.4f}")
    plt.figure(figsize=(12, 8))
    
    plt.imshow(elevation, cmap='terrain', extent=[lons.min(), lons.max(), lats.min(), lats.max()], alpha=0.5)
    
    water_masked = np.ma.masked_where(depth_map < 0.001, depth_map)
    water_plot = plt.imshow(water_masked, cmap='Blues', extent=[lons.min(), lons.max(), lats.min(), lats.max()], alpha=0.9)
    
    plt.colorbar(water_plot, label='Predicted Water Depth (meters)')
    plt.title(f'AI Predicted Flood Inundation - Ahr Valley (Hour {target_hour})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.show()

if __name__ == "__main__":
    visualize_flood()