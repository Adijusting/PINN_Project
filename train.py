import torch
import torch.optim as optim
import numpy as np
import xarray as xr
from model import PINN
from physics import shallow_water_loss

def train_pinn():
    print("1. Initializing AI Architecture...")
    # Auto-detect GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("2. Loading Master Training Dataset...")
    ds = xr.open_dataset('data/pinn_training_data.nc')
    
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    elevation = ds['elevation'].values
    precip_1d = ds['precipitation'].values
    friction = ds['friction'].values
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    num_points = lon_grid.size

    print("3. Beginning Physics-Informed Training Loop...")
    epochs = 1000  # Adjust as needed for better accuracy
    
    # We will train on the peak storm hour (e.g., Hour 48)
    time_step = 48.0
    precip_val = precip_1d[-1] if len(precip_1d) > 0 else 0.0

    # Ensure massive arrays are forced to 32-bit BEFORE stacking to save RAM
    lat_flat = lat_grid.flatten().astype(np.float32)
    lon_flat = lon_grid.flatten().astype(np.float32)
    elev_flat = elevation.flatten().astype(np.float32)
    friction_flat = friction.flatten().astype(np.float32)

    # Build the 7-Variable Input Stack
    inputs_np = np.column_stack((
        lat_flat, 
        lon_flat,
        np.full(num_points, time_step, dtype=np.float32),         # Time
        elev_flat,                                                # Topography
        np.full(num_points, float(precip_val), dtype=np.float32), # Rain
        np.full(num_points, 290.0, dtype=np.float32),             # Init Depth
        friction_flat                                             # Manning's Friction
    ))
    
    # Convert to PyTorch Tensor and push to GPU/CPU
    inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32).to(device)

    # The actual Training Loop
    # The actual Training Loop
    model.train()
    
    # THE FIX: Set a safe bite-size for your CPU
    batch_size = 10000 
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Randomly sample a batch of pixels from the map
        # (This prevents the C++ Segfault by keeping the calculus graph small!)
        idx = torch.randperm(num_points)[:batch_size]
        batch_inputs = inputs_tensor[idx]
        
        # 2. Calculate how badly the AI is breaking the laws of physics on THIS batch
        loss = shallow_water_loss(model, batch_inputs)
        
        # 3. Backpropagation (AI corrects its mistakes)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Physics Loss: {loss.item():.6f}")

    print("4. Saving Trained Weights...")
    torch.save(model.state_dict(), 'models/pinn_ahr_valley.pth')
    print("Training Complete! Brain saved to 'models/pinn_ahr_valley.pth'")

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    train_pinn()