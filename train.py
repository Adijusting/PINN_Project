import torch
import torch.optim as optim
import xarray as xr
import numpy as np
from model import FloodPINN
from physics import calculate_physics_loss

def prepare_training_data():
    print("Loading Master PINN dataset...")
    ds = xr.open_dataset('data/pinn_training_data.nc')
    
    # Extract the exact bounds of your Ahr valley data
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # convert time to hours from start
    times = np.arange(len(ds['valid_time']))
    
    # we will use these bounds to generate random training points
    bounds = {
        'lat_min': lats.min(), 'lat_max': lats.max(),
        'lon_min': lons.min(), 'lon_max': lons.max(),
        'time_min': times.min(), 'time_max': times.max(),
        
        # get max/min for basic normalization
        'z_max': ds['elevation'].max().values, 'z_min': ds['elevation'].min().values,
        'p_max': ds['precipitation'].max().values, 'p_min': ds['precipitation'].min().values
    }
    
    return ds, bounds

def generate_collocation_points(ds, bounds, num_points=2000):
    
    lats = np.random.uniform(bounds['lat_min'], bounds['lat_max'], num_points)
    lons = np.random.uniform(bounds['lon_min'], bounds['lon_max'], num_points)
    times = np.random.uniform(bounds['time_min'], bounds['time_max'], num_points)
    
    # In a full-sclae model, we would precisely interpolate elevation/weather 
    z = np.random.uniform(bounds['z_min'], bounds['z_max'], num_points)
    P = np.random.uniform(bounds['p_min'], bounds['p_max'], num_points)
    
    # We don't strictly need temperature for pure shallow water equations
    temp = np.full(num_points, 290.0)
    
    # Stack them into our 6-feature input tensor
    points = np.column_stack((lats, lons, times, z, P, temp))
    
    # Convert to PyTorch tensor and enable gradients for the physics engine
    inputs = torch.tensor(points, dtype=torch.float32)
    inputs.requires_grad_(True)
    
    return inputs

def train_pinn():
    ds, bounds = prepare_training_data()
    print("\nInitializing PyTorch Model and Adam optimizer")
    model = FloodPINN()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs=500
    points_per_epoch = 2000
    
    print("\nStarting PINN Training")
    for epoch in range(epochs):
        # 1. Clear old gradients
        optimizer.zero_grad()
        
        # 2. Generate random coordinate points across the Ahr valley
        inputs = generate_collocation_points(ds, bounds, points_per_epoch)
        
        # 3. Calculate how badly the network breaks physics at those points
        physics_loss = calculate_physics_loss(model, inputs)
        
        # 4. Initial condition loss
        initial_inputs = inputs[:100].clone()
        initial_inputs[:,2] = 0.0
        initial_predictions = model(initial_inputs)
        initial_depths = initial_predictions[:,0:1]
        inital_loss = torch.mean((initial_depths - 0.0)**2)
        
        # 5. Total Loss
        total_loss = physics_loss + inital_loss
        
        # 6. Backpropogation
        total_loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.6f} | Physics: {physics_loss.item():.6f} | Initial: {inital_loss.item():.6f}")
            
    print("\nTraining Complete")
    torch.save(model.state_dict(), 'models/trained_flood_pinn.pth')
    print("Network weights saved to 'trained_flood_pinn.pth'")

if __name__ == "__main__":
    train_pinn()