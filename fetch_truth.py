from pystac_client import Client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds  
import matplotlib.pyplot as plt
import numpy as np
import sys

def fetch_satellite_truth():
    print("1. Connecting to the Global Satellite Catalog (STAC)...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # Your exact Ahr Valley bounding box (in GPS Degrees)
    bbox = [6.8, 50.4, 7.2, 50.7] # [west, south, east, north]
    
    # Widened the window slightly to guarantee a perfectly centered orbital pass
    time_of_interest = "2021-07-14/2021-07-18"

    print("2. Searching for Sentinel-1 Radar imagery over the valley...")
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=time_of_interest
    )
    
    items = list(search.items())
    if not items:
        print("Error: No satellite images found for this exact date and location.")
        sys.exit(1)

    print(f"   --> Found {len(items)} possible satellite passes.")
    
    valid_data = None
    capture_date = None
    
    for i, item in enumerate(items):
        print(f"\n3. Checking Image #{i+1}...")
        vh_asset_url = item.assets["vh"].href
        
        try:
            with rasterio.open(vh_asset_url) as src:
                # THE FIX: Translate our GPS degrees into the satellite's meter-grid!
                proj_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
                
                # Now cut the window using the translated coordinates
                window = from_bounds(*proj_bbox, transform=src.transform)
                temp_data = src.read(1, window=window)
                
            if np.isnan(temp_data).all() or np.nanmax(temp_data) == 0:
                print("   [Skip] This image is empty (satellite swath missed).")
                continue
                
            valid_data = temp_data
            capture_date = item.datetime.strftime('%Y-%m-%d %H:%M UTC')
            print(f"   [Success] Found a clean radar sweep captured on: {capture_date}")
            break 
            
        except Exception as e:
            print(f"   [Error reading image]: {e}")
            continue

    if valid_data is None:
        print("\nFATAL ERROR: All satellite passes were empty.")
        sys.exit(1)

    print("\n4. Diagnosing Radar Data...")
    print(f"   --> Data Shape: {valid_data.shape}")
    print(f"   --> Max Value: {np.nanmax(valid_data):.4f}")
    
    print("\n5. Rendering the True Ground Image...")
    print("\n5. Rendering the True Ground Image (Memory-Safe Mode)...")
    plt.figure(figsize=(10, 8))
    
    # THE FIX: Downsample the image! Grab every 4th pixel to save massive amounts of RAM
    factor = 4
    memory_safe_data = valid_data[::factor, ::factor]
    
    # Calculate the 95th percentile on the smaller array
    vmax_auto = np.nanpercentile(memory_safe_data, 95)
    
    # Render the smaller array
    plt.imshow(memory_safe_data, cmap='gray', vmin=0, vmax=vmax_auto)
    
    plt.title(f"True Ground Reality - Sentinel-1 Radar\nCaptured: {capture_date}")
    plt.colorbar(label='Radar Backscatter (Pitch Black = Water)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    print("\nLook for the thick, dark black veins running through the image—that is the floodwater!")
    plt.show()

if __name__ == "__main__":
    fetch_satellite_truth()