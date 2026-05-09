import requests
from tqdm import tqdm
import sys
from dotenv import load_dotenv

def download_terrain_data():
    print("Preparing to download SRTM DEM (30m Resolution) data...")
    
    # [North, West, South, East]
    north, west, south, east = 50.7, 6.8, 50.4, 7.2
    
    output_file = 'srtm_dem_ahr_valley.tif'
    api_key = "6b34c82f23112102afda37597b8f3b3b"
    
    if not api_key == "6b34c82f23112102afda37597b8f3b3b":
        print("Error: Please insert your OpenTopogrpahy API key in the script")
        sys.exit(1)
        
    # OpenTopography REST API endpoint for SRTM GL1
    url = (f"https://portal.opentopography.org/API/globaldem"
           f"?demtype=SRTMGL1"
           f"&south={south}&north={north}&west={west}&east={east}"
           f"&outputFormat=GTiff"
           f"&API_Key={api_key}")
    
    print(f"Requesting precise terrain data for bounding box: [{north}, {west}, {south}, {east}]")
    
    # Start the download stream
    response = requests.get(url, stream=True)
    
    # handle authentication errors
    if response.status_code == 401:
        print("\nError 401: Unauthorized. Check if your API key is correct.")
        sys.exit(1)
    
    response.raise_for_status()
    
    # Get file size for the progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    print("\nProcessing and downloading GeoTIFF...")
    with open(output_file, 'wb') as file, tqdm(
        desc=output_file,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = file.write(chunk)
                bar.update(size)
    
    print(f"\nSuccess! High-Resolution terrain map saved as '{output_file}'")
    

if __name__ == "__main__":
    download_terrain_data()
    
    