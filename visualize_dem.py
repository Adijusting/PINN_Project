import rasterio
import matplotlib.pyplot as plt

def plot_terrain():
    file_path = 'data/srtm_dem_ahr_valley.tif'
    print(f"Loading terrain data from {file_path}")
    
    # Open the GeoTIFF file
    with rasterio.open(file_path) as src:
        # The DEM data is stored in the first and only band
        elevation = src.read(1)
        
        # Create a plot
        plt.figure(figsize=(10,8))
        
        # cmap='terrain' gives it realistic map colors
        img = plt.imshow(elevation, cmap='terrain')
        
        # Add a legend and labels
        plt.colorbar(img, label='Elevation (meters above sea level)')
        plt.title('Ahr Valley - SRTM Digital elevation model (30m resolution)')
        plt.xlabel('Longitude (Grid Columns)')
        plt.ylabel('Latitude (Grid Rows)')
        
        print("Rendering map...")
        plt.show()
        
if __name__ == "__main__":
    plot_terrain()