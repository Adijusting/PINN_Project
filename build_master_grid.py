import xarray as xr
import rioxarray
import numpy as np

def build_pinn_dataset():
    print("1. Loading Topography (DEM)...")
    dem = rioxarray.open_rasterio('data/srtm_dem_ahr_valley.tif').squeeze(drop=True)
    
    if 'x' in dem.coords:
        dem = dem.rename({'x': 'longitude', 'y': 'latitude'})

    print("2. Loading and Aligning Friction Map...")
    friction = rioxarray.open_rasterio('data/mannings_n_ahr_valley.tif').squeeze(drop=True)
    friction_aligned = friction.rio.reproject_match(dem)

    print("3. Loading ERA5 Climate Data...")
    era5 = xr.open_dataset('data/era5_extracted/data_stream-oper_stepType-accum.nc')
    tp = era5['tp']

    print("4. Bypassing Memory Trap (Storing Weather as 1D Timeline)...")
    precip_1d = tp.values.squeeze()

    print("5. Fusing the Master AI Brain Dataset...")
    master_ds = xr.Dataset(
        {
            'elevation': (['latitude', 'longitude'], dem.values),
            'friction': (['latitude', 'longitude'], friction_aligned.values),
            'precipitation': (['valid_time'], precip_1d) 
        },
        coords={
            'longitude': dem.longitude.values,
            'latitude': dem.latitude.values,
            'valid_time': tp.valid_time.values
        }
    )

    print("6. Saving to disk...")
    master_ds.to_netcdf('data/pinn_training_data.nc')
    print("Success! Master grid saved to 'data/pinn_training_data.nc'")

if __name__ == "__main__":
    build_pinn_dataset()