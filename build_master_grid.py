import xarray as xr
import rioxarray

def build_pinn_dataset():
    print("1. Loading high-resolution DEM...\n")
    
    # Load GeoTIFF and squeeze out the aribitrary 'band'
    dem = rioxarray.open_rasterio('data/srtm_dem_ahr_valley.tif').squeeze(drop=True)
    
    # Rename the spatial coordinates from image standards
    dem = dem.rename({'x':'longitude', 'y':'latitude'})
    dem.name = 'elevation'
    
    print("2. Loading ERA5 climate data...\n")
    file_accum = 'data/era5_extracted/data_stream-oper_stepType-accum.nc'
    file_instant = 'data/era5_extracted/data_stream-oper_stepType-instant.nc'
    
    era5_precip = xr.open_dataset(file_accum, engine='netcdf4')
    era5_temp = xr.open_dataset(file_instant, engine='netcdf4')
    
    precip_id = era5_precip['tp'].squeeze(drop=True)
    temp_1d = era5_temp['t2m'].squeeze(drop=True)
    
    print("3. Fusing datasets into Master PINN Grid\n")
    master_ds = xr.Dataset({
        'elevation':dem,
        'precipitation': precip_id,
        'temperature': temp_1d,
    })
    
    output_file = 'data/pinn_training_data.nc'
    master_ds.to_netcdf(output_file)
    
    print(f"Master dataset saved as '{output_file}'")
    
    print("\nMaster Data Summary")
    print(master_ds)
    
if __name__ == "__main__":
    build_pinn_dataset()