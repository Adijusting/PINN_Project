import xarray as xr

def inspect_era5():
    file_accum = 'era5_extracted/data_stream-oper_stepType-accum.nc'
    file_instant = 'era5_extracted/data_stream-oper_stepType-instant.nc'
    
    print("\nInstantaneous Variables (Temperature)")
    ds_instant = xr.open_dataset(file_instant, engine='netcdf4')
    print(ds_instant)
    print("\n2-Meter Temperature (t2m) metadata")
    print(ds_instant['t2m'].attrs)
    
    print("\n" + "-"*50 + "\n")
    print("Accumulated Variables (Precipitation)\n")
    ds_accum = xr.open_dataset(file_accum, engine='netcdf4')
    print(ds_accum)
    print("Total Precipitation\n")
    print(ds_accum['tp'].attrs)
    
if __name__ == "__main__":
    inspect_era5()