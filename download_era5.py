import cdsapi
from dotenv import load_dotenv


def download_climate_data():
    print("Connecting to Copernicus Climate Data Store...")
    c = cdsapi.Client()

    bounding_box = [50.6, 6.8, 50.4, 7.2] 
    hours = [f"{str(i).zfill(2)}:00" for i in range(24)]
    output_file = 'era5_ahr_valley_2021.nc'

    print(f"Requesting ERA5 data for bounding box: {bounding_box}")
    print("The CDS API will queue the request, then automatically show a progress bar...")
    
    # The native cdsapi library handles all redirects and shows a progress bar automatically
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'download_format':'unarchived',
            'variable': [
                'total_precipitation', 
                '2m_temperature',
            ],
            'year': '2021',
            'month': '07',
            'day': ['13', '14', '15'],
            'time': hours,
            'area': bounding_box,
        },
        output_file
    )
    
    print(f"\nSuccess! Data safely downloaded as '{output_file}'")

if __name__ == "__main__":
    download_climate_data()