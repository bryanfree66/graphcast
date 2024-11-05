# downloader.py
import cdsapi
import datetime

client = cdsapi.Client()

singlelevelfields = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'geopotential',
    'land_sea_mask',
    'mean_sea_level_pressure',
    'toa_incident_solar_radiation',
    'total_precipitation'
]
pressurelevelfields = [
    'u_component_of_wind',
    'v_component_of_wind',
    'geopotential',
    'specific_humidity',
    'temperature',
    'vertical_velocity'
]
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

def download_data(year):
    """Downloads ERA5 data for a single year."""

    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': singlelevelfields,
            'grid': '1.0/1.0',
            'year': [str(year)],
            'month': ['{:02d}'.format(month) for month in range(1, 13)],
            'day': ['{:02d}'.format(day) for day in range(1, 32)],
            'time': ['{:02d}:00'.format(hour) for hour in range(0, 24)],
            'format': 'netcdf'
        },
        f'Dataset/single-level-{year}.nc'
    )

    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': pressurelevelfields,
            'grid': '1.0/1.0',
            'year': [str(year)],
            'month': ['{:02d}'.format(month) for month in range(1, 13)],
            'day': ['{:02d}'.format(day) for day in range(1, 32)],
            'time': ['06:00', '12:00'],
            'pressure_level': pressure_levels,
            'format': 'netcdf'
        },
        f'Dataset/pressure-level-{year}.nc'
    )

if __name__ == '__main__':
    for year in range(2022, 2025):
        download_data(year)