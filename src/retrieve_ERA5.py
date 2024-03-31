import sys
from datetime import datetime
import time
from cdsapi.api import Client as ClientCDS
from collections.abc import Iterator
from pathlib import Path
import xarray as xr
import argparse
import sys
from enum import Enum
import globals

"""
    For using the CDS API to download ERA-5 data consult: https://cds.climate.copernicus.eu/api-how-to
"""

# North -22°, West -44°, South -23°, East -42°
REGION_OF_INTEREST = [-22, -44, -23, -42]

class Months(Enum):
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12

def _download_CDS_data(clientCDS: ClientCDS, month: str, year: str):
    file_exist = Path(f"{globals.NWP_DATA_DIR}ERA5/montly_data/RJ_{year}_{month}.nc")
    if file_exist.is_file():
        print(f"ERA5 data already downloaded for the month {month} of year {year}.")
        return

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "geopotential",
            "relative_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "pressure_level": ['200', '700', '1000'],
        "year": [year],
        "month": [month],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": REGION_OF_INTEREST,
    }

    try:
        print(f"Downloading ERA5 data at month {month} of year {year}...")
        clientCDS.retrieve(
            name="reanalysis-era5-pressure-levels",
            request=request,
            target=f"{globals.NWP_DATA_DIR}ERA5/montly_data/RJ_{year}_{month}.nc"
        )
        print(f"Downloaded ERA5 data at month {month} of year {year}")
    except Exception as e:
        print(f"Unexpected error! {repr(e)}")
        sys.exit(2)

def _get_datasets_generator(years: Iterator, months: Iterator):
    for year in years:
        for month in months:
            target = f"{globals.NWP_DATA_DIR}ERA5/montly_data/RJ_{year}_{month}.nc"
            yield xr.open_dataset(target)

def get_data(start_year: int, end_year: int) -> None:
    end_year = min([end_year, datetime.today().year])

    target_path = f"{globals.NWP_DATA_DIR}ERA5/RJ_{start_year}_{end_year}.nc"

    file_exist = Path(target_path)
    if file_exist.is_file():
        print(f"ERA5 data already downloaded for the period {start_year} to {end_year}.")
        return
    
    clientCDS = ClientCDS()
    years = map(str, range(start_year, end_year + 1))
    months = map(str, range(Months.JANUARY.value, Months.DECEMBER.value + 1))
    for year in years:
        for month in months:
            _download_CDS_data(clientCDS, month, year)

    datasets_generator = _get_datasets_generator(years, months)
    ds = next(datasets_generator)
    for dataset in datasets_generator:
        ds = ds.merge(dataset)

    print(f"ERA5 data downloaded for the period {start_year} to {end_year}")
    print(f"Saving dowloaded data to {target_path}")
    ds.to_netcdf(target_path)

def main(argv):
    parser = argparse.ArgumentParser(description='Retrieve ERA5 data between two given years.')
    parser.add_argument('-b', '--start_year', type=int, required=True, help='Start year')
    parser.add_argument('-e', '--end_year', type=int, required=True, help='End year')

    args = parser.parse_args(argv[1:])

    start_year = args.start_year
    end_year = args.end_year

    # ERA5 data goes back to the year 1940. 
    # see https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview
    assert start_year >= 1940 
    assert start_year <= end_year

    get_data(start_year, end_year)

if __name__ == "__main__":
    main(sys.argv)
