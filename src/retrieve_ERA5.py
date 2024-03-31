import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections.abc import Generator

from cdsapi.api import Client as ClientCDS
import xarray as xr
from xarray.core.dataset import Dataset

import globals

"""
For using the CDS API to download ERA-5 data consult: https://cds.climate.copernicus.eu/api-how-to
"""

REGION_OF_INTEREST = {'north': -22, 'west': -44, 'south': -23, 'east': -42}

class DatasetClient:
    def __init__(self) -> None:
        self.clientCDS = ClientCDS(timeout=10, retry_max=2, sleep_max=1)

    def _convert_grib_to_netcdf(self, target: str):
        # please see, https://github.com/ecmwf/cfgrib
        # also, https://docs.xarray.dev/en/stable/user-guide/io.html#grib-format-via-cfgrib
        assert target.endswith('.grib'), "The target file must be a grib file"
        data = xr.open_dataset(target, engine='cfgrib')
        target = target.replace('.grib', '.nc')
        data.to_netcdf(target)
        print(f"Converted grib data to netcdf")

    def call_retrieve(self,name: str, request: dict, target: str):
        try:
            self.clientCDS.retrieve(name=name, request=request, target=target)
            print(f"Downloaded ERA5 data - {request['format']} format")
            if request['format'] == 'grib': self._convert_grib_to_netcdf(target)
        except Exception as e:
            if request['format'] == 'grib':
                print(f"Failed to download ERA5 data in grib format. Error {repr(e)}")
                raise e

            print(f"Failed to download ERA5 data in netcdf format. Downloading in grib format")
            request['format'] = 'grib'
            target = target.replace('.nc', '.grib')
            self.call_retrieve(request, target)

class CDSDatasetDownloader:
    def __init__(self, start_year: int, end_year: int) -> None:
        self.start_year = start_year
        self.end_year = min([end_year, datetime.today().year])
        self.dataset_client = DatasetClient()

    def _get_datasets_generator(self) -> Generator[Dataset, None, None]:
        for year in map(str, range(self.start_year, self.end_year + 1)):
            for month in map(str, range(1, 12 + 1)):
                yield xr.open_dataset(f"{globals.NWP_DATA_DIR}ERA5/montly_data/RJ_{year}_{month}.nc")

    def _download_dataset(self, month: str, year: str):
        target_path_nc = Path(f"{globals.NWP_DATA_DIR}ERA5/montly_data/RJ_{year}_{month}.nc")
        if target_path_nc.is_file():
            print(f"ERA5 data already downloaded for the month {month} of year {year}.")
            return
        
        target_path_grib = Path(f"{globals.NWP_DATA_DIR}ERA5/montly_data/RJ_{year}_{month}.grib")
        if target_path_grib.is_file():
            print(f"ERA5 data already downloaded for the month {month} of year {year} with GRIB format")
            self.dataset_client._convert_grib_to_netcdf(str(target_path_grib.resolve()))
            return
        
        request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "geopotential", "relative_humidity", "temperature",
                "u_component_of_wind", "v_component_of_wind"
            ],
            "pressure_level": ['200', '700', '1000'],
            "year": year,
            "month": month,
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "area": [REGION_OF_INTEREST[key] for key in ['north', 'west', 'south', 'east']]
        }

        print(f"Downloading ERA5 data at month {month} of year {year}...")
        self.dataset_client.call_retrieve(
            name="reanalysis-era5-pressure-levels",
            request=request,
            target=str(target_path_nc.resolve())
        )
        print(f"Downloaded ERA5 data at month {month} of year {year}")

    def download_datasets(self):
        print(f"Downloading ERA5 data for the period {self.start_year} to {self.end_year}...")
        for year in map(str, range(self.start_year, self.end_year + 1)):
            for month in map(str, range(1, 12 + 1)):
                self._download_dataset(month, year)
        print(f"Downloaded ERA5 data for the period {self.start_year} to {self.end_year}")

    def merge_datasets(self):
        print(f"Merging ERA5 data for the period {self.start_year} to {self.end_year}...")
        datasets_generator = self._get_datasets_generator()
        ds = next(datasets_generator)
        for dataset in datasets_generator:
            ds = ds.merge(dataset)
        ds.to_netcdf(f"{globals.NWP_DATA_DIR}ERA5/RJ_{self.start_year}_{self.end_year}.nc")
        print(f"ERA5 data merged for the period {self.start_year} to {self.end_year}")

def main(argv):
    parser = argparse.ArgumentParser(description='Retrieve ERA5 data between two given years.')
    parser.add_argument('-b', '--start_year', type=int, required=True, help='Start year')
    parser.add_argument('-e', '--end_year', type=int, required=True, help='End year')

    args = parser.parse_args(argv[1:])

    start_year = args.start_year
    end_year = args.end_year

    # ERA5 data goes back to the year 1940. 
    # see https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form
    assert start_year >= 1940, "ERA5 start year must be greater than or equal to 1940"
    assert start_year <= end_year, "ERA5 start year must be less than or equal to end year"

    dataset_downloader = CDSDatasetDownloader(start_year, end_year)
    dataset_downloader.download_datasets()
    dataset_downloader.merge_datasets()

if __name__ == "__main__":
    main(sys.argv)
