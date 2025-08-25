import time
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections.abc import Generator

import cdsapi
import xarray as xr
from xarray.core.dataset import Dataset
from tqdm import tqdm

import globals

"""
For using the CDS API to download ERA-5 data consult: https://cds.climate.copernicus.eu/api-how-to
"""

REGION_OF_INTEREST = {'north': -22, 'west': -44, 'south': -23, 'east': -42}
download_folder = 'ERA5_single_levels_2025'

# Group variables to avoid CDS API request limits
VARIABLE_GROUPS = [
    ["convective_available_potential_energy", "convective_inhibition"],
    ["total_column_water_vapour", "total_cloud_cover"],
    ["surface_net_solar_radiation", "geopotential"],
    ["total_precipitation", "2m_dewpoint_temperature"],
    ["k_index", "total_totals_index"]
]

class DatasetClient:
    def __init__(self) -> None:
        self.clientCDS = cdsapi.Client(timeout=30, retry_max=5, sleep_max=30)

    def _convert_grib_to_netcdf(self, target: str):
        print("Converting grib data to netcdf...")
        assert target.endswith('.grib'), "The target file must be a grib file"
        data = xr.open_dataset(target, engine='cfgrib')
        target = target.replace('.grib', '.nc')
        data.to_netcdf(target)
        print("Converted grib data to netcdf")

    def call_retrieve(self, name: str, request: dict, target: str):
        MAX_RETRIES = 5
        base_sleep = 60
        for i in range(MAX_RETRIES):
            try:
                self.clientCDS.retrieve(name=name, request=request, target=target)
                print(f"Downloaded ERA5 data - {request['format']} format")
                break
            except Exception as e:
                print("Failed to download ERA5 data.")
                print(f"Error message: {e} - {repr(e)}")
                print(f"{i + 1}/{MAX_RETRIES} retries")
                if i == MAX_RETRIES - 1:
                    raise e
                sleep_time = base_sleep * (2 ** i)
                print(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)

class CDSDatasetDownloader:
    def __init__(self, begin_year: int, begin_month: int, end_year: int, end_month: int) -> None:
        self.begin_year = begin_year
        self.begin_month = begin_month
        self.end_year = min([end_year, datetime.today().year])
        self.end_month = end_month
        self.dataset_client = DatasetClient()

    def _get_dates_generator(self) -> Generator[tuple[int, int], None, None]:
        current_date = datetime.strptime(f"{self.begin_year}-{self.begin_month}", "%Y-%m")
        end_date = datetime.strptime(f"{self.end_year}-{self.end_month}", "%Y-%m")
        while current_date <= end_date:
            year = int(current_date.year)
            month = int(current_date.month)
            yield year, month
            if month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

    def _get_datasets_generator(self) -> Generator[Dataset, None, None]:
        for year, month in self._get_dates_generator():
            yield xr.open_dataset(f"{globals.NWP_DATA_DIR}{download_folder}/montly_data/RJ_{year}_{month}_merged.nc")

    def _download_dataset_split_vars(self, month: int, year: int):
        datasets = []
        for idx, group in enumerate(VARIABLE_GROUPS, start=1):
            target_path = Path(f"{globals.NWP_DATA_DIR}{download_folder}/montly_data/RJ_{year}_{month}_grp{idx}.nc")
            if not target_path.parent.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.is_file():
                request = {
                    "product_type": ["reanalysis"],
                    "format": "netcdf",
                    "variable": group,
                    "year": [str(year)],
                    "month": [f"{month:02d}"],
                    "day": [f"{day:02d}" for day in range(1, 32)],
                    "time": [f"{hour:02d}:00" for hour in range(24)],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    "area": [REGION_OF_INTEREST[key] for key in ['north', 'west', 'south', 'east']]
                }
                print(f"Downloading ERA5 group{idx} for month {month}, year {year}...")
                self.dataset_client.call_retrieve(
                    name="reanalysis-era5-single-levels",
                    request=request,
                    target=str(target_path.resolve())
                )
            ds = xr.open_dataset(target_path)
            datasets.append(ds)

        # Merge all groups
        merged_ds = xr.merge(datasets)
        merged_path = f"{globals.NWP_DATA_DIR}{download_folder}/montly_data/RJ_{year}_{month}_merged.nc"
        merged_ds.to_netcdf(merged_path)
        print(f"Merged dataset saved to {merged_path}")

        # Optionally delete intermediate files
        for idx in range(1, len(VARIABLE_GROUPS) + 1):
            target_path = Path(f"{globals.NWP_DATA_DIR}{download_folder}/montly_data/RJ_{year}_{month}_grp{idx}.nc")
            target_path.unlink()
            print(f"Deleted variable group {target_path}")

    def download_and_merge_monthly(self):
        print("Downloading and merging ERA5 single level data...")
        dates = list(self._get_dates_generator())
        for year, month in tqdm(dates, desc="Downloading ERA5 monthly datasets"):
            merged_path = f"{globals.NWP_DATA_DIR}{download_folder}/montly_data/RJ_{year}_{month}_merged.nc"
            if Path(merged_path).is_file():
                print(f"Merged file already exists for {year}-{month}, skipping.")
                continue
            self._download_dataset_split_vars(month, year)

    def check_datasets(self):
        target_dir = Path(f"{globals.NWP_DATA_DIR}{download_folder}/montly_data")
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {target_dir}")
        downloaded_files = list(target_dir.glob("*.nc"))
        expected_files = []
        for year, month in self._get_dates_generator():
            expected_files.append(f"RJ_{year}_{month}_merged.nc")
        print(f"len downloaded_files: {len(downloaded_files)}")
        print(f"len expected_files: {len(expected_files)}")
        missing_files = set(expected_files) - set([file.name for file in downloaded_files])
        if len(missing_files) != 0:
            print(f"Missing files: {missing_files}")
            raise FileNotFoundError("Not all datasets were downloaded")
        print("All datasets were downloaded")

    def prepend_dataset(self, prepend_dataset: str):
        if not Path(prepend_dataset).is_file():
            raise FileNotFoundError(f"Dataset to prepend not found: {prepend_dataset}")
        prepend_dataset_name = Path(prepend_dataset).name
        prepend_begin_year = int(prepend_dataset_name.split('_')[1])
        prepend_end_year = int(prepend_dataset_name.split('_')[-1].split('.')[0])
        target_path = Path(f"{globals.NWP_DATA_DIR}{download_folder}/RJ_{prepend_begin_year}_{self.end_year}.nc")
        if target_path.is_file():
            print(f"ERA5 data already prepended for the period {prepend_begin_year} to {self.end_year}")
            return
        print(f"Prepending ERA5 data {prepend_begin_year}-{prepend_end_year} to {self.begin_year}-{self.end_year}...")
        prepend_ds = xr.open_dataset(prepend_dataset)
        append_ds = xr.open_dataset(f"{globals.NWP_DATA_DIR}{download_folder}/RJ_{self.begin_year}_{self.end_year}.nc")
        ds = prepend_ds.merge(append_ds)
        ds.to_netcdf(str(target_path.resolve()))
        print(f"ERA5 data prepended for the period {prepend_begin_year} to {self.end_year}")

    def merge_datasets(self):
        target_path = Path(f"{globals.NWP_DATA_DIR}{download_folder}/RJ_{self.begin_year}_{self.end_year}.nc")
        if target_path.is_file():
            print(f"ERA5 data already merged for the period {self.begin_year} to {self.end_year}")
            return
        print(f"Merging ERA5 data for the period {self.begin_year} to {self.end_year}...")
        datasets_generator = self._get_datasets_generator()
        ds = next(datasets_generator)
        for dataset in datasets_generator:
            ds = ds.merge(dataset)
        ds.to_netcdf(str(target_path.resolve()))
        print(f"ERA5 data merged for the period {self.begin_year} to {self.end_year}")

def valid_date(arg: str):
    try:
        year_str, month_str = arg.split('-')
        if len(month_str) != 2 or len(year_str) != 4: raise ValueError
        year = int(year_str)
        month = int(month_str)
        return year, month
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid date format. Please use YYYY-MM")

def main(argv):
    global download_folder
    parser = argparse.ArgumentParser(description='Retrieve ERA5 single level data between two given years.')
    parser.add_argument('-b', '--begin', type=valid_date, required=True, help='Begin date (YYYY-MM)')
    parser.add_argument('-e', '--end', type=valid_date, required=True, help='End date (YYYY-MM)')
    parser.add_argument('-pd', '--prepend_dataset', type=str, default=None, help='Dataset to merge datasets')
    parser.add_argument('-north', '--north', type=float, default=REGION_OF_INTEREST['north'], help='Northernmost latitude')
    parser.add_argument('-west', '--west', type=float, default=REGION_OF_INTEREST['west'], help='Westernmost longitude')
    parser.add_argument('-south', '--south', type=float, default=REGION_OF_INTEREST['south'], help='Southernmost latitude')
    parser.add_argument('-east', '--east', type=float, default=REGION_OF_INTEREST['east'], help='Easternmost longitude')
    parser.add_argument('-d', '--download_folder', type=str, default=download_folder, help='Folder to download datasets')

    args = parser.parse_args(argv[1:])

    begin_year, begin_month = args.begin
    end_year, end_month = args.end
    prepend_dataset = args.prepend_dataset

    REGION_OF_INTEREST['north'] = args.north
    REGION_OF_INTEREST['west'] = args.west
    REGION_OF_INTEREST['south'] = args.south
    REGION_OF_INTEREST['east'] = args.east

    download_folder = args.download_folder

    assert begin_year >= 1940, "ERA5 start year must be greater than or equal to 1940"
    assert begin_year <= end_year, "ERA5 start year must be less than or equal to end year"

    dataset_downloader = CDSDatasetDownloader(
        begin_year=begin_year,
        begin_month=begin_month,
        end_year=end_year,
        end_month=end_month
    )
    dataset_downloader.download_and_merge_monthly()
    dataset_downloader.check_datasets()
    # dataset_downloader.merge_datasets()

    if prepend_dataset:
        dataset_downloader.prepend_dataset(prepend_dataset)

if __name__ == "__main__":
    main(sys.argv)