import s3fs
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
from netCDF4 import Dataset
import xarray as xr
import sys
import argparse
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define coordinate boundaries of interest (Rio de Janeiro)
lon_min, lon_max = -45.05290312102409, -42.35676996062447
lat_min, lat_max = -23.801876626302175, -21.699774257353113

# Output directories
output_directory = "data/goes16/glm/glm_files/"
temp_directory = "data/goes16/glm/temp_glm_files/"
final_directory = "data/goes16/glm/aggregated_glm_files/"

def create_directory(directory):
    """Create the directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)
    logging.info(f"Directory {directory} created (or already existed).")

def clear_directory(directory):
    """Clear the directory and recreate it."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logging.info(f"Directory {directory} cleared.")
    create_directory(directory)

def download_files(start_date, end_date, ignored_months):
    """Download and process GLM files for a specified date range."""
    current_date = start_date
    fs = s3fs.S3FileSystem(anon=True)

    clear_directory(temp_directory)
    create_directory(final_directory)

    temp_files = []
    while current_date <= end_date:
        if (current_date.month in ignored_months):
            day = current_date.strftime('%Y_%m_%d')
            logging.info(f"Ignoring data for {day}")
            current_date += timedelta(days=1)
            continue
        year = current_date.year
        day_of_year = current_date.timetuple().tm_yday 
        bucket_path = f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year:03d}/'

        logging.info(f"Fetching files for {current_date.strftime('%Y-%m-%d')} (day {day_of_year})")

        for hour in range(24):
            hour_path = bucket_path + f"{hour:02d}/"
            hourly_files = []
            try:
                hourly_files = fs.ls(hour_path)
            except FileNotFoundError:
                logging.warning(f"Hour {hour:02d} not found in the bucket. Skipping...")

            for file in hourly_files:
                file_name = file.split('/')[-1]
                local_file_path = os.path.join(temp_directory, file_name)
                logging.info(f"Downloading: {file} to {local_file_path}")
                fs.get(file, local_file_path)
                
                if not filter_by_coordinates(local_file_path):
                    open(local_file_path, 'w').close()  # Create empty file if there is no data in Rio

                temp_files.append(local_file_path)

                if len(temp_files) == 20:
                    aggregate_files(temp_files, current_date, hour)
                    clear_directory(temp_directory)  # Clear the temporary folder for the next cycle
                    temp_files.clear()

        current_date += timedelta(days=1)

def filter_by_coordinates(file_path):
    """Filter GLM events in a NetCDF file based on the provided coordinates."""
    try:
        dataset = Dataset(file_path, 'r')
        longitudes = dataset.variables['flash_lon'][:]
        latitudes = dataset.variables['flash_lat'][:]
        dataset.close()

        mask = (
            (longitudes >= lon_min) & (longitudes <= lon_max) &
            (latitudes >= lat_min) & (latitudes <= lat_max)
        )

        if np.sum(mask) == 0:
            logging.info(f"No events within the filter found in {file_path}. Removing file.")
            return False
        else:
            logging.info(f"Events within the filter found in file {file_path}.")
            return True
    except Exception as e:
        logging.error(f"Error filtering file {file_path}: {e}")
        return False

def aggregate_files(files, current_date, hour):
    """Aggregate 30 valid files and save them as a single NetCDF file."""
    datasets = []
    
    for file in files:
        if os.path.getsize(file) > 0:  # Ignore empty files
            ds = xr.open_dataset(file)
            datasets.append(ds)

    if datasets:
        try:
            # Remove 'number_of_events' and 'number_of_groups' at the time of concatenation
            combined = xr.concat(
                [ds.drop_dims(['number_of_events', 'number_of_groups', 'number_of_flashes'], errors="ignore") for ds in datasets],
                dim='time',
                data_vars='minimal',
                coords='minimal',
                compat='override'
            )
            output_file_name = f"glm_agg_{current_date.strftime('%Y%m%d')}_{hour:02d}.nc"
            output_file_path = os.path.join(final_directory, output_file_name)
            combined.to_netcdf(output_file_path)
            logging.info(f"Aggregation saved in {output_file_path}")
        except ValueError as e:
            logging.error(f"Error concatenating files: {e}")
    else:
        logging.info("No data to aggregate in this round.")

def main(argv):
    '''
    Example usage:
    python src/goes16_retrieve_glm.py --start_date "2024-01-13" --end_date "2024-01-13"
    '''
    parser = argparse.ArgumentParser(description='Download and filter GLM files by coordinates.')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in the format YYYY-MM-DD')
    parser.add_argument("--ignored_months", nargs='+', type=int, required=False, default=[6, 7, 8],
                        help="Months to ignore (e.g., --ignored_months 6 7 8)")
    args = parser.parse_args(argv[1:])
    ignored_months = args.ignored_months

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    assert start_date <= end_date, "Start date must be before or equal to the end date."

    download_files(start_date, end_date, ignored_months)

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main(sys.argv)
    end_time = time.time()  # Record the end time
    duration = (end_time - start_time) / 60  # Calculate duration in minutes
    print(f"Script execution time: {duration:.2f} minutes.")