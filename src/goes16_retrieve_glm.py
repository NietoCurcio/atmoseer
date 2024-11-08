import s3fs
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
from netCDF4 import Dataset
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define coordinate bounds of interest
lon_min, lon_max = -45.05290312102409, -42.35676996062447
lat_min, lat_max = -23.801876626302175, -21.699774257353113

# Output directory
output_directory = "data/goes16/glm_files/"

def create_directory(directory):
    """Creates the directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    logging.info(f"Directory {directory} created (or already existed).")

def clear_directory(directory):
    """Clears the output directory and recreates it."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logging.info(f"Directory {directory} cleared.")
    create_directory(directory)

def download_files(start_date, end_date):
    """Downloads GLM files for a specified date range and crops by coordinates."""
    current_date = start_date
    fs = s3fs.S3FileSystem(anon=True)

    while current_date <= end_date:
        year = current_date.year
        day_of_year = current_date.timetuple().tm_yday 
        bucket_path = f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year:03d}/'

        logging.info(f"Searching for files for {current_date.strftime('%Y-%m-%d')} (day {day_of_year})")
        
        # Specific output directory for the current day
        day_output_directory = os.path.join(output_directory, current_date.strftime('%Y-%m-%d'))
        clear_directory(day_output_directory)

        # Iterate through the subfolders of each hour (00 to 23)
        files = []
        for hour in range(24):
            hour_path = bucket_path + f"{hour:02d}/"  
            try:
                hourly_files = fs.ls(hour_path)
                files.extend(hourly_files)  
            except FileNotFoundError:
                logging.warning(f"Hour {hour:02d} not found in bucket. Skipping...")

        logging.info(f"Total files found for {current_date.strftime('%Y-%m-%d')}: {len(files)}")

        for file in files:
            file_name = file.split('/')[-1]
            local_file_path = os.path.join(day_output_directory, file_name)
            logging.info(f"Downloading: {file} to {local_file_path}")
            fs.get(file, local_file_path)
            filter_by_coordinates(local_file_path)

        logging.info(f"Download and filter for {current_date.strftime('%Y-%m-%d')} completed.")
        current_date += timedelta(days=1)

def filter_by_coordinates(file_path):
    """Filters GLM events from a NetCDF file based on provided coordinates."""
    dataset = None
    try:
        dataset = Dataset(file_path, 'r')

        longitudes = dataset.variables['flash_lon'][:]
        latitudes = dataset.variables['flash_lat'][:]

        logging.info(f"Minimum longitude: {longitudes.min()}, maximum: {longitudes.max()}")
        logging.info(f"Minimum latitude: {latitudes.min()}, maximum: {latitudes.max()}")

        mask = (
            (longitudes >= lon_min) & (longitudes <= lon_max) &
            (latitudes >= lat_min) & (latitudes <= lat_max)
        )

        if np.sum(mask) == 0:
            logging.info(f"No events within filter found in {file_path}. Removing file.")
            dataset.close()
            os.remove(file_path)
        else:
            logging.info(f"Events within filter found in file {file_path}.")
        
    except Exception as e:
        logging.error(f"Error filtering file {file_path}: {e}")

def main(argv):
    parser = argparse.ArgumentParser(description='Download and filter GLM files by coordinates.')
    parser.add_argument('-b', '--start_date', required=True, help='Start date in the format YYYY-MM-DD')
    parser.add_argument('-e', '--end_date', required=True, help='End date in the format YYYY-MM-DD')
    args = parser.parse_args(argv[1:])

    # Convert date strings to datetime objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Check if the start date is less than or equal to the end date
    assert start_date <= end_date, "The start date must be earlier or equal to the end date."

    # Start the download and filter process
    download_files(start_date, end_date)

if __name__ == "__main__":
    main(sys.argv)
