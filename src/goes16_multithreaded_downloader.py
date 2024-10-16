import s3fs
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time
import argparse
import logging

# Function to convert a regular date to the Julian day of the year
def get_julian_day(date):
    return date.strftime('%j')

# Function to download a single file
def download_file(fs, file, local_path):
    try:
        fs.get(file, local_path)
        print(f"Downloaded {os.path.basename(file)}")
    except Exception as e:
        print(f"Error downloading {os.path.basename(file)}: {e}")

# Function to download files from GOES-16 for a specific hour
def download_goes16_hour_data(fs, s3_path, channel, save_dir):
    # List all files in the given hour directory
    try:
        files = fs.ls(s3_path)
    except Exception as e:
        print(f"Error accessing S3 path {s3_path}: {e}")
        return

    # Filter files for the specific channel (e.g., C01 for channel 1)
    channel_files = [file for file in files if f'C{channel:02d}' in file]

    # Download each file using threads
    with ThreadPoolExecutor() as executor:
        for file in channel_files:
            file_name = os.path.basename(file)
            local_path = os.path.join(save_dir, file_name)
            if not os.path.exists(local_path):
                # Submit the download task to the thread pool
                executor.submit(download_file, fs, file, local_path)

# Function to download all files for a given day
def download_goes16_data(date, channel, save_dir='goes16_data'):
    # Set up S3 access
    fs = s3fs.S3FileSystem(anon=True)
    bucket = 'noaa-goes16'
    product = f'ABI-L2-CMIPF'

    # Format the date
    year = date.strftime('%Y')
    julian_day = get_julian_day(date)

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Download data for each hour concurrently
    with ThreadPoolExecutor() as executor:
        for hour in range(24):
            hour_str = f'{hour:02d}'
            s3_path = f'{bucket}/{product}/{year}/{julian_day}/{hour_str}/'
            
            # Submit each hour's download process to the thread pool
            executor.submit(download_goes16_hour_data, fs, s3_path, channel, save_dir)

# Main function to download data for a range of dates
def download_goes16_data_for_period(start_date, end_date, ignored_months, channel, save_dir='goes16_data'):
    current_date = start_date
    if current_date.month not in ignored_months:
        while current_date <= end_date:
            print(f"Downloading data for {current_date.strftime('%Y-%m-%d')}")
            download_goes16_data(current_date, channel, save_dir)
            current_date += timedelta(days=1)

if __name__ == "__main__":
    '''
    Example usage:
    one-day test
    python src/goes16_multithreaded_downloader.py --start_date "2024-02-08" --end_date "2024-02-08" --channel 7 --save_dir "./data/goes16/cmi/fulldisk"

    librae
    python src/goes16_multithreaded_downloader.py --start_date "2023-01-01" --end_date "2023-12-31" --channel 7 --save_dir "./data/goes16/cmi/fulldisk"

    tauri
    python src/goes16_multithreaded_downloader.py --start_date "2022-01-01" --end_date "2023-12-31" --channel 9 --save_dir "./data/goes16/cmi/fulldisk"
    '''
    # Create an argument parser to accept start and end dates, and channel number from the command line
    parser = argparse.ArgumentParser(description="Download GOES-16 data for a specific date range.")
    parser.add_argument('--start_date', type=str, required=True, help="Start date (format: YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, required=True, help="End date (format: YYYY-MM-DD)")
    parser.add_argument('--channel', type=int, required=True, help="GOES-16 channel number (1-16)")
    parser.add_argument('--save_dir', type=str, default='goes16_data', help="Directory to save downloaded files")
    parser.add_argument("--ignored_months", nargs='+', type=int, required=False, default=[6, 7, 8],
                        help="Months to ignore (e.g., --ignored_months 6 7 8)")
    
    fmt = "[%(levelname)s] %(funcName)s():%(lineno)i: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    
    args = parser.parse_args()

    # Parse the start and end dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    channel = args.channel
    save_dir = args.save_dir
    ignored_months = args.ignored_months

    # Start the download process
    start_time = time.time()  # Record the start time
    download_goes16_data_for_period(start_date, end_date, ignored_months, channel, save_dir)
    end_time = time.time()  # Record the end time
    duration = (end_time - start_time) / 60  # Calculate duration in minutes
    
    print(f"Script execution time: {duration:.2f} minutes.")
