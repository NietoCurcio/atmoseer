import s3fs
import os
from datetime import datetime

# Function to convert a regular date to the Julian day of the year
def get_julian_day(date):
    return date.strftime('%j')

# Function to download files from GOES-16
def download_goes16_data(date, channel, save_dir='goes16_data'):
    # Set up S3 access
    fs = s3fs.S3FileSystem(anon=True)
    bucket = 'noaa-goes16'
    product = f'ABI-L1b-RadF'  # Full-disk Radiances, replace with other product if needed

    # Format the date
    year = date.strftime('%Y')
    julian_day = get_julian_day(date)

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loop through the 24 hours of the day to download files
    for hour in range(24):
        hour_str = f'{hour:02d}'
        s3_path = f'{bucket}/{product}/{year}/{julian_day}/{hour_str}/'
        
        # List all files in the given hour directory
        try:
            files = fs.ls(s3_path)
        except Exception as e:
            print(f"Error accessing S3 path {s3_path}: {e}")
            continue

        # Filter files for the specific channel (e.g., C01 for channel 1)
        channel_files = [file for file in files if f'C{channel:02d}' in file]

        # Download each file
        for file in channel_files:
            file_name = os.path.basename(file)
            local_path = os.path.join(save_dir, file_name)
            
            # Download the file if it doesn't exist locally
            if not os.path.exists(local_path):
                try:
                    fs.get(file, local_path)
                    print(f"Downloaded {file_name}")
                except Exception as e:
                    print(f"Error downloading {file_name}: {e}")

if __name__ == "__main__":
    # Input: User provides the date and channel
    user_date = input("Enter the date (YYYY-MM-DD): ")
    user_channel = int(input("Enter the GOES16 channel number (1-16): "))

    # Convert user input to a datetime object
    date = datetime.strptime(user_date, "%Y-%m-%d")

    # Call the download function
    download_goes16_data(date, user_channel)
