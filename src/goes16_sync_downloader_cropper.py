import s3fs
import os
import time
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from osgeo import osr
from osgeo import gdal
from netCDF4 import Dataset 

########################################################################
### DOWNLOADER
########################################################################

# Queue to hold files that have been completely downloaded
file_queue = queue.Queue()

# Lock to synchronize access to shared resources
download_lock = threading.Lock()

# Full disk download function
def download_file(fs, file_name, local_path):
    """Download a file and put it into the download directory."""
    # with download_lock:
    #     try:
    #         fs.get(file_name, local_path)
    #         # print(f"Downloaded {os.path.basename(file_name)}")
    #         file_queue.put(local_path)  # Notify that the file is ready for processing
    #     except Exception as e:
    #         print(f"Error downloading {os.path.basename(file_name)}: {e}")
    try:
        fs.get(file_name, local_path)
        print(f"Downloaded {os.path.basename(file_name)}")
        file_queue.put(local_path)  # Notify that the file is ready for processing
    except Exception as e:
        print(f"Error downloading {os.path.basename(file_name)}: {e}")

# Simulated file processing function
def process_file(file_path):
    os.remove(file_path)  # Delete the file after processing
    print(f"Cropper generate file: {os.path.basename(file_path)}")

# Function to convert a regular date to the Julian day of the year
def get_julian_day(date):
    return date.strftime('%j')

# Function to download files from GOES-16 for a specific hour
def download_goes16_data_for_hour(fs, s3_path, channel, save_dir):
    # List all files in the given hour directory
    try:
        files = fs.ls(s3_path)
    except Exception as e:
        print(f"Error accessing S3 path {s3_path}: {e}")
        return

    print(f's3_path: {s3_path}')

    # Filter files for the specific channel (e.g., C01 for channel 1)
    channel_files = [file for file in files if f'C{channel:02d}' in file]

    print(f'len(channel_files): {len(channel_files)}')

    # Download each file using threads
    with ThreadPoolExecutor() as executor:
        for file in channel_files:
            file_name = os.path.basename(file)
            local_path = os.path.join(save_dir, file_name)
            if not os.path.exists(local_path):
                executor.submit(download_file, fs, file, local_path)

# Function to download all files for a given day
def download_goes16_data_for_day(date, channel, save_dir):
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
    with download_lock:
        with ThreadPoolExecutor() as executor:
            for hour in range(24):
                hour_str = f'{hour:02d}'
                s3_path = f'{bucket}/{product}/{year}/{julian_day}/{hour_str}/'
                
                # Submit each hour's download process to the thread pool
                executor.submit(download_goes16_data_for_hour, fs, s3_path, channel, save_dir)

# Main function to download data for a range of dates
def download_goes16_data_for_period(start_date, end_date, ignored_months, channel, save_dir):
    current_date = start_date
    while current_date <= end_date:
        if current_date.month in ignored_months:
            continue
        print(f"Downloading data for {current_date.strftime('%Y-%m-%d')}")
        download_goes16_data_for_day(current_date, channel, save_dir)
        current_date += timedelta(days=1)

# Thread for downloading files
def downloader_thread(download_dir):
    # files_to_download = ['file1.txt', 'file2.txt', 'file3.txt']  # Example file list
    # for file_name in files_to_download:
        # download_file(file_name, download_dir)
    start_date = datetime.strptime("2024-10-17", "%Y-%m-%d")
    end_date = datetime.strptime("2024-10-17", "%Y-%m-%d")
    download_goes16_data_for_period(start_date, end_date, ignored_months = [6,7,8], channel = 7, save_dir = "./downloads")

########################################################################
### CROPPER
########################################################################

# Thread for cropping files
def cropper_thread():
    while True:
        file_path = file_queue.get()  # Wait for a file to be added to the queue
        if file_path is None:
            break  # Stop thread if a None sentinel is received
        process_file(file_path)
        file_queue.task_done()  # Mark the task as done


def extract_middle_part(file_path):
    # Split the file path by '/' and get the last part (the filename)
    filename = file_path.split('/')[-1]
    
    # The middle part ends right before the '_s' section
    middle_part = filename.split('_s')[0]
    
    return middle_part

def crop_full_disk_and_save(full_disk_filename, variable_names, extent, dest_path):
    file = Dataset(full_disk_filename)
    dtime = datetime.strptime(file.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')
    # print(f'dtime1: {dtime}')
    yyyymmddhhmn = dtime.strftime('%Y_%m_%d_%H_%M')

    for var in variable_names:
   
        # Open the file
        img = gdal.Open(f'NETCDF:{full_disk_filename}:' + var)

        assert (img is not None)

        # Read the header metadata
        metadata = img.GetMetadata()
        scale = float(metadata.get(var + '#scale_factor'))
        offset = float(metadata.get(var + '#add_offset'))
        undef = float(metadata.get(var + '#_FillValue'))

        # dtime = metadata.get('NC_GLOBAL#time_coverage_start')
        # dtime = datetime.strptime(dtime, '%Y-%m-%dT%H:%M:%S.%fZ')
        # print(f'dtime2: {dtime}')
        # yyyymmddhhmn = dtime.strftime('%Y_%m_%d_%H_%M')

        # Load the data
        ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)

        # Apply the scale and offset
        ds = (ds * scale + offset)

        # Read the original file projection and configure the output projection
        source_prj = osr.SpatialReference()
        source_prj.ImportFromProj4(img.GetProjectionRef())

        target_prj = osr.SpatialReference()
        target_prj.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

        # Reproject the data
        GeoT = img.GetGeoTransform()
        driver = gdal.GetDriverByName('MEM')
        raw = driver.Create('raw', ds.shape[0], ds.shape[1], 1, gdal.GDT_Float32)
        raw.SetGeoTransform(GeoT)
        raw.GetRasterBand(1).WriteArray(ds)

        # Define the parameters of the output file  
        options = gdal.WarpOptions(format = 'netCDF', 
                srcSRS = source_prj, 
                dstSRS = target_prj,
                outputBounds = (extent[0], extent[3], extent[2], extent[1]), 
                outputBoundsSRS = target_prj, 
                outputType = gdal.GDT_Float32, 
                srcNodata = undef, 
                dstNodata = 'nan', 
                resampleAlg = gdal.GRA_NearestNeighbour)
        
        img = None  # Close file

        # Write the reprojected file on disk
        prefix = extract_middle_part(full_disk_filename)
        # print("prefix: ", prefix)
        filename_reprojected = f'{dest_path}/{prefix}_{var}_{yyyymmddhhmn}.nc'
        print(f"Saving crop: {filename_reprojected}")
        gdal.Warp(filename_reprojected, raw, options=options)

# Simulate file preprocessing
def preprocess_file(file_path, variable_names, save_dir):
    # print(f"Processing {file_path}")
    lat_max, lon_max = (
        -21.699774257353113,
        -42.35676996062447,
    )  # canto superior direito
    lat_min, lon_min = (
        -23.801876626302175,
        -45.05290312102409,
    )  # canto inferior esquerdo
    extent = [lon_min, lat_min, lon_max, lat_max]
    crop_full_disk_and_save(full_disk_filename = file_path, 
                            variable_names = variable_names, 
                            extent = extent, 
                            dest_path = save_dir)

########################################################################
### MAIN
########################################################################

def main():
    download_dir = './downloads'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # Start the downloading thread
    downloader = threading.Thread(target=downloader_thread, args=(download_dir,))
    downloader.start()

    # Start the processing thread
    processor = threading.Thread(target=cropper_thread)
    processor.start()

    # Wait for the downloading thread to finish
    downloader.join()

    # Signal the processing thread to exit
    file_queue.put(None)
    
    # Wait for the processing thread to finish
    processor.join()

if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()    
    end_time = time.time()  # Record the end time
    duration = (end_time - start_time) / 60  # Calculate duration in minutes
    print(f"Script execution time: {duration:.2f} minutes.")
