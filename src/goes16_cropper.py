import os
import time
from osgeo import osr
from osgeo import gdal
from netCDF4 import Dataset 
from datetime import datetime

def crop_full_disk_and_save(full_disk_filename, variable_names, extent, dest_path, band):
    # datetimeAgain = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M')
    # formatted_date = datetimeAgain.strftime('%Y-%m-%d')

    file = Dataset(full_disk_filename)
    date = (datetime.strptime(file.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ'))
    # print('date1: ', date.strftime('%Y_%m_%d_%H_%M'))
    # print('date2: ', yyyymmddhhmn)

    yyyymmddhhmn = date.strftime('%Y_%m_%d_%H_%M')

    for var in variable_names:
   
        # Open the file
        img = gdal.Open(f'NETCDF:{full_disk_filename}:' + var)

        # Read the header metadata
        metadata = img.GetMetadata()
        scale = float(metadata.get(var + '#scale_factor'))
        offset = float(metadata.get(var + '#add_offset'))
        undef = float(metadata.get(var + '#_FillValue'))
        # dtime = metadata.get('NC_GLOBAL#time_coverage_start')

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
        filename_reprojected = f'{dest_path}/{var}_band{band}_{yyyymmddhhmn}.nc'
        print(f"Saving {filename_reprojected}")
        gdal.Warp(filename_reprojected, raw, options=options)

# Simulate file preprocessing
def preprocess_file(file_path):
    print(f"Processing {file_path}")
    lat_max, lon_max = (
        -21.699774257353113,
        -42.35676996062447,
    )  # canto superior direito
    lat_min, lon_min = (
        -23.801876626302175,
        -45.05290312102409,
    )  # canto inferior esquerdo
    extent = [lon_min, lat_min, lon_max, lat_max]
    crop_full_disk_and_save(full_disk_filename = file_path, variable_names = ['CMI'], extent = extent, dest_path = './goes16_data_cropped', band = '7')

def monitor_folder_and_preprocess(local_folder):
    processed_files = set()

    while True:
        # List files in the folder
        files = set(os.listdir(local_folder))

        # Identify new files
        new_files = files - processed_files

        for file in new_files:
            file_path = os.path.join(local_folder, file)

            try:
                preprocess_file(file_path)

                # Delete the file after processing
                os.remove(file_path)
                print(f"Deleted {file_path} after processing.")

                # Mark file as processed
                processed_files.add(file)

            except OSError:
                print(f'File {file_path} still being downloaded...moving to the next available file.')
                continue

        # Sleep for a short period before checking again
        time.sleep(1)

if __name__ == "__main__":
    local_folder = './goes16_data'  # Folder to monitor

    print("Starting folder monitoring and preprocessing process...")
    monitor_folder_and_preprocess(local_folder)
