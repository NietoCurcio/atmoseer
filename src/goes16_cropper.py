import os
import time
from osgeo import osr
from osgeo import gdal
from netCDF4 import Dataset 
from datetime import datetime
import argparse

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
        print(f"Saving {filename_reprojected}")
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

def monitor_folder_and_preprocess(full_disk_dir, variable_names, save_dir):
    processed_files = set()

    while True:
        # List full disk files in the folder
        files = set(os.listdir(full_disk_dir))

        # Identify new files
        new_files = files - processed_files

        for file in new_files:
            file_path = os.path.join(full_disk_dir, file)

            try:
                preprocess_file(file_path, variable_names, save_dir)

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
    '''
    Example usage:
    python src/goes16_cropper.py --input_dir "./data/goes16/cmi/fulldisk" --save_dir "./data/goes16/cmi/cropped" --var CMI
    '''

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Retrieve GOES16's data for (user-provided) band, variable, and date range.")

    # Add command line arguments
    parser.add_argument("--vars", nargs='+', type=str, required=True, help="At least one variable name (CMI, ...)")
    parser.add_argument('--input_dir', type=str, default='./data/goes16/cmi/fulldisk', help="Directory of fulldisk files")
    parser.add_argument('--save_dir', type=str, default='./data/goes16/cmi/cropped', help="Directory to save cropped files")

    args = parser.parse_args()
    fulldisk_dir = args.input_dir  # Folder to monitor for new files
    save_dir = args.save_dir
    variable_names = args.vars

    print("Starting folder monitoring and preprocessing process...")
    monitor_folder_and_preprocess(fulldisk_dir, variable_names, save_dir)
