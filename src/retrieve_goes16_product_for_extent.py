from netCDF4 import Dataset                     # Read / Write NetCDF4 files
from datetime import timedelta, date, datetime  # Basic Dates and time types
import os                                       # Miscellaneous operating system interfaces
from osgeo import osr                           # Python bindings for GDAL
from osgeo import gdal                          # Python bindings for GDAL
import numpy as np                              # Scientific computing with Python
from goes16_utils import download_PROD, reproject, geo2grid
import sys
import argparse
from typing import List
import time
import pickle
import logging

def save_extent_data(full_disk_filename, yyyymmddhhmn, variable_names, extent, dest_path):
    dqf_saved = False

    for var in variable_names:
        # Save Data Quality Flag (DQF)
        if not dqf_saved:
            full_disk_ds = Dataset(full_disk_filename)
            # Convert lat/lon to grid-coordinates
            lly, llx = geo2grid(extent[1], extent[0], full_disk_ds)
            ury, urx = geo2grid(extent[3], extent[2], full_disk_ds)
            data = full_disk_ds.variables['DQF_Overall'][ury:lly, llx:urx]
            dqf_file_name = f'{dest_path}/{yyyymmddhhmn}_DQF.pkl'
            dqf_file = open(dqf_file_name, 'wb')
            pickle.dump(data, dqf_file)
            dqf_saved = True

        # Open the file
        img = gdal.Open(f'NETCDF:{full_disk_filename}:' + var)

        # Read the header metadata
        metadata = img.GetMetadata()
        scale = float(metadata.get(var + '#scale_factor'))
        offset = float(metadata.get(var + '#add_offset'))
        undef = float(metadata.get(var + '#_FillValue'))
        dtime = metadata.get('NC_GLOBAL#time_coverage_start')

        # Load the data
        ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)

        # Apply the scale and offset
        ds = (ds * scale + offset)

        # Apply NaN's where the quality flag is greater than 1
        # ds[ds_dqf > 1] = np.nan

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

        # Write the reprojected file on disk
        filename_reprojected = f'{dest_path}/{yyyymmddhhmn}_{var}.nc'
        gdal.Warp(filename_reprojected, raw, options=options)

#------------------------------------------------------------------------------
def download_data_for_a_day(extent: List[float], 
                            dest_path: str,
                            yyyymmdd: str, 
                            product_name: str, 
                            variable_names: List[str], 
                            temporal_resolution: int, 
                            remove_full_disk_file: bool = True):
    """
    Downloads values of a specific variable from a specific product and for a specific day from GOES-16 satellite.
    These values are downloaded only for the locations (lat/lon) of a list of stations of interest. 
    These downloaded values are appended (as new rows) to the provided DataFrame. 
    Each row will have the following columns: (timestamp, station_id, variable names's value)

    Args:
    - df (pandas.DataFrame): DataFrame to which downloaded values for stations of interest will be appended as a new column.
    - yyyymmdd (str): Date in 'YYYYMMDD' format specifying the day for which data will be downloaded.
    - stations_of_interest (dict): Dictionary containing stations of interest with their IDs as keys
                                   and their corresponding latitude and longitude coordinates as values.
    - product_name (str): The name of the GOES-16 product from which data will be downloaded.
    - variable_name (str): The specific variable to be retrieved from the product.
    """

    # Directory to temporarily store each downloaded full disk file.
    TEMP_DIR  = "./data/goes16/temp"

    # Initial time and date
    yyyy = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%Y')
    mm = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%m')
    dd = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%d')

    hour_ini = 0
    date_ini = datetime(int(yyyy),int(mm),int(dd),hour_ini,0)
    date_end = datetime(int(yyyy),int(mm),int(dd),hour_ini,0) + timedelta(hours=23)

    time_step = date_ini
  
    while (time_step <= date_end):
        # Date structure
        yyyymmddhhmn = datetime.strptime(str(time_step), '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')

        logging.info(f'-Getting data for {yyyymmddhhmn}...')

        # Download the full disk file from the Amazon cloud.
        file_name = download_PROD(yyyymmddhhmn, product_name, TEMP_DIR)

        if file_name != -1:
            try:
                full_disk_filename = f'{TEMP_DIR}/{file_name}.nc'

                save_extent_data(full_disk_filename, yyyymmddhhmn, variable_names, extent, dest_path)

                if remove_full_disk_file:
                    try:
                        os.remove(full_disk_filename)  # Use os.remove() to delete the file
                    except FileNotFoundError:
                        logging.info(f"Error: File '{full_disk_filename}' not found.")
                    except PermissionError:
                        logging.info(f"Error: Permission denied to remove file '{full_disk_filename}'.")
                    except Exception as e:
                        logging.info(f"An error occurred: {e}")
            except FileNotFoundError:
                logging.info(f"Error: File '{full_disk_filename}' not found.")

        # Increment to get the next full disk observation.
        time_step = time_step + timedelta(minutes=temporal_resolution)

def main(argv):
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Retrieve GOES16's data for (user-provided) product, variable, and date range.")

    # Add command line arguments for date_ini and date_end
    parser.add_argument("--date_ini", type=str, required=True, help="Start date (format: YYYY-MM-DD)")
    parser.add_argument("--date_end", type=str, required=True, help="End date (format: YYYY-MM-DD)")
    parser.add_argument("--prod", type=str, required=True, help="GOES16 product name (e.g., 'ABI-L2-TPWF', 'ABI-L2-DSIF')")
    parser.add_argument("--vars", nargs='+', type=str, required=True, help="At least one variable name (TPW, CAPE, CIN, ...)")
    parser.add_argument("--temporal_resolution", type=int, default=10, help="Temporal resolution of the observations, in minutes (default: 10)")
    parser.add_argument("--ignored_months", nargs='+', type=int, required=False, default=[6, 7, 8],
                        help="Months to ignore (e.g., --ignored_months 6 7 8)")
    
    # TODO - check compatibility between the following cmd line args: "prod" and "vars"

    dest_path = './data/goes16/DSI'

    args = parser.parse_args()
    start_date = args.date_ini
    end_date = args.date_end
    product_name = args.prod
    variable_names = args.vars
    temporal_resolution = args.temporal_resolution
    ignored_months = args.ignored_months

    # Add command line arguments for extent (bounding box)
    parser.add_argument("--lon_min", type=float, required=True, help="Minimum longitude (west)")
    parser.add_argument("--lat_min", type=float, required=True, help="Minimum latitude (south)")
    parser.add_argument("--lon_max", type=float, required=True, help="Maximum longitude (east)")
    parser.add_argument("--lat_max", type=float, required=True, help="Maximum latitude (north)")

    # Use the provided extent values
    extent = [args.lon_min, args.lat_min, args.lon_max, args.lat_max]

    # Convert start_date and end_date to datetime objects
    from datetime import datetime
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

    # Iterate through the range of user-provided days, 
    # one day at a time, to retrieve corresponding data.
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        # Ignore winter months
        if current_datetime.month not in ignored_months:
            yyyymmdd = current_datetime.strftime('%Y%m%d')
            df = download_data_for_a_day(extent, dest_path, yyyymmdd, product_name, variable_names, temporal_resolution=temporal_resolution)
        # Increment the current date by one day
        current_datetime += timedelta(days=1)


if __name__ == "__main__":
    ### Example usage:
    # python src/retrieve_goes16_product_for_extent.py --date_ini "2024-01-13" --date_end "2024-01-13" --prod ABI-L2-DSIF --vars CAPE LI TT SI KI --ignored_months 6 7 8 --lon_min -45.05290312102409 --lat_min -23.801876626302175 --lon_max -42.35676996062447 --lat_max -21.699774257353113

    fmt = "[%(levelname)s] %(funcName)s():%(lineno)i: %(message)s"
    logging.basicConfig(level=logging.INFO, format = fmt)

    start_time = time.time()  # Record the start time

    main(sys.argv)
    
    end_time = time.time()  # Record the end time
    duration = (end_time - start_time) / 60  # Calculate duration in minutes
    
    print(f"Script duration: {duration:.2f} minutes")    
