from netCDF4 import Dataset              # Read / Write NetCDF4 files
import matplotlib.pyplot as plt          # Plotting library
from datetime import datetime            # Basic Dates and time types
import cartopy, cartopy.crs as ccrs      # Plot maps
import os                                # Miscellaneous operating system interfaces
from goes16_utils import download_CMI       # Our own utilities
from goes16_utils import geo2grid, convertExtent2GOESProjection      # Our own utilities
from globals import GOES_DATA_DIR
import argparse

import boto3
import botocore
from botocore import UNSIGNED            # boto3 config
from botocore.config import Config       # boto3 config

from datetime import datetime, timedelta
import logging

from osgeo import osr                           # Python bindings for GDAL
from osgeo import gdal                          # Python bindings for GDAL

"""Builds a reprojected netCDF file from a full disk netCDF file.

Args:
in_filename (str): Path to the full disk netCDF file.
out_filename (str): Path to write the reprojected netCDF file.
var (str): Name of the variable to reproject within the input file.
extent (tuple): A tuple of four floats defining the output bounding box in
geographic coordinates (lon_min, lat_max, lon_max, lat_min).

Returns:
str: The starting time of the data coverage extracted from the metadata
of the input file.

Raises:
RuntimeError: If there is an error opening the input file or writing
the reprojected file.
"""
def build_projection_from_full_disk(in_filename, out_filename, var, extent):

    # Open the file
    img = gdal.Open(f'NETCDF:{in_filename}:' + var)

    # Read the header metadata
    metadata = img.GetMetadata()
    scale = float(metadata.get(var + '#scale_factor'))
    offset = float(metadata.get(var + '#add_offset'))
    undef = float(metadata.get(var + '#_FillValue'))
    dtime = metadata.get('NC_GLOBAL#time_coverage_start')

    # print(f'scale/offset/undef/dtime: {scale}/{offset}/{undef}/{dtime}')

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
            xRes = 0.02,
            yRes = 0.02,
            resampleAlg = gdal.GRA_NearestNeighbour)

    # print(options)

    # Write the reprojected file on disk
    gdal.Warp(f'{out_filename}', raw, options=options)

    return dtime

def download_goes16_product(product_name, date_str, save_path="."):
    # Convert date string to datetime object
    date_obj = datetime.strptime(date_str, "%Y%m%d")

    yyyymmddhh = date_str
    year = datetime.strptime(yyyymmddhh, '%Y%m%d').strftime('%Y')
    day_of_year = datetime.strptime(yyyymmddhh, '%Y%m%d').strftime('%j')

    satellite = "noaa-goes16"

    # Create an S3 client
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Generate the prefix based on the date
    prefix = f'{product_name}/{year}/{day_of_year}'
    print(f'prefix: {prefix}')
    
    # List objects in the specified S3 bucket and prefix
    try:
        response = s3_client.list_objects_v2(Bucket="noaa-goes16", Prefix=prefix, )

        if 'Contents' not in response:
            # There are no files
            print(f'No files found for the date: {date_str}, Product: {product_name}')
            return -1
        else:
            # print(f'Contents: {response.get("Contents", [])}')
            # Download each file
            for content in response.get("Contents", []):
                filename = content["Key"]

                # Find the index (within the key) in which the date information begins, and extract the date part
                s_index = filename.find('s')
                temp = filename[s_index+1: -1]
                underline_index = temp.find('_')
                date_string = temp[0: underline_index-3]

                # Parse the date string
                parsed_date = datetime.strptime(date_string, '%Y%j%H%M')

                # Format the parsed date as YYYYMMDDHM
                formatted_date = parsed_date.strftime('%Y_%m_%d_%H_%M')

                print(f'formatted data: {formatted_date}')

                full_disk_filename = f"{save_path}/{product_name}_{formatted_date}.nc"

                print(f"Downloading {filename} to {full_disk_filename}")
                s3_client.download_file("noaa-goes16", filename, full_disk_filename)

            print("Download complete!")

    except botocore.exceptions.NoCredentialsError:
        print("Credentials not available. Please configure AWS credentials.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GOES-16 data for a specified product, date, and save path.")
    
    parser.add_argument("--product_name", type=str, default="ABI-L2-DSIF", help="Specify the product name")
    parser.add_argument("--target_date", type=str, default="20220911", help="Specify the target date in YYYYMMDD format")
    parser.add_argument("--local_save_path", type=str, default="./data/goes16/goes16_dsif", help="Specify the local path to save the downloaded files")

    args = parser.parse_args()

    download_goes16_product(args.product_name, args.target_date, args.local_save_path)
