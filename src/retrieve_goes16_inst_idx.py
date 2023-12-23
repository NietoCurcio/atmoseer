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
            # Download each file
            for content in response.get("Contents", []):
                key = content["Key"]
                print(f'key: {key}')

                # Find the index (within the key) in which the date information begins, and extract the date part
                s_index = key.find('s')
                temp = key[s_index+1: -1]
                underline_index = temp.find('_')
                date_string = temp[0: underline_index-3]

                print(f'date: {date_string}')

                # Parse the date string
                parsed_date = datetime.strptime(date_string, '%Y%j%H%M')

                # Format the parsed date as YYYYMMDDHM
                formatted_date = parsed_date.strftime('%Y_%m_%d_%H_%M')

                print(f'formatted data: {formatted_date}')

                local_path = f"{save_path}/{product_name}_{formatted_date}.nc"

                print(f"Downloading {key} to {local_path}")
                s3_client.download_file("noaa-goes16", key, local_path)

            print("Download complete!")

    except botocore.exceptions.NoCredentialsError:
        print("Credentials not available. Please configure AWS credentials.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GOES-16 data for a specified product, date, and save path.")
    
    parser.add_argument("--product_name", type=str, default="ABI-L2-DSIF", help="Specify the product name")
    parser.add_argument("--target_date", type=str, default="20220911", help="Specify the target date in YYYYMMDD format")
    parser.add_argument("--local_save_path", type=str, default="../data/goes16_dsif", help="Specify the local path to save the downloaded files")

    args = parser.parse_args()

    download_goes16_product(args.product_name, args.target_date, args.local_save_path)
