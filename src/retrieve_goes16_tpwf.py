from netCDF4 import Dataset                     # Read / Write NetCDF4 files
import matplotlib.pyplot as plt                 # Plotting library
from datetime import timedelta, date, datetime  # Basic Dates and time types
import cartopy, cartopy.crs as ccrs             # Plot maps
import os                                       # Miscellaneous operating system interfaces
from osgeo import gdal                          # Python bindings for GDAL
import numpy as np                              # Scientific computing with Python
from matplotlib import cm                       # Colormap handling utilities
from goes16_utils import download_PROD             # Our function for download
from goes16_utils import reproject                 # Our function for reproject
from goes16.processing_data import find_pixel_of_coordinate
from goes16.processing_data import open_dataset

import pandas as pd

import sys

import argparse
from globals import INMET_WEATHER_STATION_IDS

gdal.PushErrorHandler('CPLQuietErrorHandler')   # Ignore GDAL warnings

#------------------------------------------------------------------------------
def store_file(product_name, yyyymmddhhmn, output, img, acum, extent, undef):
  # Reproject the file
  filename_acum = f'{output}/{product_name}_{yyyymmddhhmn}.nc'
  reproject(filename_acum, img, acum, extent, undef)


#------------------------------------------------------------------------------
def download_data_for_a_day(df, yyyymmdd, stations_of_interest):
    # Input and output directories
    input  = "./data/goes16/Samples"; os.makedirs(input, exist_ok=True)
    output = "./data/goes16/Output"; os.makedirs(output, exist_ok=True)

    extent = [-44.0, -23.0, -43.0, -22.0]  # Min lon, Min lat, Max lon, Max lat

    # Initial time and date
    yyyy = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%Y')
    mm = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%m')
    dd = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%d')

    hour_ini = 0
    date_ini = datetime(int(yyyy),int(mm),int(dd),hour_ini,0)
    date_end = datetime(int(yyyy),int(mm),int(dd),hour_ini,0) + timedelta(hours=23)

    temp = date_ini
  
    var = 'TPW'
    product_name = 'ABI-L2-TPWF'

    #-----------------------------------------------------------------------------------------------------------
    # Accumulation loop
    while (temp <= date_end):
        # Date structure
        yyyymmddhhmn = datetime.strptime(str(temp), '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')

        print(f'-Getting data for {yyyymmddhhmn}...')

        # Download the file
        file_name = download_PROD(yyyymmddhhmn, product_name, input)

        # Open the file
        img = gdal.Open(f'NETCDF:{input}/{file_name}.nc:' + var)

        if img is not None:

            ds = open_dataset(f'{input}/{file_name}.nc')
            RRQPE, LonCen, LatCen = ds.image(var, lonlat='center')

            for wsoi_id in stations_of_interest:
                lon = stations_of_interest[wsoi_id][1]
                lat = stations_of_interest[wsoi_id][0]
                x, y = find_pixel_of_coordinate(LonCen, LatCen, lon, lat)
                value1 = RRQPE.data[y,x]
                new_row = {'timestamp': yyyymmddhhmn, 'station_id': wsoi_id, 'tpw_value': value1}
                df = df.append(new_row, ignore_index=True)

            acum = np.zeros((5424,5424))
        
            # Read the header metadata
            metadata = img.GetMetadata()
            undef = float(metadata.get(var + '#_FillValue'))

            store_file(product_name, yyyymmddhhmn, output, img, acum, extent, undef)

        # Increment 10 minutes
        temp = temp + timedelta(minutes=10)

        try:
            file_path = f'{input}/{file_name}.nc'
            # print(f'Removing file {file_path}')
            os.remove(file_path)  # Use os.remove() to delete the file
            # print(f"File '{file_path}' has been successfully removed.")
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except PermissionError:
            print(f"Error: Permission denied to remove file '{file_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
        #-----------------------------------------------------------------------------------------------------------
    return df


def main(argv):
    periods = [
        ('20200101', '20200531'),
        ('20200901', '20201231'),

        ('20210101', '20210531'),
        ('20210901', '20211231'),

        ('20220101', '20220531'),
        ('20220901', '20221231'),

        ('20230001', '20230531'),
    ]

    # # Create an argument parser
    # parser = argparse.ArgumentParser(description="Process TPW data for a date range.")
    
    # # Add command line arguments for date_ini and date_end
    # parser.add_argument("--date_ini", required=True, help="Start date (YYYYMMDD)")
    # parser.add_argument("--date_end", required=True, help="End date (YYYYMMDD)")
    
    # args = parser.parse_args()
    # date_ini = args.date_ini
    # date_end = args.date_end

    stations_of_interest = dict()
    stations_filename = "./data/ws/WeatherStations.csv"
    df_stations = pd.read_csv(stations_filename)
    for wsoi_id in INMET_WEATHER_STATION_IDS:
        row = df_stations[df_stations["STATION_ID"] == wsoi_id].iloc[0]
        wsoi_lat_lon = (row["VL_LATITUDE"], row["VL_LONGITUDE"])
        stations_of_interest[row["STATION_ID"]] = wsoi_lat_lon

    for period in periods:
        # Create an empty DataFrame
        df = pd.DataFrame(columns=['timestamp', 'station_id', 'tpw_value'])

        start_date = period[0]
        end_date = period[1]

        # Convert start_date and end_date to datetime objects
        from datetime import datetime
        start_datetime = datetime.strptime(start_date, '%Y%m%d')
        end_datetime = datetime.strptime(end_date, '%Y%m%d')

        # Iterate through the range of dates
        current_datetime = start_datetime
        while current_datetime <= end_datetime:
            yyyymmdd = current_datetime.strftime('%Y%m%d')
            df = download_data_for_a_day(df, yyyymmdd, stations_of_interest)
            # Increment the current date by one day
            current_datetime += timedelta(days=1)
    
        print(f'Shape in the end: {df.shape}')
        df.to_parquet(f'tpw_{start_datetime}_to_{end_datetime}.parquet')

if __name__ == "__main__":
    main(sys.argv)
