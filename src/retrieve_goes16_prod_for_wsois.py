from netCDF4 import Dataset                     # Read / Write NetCDF4 files
import matplotlib.pyplot as plt                 # Plotting library
from datetime import timedelta, date, datetime  # Basic Dates and time types
import cartopy, cartopy.crs as ccrs             # Plot maps
import os                                       # Miscellaneous operating system interfaces
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

# gdal.PushErrorHandler('CPLQuietErrorHandler')   # Ignore GDAL warnings

#------------------------------------------------------------------------------
# def store_file(product_name, yyyymmddhhmn, output, img, acum, extent, undef):
#   # Reproject the file
#   filename_acum = f'{output}/{product_name}_{yyyymmddhhmn}.nc'
#   reproject(filename_acum, img, acum, extent, undef)


#------------------------------------------------------------------------------
def download_data_for_a_day(df: pd.DataFrame, yyyymmdd: str, stations_of_interest: dict, product_name: str, variable_name: str):
    """
    Downloads TPW (Total Precipitable Water) data for a specific day from GOES-16 satellite,
    extracts TPW values for stations of interest, and appends them to a DataFrame.

    Args:
    - df (pandas.DataFrame): DataFrame to which TPW values for stations of interest will be appended.
    - yyyymmdd (str): Date in 'YYYYMMDD' format specifying the day for which data will be downloaded.
    - stations_of_interest (dict): Dictionary containing stations of interest with their IDs as keys
                                   and their corresponding latitude and longitude coordinates as values.

    Returns:
    - pandas.DataFrame: Updated DataFrame with appended TPW values for stations of interest.
    """

    # Directory to temporarily store each downloaded full disk file.
    temp_dir  = "./data/goes16/temp"

    # Initial time and date
    yyyy = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%Y')
    mm = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%m')
    dd = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%d')

    hour_ini = 0
    date_ini = datetime(int(yyyy),int(mm),int(dd),hour_ini,0)
    date_end = datetime(int(yyyy),int(mm),int(dd),hour_ini,0) + timedelta(hours=23)

    time_step = date_ini
  
    #-----------------------------------------------------------------------------------------------------------
    # Accumulation loop. Scans all of the files for the given day. 
    # For each file, gets the TPW values for the locations where 
    # the stations of interested are located.
    while (time_step <= date_end):
        # Date structure
        yyyymmddhhmn = datetime.strptime(str(time_step), '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')

        print(f'-Getting data for {yyyymmddhhmn}...')

        # Download the full disk file from the Amazon cloud.
        file_name = download_PROD(yyyymmddhhmn, product_name, temp_dir)

        try:
            filename = f'{temp_dir}/{file_name}.nc'
            ds = open_dataset(filename)

            if ds is not None:
                field, LonCen, LatCen = ds.image(variable_name, lonlat='center')

                for wsoi_id in stations_of_interest:
                    lon = stations_of_interest[wsoi_id][1]
                    lat = stations_of_interest[wsoi_id][0]
                    x, y = find_pixel_of_coordinate(LonCen, LatCen, lon, lat)
                    value1 = field.data[y,x]
                    new_row = {'timestamp': yyyymmddhhmn, 'station_id': wsoi_id, variable_name: value1}
                    df = df.append(new_row, ignore_index=True)

            # try:
            #     os.remove(filename)  # Use os.remove() to delete the file
            # except FileNotFoundError:
            #     print(f"Error: File '{filename}' not found.")
            # except PermissionError:
            #     print(f"Error: Permission denied to remove file '{filename}'.")
            # except Exception as e:
            #     print(f"An error occurred: {e}")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")

        # Increment 10 minutes
        time_step = time_step + timedelta(minutes=10)
    return df

###
# python src/retrieve_goes16_prod_for_wsois.py --date_ini "2024-03-09" --date_end "2024-03-09" --prod ABI-L2-DSIF --var CAPE
# python src/retrieve_goes16_prod_for_wsois.py --date_ini "2024-01-12" --date_end "2024-01-12" --prod ABI-L2-DSIF --var CAPE
# python src/retrieve_goes16_prod_for_wsois.py --date_ini "2024-01-12" --date_end "2024-01-12" --prod ABI-L2-TPWF --var TPW
def main(argv):
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Retrieve GOES16's data for (user-provided) product, variable, and date range.")
    
    # Add command line arguments for date_ini and date_end
    parser.add_argument("--date_ini", type=str, required=True, help="Start date (format: YYYY-MM-DD)")
    parser.add_argument("--date_end", type=str, required=True, help="End date (format: YYYY-MM-DD)")
    parser.add_argument("--prod", type=str, required=True, help="GOES16 product name (e.g., 'ABI-L2-TPWF', 'ABI-L2-DSIF')")
    parser.add_argument("--var", type=str, required=True, help="Variable name (e.g., CAPE, TPW, ...)")
    
    args = parser.parse_args()
    start_date = args.date_ini
    end_date = args.date_end

    product_name = args.prod
    variable_name = args.var

    # try:
    #     if (station_id in globals.ALERTARIO_WEATHER_STATION_IDS or station_id in globals.ALERTARIO_GAUGE_STATION_IDS):
    #         # This UTC thing is really anonying!
    #         start_date = pd.to_datetime(args.date_ini, utc=True)
    #         end_date = pd.to_datetime(args.date_end, utc=True)
    #     else:
    #         start_date = pd.to_datetime(args.date_ini)
    #         end_date = pd.to_datetime(args.date_end)
    # except ParserError:
    #     print(f"Invalid date format: {args.date_ini}, {args.date_end}.")
    #     parser.print_help()
    #     sys.exit(2)

    stations_of_interest = dict()
    stations_filename = "./data/ws/WeatherStations.csv"
    df_stations = pd.read_csv(stations_filename)
    for wsoi_id in INMET_WEATHER_STATION_IDS:
        row = df_stations[df_stations["STATION_ID"] == wsoi_id].iloc[0]
        wsoi_lat_lon = (row["VL_LATITUDE"], row["VL_LONGITUDE"])
        stations_of_interest[row["STATION_ID"]] = wsoi_lat_lon

    # Create an empty DataFrame to store historical product values for each station of interest.
    df = pd.DataFrame(columns=['timestamp', 'station_id', variable_name])

    # Convert start_date and end_date to datetime objects
    from datetime import datetime
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

    # Iterate through the range of user-provided days, 
    # one day at a time, to retrieve corresponding TPW data.
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        yyyymmdd = current_datetime.strftime('%Y%m%d')
        df = download_data_for_a_day(df, yyyymmdd, stations_of_interest, product_name, variable_name)
        # Increment the current date by one day
        current_datetime += timedelta(days=1)

    print(f'Shape in the end: {df.shape}')
    df.to_parquet(f'tpw_{start_datetime}_to_{end_datetime}.parquet')

if __name__ == "__main__":
    main(sys.argv)
