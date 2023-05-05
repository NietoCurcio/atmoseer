import pandas as pd
import sys, getopt
from datetime import datetime
from util import is_posintstring
from globals import *
import s3fs
import numpy as np
import xarray as xr
import pandas as pd
import os
import concurrent.futures

# Use the anonymous credentials to access public data
fs = s3fs.S3FileSystem(anon=True)

# Latitude e longitude dos municipios que o forte de copacabana pega
# Latitude: entre -22.717° e -23.083°
# Longitude: entre -43.733° e -42.933°
# estacoes = [{"nome": "copacabana",
#             "n_lat": -22.717,
#             "s_lat": -23.083,
#             'w_lon': -43.733,
#             'e_lon': -42.933}]

# Latitude e Longitude do forte de copacaban
def filter_coordinates(ds:xr.Dataset):
  """
    Filter lightning event data in an xarray Dataset based on latitude and longitude boundaries.

    Args:
        ds (xarray.Dataset): Dataset containing lightning event data with variables `event_energy`, `event_lat`, and `event_lon`.
    Returns:
        xarray.Dataset: A new dataset with the same variables as `ds`, but with lightning events outside of the specified latitude and longitude boundaries removed.
  """
  return ds['event_energy'].where(
      (ds['event_lat'] >= -23.083) & (ds['event_lat'] <= -22.717) &
      (ds['event_lon'] >= -42.933) & (ds['event_lon'] <= -43.733),
      drop=True)

def processing_goes_16_file(filename):
    """
    Processes a GOES-16 data file, filters its coordinates, and returns a filtered xarray dataset.
    This function reads a GOES-16 data file using xarray, applies a filter to its coordinates, drops unnecessary columns,
    renames columns, sets the index to the 'time' column, and removes the original file. If the number of events in the
    dataset is zero, the function returns an empty xarray dataset object.

    Args:
        filename (str): The path to the GOES-16 data file to process.

    Returns:
        xr.Dataset: A filtered xarray dataset object.
    """
    df = []
    ds = xr.open_dataset(filename)
    ds = filter_coordinates(ds)
    if ds.number_of_events.nbytes != 0:
        df = ds.to_dataframe()
        df.drop(df.columns[4:-1], axis=1, inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        df.rename(columns={'event_time_offset': 'time',
                            'event_lat': 'lat',
                            'event_lon': 'lon',
                            'event_energy': 'energy'},
                inplace=True)
        df.set_index('time', inplace=True)
    os.remove(filename)
    return df

# Download all files in parallel, and rename them the same name (without the directory structure)
def download_file(files):
    """
    Downloads a GOES-16 netCDF file from an S3 bucket, filters it for events that fall within a specified set of coordinates, 
    and saves the filtered file to disk.
    Args:
        file (str): A string representing the name of the file to be downloaded from an S3 bucket.
    """
    files_process = []
    for file in files:
        filename = file.split('/')[-1]
        fs.get(file, filename)
        files_process.append(processing_goes_16_file(filename))
    # concatenate dataframes along the rows
    merged_df = pd.concat(files_process, axis=0)

    # save merged dataframe to a parquet file
    merged_df.to_parquet("merged_file.parquet.gzip")

def import_data(station_code, initial_year, final_year):
    """
    Downloads and saves GOES-16 data files from Amazon S3 for a given station code and time period.

    Args:
        station_code (str): The station code to download data for.
        initial_year (int): The initial year of the time period to download data for.
        final_year (int): The final year of the time period to download data for.

    Returns:
        None

    This function first reads a CSV file with relevant dates to download data for, then constructs a list of
    file paths for the requested station code and time period using these dates. The files are then downloaded
    using a thread pool executor for parallel processing.

    Note: This function assumes that the relevant data files are stored in the Amazon S3 bucket 'noaa-goes16'.
    """
    # Get files of GOES-16 data (multiband format) on multiple dates
    # format: <Product>/<Year>/<Day of Year>/<Hour>/<Filename>
    hours = [f'{h:02d}' for h in range(24)]  # Download all 24 hours of data

    start_date = pd.to_datetime(f'{initial_year}-01-01')
    end_date = pd.to_datetime(f'{final_year}-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    files = []
    for date in dates:
        year = str(date.year)
        day_of_year = f'{date.dayofyear:03d}'
        print(f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year}')
        if day_of_year == '005':
            break
        for hour in hours:
            target = f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year}/{hour}'
            files.extend(fs.ls(target))

    download_file(files)

def main(argv):
    station_code = ""

    start_goes_16 = 2017
    start_year = 2017
    end_year = datetime.now().year

    help_message = "{0} -s <station_id> -b <begin> -e <end>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:b:e:t:", ["help", "station=", "begin=", "end="])
    except:
        print(help_message)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)
            sys.exit(2)
        # elif opt in ("-s", "--station"):
        #     station_code = arg
        #     if not ((station_code == "all") or (station_code in INMET_STATION_CODES_RJ)):
        #         print(help_message)
        #         sys.exit(2)
        elif opt in ("-b", "--begin"):
            if not is_posintstring(arg):
                sys.exit("Argument start_year must be an integer. Exit.")
            start_year = int(arg)
        elif opt in ("-e", "--end"):
            if not is_posintstring(arg):
                sys.exit("Argument end_year must be an integer. Exit.")
            end_year = int(arg)

    # assert (station_code is not None) and (station_code != '')
    assert (start_year <= end_year) and (start_year >= start_goes_16)

    station_code = 'copacabana'
    start_year = 2019
    end_year = 2019

    import_data(station_code, start_year, end_year)


if __name__ == "__main__":
    main(sys.argv)
