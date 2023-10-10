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

import sys

gdal.PushErrorHandler('CPLQuietErrorHandler')   # Ignore GDAL warnings

product_name = 'ABI-L2-RRQPEF'


def store_file(product_name, yyyymmddhhmn, output, img, acum, extent, undef):
  # Reproject the file
  filename_acum = f'{output}/{product_name}_{yyyymmddhhmn}.nc'
  reproject(filename_acum, img, acum, extent, undef)


def download_data_for_a_day(yyyymmdd):
  #-----------------------------------------------------------------------------------------------------------
  # Input and output directories
  input = "./temp/Samples"; os.makedirs(input, exist_ok=True)
  output = "./temp/Output"; os.makedirs(output, exist_ok=True)

  extent = [-44.0, -23.0, -43.0, -22.0]  # Min lon, Min lat, Max lon, Max lat

  # Initial time and date
  yyyy = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%Y')
  mm = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%m')
  dd = datetime.strptime(yyyymmdd, '%Y%m%d').strftime('%d')

  hour_ini = 0
  date_ini = datetime(int(yyyy),int(mm),int(dd),hour_ini,0)
  date_end = datetime(int(yyyy),int(mm),int(dd),hour_ini,0) + timedelta(hours=23)

  temp = date_ini

  #-----------------------------------------------------------------------------------------------------------
  # Accumulation loop
  while (temp <= date_end):
      # Date structure
      yyyymmddhhmn = datetime.strptime(str(temp), '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')

      print(f'Getting {product_name} data for {yyyymmddhhmn}...')

      # Download the file
      file_name = download_PROD(yyyymmddhhmn, product_name, input)

      #-----------------------------------------------------------------------------------------------------------
      # Variable
      var = 'RRQPE'

      # Open the file
      img = gdal.Open(f'NETCDF:{input}/{file_name}.nc:' + var)
      dqf = gdal.Open(f'NETCDF:{input}/{file_name}.nc:DQF')

      if img is not None:
        acum = np.zeros((5424,5424))
      
        # Read the header metadata
        metadata = img.GetMetadata()
        scale = float(metadata.get(var + '#scale_factor'))
        offset = float(metadata.get(var + '#add_offset'))
        undef = float(metadata.get(var + '#_FillValue'))
        dtime = metadata.get('NC_GLOBAL#time_coverage_start')
      
        # Load the data
        ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)
        ds_dqf = dqf.ReadAsArray(0, 0, dqf.RasterXSize, dqf.RasterYSize).astype(float)

        # Remove undef
        ds[ds == undef] = np.nan

        # Apply the scale and offset
        ds = (ds * scale + offset)

        # Apply NaN's where the quality flag is greater than 1
        ds[ds_dqf > 0] = np.nan

        # Sum the instantaneous value in the accumulation
        acum = np.nansum(np.dstack((acum, ds)),2)

        store_file(product_name, yyyymmddhhmn, output, img, acum, extent, undef)

      # Increment 1 hour
      temp = temp + timedelta(hours=1)

      try:
        file_path = f'{input}/{file_name}.nc'
        print(f'Removing file {file_path}')
        os.remove(file_path)  # Use os.remove() to delete the file
        print(f"File '{file_path}' has been successfully removed.")
      except FileNotFoundError:
          print(f"Error: File '{file_path}' not found.")
      except PermissionError:
          print(f"Error: Permission denied to remove file '{file_path}'.")
      except Exception as e:
          print(f"An error occurred: {e}")
      #-----------------------------------------------------------------------------------------------------------


def main(argv):

  periods = [
       ('20191204', '20200531'),
       ('20200901', '20210531'),
       ('20210901', '20220531'),
       ('20220901', '20230531')
  ]

  for period in periods:
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
        download_data_for_a_day(yyyymmdd)
        # Increment the current date by one day
        current_datetime += timedelta(days=1)

if __name__ == "__main__":
    main(sys.argv)
