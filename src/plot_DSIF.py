# Training: Python and GOES-R Imagery: Script 17 - Level 2 Products (RRQPE) and Data Accumulation
#-----------------------------------------------------------------------------------------------------------
# Required modules
from netCDF4 import Dataset                     # Read / Write NetCDF4 files
import matplotlib.pyplot as plt                 # Plotting library
from datetime import timedelta, datetime  # Basic Dates and time types
import cartopy, cartopy.crs as ccrs             # Plot maps
import os          
import sys
import argparse
import pandas as pd  # Miscellaneous operating system interfaces
from osgeo import gdal                          # Python bindings for GDAL
import numpy as np                              # Scientific computing with Python
from matplotlib import cm                       # Colormap handling utilities
from goes16_utils import download_PROD             # Our function for download
from goes16_utils import reproject                 # Our function for reproject
from goes16_utils import geo2grid
gdal.PushErrorHandler('CPLQuietErrorHandler')   # Ignore GDAL warnings


def fill_pixel_values(sat_array, x, y, resolution):
    list_pixel_values = []
    for i in range(x-resolution, x+resolution+1):
        for j in range(y-resolution, y+resolution+1):
            list_pixel_values = np.append(list_pixel_values, sat_array[j, i])
    print(list_pixel_values)
    return np.array(list_pixel_values)


def plot_data_for_this_timestamp(filename_reprojected, extent, index_name, yyyymmddhhmn, ws_id, dtime):
    #-----------------------------------------------------------------------------------------------------------
    # Open the reprojected GOES-R image
    file_reprojected = Dataset(filename_reprojected)

    #  Read file netcdf with estimated rain from GOES-16
    sat_data = gdal.Open(f'{filename_reprojected}')
    # Read number of cols and rows
    ncol = sat_data.RasterXSize
    nrow = sat_data.RasterYSize
    # Load the data
    sat_array = sat_data.ReadAsArray(0, 0, ncol, nrow).astype(float)
    # print(type(sat_array))
    # print(f'sat_array.shape = {sat_array.shape}')

    # Get geotransform
    transform = sat_data.GetGeoTransform()
    # print(f'ncol, nrow = {(ncol, nrow)}')
    # lat, lon = -22.98833333,-43.19055555
    # x_float = (lon - transform[0]) / transform[1]
    # y_float = (transform[3] - lat) / -transform[5]
    # x = int(x_float)
    # y = int(y_float)
    # print(f'(x_float,y_float) = {(x,y)}')
    # print(f'(x,y) = {(x,y)}')
    # sat = sat_array[y,x]
    # print(f'*VALUE = {sat}')

    # -----------------------------------------------------------------------------------------------------------
    # Choose the plot size (width x height, in inches)
    plt.figure(figsize=(10,10))

    # Use the Geostationary projection in cartopy
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Define the image extent
    img_extent = [extent[0], extent[2], extent[1], extent[3]]

    # Modify the colormap to zero values are white
    colormap = plt.get_cmap('rainbow').resampled(240)

    newcolormap = colormap(np.linspace(0, 1, 240))
    newcolormap[:1, :] = np.array([1, 1, 1, 1])
    cmap = cm.colors.ListedColormap(newcolormap)

    # #- Plot WSoI location ----------------------------------------------------------------------------------------
    # Define the coordinates to extract the data (lat,lon) <- pairs
    coordinates = [(ws_id, -22.98833333, -43.19055555)]

    for label, lat, lon in coordinates:
        # Reading coordinate of a WSoI
        lat_point = lat
        lon_point = lon

        x = int((lon - transform[0]) / transform[1])
        y = int((transform[3] - lat) / -transform[5])
        variable_list_values = fill_pixel_values(sat_array, x, y, 1)
        # print(f'**VALUE = {variable_point}')

        # Adding WSoI as an annotation
        ax.plot(lon_point, lat_point, 'x', color='black', markersize=2, transform = ccrs.Geodetic(), markeredgewidth=1.0, markeredgecolor=(0, 0, 0, 1), zorder=8)

        # Add a text
        txt_offset_x = 0.4
        txt_offset_y = 0.4

        plt.annotate(label + "\n" + "Lat: " + str(lat_point) + "\n" + "Lon: " + str(lon_point) + "\n" + index_name + ": " + str(index_name) + '', 
                    xy=(lon_point + txt_offset_x, lat_point + txt_offset_y), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), 
                    fontsize=10, fontweight='bold', color='gold', bbox=dict(boxstyle="round",fc=(0.0, 0.0, 0.0, 0.5), ec=(1., 1., 1.)), alpha = 1.0, zorder=9) 
        
    #-----------------------------------------------------------------------------------------------------------

    #- Plot product image ----------------------------------------------------------------------------------------
    img = ax.imshow(sat_array, vmin=0, vmax=150, cmap=cmap, origin='upper', extent=img_extent)

    # Add coastlines, borders and gridlines
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    plt.xlim(extent[0], extent[2])
    plt.ylim(extent[1], extent[3])
        
    # Add a colorbar
    plt.colorbar(img, label=index_name, extend='max', orientation='horizontal', pad=0.05, fraction=0.05)

    # Extract date
    date = (datetime.strptime(dtime, '%Y-%m-%dT%H:%M:%S.%fZ'))

    # Add a title
    plt.title(f'G-16-DSIF {index_name}', fontweight='bold', fontsize=10, loc='left')
    plt.title(f'Timestamp: {yyyymmddhhmn}', fontsize=10, loc='right')
    #-----------------------------------------------------------------------------------------------------------
    # Save the image
    plt.savefig(f'{output}/{index_name}_{yyyymmddhhmn}.png', bbox_inches='tight', pad_inches=0, dpi=300)

    plt.close()
    return variable_list_values


def main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], 
                                     description="""This script provides a simple interface for retrieving observations values 
                                       of DSIT indexes.""")
    parser.add_argument("-i", "--index", required=True, help="DSIT index", metavar='')
    parser.add_argument("-s", "--ws_id", required=True, help="Weather station ID", metavar='')
    parser.add_argument('-l', "--latlonlist", nargs='+', help='<Required> Set min and max latitude and longitude values', required=True)
    parser.add_argument('-d', "--day_of_retrieve_data", help='<Required> Set day to retrieve data', required=True )

    args = parser.parse_args(argv[1:])

    index_name = args.index
    station_id = args.ws_id
    date_to_retrieve_data = args.day_of_retrieve_data
    extent = [float(x) for x in args.latlonlist]

    # Desired extent
    # extent = [-74.0, -34.1, -34.8, 5.5] -74.0, -34.1 -34.8 5.5 # Min lon, Max lon, Min lat, Max lat
    # extent = [-64.0, -35.0, -35.0, -15.0] -64.0 -35.0 -35.0 -15.0 # Min lon, Min lat, Max lon, Max lat
    # extent = [-45, -25, -40, -20] -45 -25 -40 -20
    # extent = [-44, -24, -42, -20] 

    # Initial time and date
    yyyy = datetime.strptime(date_to_retrieve_data, '%Y%m%d').strftime('%Y')
    mm = datetime.strptime(date_to_retrieve_data, '%Y%m%d').strftime('%m')
    dd = datetime.strptime(date_to_retrieve_data, '%Y%m%d').strftime('%d')

    date_ini = str(datetime(int(yyyy), int(mm), int(dd), 12, 0) - timedelta(hours=12))
    date_end = str(datetime(int(yyyy), int(mm), int(dd), 12, 0) + timedelta(hours=12))
    
    acum = np.zeros((1086,1086))    

    columns = ['datetime', 'x-1,y-1', 'x-1,y', 'x-1,y+1', 'x,y-1', 'x,y', 'x,y+1', 'x+1,y-1', 'x+1,y', 'x+1,y+1']
    df = pd.DataFrame(columns=columns)
    #-----------------------------------------------------------------------------------------------------------
    # Loop to produce sequence of images (animation)
    while (date_ini <= date_end):
        
        # Date structure
        yyyymmddhhmn = datetime.strptime(date_ini, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')

        # Download file for current timestamp
        filename_downloaded = download_PROD(yyyymmddhhmn, product_name, temp_dir)
        #-----------------------------------------------------------------------------------------------------------

    
        # Open the file
        img = gdal.Open(f'NETCDF:{temp_dir}/{filename_downloaded}.nc:' + index_name)
        dqf = gdal.Open(f'NETCDF:{temp_dir}/{filename_downloaded}.nc:DQF_Overall')

        # Read the header metadata
        metadata = img.GetMetadata()
        scale = float(metadata.get(index_name + '#scale_factor'))
        offset = float(metadata.get(index_name + '#add_offset'))
        undef = float(metadata.get(index_name + '#_FillValue'))
        dtime = metadata.get('NC_GLOBAL#time_coverage_start')

        # Load the data
        ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)
        ds_dqf = dqf.ReadAsArray(0, 0, dqf.RasterXSize, dqf.RasterYSize).astype(float)

        # Remove undef
        ds[ds == undef] = np.nan

        # Apply the scale and offset
        ds = (ds * scale + offset)

        # Apply NaN's where the quality flag is greater than 0
        ds[ds_dqf > 0] = np.nan

        # Reproject the file
        filename_reprojected = f'{output}/{index_name}_{yyyymmddhhmn}.nc'
        reproject(filename_reprojected, img, ds, extent, undef)

        variable_values = plot_data_for_this_timestamp(filename_reprojected, extent, index_name, yyyymmddhhmn, station_id, dtime)
        variable_values = np.append(date_ini, variable_values)

        df.loc[len(df)] = variable_values

        date_ini = str(datetime.strptime(date_ini, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=10))

    df.to_csv('./data/goes16/data_' + index_name + '_' +
              date_to_retrieve_data + '.csv', index=False)


if __name__ == "__main__":
    temp_dir = "./data/goes16/temp"
    output = "./data/goes16/Animation"
    # yyyymmdd = '20201217'
    product_name = 'ABI-L2-DSIF'
    
    main(sys.argv)
