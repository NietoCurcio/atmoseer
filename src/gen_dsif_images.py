import os
from netCDF4 import Dataset                     # Read / Write NetCDF4 files
import matplotlib.pyplot as plt                 # Plotting library
from datetime import datetime                   # Basic Dates and time types
import cartopy, cartopy.crs as ccrs             # Plot maps
import os                                       # Miscellaneous operating system interfaces
from osgeo import osr                           # Python bindings for GDAL
from osgeo import gdal                          # Python bindings for GDAL
import numpy as np                              # Scientific computing with Python

def plot_derived_instability_index(input_folder, output_folder, file_name, var, extent):

    # Open the file
    img = gdal.Open(f'NETCDF:{input_folder}{file_name}:' + var)

    # Read the header metadata
    metadata = img.GetMetadata()
    scale = float(metadata.get(var + '#scale_factor'))
    offset = float(metadata.get(var + '#add_offset'))
    undef = float(metadata.get(var + '#_FillValue'))
    dtime = metadata.get('NC_GLOBAL#time_coverage_start')

    # print(f'scale/offset/undef/dtime: {scale}/{offset}/{undef}/{dtime}')

    # Load the data
    ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)

    # Apply the scale, offset and convert to celsius
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
    temp = file_name[0:-3]
    temp = f'{output_folder}{temp}_reprojected.nc'
    print(temp)
    gdal.Warp(f'{temp}', raw, options=options)
    #-----------------------------------------------------------------------------------------------------------
    # Open the reprojected GOES-R image
    file = Dataset(f'{temp}')

    # print(f'file.variables = {file.variables}')

    # Get the pixel values
    data = file.variables['Band1'][:]

    #-----------------------------------------------------------------------------------------------------------
    # Choose the plot size (width x height, in inches)
    plt.figure(figsize=(10,10))

    # Use the Geostationary projection in cartopy
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Define the image extent
    img_extent = [extent[0], extent[2], extent[1], extent[3]]

    # Define the color scale based on the channel
    colormap = "jet" # White to black for IR channels
    # colormap = "gray_r" # White to black for IR channels

    # Plot the image
    img = ax.imshow(data, origin='upper', extent=img_extent, cmap=colormap)

    # Add coastlines, borders and gridlines
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                        color='gray', 
                        alpha=1.0, 
                        linestyle='--', 
                        linewidth=0.25, 
                        xlocs=np.arange(-180, 180, 5), 
                        ylocs=np.arange(-90, 90, 5), 
                        draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # Add a colorbar
    plt.colorbar(img, label=var, extend='both', orientation='horizontal', pad=0.05, fraction=0.05)

    # Extract date
    date = (datetime.strptime(dtime, '%Y-%m-%dT%H:%M:%S.%fZ'))

    # Add a title
    plt.title('GOES-16 ' + var + ' ' + date.strftime('%Y-%m-%d %H:%M') + ' UTC', fontweight='bold', fontsize=10, loc='left')
    plt.title('Reg.: ' + str(extent) , fontsize=10, loc='right')


    #-----------------------------------------------------------------------------------------------------------
    # Save the image
    temp = file_name[0:-3]
    plt.savefig(f'{output_folder}{temp}_{var}.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # Show the image

input_folder = './data/goes16/goes16_dsif/'

# Get the list of file names in the folder
file_names = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

extent = [-64.0, -35.0, -35.0, -15.0] # Min lon, Min lat, Max lon, Max lat

# Options:
# - Convective Available Potential Energy (CAPE)
# - Lifted Index (LI)
# - Total Totals (TT)
# - Showalter Index (SI)
# - K-index (KI)

# vars = ['CAPE', 'LI', 'TT', 'SI', 'KI']
vars = ['TT', 'SI', 'KI']
# vars = ['LI']

file_names.sort()

# Print the list of file names
print(file_names)

output_folder = './data/goes16/goes16_dsif_imgs/'

for file_name in file_names:
    print(f'Generating images for {file_name}')
    for var in vars:
        plot_derived_instability_index(input_folder, output_folder, file_name, var, extent)