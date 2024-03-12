#-----------------------------------------------------------------------------------------------------------
# Agrometeorology Course - Demonstration: Normalized Difference Vegetation Index (NDVI) - Extract Values
# Author: Diego Souza
#-----------------------------------------------------------------------------------------------------------
# Required modules
from netCDF4 import Dataset                    # Read / Write NetCDF4 files
import matplotlib.pyplot as plt                # Plotting library
from datetime import datetime                  # Basic Dates and time types
from datetime import timedelta, date, datetime # Basic Dates and time types
import cartopy, cartopy.crs as ccrs            # Plot maps
import cartopy.io.shapereader as shpreader     # Import shapefiles
import cartopy.feature as cfeature             # Cartopy features
import numpy as np                             # Scientific computing with Python
import os                                      # Miscellaneous operating system interfaces
from goes16_utils import download_PROD             # Our own utilities
from goes16_utils import geo2grid, latlon2xy, convertExtent2GOESProjection      # Our own utilities
#-----------------------------------------------------------------------------------------------------------
# Input and output directories
temp_dir = "./data/goes16/temp"
output = "./data/goes16/Animation"

# # Desired extent
extent = [-93.0, -60.00, -25.00, 15.00] # Min lon, Min lat, Max lon, Max lat
# Coordinates of rectangle for region of interest (Rio de Janeiro municipality)
# extent = [-64.0, -35.0, -35.0, -15.0] # Min lon, Min lat, Max lon, Max lat

# Initial date and time to process
date_ini = '202201011800'

# Number of days to accumulate
ndays = 1

prod_name = 'ABI-L2-DSIF'
variable_name = 'CAPE'

# prod_name = 'ABI-L2-SSTF'
# variable_name = 'SST'

# product_name = 'ABI-L2-RRQPEF'
# variable_name = 'RRQPE'


# Convert to datetime
date_ini = datetime(int(date_ini[0:4]), int(date_ini[4:6]), int(date_ini[6:8]), int(date_ini[8:10]), int(date_ini[10:12]))
date_end = date_ini + timedelta(days=ndays)

# Create our references for the loop
date_loop = date_ini

#-----------------------------------------------------------------------------------------------------------
# NDVI colormap creation 
import matplotlib
colors = ["#8f2723", "#8f2723", "#8f2723", "#8f2723", "#af201b", "#af201b", "#af201b", "#af201b", "#ce4a2e", "#ce4a2e", "#ce4a2e", "#ce4a2e", 
          "#df744a", "#df744a", "#df744a", "#df744a", "#f0a875", "#f0a875", "#f0a875", "#f0a875", "#fad398", "#fad398", "#fad398", "#fad398",
          "#fff8ba",
          "#d8eda0", "#d8eda0", "#d8eda0", "#d8eda0", "#bddd8a", "#bddd8a", "#bddd8a", "#bddd8a", "#93c669", "#93c669", "#93c669", "#93c669", 
          "#5da73e", "#5da73e", "#5da73e", "#5da73e", "#3c9427", "#3c9427", "#3c9427", "#3c9427", "#235117", "#235117", "#235117", "#235117"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
cmap.set_over('#235117')
cmap.set_under('#8f2723')
vmin = 0.1
vmax = 0.8
#-----------------------------------------------------------------------------------------------------------

# Variable to check if is the first iteration
first = True

# Create our references for the loop
date_loop = date_ini

while (date_loop <= date_end):

    # Date structure
    yyyymmddhhmn = date_loop.strftime('%Y%m%d%H%M') 
    print(yyyymmddhhmn)

    # Download the Full Disk file
    file_name = download_PROD(yyyymmddhhmn, prod_name, temp_dir)

    #-----------------------------------------------------------------------------------------------------------
    
    # If file hasn't been downloaded for some reason, skip the current iteration
    if not (os.path.exists(f'{temp_dir}/{file_name}.nc')):
      # Increment the date_loop
      date_loop = date_loop + timedelta(days=1)
      print("The file is not available, skipping this iteration.")
      continue

    # Open the Full Disk file
    file = Dataset(f'{temp_dir}/{file_name}.nc')
    print(f'file.dimensions from Dataset instantiation: {file.dimensions}')

    ##############################
    lat_point, lon_point = -22.98833333,-43.19055555
    lat_ind, lon_ind = geo2grid(lat_point, lon_point, file)    
    print(f'lat_ind, lon_ind: {(lat_ind, lon_ind)}\n')
    ##############################


    # Convert lat/lon to grid-coordinates
    lly, llx = geo2grid(extent[1], extent[0], file)
    ury, urx = geo2grid(extent[3], extent[2], file)
    print(f'lly, llx, ury, urx: {(lly, llx, ury, urx)}\n')
    
    # print(f'file.variables: {file.variables}\n')

    # Get the pixel values
    data = file.variables[variable_name][ury:lly, llx:urx]
    # data = file.variables[variable_name][ury:lly, llx:urx][::2 ,::2]
    print(type(data))
    print(f'data.shape: {data.shape}')
    
    #-----------------------------------------------------------------------------------------------------------
    
    # Calculate the NDVI
    # TODO, TODO
    # data = (data_ch03 - data_ch02) / (data_ch03 + data_ch02)
    
    #-----------------------------------------------------------------------------------------------------------
    
    # Compute data-extent in GOES projection-coordinates
    img_extent = convertExtent2GOESProjection(extent)
    
    #-----------------------------------------------------------------------------------------------------------

    # Choose the plot size (width x height, in inches)
    plt.figure(figsize=(15,15))

    # Use the Geostationary projection in cartopy
    ax = plt.axes(projection=ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0))
   
    # Add a land mask
    ax.stock_img()       
    land = ax.add_feature(cfeature.LAND, facecolor='gray', zorder=1)

    #-----------------------------------------------------------------------------------------------------------
    # Define the coordinates to extract the data (lat,lon) <- pairs
    coordinates = [('A652', -22.98833333,-43.19055555), ('A636', -22.93999999,-43.40277777)]

    for label, lat, lon in coordinates:
      # Reading the data from a coordinate
      lat_point = lat
      lon_point = lon
      
      # Convert lat/lon to grid-coordinates
      lat_ind, lon_ind = geo2grid(lat_point, lon_point, file)
      print(f'Current (x,y)={(lat_ind, lon_ind)}') 
      print(f'lat_ind - ury, lon_ind - llx = {(lat_ind - ury, lon_ind - llx)}')
      variable_point = (data[lat_ind - ury, lon_ind - llx]).round(2)  

      # Adding the data as an annotation
      # Add a circle
      ax.plot(lon_point, lat_point, 'o', color='yellow', markersize=3, transform=ccrs.Geodetic(), markeredgewidth=1.0, markeredgecolor=(0, 0, 0, 1), zorder=8)
      
      # Add a text
      txt_offset_x = 0.8
      txt_offset_y = 0.8
      
      plt.annotate(label + "\n" + "Lat: " + str(lat_point) + "\n" + "Lon: " + str(lon_point) + "\n" + variable_name + ": " + str(variable_point) + '', 
                  xy=(lon_point + txt_offset_x, lat_point + txt_offset_y), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), 
                  fontsize=10, fontweight='bold', color='gold', bbox=dict(boxstyle="round",fc=(0.0, 0.0, 0.0, 0.5), ec=(1., 1., 1.)), alpha = 1.0, zorder=9) 
      
    #-----------------------------------------------------------------------------------------------------------

    # Plot the image
    img = ax.imshow(data, origin='upper', vmin=vmin, vmax=vmax, extent=img_extent, cmap=cmap, zorder=2)
   
    # Add a shapefile
    shapefile = list(shpreader.Reader('./data/natural_earth/ne_10m_admin_1_states_provinces.shp').geometries())
    ax.add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='dimgray',facecolor='none', linewidth=0.3, zorder=4)

    # Add coastlines, borders and gridlines
    ax.coastlines(resolution='10m', color='black', linewidth=0.8, zorder=5)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5, zorder=6)
    ax.gridlines(color='gray', alpha=0.5, linestyle='--', linewidth=0.5, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), zorder=7)

    # Add a colorbar
    plt.colorbar(img, label=variable_name, extend='both', orientation='vertical', pad=0.05, fraction=0.05)

    # Extract date
    date = (datetime.strptime(file.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ'))

    # Add a title
    plt.title('GOES-16 CAPE ' + date.strftime('%Y-%m-%d %H:%M') + ' UTC', fontweight='bold', fontsize=10, loc='left')
    plt.title('ROI: ' + str(extent) , fontsize=10, loc='right')
        
    # Save the image
    plt.savefig(f'{output}/G16_{variable_name}_{yyyymmddhhmn}.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # Show the image
    plt.show()

    # Increment the date_loop
    date_loop = date_loop + timedelta(days=1)