from netCDF4 import Dataset              # Read / Write NetCDF4 files
import matplotlib.pyplot as plt          # Plotting library
from datetime import datetime            # Basic Dates and time types
import cartopy, cartopy.crs as ccrs      # Plot maps
import os                                # Miscellaneous operating system interfaces
from goes16_utils import download_CMI       # Our own utilities
from goes16_utils import geo2grid, convertExtent2GOESProjection      # Our own utilities
from globals import GOES_DATA_DIR

def plot_instability_index(instability_index: str):
    #-----------------------------------------------------------------------------------------------------------
    # Input and output directories
    input = "Samples"; os.makedirs(input, exist_ok=True)
    output = "Output"; os.makedirs(output, exist_ok=True)

    # North -22°, West -44°, South -23°, East -42°
    # REGION_OF_INTEREST = [-22, -44, -23, -42]

    # Desired extent
    extent = [-44.0, -23.0, -43.0, -22.0]  # Min lon, Min lat, Max lon, Max lat

    #-----------------------------------------------------------------------------------------------------------
    # Open the GOES-R image
    file = Dataset(f'{GOES_DATA_DIR}/OR_ABI-L2-DSIF-M6_G16_s20220911700205_e20220911709513_c20220911711480.nc')
    # https://tempoagora.uol.com.br/noticia/2022/04/01/chuva-muito-volumosa-ainda-deixa-rj-em-alerta-4746

    # Convert lat/lon to grid-coordinates
    lly, llx = geo2grid(extent[1], extent[0], file)
    ury, urx = geo2grid(extent[3], extent[2], file)

    # Get the pixel values
    data = file.variables[instability_index][ury:lly, llx:urx]

    print(file.variables.keys())
    print('---')
    print(file.variables[instability_index])
    print('---')
    print(type(data))
    print(data.shape)
    import numpy.ma as ma
    min_indices = ma.where(data == data.min())
    print(min_indices)

    #-----------------------------------------------------------------------------------------------------------
    # Compute data-extent in GOES projection-coordinates
    img_extent = convertExtent2GOESProjection(extent)
    #-----------------------------------------------------------------------------------------------------------
    # Choose the plot size (width x height, in inches)
    plt.figure(figsize=(10,10))

    # Use the Geostationary projection in cartopy
    ax = plt.axes(projection=ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0))

    # Define the color scale based on the channel
    colormap = "jet" # White to black for IR channels

    # Plot the image
    img = ax.imshow(data, vmin=0, vmax=500, origin='upper', extent=img_extent, cmap=colormap)

    # Add coastlines, borders and gridlines
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
    ax.gridlines(color='black', alpha=0.5, linestyle='--', linewidth=0.5)

    # Add a colorbar
    plt.colorbar(img, label=instability_index, extend='both', orientation='horizontal', pad=0.05, fraction=0.05)

    # Extract the date
    date = (datetime.strptime(file.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ'))

    # Add a title
    plt.title('GOES-16 (' + instability_index + ') ' + date.strftime('%Y-%m-%d %H:%M') + ' UTC', fontweight='bold', fontsize=10, loc='left')
    plt.title('Reg.: ' + str(extent) , fontsize=10, loc='right')
    #-----------------------------------------------------------------------------------------------------------
    # Save the image
    plt.savefig(f'OR_ABI-L2-DSIF-M6_G16_s20210632010164_e20210632019472_c20210632020547-{instability_index}.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # Show the image
    plt.show()

instability_indices = ['LI', 'CAPE', 'TT', 'SI', 'KI']
for idx in instability_indices:
    plot_instability_index(idx)
