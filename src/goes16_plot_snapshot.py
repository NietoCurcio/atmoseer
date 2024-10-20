import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

def plot_netcdf_array(netcdf_file, timestamp, output_image_file):
    """
    Plots the numpy array corresponding to the user-provided timestamp from a netCDF file
    and saves the result as an image.
    
    Args:
    netcdf_file (str): Path to the netCDF file.
    timestamp (str): Timestamp in the format '%Y_%m_%d_%H_%M' to find the corresponding data.
    output_image_file (str): Path and name of the output image file (e.g., 'output.png').
    """
    # Sanitize the timestamp to match the variable name in the netCDF file
    sanitized_name = timestamp.replace(":", "_").replace(" ", "_")
    
    try:
        # Open the netCDF file
        with nc.Dataset(netcdf_file, 'r') as dataset:
            # Check if the variable exists in the file
            if sanitized_name in dataset.variables:
                data_array = dataset.variables[sanitized_name][:]
            else:
                print(f"No data found for timestamp: {timestamp}")
                return
        
        # Plot the array using matplotlib
        plt.imshow(data_array, cmap='viridis', origin='lower')
        plt.colorbar(label='Value')
        plt.title(f"Array Data for Timestamp: {timestamp}")
        
        # Save the plot as an image
        plt.savefig(output_image_file)
        plt.close()
        
        print(f"Plot saved as '{output_image_file}'")
    
    except Exception as e:
        print(f"Error while processing netCDF file: {e}")

# Example usage:
# Provide the path to your netCDF file, timestamp, and desired output image file
netcdf_file = './cropped/2024_02_08.nc'
timestamp = 'CMI_2024_02_08_08_50'  # Example timestamp
output_image_file = 'CMI_2024_02_08_08_50.png'

plot_netcdf_array(netcdf_file, timestamp, output_image_file)
