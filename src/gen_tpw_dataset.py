import pandas as pd
from globals import INMET_WEATHER_STATION_IDS

import os
import pandas as pd

def hourly_average_with_nan_handling(df):
    # Resample the DataFrame to hourly frequency and compute the mean
    df_resampled = df['tpw_value'].resample('H').mean()
    
    # Drop rows with NaN values in 'tpw'
    # df_resampled = df_resampled.dropna()
    
    return pd.DataFrame({'tpw_value': df_resampled.values},
                        index=df_resampled.index + pd.DateOffset(hours=1))

# def hourly_average_with_nan_handling(df):
#     # First, make sure the DataFrame is sorted by the datetime index
#     df = df.sort_index()

#     # Use resample to group the data by hour and calculate the mean
#     hourly_avg_df = df.resample('H').mean()

#     return hourly_avg_df

# hourly_avg_df = hourly_average_with_nan_handling(df)
# print(hourly_avg_df)

# Set the folder containing the Parquet files
folder_path = './data/goes16/tpw'  # Replace with the path to your folder

# Initialize an empty list to store DataFrames
dataframes = []

# List all Parquet files in the folder
parquet_files = [file for file in os.listdir(folder_path) if file.endswith('.parquet')]

# Loop through the Parquet files, read them, and append to the list
for file in parquet_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_parquet(file_path)
    print(f'Merging dataframe {file_path} of shape {df.shape}')
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame, so that
# merged_df contains the combined data from all Parquet files
merged_df = pd.concat(dataframes)

# print(f'merged_df.columns (1): {merged_df.columns}')

# # Reset the index to 'timestamp' and drop the old index
# merged_df.set_index('timestamp', inplace=True)
# merged_df.index = pd.to_datetime(merged_df.index[0])

# print(f'merged_df.columns (2): {merged_df.columns}')

# If you want to sort the DataFrame by the 'timestamp' column
# merged_df.sort_index(inplace=True)


# Print the first few rows to verify
print('MERGED DATAFRAME:')
print(merged_df.head())
print(merged_df.shape)

for wsoi_id in INMET_WEATHER_STATION_IDS:
    print(f'Processing data for station {wsoi_id}')

    # Create a boolean mask to filter rows with 'station_id' equal to 'wsoi_id'
    mask = merged_df['station_id'] == wsoi_id

    # Use the mask to select rows from the DataFrame
    df_wsoi = merged_df[mask]
    print(df_wsoi)

    print('columns (1):', df_wsoi.columns)
    df_wsoi.reset_index(inplace=True)
    print('columns (2):', df_wsoi.columns)

    # Now, the `df_wsoi` DataFrame contains rows where 'station_id' matches 'wsoi_id'
    print(f'df_wsoi.shape: {df_wsoi.shape}')
    print('Number of NaN values:', df_wsoi['tpw_value'].isna().sum())

    print(f'df_wsoi.index (1): {df_wsoi.index}')

    print(f'---(1):')
    print(df_wsoi.head())

    df_wsoi = df_wsoi[['timestamp', 'tpw_value']]

    df_wsoi.index = pd.to_datetime(df_wsoi.timestamp)
    df_wsoi = df_wsoi.drop(columns=['timestamp'])

    df_wsoi = df_wsoi.sort_index()

    print(f'---(2):')
    print(df_wsoi.head(80))

    df_wsoi = hourly_average_with_nan_handling(df_wsoi)

    print(f'df_wsoi.index (2): {df_wsoi.index}')

    print(f'---(3):')
    print(df_wsoi.head(30))

    print(df_wsoi['tpw_value'].isna().sum())
    df_wsoi.to_parquet(f'./data/goes16/wsoi/{wsoi_id}.parquet')
    print('~~~')
    # break

# print('A636')
# df = pd.read_parquet('./data/goes16/wsoi/A636.parquet')
# print(df.head(30))