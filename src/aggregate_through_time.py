import pandas as pd
import os
from netCDF4 import Dataset
from collections import OrderedDict
import logging

# def generate_timestamps(initial_timestamp, final_timestamp, interval_in_minutes = 10):
#     # Parse the input timestamps to datetime objects
#     initial_dt = datetime.strptime(initial_timestamp, '%Y%m%d%H%M')
#     final_dt = datetime.strptime(final_timestamp, '%Y%m%d%H%M')

#     # Generate timestamps in 10-minute intervals
#     timestamps = []
#     current_dt = initial_dt

#     while current_dt <= final_dt:
#         timestamps.append(current_dt.strftime('%Y%m%d%H%M'))
#         current_dt += timedelta(minutes=interval_in_minutes)

#     return timestamps

def aggregate_to_hourly_resolution(df: pd.DataFrame, col_name: str):
    """
    Resamples a DataFrame containing time series data to an hourly frequency and computes the mean of 
    the 'col_name' column.
    
    Parameters:
    df (pd.DataFrame): A pandas DataFrame with a DateTime index and a column named 'col_name' containing 
                    the time series data.
    
    Returns:
    pd.DataFrame: A DataFrame with the hourly resampled mean values of the 'col_name' column. 
                    The index of the returned DataFrame is shifted by one hour.
    
    Notes:
    - The function resamples the input DataFrame to an hourly frequency and calculates the mean of the 'col_name' column for each hour.
    - NaN values are ignored in the mean calculation.
    - The index of the resulting DataFrame is shifted by one hour using `pd.DateOffset(hours=1)` to correctly represent the aggregated value in the previous hour.
    - If all the values in a hour are NaN, the the corresponding entry in the resulting dataframe will also be NaN.
    
    Example:
    df = pd.DataFrame({'value': [2, 2, 2, np.NaN, 2, 2, 3, 3, 3, 3, 3, 3, np.NaN, 4, 4, 4, np.NaN]}, 
                        index=pd.date_range('2023-01-01', periods=17, freq='10T'))
    print(df)
    print(30*'~')
    print(aggregate_to_hourly_resolution(df), 'value')

                                  value
        2023-01-01 00:00:00        2.0
        2023-01-01 00:10:00        2.0
        2023-01-01 00:20:00        2.0
        2023-01-01 00:30:00        NaN
        2023-01-01 00:40:00        2.0
        2023-01-01 00:50:00        2.0
        2023-01-01 01:00:00        3.0
        2023-01-01 01:10:00        3.0
        2023-01-01 01:20:00        3.0
        2023-01-01 01:30:00        3.0
        2023-01-01 01:40:00        3.0
        2023-01-01 01:50:00        3.0
        2023-01-01 02:00:00        NaN
        2023-01-01 02:10:00        4.0
        2023-01-01 02:20:00        4.0
        2023-01-01 02:30:00        4.0
        2023-01-01 02:40:00        NaN
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                  value
        2023-01-01 01:00:00        2.0
        2023-01-01 02:00:00        3.0
        2023-01-01 03:00:00        4.0
    """
    # Resample the DataFrame to hourly frequency and compute the mean
    hourly_df = df.resample('H').mean()

    hourly_df.set_index(hourly_df.index + pd.DateOffset(hours=1), inplace=True)

    return hourly_df

def main(argv):
    parser = argparse.ArgumentParser(description="Aggregate observations of a Pandas dataframe with a datatime as index to hourly temporal resolutio.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input Parquet file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output Parquet file')

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    df = pd.read_parquet(input_file)
    agg_df = aggregate_to_hourly_resolution(df)
    df.to_parquet(output_file)
    logging.info(f'A Pandas dataframe with shape {df.shape} was created and saved in the file {output_file}.')

# python aggregate_through_time.py --input_file ./data/goes16/DSI/CAPE.parquet --output_file ./data/goes16/DSI/CAPE_1H.parquet
if __name__ == "__main__":
    fmt = "[%(levelname)s] %(funcName)s():%(lineno)i: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format = fmt)

    start_time = time.time()  # Record the start time

    main(sys.argv)
    
    end_time = time.time()  # Record the end time
    duration = (end_time - start_time) / 60  # Calculate duration in minutes
    
    logging.info(f"Script duration: {duration:.2f} minutes")    
