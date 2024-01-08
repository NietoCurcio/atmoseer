import pandas as pd
import numpy as np
import sys
import pickle
from utils.near_stations import prox

from era5_data_source import Era5ReanalisysDataSource

import globals

from util import find_contiguous_observation_blocks, add_missing_indicator_column, split_dataframe_by_date
from utils.windowing import apply_windowing
import util as util
import argparse
from subsampling import apply_subsampling
import logging
import yaml
import datetime

from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

# def format_for_binary_classification(y_train, y_val, y_test):
#     y_train_oc = map_to_binary_precipitation_levels(y_train)
#     y_val_oc = map_to_binary_precipitation_levels(y_val)
#     y_test_oc = map_to_binary_precipitation_levels(y_test)
#     return y_train_oc, y_val_oc, y_test_oc

# def format_for_ordinal_classification(y_train, y_val, y_test):
#     y_train_oc = map_to_precipitation_levels(y_train)
#     y_val_oc = map_to_precipitation_levels(y_val)
#     y_test_oc = map_to_precipitation_levels(y_test)
#     return y_train_oc, y_val_oc, y_test_oc

def apply_sliding_window(df: pd.DataFrame, target_idx: int, window_size: int):
    """
    This function applies the sliding window preprocessing technique to generate data and response 
    matrices (that is, X and y) from an input time series represented as a pandas DataFrame. This 
    DataFrame is supposed to have a datetime index that corresponds to the timestamps in the time series.

    @see: https://stackoverflow.com/questions/8269916/what-is-sliding-window-algorithm-examples

    Note that this function takes the eventual existence of gaps in the input time series 
    into account. In particular, the windowing operation is performed in each separate 
    contiguous block of observations.
    """
    contiguous_observation_blocks = list(
        find_contiguous_observation_blocks(df))

    is_first_block = True

    for block in contiguous_observation_blocks:
        start = block[0]
        end = block[1]

        # logging.info(df[start:end].shape)
        if df[start:end].shape[0] < window_size + 1:
            continue

        arr = np.array(df[start:end])
        X_block, y_block = apply_windowing(arr,
                                           initial_time_step=0,
                                           max_time_step=len(arr)-window_size-1,
                                           window_size=window_size,
                                           target_idx=target_idx)
        y_block = y_block.reshape(-1, 1)
        if is_first_block:
            X = X_block
            y = y_block
            is_first_block = False
        else:
            X = np.concatenate((X, X_block), axis=0)
            y = np.concatenate((y, y_block), axis=0)

    return X, y
    
def generate_windowed_split(train_df, val_df, test_df, target_name, window_size):
    target_idx = train_df.columns.get_loc(target_name)
    logging.info(f"Position (index) of target variable {target_name}: {target_idx}")
    X_train, y_train = apply_sliding_window(train_df, target_idx, window_size)
    X_val, y_val = apply_sliding_window(val_df, target_idx, window_size)
    X_test, y_test = apply_sliding_window(test_df, target_idx, window_size)
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_goes16_data_for_weather_station(df: pd.DataFrame, max_event: bool = False) -> pd.DataFrame:
    """
    Filters lightning event data in a DataFrame based on latitude and longitude boundaries for a specific weather station
    and calculates the maximum or median value of the 'event_energy' column on an hourly basis.

    Args:
        df (pd.DataFrame): DataFrame containing lightning event data with columns 'event_energy', 'event_lat', and 'event_lon'.
        station_id (str): Identifier for the weather station to filter coordinates.
        max_event (bool, optional): Flag to determine whether to calculate the maximum event energy or the median event energy.
            Defaults to True, calculating the maximum event energy.

    Returns:
        pd.DataFrame: A new DataFrame with the same columns as the input DataFrame, but with lightning events outside of the
            specified latitude and longitude boundaries removed, and the maximum or median value of 'event_energy' for each hour.

    """

    filtered_df = util.min_max_normalize(df)

    result_df = filtered_df[["event_energy"]]

    if max_event:
        result_df = result_df.resample('H').max()
    else:
        result_df = result_df.resample('H').mean()

    return result_df

def gaussian_noise(df, column_name, mu=0, sigma=1):

    # Generate Gaussian noise
    noise = np.random.normal(mu, sigma, size=df.shape[0])

    # Add noise to the column
    df[column_name] = df[column_name] + noise

    return df


def add_user_specified_data_sources(
        station_id, 
        join_AS_datasource, 
        join_reanalisys_datasource, 
        join_goes16_glm_datasource, 
        join_goes16_tpw_datasource,
        join_colorcord_datasource,
        join_conv2d_datasource,
        join_autoencoder_datasource,   
        df_ws, 
        min_datetime, 
        max_datetime):
    joined_df = df_ws

    if join_reanalisys_datasource:
        logging.info(f"Loading reanalisys (ERA5) data near the weather station {station_id}...")
        data_source = Era5ReanalisysDataSource()
        df_era5_reanalisys = data_source.get_data(station_id, min_datetime, max_datetime)
        logging.info(f"Done! Shape = {df_era5_reanalisys.shape}.")
        assert (not df_era5_reanalisys.isnull().values.any().any())

        joined_df = pd.merge(joined_df, df_era5_reanalisys, how='left', left_index=True, right_index=True)

        logging.info(f"Reanalisys data successfully joined; resulting shape = {joined_df.shape}.")
        logging.info(df_ws.index.difference(joined_df.index).shape)
        logging.info(joined_df.index.difference(df_ws.index).shape)

        logging.info(df_era5_reanalisys.index.intersection(df_ws.index).shape)
        logging.info(df_era5_reanalisys.index.difference(df_ws.index).shape)
        logging.info(df_ws.index.difference(df_era5_reanalisys.index).shape)
        logging.info(df_ws.index.difference(df_era5_reanalisys.index))

        shape_before_dropna = joined_df.shape
        joined_df = joined_df.dropna()
        shape_after_dropna = joined_df.shape
        logging.info(f"Removed NaN rows in merged dataset; Shapes before/after dropna: {shape_before_dropna}/{shape_after_dropna}.")

    assert (not joined_df.isnull().values.any().any())

    if join_AS_datasource:
        filename = globals.AS_DATA_DIR + 'SBGL_indices_1997_2023.parquet.gzip'
        logging.info(f"Loading atmospheric sounding indices from {filename}...")
        df_as = pd.read_parquet(filename)
        logging.info(f"Done! Shape = {df_as.shape}.")

        format_string = '%Y-%m-%d %H:%M:%S'

        #
        # Add index to dataframe using the observation's timestamps.
        df_as['Datetime'] = pd.to_datetime(df_as['time'], format=format_string)
        df_as = df_as.set_index(pd.DatetimeIndex(df_as['Datetime']))
        logging.info(f"Range of timestamps in the atmospheric sounding data source: [{min(df_as.index)}, {max(df_as.index)}]")

        #
        # Remove time-related columns since now this information is in the index.
        df_as = df_as.drop(['time', 'Datetime'], axis = 1)

        joined_df = pd.merge(joined_df, df_as, how='left', left_index=True, right_index=True)

        logging.info(f"Atmospheric sounding data successfully joined; resulting shape: {joined_df.shape}.")

        joined_df = add_missing_indicator_column(joined_df, "asi_idx_missing")
        logging.info(f"Shape after adding missing indicator column: {joined_df.shape}.")

        logging.info(f"Doing interpolation to imput missing values on the sounding indices...")
        joined_df['cape'] = joined_df['cape'].interpolate(method='linear')
        joined_df['cin'] = joined_df['cin'].interpolate(method='linear')
        joined_df['lift'] = joined_df['lift'].interpolate(method='linear')
        joined_df['k'] = joined_df['k'].interpolate(method='linear')
        joined_df['total_totals'] = joined_df['total_totals'].interpolate(method='linear')
        joined_df['showalter'] = joined_df['showalter'].interpolate(method='linear')
        logging.info("Done!")

        # At the beggining of the joined dataframe, a few entries may remain with NaN values. 
        # The code below gets rid of these entries.
        # see https://stackoverflow.com/questions/27905295/how-to-replace-nans-by-preceding-or-next-values-in-pandas-dataframe
        joined_df.fillna(method='bfill', inplace=True)

        # TODO: data normalization 
        # TODO: implement interpolation
        # TODO: deal with missing values (see https://youtu.be/DKmDJJzayZw)
        # TODO: Imputing with MICE (see https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87)
        # TODO: use other sounding stations (?) (see tempo.inmet.gov.br/Sondagem/)

    if join_goes16_tpw_datasource:
        logging.info(f"Loading GOES16 TPW product for weather station {station_id}...")
        df_tpw = pd.read_parquet(f'{globals.TPW_DATA_DIR}/{station_id}.parquet')
        logging.info(f"Done! Shape = {df_tpw.shape}.")

        logging.info(f"Range of timestamps in the TPW data: [{min(df_tpw.index)}, {max(df_tpw.index)}]")

        joined_df = joined_df.join(df_tpw, how='inner')

        logging.info(f"TPW data successfully joined; resulting shape: {joined_df.shape}.")

        # print(joined_df.columns)

        logging.info(f"Adding missing indicator column...")
        joined_df = add_missing_indicator_column(joined_df, "tpw_idx_missing")
        logging.info(f"Done! New shape: {joined_df.shape}.")

        logging.info(f"Doing interpolation to imput missing values on the TPW values...")
        joined_df['tpw_value'] = joined_df['tpw_value'].interpolate(method='linear')
        logging.info(f"Done!")

        # At the beggining of the joined dataframe, a few entries may remain with NaN values. 
        # The code below gets rid of these entries.
        # see https://stackoverflow.com/questions/27905295/how-to-replace-nans-by-preceding-or-next-values-in-pandas-dataframe
        joined_df.fillna(method='bfill', inplace=True)

        shape_before_dropna = joined_df.shape
        joined_df = joined_df.dropna()
        shape_after_dropna = joined_df.shape
        assert shape_before_dropna == shape_after_dropna

    assert (not joined_df.isnull().values.any().any())

    if join_goes16_glm_datasource:
        print(f"Loading GLM (Goes 16) data near the weather station {station_id}...", end= "")
        df_lightning = pd.read_parquet(f'data/parquet_files/glm_{station_id}_preprocessed_file.parquet')
        df_lightning_filtered = get_goes16_data_for_weather_station(df_lightning)
        print(df_lightning_filtered.isnull().sum())
        df_lightning_filtered.fillna(method="bfill", inplace=True)
        assert (not df_lightning_filtered.isnull().values.any().any())
        joined_df = pd.merge(df_ws, df_lightning_filtered, how='left', left_index=True, right_index=True)

        joined_df['event_energy'].fillna(method="bfill", inplace=True)

        # Ruido branco
        lb_test_stat = acorr_ljungbox(joined_df['event_energy'], lags=10)
        print("Estatísticas do Teste:", lb_test_stat)
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.title('Valores p do Teste de Ljung-Box')
        plt.xlabel('Lag')
        plt.ylabel('Valor p')
        plt.show()

        print(f"GLM data successfully joined; resulting shape = {joined_df.shape}.")
        print(df_ws.index.difference(joined_df.index).shape)
        print(joined_df.index.difference(df_ws.index).shape)

        print(df_lightning_filtered.index.intersection(df_ws.index).shape)
        print(df_lightning_filtered.index.difference(df_ws.index).shape)
        print(df_ws.index.difference(df_lightning_filtered.index).shape)
        print(df_ws.index.difference(df_lightning_filtered.index))

        shape_before_dropna = joined_df.shape
        joined_df = joined_df.dropna()
        shape_after_dropna = joined_df.shape
        print(f"Removed NaN rows in merge data; Shapes before/after dropna: {shape_before_dropna}/{shape_after_dropna}.")

    assert (not joined_df.isnull().values.any().any())

    if join_colorcord_datasource:
        filename = f'FEATURE_{station_id}_COLORCORD.csv'
        logging.info(f"Loading image features {filename}...")
        df_new = pd.read_csv(filename)
        logging.info(f"Done! Shape = {df_new.shape}.")
        df_new['date'] = pd.to_datetime(df_new['date'], format="%Y-%m-%d--%H%M%S")
        df_new['date'] = df_new['date'].dt.floor('H')
        del df_new["Estação"]
        df_new = df_new.groupby('date', as_index=False).mean()

        format_string = '%Y-%m-%d %H:%M:%S'

        df_new['Datetime'] = pd.to_datetime(df_new['date'], format=format_string)
        df_new = df_new.set_index(pd.DatetimeIndex(df_new['Datetime']))
        logging.info(f"Range of timestamps in the image feature data source: [{min(df_new.index)}, {max(df_new.index)}]")

        df_new = df_new.drop(['date', 'Datetime', "Estação"], axis = 1)
        joined_df = pd.merge(joined_df, df_new, how='left', left_index=True, right_index=True)
        joined_df = joined_df.fillna(0)

        logging.info(f"Image features data successfully joined; resulting shape: {joined_df.shape}.")
    
    elif join_conv2d_datasource:
        filename = f'FEATURE_{station_id}_CONV2D.csv'
        logging.info(f"Loading image features {filename}...")
        df_new = pd.read_csv(filename)
        logging.info(f"Done! Shape = {df_new.shape}.")
        df_new['date'] = pd.to_datetime(df_new['date'], format="%Y-%m-%d--%H%M%S")
        df_new['date'] = df_new['date'].dt.floor('H')
        del df_new["Estação"]
        df_new = df_new.groupby('date', as_index=False).mean()

        format_string = '%Y-%m-%d %H:%M:%S'

        df_new['Datetime'] = pd.to_datetime(df_new['date'], format=format_string)
        df_new = df_new.set_index(pd.DatetimeIndex(df_new['Datetime']))
        logging.info(f"Range of timestamps in the image feature data source: [{min(df_new.index)}, {max(df_new.index)}]")

        df_new = df_new.drop(['date', 'Datetime'], axis = 1)
        joined_df = pd.merge(joined_df, df_new, how='left', left_index=True, right_index=True)
        joined_df = joined_df.fillna(0)

        logging.info(f"Image features data successfully joined; resulting shape: {joined_df.shape}.")

    elif join_autoencoder_datasource:
        filename = f'FEATURE_{station_id}_AUTOENCODER.csv'
        logging.info(f"Loading image features {filename}...")
        df_new = pd.read_csv(filename)
        logging.info(f"Done! Shape = {df_new.shape}.")
        df_new['date'] = pd.to_datetime(df_new['date'], format="%Y-%m-%d--%H%M%S")
        df_new['date'] = df_new['date'].dt.ceil('H')
        del df_new["Estação"]
        df_new = df_new.groupby('date', as_index=False).mean()

        format_string = '%Y-%m-%d %H:%M:%S'

        df_new['Datetime'] = pd.to_datetime(df_new['date'], format=format_string)
        df_new = df_new.set_index(pd.DatetimeIndex(df_new['Datetime']))
        logging.info(f"Range of timestamps in the image feature data source: [{min(df_new.index)}, {max(df_new.index)}]")

        df_new = df_new.drop(['date', 'Datetime'], axis = 1)
        joined_df = pd.merge(joined_df, df_new, how='left', left_index=True, right_index=True)
        joined_df = joined_df.fillna(0)

        logging.info(f"Image features data successfully joined; resulting shape: {joined_df.shape}.")

    assert (not joined_df.isnull().values.any().any())

    return joined_df

def build_datasets(station_id: str, 
                   input_folder: str,
                   train_start_threshold: datetime.datetime,
                   train_test_threshold: datetime.datetime,
                   join_AS_data_source: bool, 
                   join_reanalisys_datasource: bool, 
                   join_goes16_glm_datasource: bool, 
                   join_goes16_tpw_datasource: bool,
                   join_colorcord_datasource: bool,
                   join_conv2d_datasource: bool,
                   join_autoencoder_datasource: bool,
                   subsampling_procedure: str):
    '''
    This function joins a set of datasources to build datasets. These resulting datasets are used to fit the 
    parameters of precipitation models down the AtmoSeer pipeline. Each datasource contributes with a group 
    of features to build the datasets that are going to be used for training and validating the prediction models.
    
    Notice that, when joining the user-specified data sources, there is always a station of interest, that is, 
    a weather station that will provide the values of the target variable (in our case, precipitation). 
    It can even be the case that the only user-specified data source is this weather station. 
    '''

    pipeline_id = station_id
    if join_reanalisys_datasource:
        pipeline_id = pipeline_id + '_N'
    if join_AS_data_source:
        pipeline_id = pipeline_id + '_R'
    if join_goes16_glm_datasource:
        pipeline_id = pipeline_id + '_L'
    if join_goes16_tpw_datasource:
        pipeline_id = pipeline_id + '_T'
    if join_colorcord_datasource:
        pipeline_id = pipeline_id + '_I'
    if join_conv2d_datasource:
        pipeline_id = pipeline_id + '_C'
    if join_autoencoder_datasource:
        pipeline_id = pipeline_id + '_A'

    logging.info(f"Loading observations for weather station {station_id}...")
    df_ws = pd.read_parquet(input_folder + station_id + "_preprocessed.parquet.gzip")
    logging.info(f"Done! Shape = {df_ws.shape}.\n")

    ####
    # Apply a filtering step, with the purpose of disregarding all observations 
    # before a user-specified timestamp. This is useful when doing training 
    # experiments with datasources whose start of operation was AFTER the one 
    # corresponding to the WSoI.
    ####
    if train_start_threshold is not None:
        logging.info(f"Applying start threshold filtering...")
        logging.info(f'Timestamp from which examples will be considered: {train_start_threshold}')
        shape_before_month_filtering = df_ws.shape
        df_ws = df_ws[df_ws.index >= train_start_threshold]
        shape_after_month_filtering = df_ws.shape
        logging.info(f"Done! Shapes before/after: {shape_before_month_filtering}/{shape_after_month_filtering}\n")

    ####
    # Apply a filtering step, with the purpose of disregarding all observations made between  
    # what we consider to be the drought period of a year (months of June, July, and August).
    ####
    logging.info(f"Applying month filtering...")
    shape_before_month_filtering = df_ws.shape
    df_ws = df_ws[df_ws.index.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5])].sort_index(ascending=True)
    shape_after_month_filtering = df_ws.shape
    logging.info(f"Done! Shapes before/after: {shape_before_month_filtering}/{shape_after_month_filtering}\n")

    assert (not df_ws.isnull().values.any().any())

    #####
    # Start with the mandatory data source (i.e., the one represented by the weather 
    # station of interest) and sequentially join the other user-specified datasources.
    #####
    joined_df = df_ws
    min_datetime = min(joined_df.index)
    max_datetime = max(joined_df.index)

    #####
    # Now add user-specified data sources.
    #####
    logging.info(f'Going to add features from the user-specified datasources...')
    joined_df = add_user_specified_data_sources(
        station_id, 
        join_AS_data_source, 
        join_reanalisys_datasource, 
        join_goes16_glm_datasource, 
        join_goes16_tpw_datasource,
        join_colorcord_datasource,
        join_conv2d_datasource,
        join_autoencoder_datasource,     
        df_ws, 
        min_datetime, 
        max_datetime)
    logging.info(f'Done!\n')

    #
    # Save train/val/test DataFrames for future error analisys.
    filename = globals.DATASETS_DIR + pipeline_id + '.parquet.gzip'
    logging.info(f'Saving joined dataset for pipeline {pipeline_id} to file {filename}.')
    joined_df.to_parquet(filename, compression='gzip')
    logging.info(f'Done!\n')

    assert (not joined_df.isnull().values.any().any())

    #####
    # Perform train/val/test split
    #####
    logging.info(f'Going to split examples (train/val/test)...')
    logging.info(f'Timestamp used to split train and test examples: {train_test_threshold}')
    df_train_val, df_test = split_dataframe_by_date(joined_df, train_test_threshold)

    # TODO: parameterize with user-defined train/val splitting proportions.
    n = len(df_train_val)
    train_val_split = 0.8
    train_upper_limit = int(n*train_val_split)
    df_train = df_train_val[0:train_upper_limit]
    df_val = df_train_val[train_upper_limit:]

    assert (not df_train.isnull().values.any().any())
    assert (not df_val.isnull().values.any().any())
    assert (not df_test.isnull().values.any().any())

    logging.info(f"Range of timestamps in the training dataset: [{min(df_train.index)}, {max(df_train.index)}]")
    logging.info(f"Range of timestamps in the validation dataset: [{min(df_val.index)}, {max(df_val.index)}]")
    logging.info(f"Range of timestamps in the test dataset: [{min(df_test.index)}, {max(df_test.index)}]")
    logging.info(f'Number of examples in each dataset (train/val/test): {len(df_train)}/{len(df_val)}/{len(df_test)}.')
    logging.info(f'Done!\n')

    #
    # Save train/val/test DataFrames for future error analisys.
    logging.info(f'Saving each train/val/test dataset for pipeline {pipeline_id} as a parquet file.')
    df_train.to_parquet(globals.DATASETS_DIR + pipeline_id + '_train.parquet.gzip', compression='gzip')
    df_val.to_parquet(globals.DATASETS_DIR + pipeline_id + '_val.parquet.gzip', compression='gzip')
    df_test.to_parquet(globals.DATASETS_DIR + pipeline_id + '_test.parquet.gzip', compression='gzip')
    logging.info(f'Done!\n')

    assert (not df_train.isnull().values.any().any())
    assert (not df_val.isnull().values.any().any())
    assert (not df_test.isnull().values.any().any())

    if (station_id in globals.ALERTARIO_GAUGE_STATION_IDS):
        df_train = gaussian_noise(df_train, "barometric_pressure", mu=1000)
        df_val = gaussian_noise(df_val, "barometric_pressure", mu=1000)
        df_test = gaussian_noise(df_test, "barometric_pressure", mu=1000)
    
    #
    # Normalize the columns in train/val/test dataframes. This is done as a preparation step for applying
    # the sliding window technique, since the target variable is going to be used as lag feature.
    # (see, e.g., https://www.mikulskibartosz.name/forecasting-time-series-using-lag-features/)
    # (see also https://datascience.stackexchange.com/questions/72480/what-is-lag-in-time-series-forecasting)
    logging.info('Normalizing the features in train/val/test dataframes...')
    _, target_name = util.get_relevant_variables(station_id)
    min_target_value_in_train, max_target_value_in_train = min(df_train[target_name]), max(df_train[target_name])
    min_target_value_in_val, max_target_value_in_val = min(df_val[target_name]), max(df_val[target_name])
    min_target_value_in_test, max_target_value_in_test = min(df_test[target_name]), max(df_test[target_name])
    df_train = util.min_max_normalize(df_train)
    df_val = util.min_max_normalize(df_val)
    df_test = util.min_max_normalize(df_test)
    logging.info(f'Done!\n')

    assert (not df_train.isnull().values.any().any())
    assert (not df_val.isnull().values.any().any())
    assert (not df_test.isnull().values.any().any())

    #
    # Apply sliding window method to build examples (instances) of train/val/test datasets 
    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    window_size = config["preproc"]["SLIDING_WINDOW_SIZE"]
    logging.info('Applying sliding window to build train/val/test datasets...')
    X_train, y_train, X_val, y_val, X_test, y_test = generate_windowed_split(
        df_train, 
        df_val, 
        df_test, 
        target_name, 
        window_size)
    logging.info("Resulting shapes:")
    logging.info(f' - (X_train/X_val/X_test): ({X_train.shape}/{X_val.shape}/{X_test.shape})')
    logging.info(f' - (y_train/y_val/y_test): ({y_train.shape}/{y_val.shape}/{y_test.shape})')
    logging.info(f'Done!\n')

    assert not np.isnan(np.sum(X_train))
    assert not np.isnan(np.sum(X_val))
    assert not np.isnan(np.sum(X_test))
    assert not np.isnan(np.sum(y_train))
    assert not np.isnan(np.sum(y_val))
    assert not np.isnan(np.sum(y_test))

    #
    # Now, we restore the target variable to their original values. This is needed in case a multiclass
    # classification task is defined down the pipeline.
    logging.info('Restoring the target variable to their original values...')
    y_train = (y_train + min_target_value_in_train) * (max_target_value_in_train - min_target_value_in_train)
    y_val = (y_val + min_target_value_in_val) * (max_target_value_in_val - min_target_value_in_val)
    y_test = (y_test + min_target_value_in_test) * (max_target_value_in_test - min_target_value_in_test)
    logging.info('Min precipitation values (train/val/test): %.5f, %.5f, %.5f' %
          (np.min(y_train), np.min(y_val), np.min(y_test)))
    logging.info('Max precipitation values (train/val/test): %.5f, %.5f, %.5f' %
          (np.max(y_train), np.max(y_val), np.max(y_test)))
    logging.info(f'Done!\n')

    if subsampling_procedure != "NONE":
        #
        # Subsampling
        logging.info('**********Subsampling************')
        logging.info(f'- Shapes before subsampling (y_train/y_val/y_test): {y_train.shape}, {y_val.shape}, {y_test.shape}')
        logging.info("Subsampling train data.")
        X_train, y_train = apply_subsampling(X_train, y_train, subsampling_procedure)
        logging.info("Subsampling val data...")

        X_val, y_val = apply_subsampling(X_val, y_val, "NEGATIVE")
        logging.info('- Min precipitation values (train/val/test) after subsampling: %.5f, %.5f, %.5f' %
              (np.min(y_train), np.min(y_val), np.min(y_test)))

        logging.info('- Max precipitation values (train/val/test) after subsampling: %.5f, %.5f, %.5f' %
            (np.max(y_train), np.max(y_val), np.max(y_test)))
        logging.info(f'- Shapes (y_train/y_val/y_test) after subsampling: {y_train.shape}, {y_val.shape}, {y_test.shape}')
        logging.info(f'Done!\n')

    #
    # Write numpy arrays for train/val/test dataset to a single pickle file
    logging.info(f'Dumping train/val/test np arrays to pickle file {filename}...')
    logging.info(
        f'Number of examples (train/val/test): {len(X_train)}/{len(X_val)}/{len(X_test)}.')
    filename = globals.DATASETS_DIR + pipeline_id + ".pickle"
    file = open(filename, 'wb')
    ndarrays = (X_train, y_train, 
                X_val, y_val, 
                X_test, y_test)
    pickle.dump(ndarrays, file)
    logging.info(f'Done!\n')
    
    logging.info('Done it all!')

def main(argv):
    parser = argparse.ArgumentParser(
        description="""This script builds the train/val/test datasets for a given weather station, by using the user-specified data sources.""")
    parser.add_argument('-s', '--station_id', type=str, required=True, help='station id')
    parser.add_argument('-tt', '--train_test_threshold', type=str, required=True, help='The limiting date between train and test examples (format: YYYY-MM-DD).')
    parser.add_argument('-ts', '--train_start_threshold', type=str, required=False, help='The limiting date from which to consider examples (format: YYYY-MM-DD).')
    parser.add_argument('-d', '--datasources', type=str, help='data source spec')
    parser.add_argument('-sp', '--subsampling_procedure', type=str, default='NONE', help='Subsampling procedure do be applied.')
    args = parser.parse_args(argv[1:])

    station_id = args.station_id
    datasources = args.datasources
    subsampling_procedure = args.subsampling_procedure

    try:
        if (station_id in globals.ALERTARIO_WEATHER_STATION_IDS or station_id in globals.ALERTARIO_GAUGE_STATION_IDS):
            train_test_threshold = pd.to_datetime(args.train_test_threshold, utc=True) # This UTC thing is really anonying!
        else:
            train_test_threshold = pd.to_datetime(args.train_test_threshold)
    except ParserError:
        print(f"Invalid date format: {args.train_test_threshold}.")
        parser.print_help()
        sys.exit(2)

    try:
        train_start_threshold = args.train_start_threshold
        if  train_start_threshold is not None:
            train_start_threshold = pd.to_datetime(args.train_start_threshold)
    except pd.errors.ParserError:
        print(f"Invalid date format: {args.train_start_threshold}.")
        parser.print_help()
        sys.exit(2)

    lst_subsampling_procedures = ["NONE", "NAIVE", "NEGATIVE"]
    if not (subsampling_procedure in lst_subsampling_procedures):
        print(f"Invalid subsampling procedure: {subsampling_procedure}. Valid values: {lst_subsampling_procedures}")
        parser.print_help()
        sys.exit(2)

    if not ((station_id in globals.INMET_WEATHER_STATION_IDS) or (station_id in globals.ALERTARIO_WEATHER_STATION_IDS) or (station_id in globals.ALERTARIO_GAUGE_STATION_IDS)):
        print(f"Invalid station identifier: {station_id}")
        parser.print_help()
        sys.exit(2)

    if (station_id in globals.INMET_WEATHER_STATION_IDS):
        input_folder = globals.WS_INMET_DATA_DIR
    elif (station_id in globals.ALERTARIO_WEATHER_STATION_IDS):
        input_folder = globals.WS_ALERTARIO_DATA_DIR
    elif (station_id in globals.ALERTARIO_GAUGE_STATION_IDS):
        # Its a gauge station.
        input_folder = globals.GS_ALERTARIO_DATA_DIR

    fmt = "[%(levelname)s] %(funcName)s():%(lineno)i: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format = fmt)

    join_goes16_tpw_data_source = False
    join_as_data_source = False
    join_nwp_data_source = False
    join_lightning_data_source = False
    join_colorcord_data_source = False
    join_conv2d_data_source = False
    join_autoencoder_data_source = False

    if datasources:
        if 'R' in datasources:
            join_as_data_source = True
        if 'N' in datasources:
            join_nwp_data_source = True
        if 'L' in datasources:
            join_lightning_data_source = True
        if 'T' in datasources:
            join_goes16_tpw_data_source = True
        if "I" in datasources:
            join_colorcord_data_source = True
        if "C" in datasources:
            join_conv2d_data_source = True
        if "A" in datasources:
            join_autoencoder_data_source = True

    assert(station_id is not None) and (station_id != "")

    build_datasets(station_id, 
                   input_folder,
                   train_start_threshold,
                   train_test_threshold,
                   join_as_data_source, 
                   join_nwp_data_source, 
                   join_lightning_data_source, 
                   join_goes16_tpw_data_source,
                   join_colorcord_data_source,
                   join_conv2d_data_source,
                   join_autoencoder_data_source,
                   subsampling_procedure)

if __name__ == "__main__":
    main(sys.argv)
