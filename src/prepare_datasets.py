import pandas as pd
import numpy as np
import sys
import getopt
import pickle
from utils.near_stations import prox
from globals import *
from util import transform_hour, find_contiguous_observation_blocks
from utils.windowing import apply_windowing
from sklearn.preprocessing import MinMaxScaler

def apply_subsampling(X, y, percentage=0.1):
    print('*BEGIN*')
    print(X.shape)
    print(y.shape)
    y_eq_zero_idxs = np.where(y == 0)[0]
    print('# original samples with target eq zero:', y_eq_zero_idxs.shape)
    y_gt_zero_idxs = np.where(y > 0)[0]
    print('# original samples with target gt zero:', y_gt_zero_idxs.shape)
    mask = np.random.choice([True, False], size=y.shape[0], p=[
                            percentage, 1.0-percentage])
    y_train_subsample_idxs = np.where(mask == True)[0]
    print('# subsample shape:', y_train_subsample_idxs.shape)
    idxs = np.intersect1d(y_eq_zero_idxs, y_train_subsample_idxs)
    print('# subsample that are eq zero:', idxs.shape)
    idxs = np.union1d(idxs, y_gt_zero_idxs)
    print('# subsample final:', idxs.shape)
    X, y = X[idxs], y[idxs]
    print(X.shape)
    print(y.shape)
    print('*END*')
    return X, y


def generate_windowed(df: pd.DataFrame, target_idx: int):
    """
    This function applies the sliding window preprocessing technique to generate data an response matrices 
    from an input time series represented as a pandas DataFrame. This DataFrame is supposed to have
    a datetime index that corresponds to the timestamps in the time series.

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

        # print(df[start:end].shape)
        if df[start:end].shape[0] < TIME_WINDOW_SIZE + 1:
            continue

        arr = np.array(df[start:end])
        X_block, y_block = apply_windowing(arr,
                                           initial_time_step=0,
                                           max_time_step=len(
                                               arr)-TIME_WINDOW_SIZE-1,
                                           window_size=TIME_WINDOW_SIZE,
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


def generate_windowed_split(train_df, val_df, test_df, target_name):
    target_idx = train_df.columns.get_loc(target_name)
    print(f"Position (index) of target variable {target_name}: {target_idx}")
    X_train, y_train = generate_windowed(train_df, target_idx)
    X_val, y_val = generate_windowed(val_df, target_idx)
    X_test, y_test = generate_windowed(test_df, target_idx)
    return X_train, y_train, X_val, y_val, X_test, y_test


def project_to_relevant_variables(df_a652):
    return ['TEM_MAX', 'PRE_MAX', 'UMD_MAX', 'wind_u', 'wind_v', 'hour_sin', 'hour_cos'], 'CHUVA'


def normalize(df: pd.DataFrame):
    df = (df - df.min()) / (df.max() - df.min())
    return df


def pre_proc(station_id, use_sounding_as_data_source, use_numerical_model_as_data_source, num_neighbors=0):

    arq_pre_proc = station_id + '_E'
    if use_numerical_model_as_data_source:
        arq_pre_proc = arq_pre_proc + '-N'
    if use_sounding_as_data_source:
        arq_pre_proc = arq_pre_proc + '-R'
    if num_neighbors > 0:
        arq_pre_proc = arq_pre_proc + '_EI+' + str(num_neighbors) + 'NN'
    else:
        arq_pre_proc = arq_pre_proc + '_EI'

    df_a652 = pd.read_parquet(
        '../data/weather_stations/A652_1997_2022_preprocessed.parquet.gzip')

    df_a652 = transform_hour(df_a652)

    predictor_names, target_name = project_to_relevant_variables(df_a652)
    df_a652 = df_a652[predictor_names + [target_name]]

    #
    # Merge datasources
    #
    merged_df = df_a652

    if use_numerical_model_as_data_source:
        df_era5 = pd.read_parquet(
            '../data/numerical_models/ERA5_A652_1997-01-01_2021-12-31_preprocessed.parquet.gzip')
        merged_df = pd.merge(merged_df, df_era5, on='Datetime', how='left')

    if use_sounding_as_data_source:
        df_sounding = pd.read_parquet(
            '../data/sounding_stations/SBGL_indices_1997-01-01_2022-12-31_preprocessed.parquet.gzip')
        merged_df = pd.merge(merged_df, df_sounding, on='Datetime', how='left')
        # TODO: implement interpolation
        # TODO: deal with missing values (see https://youtu.be/DKmDJJzayZw)

    if num_neighbors != 0:
        pass

    ####
    # TODO: add a data filtering step (e.g., to disregard all observations made between May and September.).
    ####

    #
    # Data splitting (train/val/test)
    #
    n = len(merged_df)
    train_df = merged_df[0:int(n*0.7)]
    val_df = merged_df[int(n*0.7):int(n*0.9)]
    test_df = merged_df[int(n*0.9):]
    print(f'Saving train/val/test datasets ({arq_pre_proc}).')
    print(
        f'Number of examples (train/val/test): {len(train_df)}/{len(val_df)}/{len(test_df)}.')
    train_df.to_parquet('../data/datasets/' + arq_pre_proc +
                        '_train.parquet.gzip', compression='gzip')
    val_df.to_parquet('../data/datasets/' + arq_pre_proc +
                      '_val.parquet.gzip', compression='gzip')
    test_df.to_parquet('../data/datasets/' + arq_pre_proc +
                       '_test.parquet.gzip', compression='gzip')

    #
    # Data normalization
    #
    # print('**********\n***Before normalization***')
    # print(f"Min precipitation values (train/val/test): {min(train_df[target_name])},{min(val_df[target_name])},{min(test_df[target_name])}")
    # print(f"Max precipitation values (train/val/test): {max(train_df[target_name])},{max(val_df[target_name])},{max(test_df[target_name])}")
    # min_y_train, max_y_train = min(
    #     train_df[target_name]), max(train_df[target_name])
    # min_y_val, max_y_val = min(val_df[target_name]), max(val_df[target_name])
    # min_y_test, max_y_test = min(test_df[target_name]), max(test_df[target_name])
    # train_df = normalize(train_df)
    # val_df = normalize(val_df)
    # test_df = normalize(test_df)
    # print('**********\n***After normalization***')
    # print(f"Min precipitation values (train/val/test): {min(train_df[target_name])},{min(val_df[target_name])},{min(test_df[target_name])}")
    # print(f"Max precipitation values (train/val/test): {max(train_df[target_name])},{max(val_df[target_name])},{max(test_df[target_name])}")

    # Data windowing
    print('**********Data windowing**********')
    print('***Before')
    print(f'Shapes (train/val/test): {train_df.shape}, {val_df.shape}, {test_df.shape}')
    X_train, y_train, X_val, y_val, X_test, y_test = generate_windowed_split(
        train_df, val_df, test_df, target_name=target_name)
    print('***After')
    print(f'Shapes (train/val/test): {X_train.shape}, {X_val.shape}, {X_test.shape}')

    return

    #
    # Subsampling
    #
    print('**********Subsampling************')
    print('***Before')
    print('Min precipitation values (train/val/test): %.5f, %.5f, %.5f' %
          (np.min(y_train), np.min(y_val), np.min(y_test)))
    print('Max precipitation values (train/val/test): %.5f, %.5f, %.5f' %
          (np.max(y_train), np.max(y_val), np.max(y_test)))
    print(f'Shapes (train/val/test): {y_train.shape}, {y_val.shape}, {y_test.shape}')

    X_train, y_train = apply_subsampling(X_train, y_train)
    X_val, y_val = apply_subsampling(X_val, y_val)
    X_test, y_test = apply_subsampling(X_test, y_test)

    print('***After')
    print('Min precipitation values (train/val/test): %.5f, %.5f, %.5f' %
          (np.min(y_train), np.min(y_val), np.min(y_test)))
    print('Max precipitation values (train/val/test): %.5f, %.5f, %.5f' %
          (np.max(y_train), np.max(y_val), np.max(y_test)))
    print(f'Shapes (train/val/test): {y_train.shape}, {y_val.shape}, {y_test.shape}')

    #
    # Data normalization
    #
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    #
    # Write numpy arrays to a parquet file
    print(f'Saving train/val/test np arrays ({arq_pre_proc}).')
    print(
        f'Number of examples (train/val/test): {len(X_train)}/{len(X_val)}/{len(X_test)}.')
    file = open('../data/datasets/' + arq_pre_proc + ".pickle", 'wb')
    ndarrays = (X_train, y_train, min_y_train, max_y_train,
                X_val, y_val, min_y_val, max_y_val,
                X_test, y_test, min_y_test, max_y_test)
    pickle.dump(ndarrays, file)

    print('Done!')


def main(argv):
    station_id = ""
    use_sounding_as_data_source = 0
    use_numerical_model_as_data_source = 0
    num_neighbors = 0
    help_message = "Usage: {0} -s <station_id> -d <data_source_spec> -n <num_neighbors>".format(
        argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "hs:d:n:", [
                                   "help", "station_id=", "datasources=", "neighbors="])
    except:
        print(help_message)
        sys.exit(2)

    num_neighbors = 0
    use_sounding_as_data_source = False
    use_numerical_model_as_data_source = False

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)  # print the help message
            sys.exit(2)
        elif opt in ("-s", "--station_id"):
            station_id = arg
            if not ((station_id in INMET_STATION_CODES_RJ) or (station_id in COR_STATION_NAMES_RJ)):
                print(f"Invalid station identifier: {station_id}")
                print(help_message)
                sys.exit(2)
        elif opt in ("-d", "--datasources"):
            if arg.find('R') != -1:
                use_sounding_as_data_source = True
            if arg.find('N') != -1:
                use_numerical_model_as_data_source = True
        elif opt in ("-n", "--neighbors"):
            num_neighbors = arg

    assert(station_id is not None) and (station_id != "")
    pre_proc(station_id, use_sounding_as_data_source,
             use_numerical_model_as_data_source, num_neighbors=num_neighbors)


# python prepare_datasets.py -s A652 -d N
if __name__ == "__main__":
    main(sys.argv)
