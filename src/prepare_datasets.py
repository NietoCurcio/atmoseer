import pandas as pd
import numpy as np
import sys
import getopt
import pickle
from utils.near_stations import prox
from globals import *
from util import transform_hour
from utils.windowing import apply_windowing

def apply_subsampling(X, y, percentage = 0.1):
  print('*BEGIN*')
  print(X.shape)
  print(y.shape)
  y_eq_zero_idxs = np.where(y==0)[0]
  print('# original samples  eq zero:', y_eq_zero_idxs.shape)
  y_gt_zero_idxs = np.where(y>0)[0]
  print('# original samples gt zero:', y_gt_zero_idxs.shape)
  mask = np.random.choice([True, False], size=y.shape[0], p=[percentage, 1.0-percentage])
  y_train_subsample_idxs = np.where(mask==True)[0]
  print('# subsample:', y_train_subsample_idxs.shape)
  idxs = np.intersect1d(y_eq_zero_idxs, y_train_subsample_idxs)
  print('# subsample that are eq zero:', idxs.shape)
  idxs = np.union1d(idxs, y_gt_zero_idxs)
  print('# subsample final:', idxs.shape)
  X, y = X[idxs], y[idxs]
  print(X.shape)
  print(y.shape)
  print('*END*')
  return X, y

def generate_windowed_split(train_df, val_df, test_df, target_name, window_size = 6):
  
    # train_df = pd.read_csv(arquivo + '_train.csv')
    # del train_df['Unnamed: 0']
  
    # val_df = pd.read_csv(arquivo + '_val.csv')
    # del val_df['Unnamed: 0']

    # test_df = pd.read_csv(arquivo + '_test.csv')
    # del test_df['Unnamed: 0']

    train_arr = np.array(train_df)
    val_arr = np.array(val_df)
    test_arr = np.array(test_df)

    idx_target = train_df.columns.get_loc(target_name)
    print(f"Position (index) of target variable {target_name}: {idx_target}")

    TIME_WINDOW_SIZE = window_size
    IDX_TARGET = target_name
      
    X_train, y_train = apply_windowing(train_arr, 
                                    initial_time_step=0, 
                                    max_time_step=len(train_arr)-TIME_WINDOW_SIZE-1, 
                                    window_size = TIME_WINDOW_SIZE, 
                                    idx_target = idx_target)
    y_train = y_train.reshape(-1,1)

    X_val, y_val = apply_windowing(val_arr, 
                                initial_time_step=0, 
                                max_time_step=len(val_arr)-TIME_WINDOW_SIZE-1, 
                                window_size = TIME_WINDOW_SIZE, 
                                idx_target = idx_target)
    y_val = y_val.reshape(-1,1)

    X_test, y_test = apply_windowing(test_arr, 
                                  initial_time_step=0, 
                                  max_time_step=len(test_arr)-TIME_WINDOW_SIZE-1, 
                                  window_size = TIME_WINDOW_SIZE, 
                                  idx_target = idx_target)
    y_test = y_test.reshape(-1,1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def project_to_relevant_variables(df_a652):
    return ['TEM_MAX', 'PRE_MAX', 'UMD_MAX', 'wind_u', 'wind_v', 'hour_sin', 'hour_cos'], 'CHUVA'

def pre_proc(station_id, use_sounding_as_data_source, use_numerical_model_as_data_source, num_neighbors = 0):

    arq_pre_proc = station_id + '_E'
    if use_numerical_model_as_data_source:
        arq_pre_proc = arq_pre_proc + '-N'
    if use_sounding_as_data_source:
        arq_pre_proc = arq_pre_proc + '-R'
    if num_neighbors > 0:
        arq_pre_proc = arq_pre_proc + '_EI+' + str(num_neighbors) + 'NN'
    else:
        arq_pre_proc = arq_pre_proc + '_EI'

    df_a652 = pd.read_parquet('../data/weather_stations/A652_1997_2022_preprocessed.parquet.gzip')

    df_a652 = transform_hour(df_a652)

    predictor_names, target_name = project_to_relevant_variables(df_a652)
    df_a652 = df_a652[predictor_names + [target_name]]

    # 
    # Merge datasources
    #
    merged_df = df_a652
    
    if use_numerical_model_as_data_source:
        df_era5 = pd.read_parquet('../data/numerical_models/ERA5_A652_1997-01-01_2021-12-31_preprocessed.parquet.gzip')
        merged_df = pd.merge(merged_df, df_era5, on = 'Datetime', how = 'left')

    if use_sounding_as_data_source:
        df_sounding = pd.read_parquet('../data/sounding_stations/SBGL_indices_1997-01-01_2022-12-31_preprocessed.parquet.gzip')
        merged_df = pd.merge(df_ws, df_sounding, on = 'Datetime', how = 'left')
        # TODO: implement interpolation

    if num_neighbors != 0:
        pass

    #
    # Data normalization
    #
    target_col = merged_df[target_name].copy()
    merged_df = ((merged_df - merged_df.min()) / (merged_df.max() - merged_df.min()))
    merged_df[target_name] = target_col
        
    # 
    # Data splitting (train/val/test)
    #
    n = len(merged_df)
    train_df = merged_df[0:int(n*0.7)]
    val_df = merged_df[int(n*0.7):int(n*0.9)]
    test_df = merged_df[int(n*0.9):]
    print(f'Saving train/val/test datasets ({arq_pre_proc}).')
    print(f'Number of examples (train/val/test): {len(train_df)}/{len(val_df)}/{len(test_df)}.')
    train_df.to_parquet('../data/datasets/' + arq_pre_proc + '_train.parquet.gzip', compression = 'gzip')  
    val_df.to_parquet('../data/datasets/' + arq_pre_proc + '_val.parquet.gzip', compression = 'gzip')  
    test_df.to_parquet('../data/datasets/' + arq_pre_proc + '_test.parquet.gzip', compression = 'gzip')

    #
    # Data windowing
    X_train, y_train, X_val, y_val, X_test, y_test = generate_windowed_split(train_df, val_df, test_df, target_name = target_name, window_size = 6)

    #
    # Subsampling
    #
    print('***Before subsampling***')
    print('Max precipitation values (train/val/test): %d, %d, %d' % (np.max(y_train), np.max(y_val), np.max(y_test)))
    print('Mean precipitation values (train/val/test): %.4f, %.4f, %.4f' % (np.mean(y_train), np.mean(y_val), np.mean(y_test)))
    X_train, y_train = apply_subsampling(X_train, y_train)
    X_val, y_val = apply_subsampling(X_val, y_val)
    X_test, y_test = apply_subsampling(X_test, y_test)
    print('***After subsampling***')
    print('Max precipitation values (train/val/test): %d, %d, %d' % (np.max(y_train), np.max(y_val), np.max(y_test)))
    print('Mean precipitation values (train/val/test): %.4f, %.4f, %.4f' % (np.mean(y_train), np.mean(y_val), np.mean(y_test)))
    
    #
    # Write numpy arrays to a parquet file
    print(f'Saving train/val/test np arrays ({arq_pre_proc}).')
    print(f'Number of examples (train/val/test): {len(X_train)}/{len(X_val)}/{len(X_test)}.')
    file = open('../data/datasets/' + arq_pre_proc + ".pickle", 'wb')
    ndarrays = (X_train, y_train, X_val, y_val, X_test, y_test)
    pickle.dump(ndarrays, file)

    print('Done!')

def main(argv):
    station_id = ""
    use_sounding_as_data_source = 0
    use_numerical_model_as_data_source = 0
    num_neighbors = 0
    help_message = "Usage: {0} -s <station_id> -d <data_source_spec> -n <num_neighbors>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:d:n:", ["help", "station_id=", "datasources=", "neighbors="])
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

    pre_proc(station_id, use_sounding_as_data_source, use_numerical_model_as_data_source, num_neighbors = num_neighbors)


# python prepare_datasets.py -s A652 -d N
if __name__ == "__main__":
    main(sys.argv)