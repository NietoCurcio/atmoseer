
import pandas as pd
import numpy as np

def apply_windowing(X, 
                    initial_time_step, 
                    max_time_step, 
                    window_size, 
                    idx_target,
                    only_y_not_nan = True,
                    only_y_gt_zero = True,
                    only_X_not_nan = True):

  assert idx_target >= 0 and idx_target < X.shape[1]
  assert initial_time_step >= 0
  assert max_time_step >= initial_time_step

  start = initial_time_step
    
  sub_windows = (
        start +
        # expand_dims converts a 1D array to 2D array.
        np.expand_dims(np.arange(window_size), 0) +
        np.expand_dims(np.arange(max_time_step + 1), 0).T
  )

  X_temp, y_temp = X[sub_windows], X[window_size:(max_time_step+window_size+1):1, idx_target]

  if only_y_not_nan and only_y_gt_zero and only_X_not_nan:
    y_train_not_nan_idx = np.where(~np.isnan(y_temp))[0]
    y_train_gt_zero_idx = np.where(y_temp>0)[0]
    x_train_is_nan_idx = np.unique(np.where(np.isnan(X_temp)))
    idxs = np.intersect1d(y_train_not_nan_idx, y_train_gt_zero_idx)
    idxs = np.setdiff1d(idxs, x_train_is_nan_idx)
    X_temp, y_temp = X_temp[idxs], y_temp[idxs]

  return X_temp, y_temp
