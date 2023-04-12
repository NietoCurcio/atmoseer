import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import sklearn.metrics as skl
from rainfall_prediction import get_events_per_precipitation_level, map_to_precipitation_levels

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

'''
  https://stackoverflow.com/questions/59935155/how-to-calculate-mean-bias-errormbe-in-python
'''
def mean_bias_error(y_true, y_pred):
    MBE = np.mean(y_pred - y_true)
    return MBE


def export_confusion_matrix_to_latex(y_true, y_pred):
    # no_rain_true, weak_rain_true, moderate_rain_true, strong_rain_true, extreme_rain_true = get_events_per_precipitation_level(
    #     y_true)
    # no_rain_pred, weak_rain_pred, moderate_rain_pred, strong_rain_pred, extreme_rain_pred = get_events_per_precipitation_level(
    #     y_pred)

    # y_true_class = np.zeros_like(y_true)
    # y_true_class[no_rain_true] = NONE
    # y_true_class[weak_rain_true] = WEAK
    # y_true_class[strong_rain_true] = STRONG
    # y_true_class[moderate_rain_true] = MODERATE
    # y_true_class[extreme_rain_true] = EXTREME

    # y_pred_class = np.zeros_like(y_pred)
    # y_pred_class[no_rain_pred] = NONE
    # y_pred_class[weak_rain_pred] = WEAK
    # y_pred_class[moderate_rain_pred] = MODERATE
    # y_pred_class[strong_rain_pred] = STRONG
    # y_pred_class[extreme_rain_pred] = EXTREME

    y_true_class = map_to_precipitation_levels(y_true)
    y_pred_class = map_to_precipitation_levels(y_pred)

    print('Classification_report: ')
    print(skl.classification_report(y_true_class, y_pred_class))
    df = pd.DataFrame(
        confusion_matrix(y_true_class, y_pred_class, labels=[0, 1, 2, 3, 4]),
        index=['None', 'Weak', 'Moderate', 'Strong', 'Extreme'],
        columns=['None', 'Weak', 'Moderate', 'Strong', 'Extreme'],
    )
    df.index.name = 'true($\\downarrow$)/pred($\\rightarrow$)'
    print(df.style.to_latex(hrules=True))


def export_results_to_latex(y_true, y_pred):
    '''
      MAE (mean absolute error) and MBE (mean bias error) values are computed for each precipitation level.
    '''

    export_confusion_matrix_to_latex(y_true, y_pred)

    no_rain_true, weak_rain_true, moderate_rain_true, strong_rain_true, extreme_rain_true = get_events_per_precipitation_level(
        y_true)
    no_rain_pred, weak_rain_pred, moderate_rain_pred, strong_rain_pred, extreme_rain_pred = get_events_per_precipitation_level(
        y_pred)

    if no_rain_pred[0].size > 0:
        mse_no_rain = skl.mean_absolute_error(
            y_true[no_rain_true], y_pred[no_rain_true])
        mbe_no_rain = mean_bias_error(
            y_true[no_rain_true], y_pred[no_rain_true])
    else:
        mse_no_rain = mbe_no_rain = 'n/a'

    if weak_rain_pred[0].size > 0:
        mse_weak_rain = skl.mean_absolute_error(
            y_true[weak_rain_true], y_pred[weak_rain_true])
        mbe_weak_rain = mean_bias_error(
            y_true[weak_rain_true], y_pred[weak_rain_true])
    else:
        mse_weak_rain = mbe_weak_rain = 'n/a'

    if moderate_rain_pred[0].size > 0:
        mse_moderate_rain = skl.mean_absolute_error(
            y_true[moderate_rain_true], y_pred[moderate_rain_true])
        mbe_moderate_rain = mean_bias_error(
            y_true[moderate_rain_true], y_pred[moderate_rain_true])
    else:
        mse_moderate_rain = mbe_moderate_rain = 'n/a'

    if strong_rain_pred[0].size > 0:
        mse_strong_rain = skl.mean_absolute_error(
            y_true[strong_rain_true], y_pred[strong_rain_true])
        mbe_strong_rain = mean_bias_error(
            y_true[strong_rain_true], y_pred[strong_rain_true])
    else:
        mse_strong_rain = mbe_strong_rain = 'n/a'

    if extreme_rain_pred[0].size > 0:
        mse_extreme_rain = skl.mean_absolute_error(
            y_true[extreme_rain_true], y_pred[extreme_rain_true])
        mbe_extreme_rain = mean_bias_error(
            y_true[extreme_rain_true], y_pred[extreme_rain_true])
    else:
        mse_extreme_rain = mbe_extreme_rain = 'n/a'

    df = pd.DataFrame()
    df['level'] = ['No rain', 'Weak', 'Moderate', 'Strong', 'Extreme']
    df['qty_true'] = [no_rain_true[0].shape[0], weak_rain_true[0].shape[0],
                      moderate_rain_true[0].shape[0], strong_rain_true[0].shape[0], extreme_rain_true[0].shape[0]]
    df['qty_pred'] = [no_rain_pred[0].shape[0], weak_rain_pred[0].shape[0],
                      moderate_rain_pred[0].shape[0], strong_rain_pred[0].shape[0], extreme_rain_pred[0].shape[0]]
    df['mae'] = [mse_no_rain, mse_weak_rain,
                 mse_moderate_rain, mse_strong_rain, mse_extreme_rain]
    df['mbe'] = [mbe_no_rain, mbe_weak_rain,
                 mbe_moderate_rain, mbe_strong_rain, mbe_extreme_rain]
    print(df.style.to_latex(hrules=True))
