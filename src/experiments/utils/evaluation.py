import numpy as np


def relative_squared_error(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / np.sum(np.square(np.mean(y_true) - y_pred))

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
