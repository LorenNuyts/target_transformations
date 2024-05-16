import numpy as np


def relative_squared_error(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / np.sum(np.square(np.mean(y_true) - y_pred))
