import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


def normalize_y(y_train: pd.Series, yval, ytest) -> (pd.Series, pd.Series, pd.Series):
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    y_train_normalized = (y_train - y_train_mean) / y_train_std
    yval_normalized = (yval - y_train_mean) / y_train_std
    ytest_normalized = (ytest - y_train_mean) / y_train_std

    return y_train_normalized, yval_normalized, ytest_normalized


class LogTransformer(TransformerMixin):
    def __init__(self, base=10):
        self.base = base
        self.offset = 0

    def fit(self, X, y=None):
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        if isinstance(y, pd.Series):
            y = y.values
        if 0 in y:
            self.offset = 1
            y += 1
        if self.base == 10:
            return np.log10(y)
        elif self.base == np.e:
            return np.log(y)
        else:
            raise ValueError("Base must be 10 or e")

    def inverse_transform(self, y):
        if self.base == 10:
            return np.power(10, y) - self.offset
        elif self.base == np.e:
            return np.exp(y) - self.offset
        else:
            raise ValueError("Base must be 10 or e")

