import pandas as pd


def normalize_y(y_train: pd.Series, yval, ytest) -> (pd.Series, pd.Series, pd.Series):
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    y_train_normalized = (y_train - y_train_mean) / y_train_std
    yval_normalized = (yval - y_train_mean) / y_train_std
    ytest_normalized = (ytest - y_train_mean) / y_train_std

    return y_train_normalized, yval_normalized, ytest_normalized
