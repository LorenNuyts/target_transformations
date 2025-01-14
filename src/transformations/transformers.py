import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class SubjectiveTransformer(TransformerMixin):
    def __init__(self):
        self.mean = None

    def fit(self, y: pd.DataFrame):
        """
        Fits the transformer to the target variable.

        Parameters
        ----------
        y: pd.DataFrame
            Target variable. Each row in the dataframe corresponds to a specific subject, with its targets in the columns.

        Returns
        -------
        pd.DataFrame
            Target variable centered by the row-wise mean.
        """
        self.mean = y.mean(axis=1)

        return y.sub(self.mean, axis=0)

    def transform(self, y: pd.DataFrame):
        """
        Transforms the target variable by applying a subjective target transformation. The subjects have to appear in the
        same order as in the fit method.

        Parameters
        ----------
        y: pd.DataFrame
            Target variable. Each row in the dataframe corresponds to a specific subject, with its targets in the columns.
            The subjects have to appear in the same order as in the fit method.

        Returns
        -------
        pd.DataFrame
            Transformed target variable.
        """
        if self.mean is None:
            raise ValueError("The transformer has not been fitted yet. Call the fit method first.")
        return y.sub(self.mean, axis=0)

    def inverse_transform(self, y: pd.DataFrame):
        """
        Inverse-transforms the target variable by applying the inverse of the subjective target transformation. The subjects
        have to appear in the same order as in the fit method.

        Parameters
        ----------
        y: pd.DataFrame
            Target variable. Each row in the dataframe corresponds to a specific subject, with its targets in the columns.
            The subjects have to appear in the same order as in the fit method.

        Returns
        -------
        pd.DataFrame
            Inverse-transformed target variable.
        """
        if self.mean is None:
            raise ValueError("The transformer has not been fitted yet. Call the fit method first.")
        return y.add(self.mean, axis=0)

class FrameDependencyTransformer:
    def fit(self, y: pd.Series, reference: pd.Series):
        """
        Point-wise division of the target variable by a reference variable.

        Parameters
        ----------
        y: pd.Series
            Target variable.
        reference: pd.Series
            Reference frame (e.g., time, area, etc.).

        Returns
        -------
        pd.Series
            Target variable divided by the reference variable.
        """
        return y / reference

    def transform(self, y: pd.Series, reference: pd.Series):
        """
        Point-wise division of the target variable by a reference variable.

        Parameters
        ----------
        y: pd.Series
            Target variable.
        reference: pd.Series
            Reference frame (e.g., time, area, etc.).

        Returns
        -------
        pd.Series
            Target variable divided by the reference variable.
        """
        return y / reference


    def inverse_transform(self, y: pd.Series, reference: pd.Series):
        """
        Point-wise multiplication of the target variable by a reference variable.

        Parameters
        ----------
        y: pd.Series
            Target variable.
        reference: pd.Series
            Reference frame (e.g., time, area, etc.).

        Returns
        -------
        pd.Series
            Target variable multiplied by the reference variable.
        """
        return y * reference


class NormalizeTransformer(TransformerMixin):
    """
    Standardizes the data by subtracting the mean and dividing by the standard deviation.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        self.mean = X.mean()
        self.std = X.std()
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean


class LogTransformer(TransformerMixin):
    """
    Applies the log transformation with a given base (default = 10), with an offset to avoid log(0). The offset is defined
    as 1 by default, or can be set to a different value.
    """
    def __init__(self, base=10):
        self.base = base
        self.offset = 1

    def fit(self, X, y=None):
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        if isinstance(y, pd.Series):
            y = y.values
        if self.base == 10:
            return np.log10(y + self.offset)
        elif self.base == np.e:
            return np.log(y + self.offset)
        else:
            raise ValueError("Base must be 10 or e")

    def inverse_transform(self, y):
        if self.base == 10:
            return np.power(10, y) - self.offset
        elif self.base == np.e:
            return np.exp(y) - self.offset
        else:
            raise ValueError("Base must be 10 or e")

