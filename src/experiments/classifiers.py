import abc
import time

import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sktime.forecasting.compose import make_reduction

from src.experiments.utils.constants import SEED
from src.experiments.data import Dataset


class AbstractClassifier(abc.ABC):
    name: str

    def __init__(self, seed=SEED):
        self.seed = seed
        self.clf = None

    def fit_coefficients(self, data: Dataset, alpha_record):
        fit_time = time.process_time()
        self.clf.fit(data.Xtrain, data.ytrain)
        fit_time = time.process_time() - fit_time

        # num_params = data.Xtrain.shape[1]
        # num_removed = np.sum(np.abs(self.clf.coef_) < 1e-6)
        # frac_removed = num_removed / num_params

        alpha_record.mtrain_clf = root_mean_squared_error(data.ytrain, self.clf.predict(data.Xtrain))
        alpha_record.mvalid_clf = root_mean_squared_error(data.yval, self.clf.predict(data.Xval))
        # alpha_record.clf = self
        alpha_record.intercept = np.copy(self.clf.intercept_)
        alpha_record.coefs = np.copy(self.clf.coef_)
        alpha_record.fit_time = fit_time

    @abc.abstractmethod
    def update_lin_clf_alpha(self, alpha):
        raise NotImplementedError

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def transform(self, X):
        return self.clf.transform(X)


class LogisticRegressionTuned(AbstractClassifier):
    def __init__(self, seed=SEED):
        super().__init__(seed)
        self.name = "LogisticRegressionTuned"
        self.clf = LogisticRegression(
            penalty='l1',
            C=1.0,
            solver="liblinear",
            max_iter=5_000,
            tol=1e-4,
            n_jobs=1,
            random_state=self.seed,
            warm_start=True,
        )

    def update_lin_clf_alpha(self, alpha):
        if alpha <= 0.0:
            raise RuntimeError("alpha == 0.0?")
        assert self.clf.penalty == "l1"
        self.clf.C = 1.0 / alpha


class LassoTuned(AbstractClassifier):
    def __init__(self, seed=SEED, precompute=False):
        super().__init__(seed)
        self.name = "LassoTuned"
        self.acronym = "Lasso"
        self.seed = seed
        self.clf = Lasso(
            alpha=1.0,
            random_state=self.seed,
            max_iter=5_000,
            tol=1e-4,
            warm_start=True,
            precompute=precompute,
        )

    def update_lin_clf_alpha(self, alpha):
        if alpha <= 0.0:
            raise RuntimeError("alpha == 0.0?")

        self.clf.alpha = 0.0001 * alpha


class RidgeRegressionTuned(AbstractClassifier):
    def __init__(self, seed=SEED):
        super().__init__(seed)
        self.name = "RidgeRegressionTuned"
        self.acronym = "Ridge"
        self.seed = seed
        self.clf = Ridge(alpha=1.0,
                         random_state=self.seed,
                         max_iter=5_000,
                         tol=1e-4)

    def update_lin_clf_alpha(self, alpha):
        if alpha <= 0.0:
            raise RuntimeError("alpha == 0.0?")

        self.clf.alpha = 0.0001 * alpha


class GradientBoostingRegressorWrapper(AbstractClassifier):
    def __init__(self, seed=SEED):
        super().__init__(seed)
        self.name = "GradientBoostingRegressorWrapper"
        self.acronym = "GBR"
        self.seed = seed
        self.clf = GradientBoostingRegressor(
                         random_state=self.seed,
                         tol=1e-4)

    def update_lin_clf_alpha(self, alpha):
        pass


class SupportVectorRegressorWrapper(AbstractClassifier):
    def __init__(self, seed=SEED):
        super().__init__(seed)
        self.name = "SupportVectorRegressorWrapper"
        self.acronym = "SVR"
        self.seed = seed
        self.clf = SVR()

    def update_lin_clf_alpha(self, alpha):
        pass


class ExponentialSmoothingWrapper:
    def __init__(self, **kwargs):
        self.model_params = kwargs
        self.acronym = "ES"
        self.clf = None

    def fit(self, data: Dataset):
        if data.Xtrain.index.nlevels == 1:
            if data.Xtrain.index.freq is None:
                data.Xtrain = data.Xtrain.asfreq(pd.infer_freq(data.Xtrain.index))
            self.clf = ExponentialSmoothing(data.Xtrain, **self.model_params)
            self.clf = self.clf.fit()
        else:
            self.clf = dict()
            for i in data.Xtrain.index.get_level_values(0).unique():
                train_data = data.Xtrain.loc[i]
                if train_data.index.freq is None:
                    train_data = train_data.asfreq(pd.infer_freq(train_data.index))

                try:
                    self.clf[i] = ExponentialSmoothing(train_data, **self.model_params)
                    self.clf[i] = self.clf[i].fit()
                except ValueError as e:
                    if train_data.shape[0] < 5:
                        print(f"Could not fit ExponentialSmoothing for {i} because of insufficient train data "
                              f"(only {train_data.shape[0]} instances).")
                    else:
                        raise e

        return self

    def forecast(self, data):
        if isinstance(self.clf, dict):
            forecasts = []
            for i in self.clf.keys():
                if i not in data.Xtest.index:
                    print(f"Could not forecast ExponentialSmoothing for {i} because of insufficient test data.")
                    continue
                n = len(data.Xtest.loc[i])
                forecast_i = self.clf[i].forecast(n).to_frame()
                forecast_i['id'] = i
                forecasts.append(forecast_i)
            forecasts_df = pd.concat(forecasts)
            forecasts_df.set_index('id', append=True, inplace=True)
            combined_df = forecasts_df.reorder_levels(['id', None])
            combined_df.sort_index(inplace=True)
            return combined_df
        else:
            return self.clf.forecast(len(data.Xtest))

class AutoArimaWrapper:
    def __init__(self, **kwargs):
        self.model_params = kwargs
        self.acronym = "AutoARIMA"
        self.clf = None

    def fit(self, data: Dataset):
        if data.Xtrain.index.nlevels == 1:
            if data.Xtrain.index.freq is None:
                data.Xtrain = data.Xtrain.asfreq(pd.infer_freq(data.Xtrain.index))

            self.clf = pm.AutoARIMA(seasonal=True, m=12,
                           suppress_warnings=True,
                           trace=True)
            self.clf = self.clf.fit(data.Xtrain)
        else:
            self.clf = dict()
            for i in data.Xtrain.index.get_level_values(0).unique():
                train_data = data.Xtrain.loc[i]
                if train_data.index.freq is None:
                    train_data = train_data.asfreq(pd.infer_freq(train_data.index))

                try:
                    self.clf[i] = pm.AutoARIMA(seasonal=True, m=12,
                           suppress_warnings=True,
                           trace=True)
                    self.clf[i] = self.clf[i].fit(train_data)
                except ValueError as e:
                    self.clf.pop(i)  # remove the failed fit
                    if train_data.shape[0] < 5:
                        print(f"Could not fit AutoArima for {i} because of insufficient train data "
                              f"(only {train_data.shape[0]} instances).")
                    else:
                        raise e

        return self

    def forecast(self, data):
        if isinstance(self.clf, dict):
            forecasts = []
            for i in self.clf.keys():
                if i not in data.Xtest.index:
                    print(f"Could not forecast AutoArima for {i} because of insufficient test data.")
                    continue
                n = len(data.Xtest.loc[i])
                forecast_i = self.clf[i].predict(n).to_frame()
                forecast_i['id'] = i
                forecasts.append(forecast_i)
            forecasts_df = pd.concat(forecasts)
            forecasts_df.set_index('id', append=True, inplace=True)
            combined_df = forecasts_df.reorder_levels(['id', None])
            combined_df.sort_index(inplace=True)
            return combined_df
        else:
            return self.clf.predict(len(data.Xtest))


class ReductionForecaster:
    def __init__(self, regression_model, window_length, strategy, **kwargs):
        self.model_params = kwargs
        self.acronym = "ReductionForecaster"
        self.clf = None
        self.regression_model = regression_model
        self.window_length = window_length
        self.strategy = strategy

    def fit(self, data: Dataset):
        if data.Xtrain.index.nlevels == 1:
            if data.Xtrain.index.freq is None:
                data.Xtrain = data.Xtrain.asfreq(pd.infer_freq(data.Xtrain.index))

            self.clf = make_reduction(self.regression_model(), window_length=self.window_length, strategy=self.strategy)
            self.clf = self.clf.fit(data.Xtrain)
        else:
            self.clf = dict()
            for i in data.Xtrain.index.get_level_values(0).unique():
                train_data = data.Xtrain.loc[i]
                if train_data.index.freq is None:
                    train_data = train_data.asfreq(pd.infer_freq(train_data.index))
                if self.window_length >= train_data.shape[0]:
                    print(f"Could not fit forecasting model for {i} because of insufficient train data "
                          f"(only {train_data.shape[0]} instances).")
                    continue
                try:
                    self.clf[i] = make_reduction(self.regression_model(), window_length=self.window_length, strategy=self.strategy)
                    self.clf[i] = self.clf[i].fit(train_data)
                except ValueError as e:
                    self.clf.pop(i)  # remove the failed fit
                    if train_data.shape[0] < 5:
                        print(f"Could not fit forecasting model for {i} because of insufficient train data "
                              f"(only {train_data.shape[0]} instances).")
                    else:
                        raise e

        return self

    def forecast(self, data):
        if isinstance(self.clf, dict):
            forecasts = []
            for i in self.clf.keys():
                if i not in data.Xtest.index:
                    print(f"Could not forecast forecasting model for {i} because of insufficient test data.")
                    continue
                n = len(data.Xtest.loc[i])
                fh = list(range(1,n + 1))
                forecast_i = self.clf[i].predict(fh)
                forecast_i['id'] = i
                forecasts.append(forecast_i)
            forecasts_df = pd.concat(forecasts)
            forecasts_df.set_index('id', append=True, inplace=True)
            combined_df = forecasts_df.reorder_levels(['id', None])
            combined_df.sort_index(inplace=True)
            return combined_df
        else:
            fh = list(range(1, len(data.Xtest) + 1))
            return self.clf.predict(fh).squeeze()

class GBForecaster(ReductionForecaster):
    def __init__(self, window_length, strategy, **kwargs):
        super().__init__(GradientBoostingRegressor, window_length, strategy, **kwargs)
        self.acronym = "GBForecaster"

