import abc
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error

from .constants import SEED
from ..data import Dataset


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
        self.seed = seed
        self.clf = GradientBoostingRegressor(
                         random_state=self.seed,
                         tol=1e-4)

    def update_lin_clf_alpha(self, alpha):
        pass

