import abc
import math
import os
import warnings
from typing import Optional, Callable

import joblib
import enum
import numpy as np
import pandas as pd
import pmdarima as pm
import multiprocessing
import time

import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.experimental import enable_iterative_imputer  # Don't remove this one, it is used!
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, TimeSeriesSplit

from ucimlrepo import fetch_ucirepo

from .utils.constants import SEED
from ..algorithms.transformers import LogTransformer

# MODEL_DIR = os.environ["MODEL_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
# DATA_DIR = "/home/loren/Code/Evaluation/data/datasets/"
NTHREADS = os.cpu_count()
base_path = os.path.dirname(os.path.abspath(__file__))


class Task(enum.Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTI_CLASSIFICATION = 3
    FORECASTING = 4


class Dataset:
    def __init__(self, task, name_suffix=""):
        self.task = task
        # self.model_dir = MODEL_DIR
        self.data_dir = DATA_DIR
        self.nthreads = NTHREADS

        self.name_suffix = name_suffix  # special parameters, name indication
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.Itrain = None
        self.Itest = None
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None
        self.Ival = None
        self.Xval = None
        self.yval = None
        self.other_params = {}
        self.missing_values = False
        self.acronym = None
        self.forecasting_horizon = None

    @property
    def name(self) -> str:
        return type(self).__name__ + self.name_suffix

    def model_params(self):
        return {}

    # def xgb_params(self, task, custom_params=None):
    #     if custom_params is None:
    #         custom_params = {}
    #     if task == Task.REGRESSION:
    #         params = {  # defaults
    #             "objective": "reg:squarederror",
    #             "eval_metric": "rmse",
    #             "tree_method": "hist",
    #             "seed": SEED,
    #             "nthread": self.nthreads,
    #         }
    #     elif task == Task.CLASSIFICATION:
    #         params = {  # defaults
    #             "objective": "binary:logistic",
    #             "eval_metric": "error",
    #             "tree_method": "hist",
    #             "seed": SEED,
    #             "nthread": self.nthreads,
    #         }
    #     elif task == Task.MULTI_CLASSIFICATION:
    #         params = {
    #             "num_class": 0,
    #             "objective": "multi:softmax",
    #             "tree_method": "hist",
    #             "eval_metric": "merror",
    #             "seed": SEED,
    #             "nthread": self.nthreads,
    #         }
    #     else:
    #         raise RuntimeError("unknown task")
    #     params.update(custom_params)
    #     return params
    #
    # def rf_params(self, custom_params):
    #     params = custom_params.copy()
    #     params["n_jobs"] = self.nthreads
    #     return params
    #
    # def extra_trees_params(self, custom_params):
    #     params = custom_params.copy()
    #     params["n_jobs"] = self.nthreads
    #     return params

    def load_dataset(self):  # populate X, y
        raise RuntimeError("not implemented")

    def to_float32(self):
        if self.X is not None: self.X = self.X.astype(np.float32)
        if self.y is not None: self.y = self.y.astype(np.float32)
        if self.Xtrain is not None: self.Xtrain = self.Xtrain.astype(np.float32)
        if self.ytrain is not None: self.ytrain = self.ytrain.astype(np.float32)
        if self.Xtest is not None: self.Xtest = self.Xtest.astype(np.float32)
        if self.ytest is not None: self.ytest = self.ytest.astype(np.float32)

    def train_and_test_set(self, seed=SEED, split_fraction=0.9, force=False):
        if self.X is None or self.y is None or force:
            raise RuntimeError("data not loaded")

        if self.Itrain is None or self.Itest is None or force:
            np.random.seed(seed)
            indices = np.random.permutation(self.X.shape[0])

            m = int(self.X.shape[0] * split_fraction)
            self.Itrain = indices[0:m]
            self.Itest = indices[m:]

        if self.Xtrain is None or self.ytrain is None or force:
            self.Xtrain = self.X.iloc[self.Itrain]
            self.ytrain = self.y[self.Itrain]

        if self.Xtest is None or self.ytest is None or force:
            self.Xtest = self.X.iloc[self.Itest]
            self.ytest = self.y[self.Itest]

    def generate_cross_validation_splits(self, nb_splits=2, nb_repeats=5, seed=SEED):
        if self.task == Task.REGRESSION:
            splitter = RepeatedKFold(n_splits=nb_splits, n_repeats=nb_repeats, random_state=seed)
            yield from splitter.split(self.X, self.y)
        elif self.task == Task.FORECASTING:
            if self.X.index.nlevels == 1:  # single instance per time step
                nb_timepoints = self.X.shape[0]
            else:
                nb_timepoints = self.X.index.levshape[1]
            p50 = int(nb_timepoints * 0.5)
            splitter = pm.model_selection.RollingForecastCV(h=self.forecasting_horizon, step=self.forecasting_horizon, initial=p50)
            # splitter = TimeSeriesSplit(n_splits=nb_splits)
            if self.X.index.nlevels == 1: # single instance per time step
                yield from splitter.split(self.X)
            else:  # multiple instances per time step
                # Find the instance with the most time steps
                counts = self.X.groupby('id').count()
                random_column_name = self.X.columns[0]
                max_id = counts[counts[random_column_name] == counts[random_column_name].max()].index[0]
                max_instance = self.X.loc[max_id]

                # Split the time steps of this instance
                # splitter = TimeSeriesSplit(n_splits=nb_splits)
                generator = splitter.split(max_instance)

                # Use the same split for all instances, but look at the exact dates, not just the n first time steps
                for (train_index, test_index) in generator:
                    X_reset = self.X.reset_index()
                    train_dates = max_instance.iloc[train_index].index
                    max_train_date = max(train_dates)
                    all_train_indices = np.empty(0, dtype=int)
                    all_test_indices = np.empty(0, dtype=int)
                    for i in X_reset['id'].unique():
                        train_indices = X_reset[X_reset['id'] == i].index[
                            X_reset[X_reset['id'] == i]['date'] <= max_train_date]
                        test_indices = X_reset[X_reset['id'] == i].index[
                            X_reset[X_reset['id'] == i]['date'] > max_train_date]
                        test_indices = test_indices[:self.forecasting_horizon]
                        all_train_indices = np.concatenate((all_train_indices, train_indices))
                        all_test_indices = np.concatenate((all_test_indices, test_indices))
                    yield all_train_indices, all_test_indices

        else:
            splitter = RepeatedStratifiedKFold(n_splits=nb_splits, n_repeats=nb_repeats, random_state=seed)
            yield from splitter.split(self.X, self.y)


    def cross_validation(self, Itrain, Itest, force=False):
        if self.X is None or (self.y is None and self.task != Task.FORECASTING):
            raise RuntimeError("data not loaded")

        if self.Itrain is None or self.Itest is None or force:
            self.Itrain = Itrain
            self.Itest = Itest

        if self.Xtrain is None or self.ytrain is None or force:
            self.Xtrain = self.X.iloc[self.Itrain]
            self.ytrain = self.y.iloc[self.Itrain] if self.task != Task.FORECASTING else self.Xtrain.squeeze()

        if self.Xtest is None or self.ytest is None or force:
            self.Xtest = self.X.iloc[self.Itest]
            self.ytest = self.y.iloc[self.Itest] if self.task != Task.FORECASTING else self.Xtest.squeeze()

    def split_validation_set(self, split_fraction=0.7, seed=SEED):
        if self.Xtrain is None or (self.ytrain is None and self.task != Task.FORECASTING):
            raise RuntimeError("training data not loaded")

        np.random.seed(seed)
        indices = np.random.permutation(self.Xtrain.index)

        m = int(self.Xtrain.shape[0] * split_fraction)
        self.Itrain = indices[0:m]
        self.Ival = indices[m:]

        # First validation set, because otherwise the training set has already changed!
        self.Xval = self.Xtrain.loc[self.Ival]
        self.yval = self.ytrain.loc[self.Ival] if self.task != Task.FORECASTING else self.Xval.squeeze()

        self.Xtrain = self.Xtrain.loc[self.Itrain]
        self.ytrain = self.ytrain.loc[self.Itrain] if self.task != Task.FORECASTING else self.Xtrain.squeeze()

    def _load_ucirepo(self, name, data_id, force=False) -> (pd.DataFrame, pd.DataFrame):
        if not os.path.exists(f"{self.data_dir}/{name}.h5") or force:
            print(f"loading {name} with fetch_ucirepo")
            dataset = fetch_ucirepo(id=data_id)
            X = dataset.data.features
            y = dataset.data.targets
            X.to_hdf(f"{self.data_dir}/{name}.h5", key="X", complevel=9)
            y.to_hdf(f"{self.data_dir}/{name}.h5", key="y", complevel=9)

        print(f"loading {name} h5 file")
        X = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="X")
        y = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="y")

        return X, y

    def _load_openml(self, name, data_id, force=False, X_type=np.float32, y_type=np.float32):
        if not os.path.exists(f"{self.data_dir}/{name}.h5") or force:
            print(f"loading {name} with fetch_openml")
            X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True)
            X = X.astype(X_type)
            y = y.astype(y_type)
            X.to_hdf(f"{self.data_dir}/{name}.h5", key="X", complevel=9)
            y.to_hdf(f"{self.data_dir}/{name}.h5", key="y", complevel=9)

        print(f"loading {name} h5 file")
        X = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="X")
        y = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="y")

        return X, y

    def _load_pmdarima(self, name, data_id: Callable, index2date=False, date_format=None, force=False):
        if not os.path.exists(f"{self.data_dir}/{name}.h5") or force:
            print(f"loading {name} with {data_id}()")
            X = data_id(as_series=True)
            if index2date:
                if date_format is None:
                    X.index = pd.to_datetime(X.index)
                else:
                    X.index = pd.to_datetime(X.index, format=date_format)
            X.to_hdf(f"{self.data_dir}/{name}.h5", key="X", complevel=9)
            
        print(f"loading {name} h5 file")
        # noinspection PyUnresolvedReferences
        X = pd.read_hdf(f"{self.data_dir}/{name}.h5", key="X")
        # noinspection PyUnresolvedReferences
        return X

    def minmax_normalize(self):
        if self.X is None:
            raise RuntimeError("data not loaded")

        X = self.X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df

    def minmax_normalize_after_split(self):
        if self.Xtrain is None:
            raise RuntimeError("data not split in train and test sets")

        # Xtrain = self.Xtrain.values
        min_max_scaler = preprocessing.MinMaxScaler()
        Xtrain_scaled = min_max_scaler.fit_transform(self.Xtrain.values)
        self.Xtrain = pd.DataFrame(Xtrain_scaled, columns=self.Xtrain.columns, index=self.Xtrain.index)
        if self.Xval is not None:
            Xval_scaled = min_max_scaler.transform(self.Xval.values)
            self.Xval = pd.DataFrame(Xval_scaled, columns=self.Xval.columns, index=self.Xval.index)
        Xtest_scaled = min_max_scaler.transform(self.Xtest.values)
        self.Xtest = pd.DataFrame(Xtest_scaled, columns=self.Xtest.columns, index=self.Xtest.index)

    def encode_object_types(self):
        cat_columns = self.X.select_dtypes(['object']).columns
        self.X.loc[:, cat_columns] = self.X[cat_columns].astype('category').apply(lambda x: x.cat.codes)

    def impute_missing_values(self):
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=10, random_state=SEED)
        if self.Xtrain is not None and self.Xtest is not None:
            imp.fit(self.Xtrain)
            self.Xtrain = pd.DataFrame(imp.transform(self.Xtrain), columns=self.Xtrain.columns, index=self.Xtrain.index)
            if self.Xval is not None:
                self.Xval = pd.DataFrame(imp.transform(self.Xval), columns=self.Xval.columns, index=self.Xval.index)
            self.Xtest = pd.DataFrame(imp.transform(self.Xtest), columns=self.Xtest.columns, index=self.Xtest.index)
        else:
            self.X = pd.DataFrame(imp.fit_transform(self.X), columns=self.X.columns, index=self.X.index)

    def encode_y(self):
        if self.y is None:
            raise RuntimeError("data not loaded")

        le = preprocessing.LabelEncoder()
        self.y = pd.Series(le.fit_transform(self.y.values.ravel()), index=self.y.index)

    def transform_target_custom(self, transformer):
        if (self.y is None and self.task != Task.FORECASTING) or (self.X is None and self.task == Task.FORECASTING):
            raise RuntimeError("data not loaded")
        if self.ytrain is None or self.ytest is None:
            raise RuntimeError("train and test sets not loaded")

        self.ytrain = pd.Series(transformer.fit_transform(self.ytrain.values.reshape(-1, 1)).ravel(), index=self.ytrain.index)
        if self.task == Task.FORECASTING:
            self.Xtrain = self.ytrain.to_frame(name=self.Xtrain.columns[0])
        if self.yval is not None:
            self.yval = pd.Series(transformer.transform(self.yval.values.reshape(-1, 1)).ravel(), index=self.yval.index)
            if self.task == Task.FORECASTING:
                self.Xval = self.yval.to_frame(name=self.Xval.columns[0])
        self.ytest = pd.Series(transformer.transform(self.ytest.values.reshape(-1, 1)).ravel(), index=self.ytest.index)
        if self.task == Task.FORECASTING:
            self.Xtest = self.ytest.to_frame(name=self.Xtest.columns[0])
        self.other_params["target_transformer"] = transformer

    def transform_features_custom(self, transformer, condition=None):
        if self.X is None:
            raise RuntimeError("data not loaded")
        if self.Xtrain is None or self.Xtest is None:
            raise RuntimeError("train and test sets not loaded")

        if condition is not None:
            columns = condition(self.Xtrain)
        else:
            columns = self.Xtrain.columns

        if isinstance(transformer, LogTransformer):
            min_train = self.Xtrain.values.min()
            min_val = self.Xval.values.min()
            min_test = self.Xtest.values.min()
            min_all = min(min_train, min_val, min_test)
            if min_all <= -transformer.offset:
                transformer.offset = math.ceil(-min_all)

        self.Xtrain.loc[:, columns] = transformer.fit_transform(self.Xtrain.loc[:, columns].values)
        if self.Xval is not None:
            self.Xval.loc[:, columns] = transformer.transform(self.Xval.loc[:, columns].values)
        # catch warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.Xtest.loc[:, columns] = transformer.transform(self.Xtest.loc[:, columns].values)
        # self.Xtrain = transformer.fit_transform(self.Xtrain.values)
        # if self.Xval is not None:
        #     self.Xval = transformer.transform(self.Xval.values)
        # self.Xtest = transformer.transform(self.Xtest.values)
        self.other_params["feature_transformer"] = transformer

    def transform_contextual(self):
        if self.other_params['contextual_transform_feature'] is None:
            raise ValueError("No feature for contextual transformation provided")
        feature_name = self.other_params['contextual_transform_feature']
        train_values = self.X.iloc[self.Itrain][feature_name]
        self.Xtrain = self.Xtrain.drop(feature_name, axis=1)
        if self.task == Task.FORECASTING:
            self.Xtrain = self.Xtrain.div(train_values, axis=0)
            self.ytrain = self.Xtrain.squeeze()
        else:
            self.ytrain = self.ytrain / train_values
        if self.Xval is not None:
            val_values = self.X.iloc[self.Ival][feature_name]
            self.Xval = self.Xval.drop(feature_name, axis=1)
            if self.task == Task.FORECASTING:
                self.Xval = self.Xval.div(val_values, axis=0)
                self.yval = self.Xval.squeeze()
            else:
                self.yval = self.yval / val_values
        test_values = self.X.iloc[self.Itest][feature_name]
        self.Xtest = self.Xtest.drop(feature_name, axis=1)
        if self.task == Task.FORECASTING:
            self.Xtest = self.Xtest.div(test_values, axis=0)
            self.ytest = self.Xtest.squeeze()
        else:
            self.ytest = self.ytest / test_values

    def inverse_contextual_transform(self, predictions):
        if self.other_params['contextual_transform_feature'] is None:
            raise ValueError("No feature for contextual transformation provided")
        feature_name = self.other_params['contextual_transform_feature']
        test_values = self.X.iloc[self.Itest][feature_name]
        return predictions * test_values.values

    def discretize(self, nb_bins: int = 10, mode: str = "equal_width") -> None:
        """
        Discretize the entire dataset. The mode "equal_width" has evenly spaced bins with equal width, calculated from
        the maximum and minimum of the corresponding column. The mode "equal_height" has bins with more or less the same
        height by dividing the values in the corresponding column evenly across the bins.
        :param nb_bins: the number of bins for each column in the dataset
        :param mode: "equal_width" (default) or "equal_height"
        :return: None
        """
        def get_bins(col: np.array) -> np.array:
            if mode == "equal_width":
                return np.linspace(col.min(), col.max(), nb_bins)
            elif mode == "equal_height":
                npt = len(col)
                return np.interp(np.linspace(0, npt, nb_bins),
                                 np.arange(npt),
                                 np.sort(col))
            else:
                raise Exception("Invalid binning mode for discretizing the dataset")

        def discretize_col(c: int):
            time.sleep(1)
            col = self.X.iloc[:, c]
            bins = get_bins(col)
            return np.digitize(col, bins)

        outputs = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
            joblib.delayed(discretize_col)(c) for c in range(self.X.shape[1]))

        for i, c in enumerate(self.X.columns):
            self.X[c] = outputs[i]

        self.X = self.X.astype(np.float32)

def _rmse_metric(self, model, best_m):
    yhat = model.predict(self.dtest, output_margin=True)
    m = metrics.mean_squared_error(yhat, self.ytest)
    m = np.sqrt(m)
    return m if best_m is None or m < best_m else best_m


def _acc_metric(self, model, best_m):
    yhat = model.predict(self.dtest, output_margin=True)
    m = metrics.accuracy_score(yhat > 0.0, self.ytest)
    return m if best_m is None or m > best_m else best_m


def _multi_acc_metric(self, model, best_m):
    yhat = model.predict(self.dtest)
    m = metrics.accuracy_score(yhat, self.ytest)
    return m if best_m is None or m > best_m else best_m


class MulticlassDataset(Dataset):
    def __init__(self, num_classes):
        super().__init__(Task.MULTI_CLASSIFICATION)
        self.num_classes = num_classes

    def get_class(self, cls):
        self.load_dataset()
        mask = np.zeros(self.X.shape[0])
        if isinstance(cls, tuple):
            for c in cls:
                if c not in range(self.num_classes):
                    raise ValueError(f"invalid class {c}")
                mask = np.bitwise_or(mask, self.y == c)
        else:
            if cls not in range(self.num_classes):
                raise ValueError(f"invalid class {cls}")
            mask = (self.y == cls)
        X = self.X.loc[mask, :]
        y = self.y[mask]
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        return X, y


class MultiBinClassDataset(Dataset):
    def __init__(self, multiclass_dataset, class1, class2):
        super().__init__(Task.CLASSIFICATION)
        self.multi_dataset = multiclass_dataset
        if not isinstance(self.multi_dataset, MulticlassDataset):
            raise ValueError("not a multiclass dataset:",
                             self.multi_dataset.name())
        if class1 not in range(self.multi_dataset.num_classes):
            raise ValueError("invalid class1")
        if class2 not in range(self.multi_dataset.num_classes):
            raise ValueError("invalid class2")
        if class1 >= class2:
            raise ValueError("take class1 < class2")
        self.class1 = class1
        self.class2 = class2

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.multi_dataset.load_dataset()
            X, y = self.multi_dataset.get_class((self.class1, self.class2))
            self.X = X
            self.y = (y == self.class2)

    def name(self):
        return f"{self.name}{self.class1}v{self.class2}"


class Abalone(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "ABA"
        # self.name = "abalone"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("abalone", data_id=1)
            self.encode_object_types()
            # self.minmax_normalize()

            self.y = self.y.squeeze()


class AutoMPG(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.missing_values = True
        self.acronym = "AMPG"
        # self.name = "autompg"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("AutoMPG", data_id=9)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class BikeSharing(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "BS"
        # self.name = "bikesharing"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("BikeSharing", data_id=275)
            self.y = self.y.squeeze()
            self.encode_object_types()
            # self.minmax_normalize()


class BikeSharingFull(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "BS"
        # self.name = "bikesharingfull"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading bikesharingfull h5 file")
            self.X = pd.read_hdf(f"{self.data_dir}/BikeSharingFull.h5", key='X')
            self.y = pd.read_hdf(f"{self.data_dir}/BikeSharingFull.h5", key='y')
            self.encode_object_types()
            # self.minmax_normalize()


class BikeSharingNormalized(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "BS"
        # self.name = "bikesharingnormalized"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading bikesharingfull h5 file")
            X = pd.read_hdf(f"{self.data_dir}/BikeSharingFull.h5", key='X')

            # noinspection PyUnresolvedReferences
            total_users = X['casual'] + X['registered']

            # noinspection PyUnresolvedReferences
            self.X = X.drop(['casual', 'registered'], axis=1)
            y = pd.read_hdf(f"{self.data_dir}/BikeSharingFull.h5", key='y')

            # noinspection PyUnresolvedReferences
            self.y = y/total_users

            self.encode_object_types()
            # self.minmax_normalize()


class Challenger(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "CH"
        # self.name = "challenger"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Challenger", data_id=92)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class CombinedCyclePowerPlant(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "CCPP"
        # self.name = "powerplant"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("CombinedCyclePowerPlant", data_id=294)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class ComputerHardware(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "CH"
        # self.name = "computerhardware"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("ComputerHardware", data_id=29)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class ConcreteCompressingStrength(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "CCS"
        # self.name = "concrete"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("ConcreteCompressingStrength", data_id=165)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class EnergyEfficiency1(Dataset):
    """
    Energy efficiency dataset from UCI repository, predicting heating load (Y1).
    """
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "EF1"
        # self.name = "energyefficiency1normalized1"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("EnergyEfficiency", data_id=242)
            self.y = self.y['Y1'].squeeze()
            # self.minmax_normalize()


class EnergyEfficiency1Normalized2(Dataset):
    """
    Energy efficiency dataset from UCI repository, predicting heating load (Y1), normalized with the surface area
    feature (X2).
    """
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "EF1"
        # self.name = "energyefficiency1normalized2"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("EnergyEfficiency", data_id=242)
            self.y = self.y['Y1'].squeeze()
            self.other_params['contextual_transform_feature'] = 'X2'


class EnergyEfficiency1Normalized3(Dataset):
    """
    Energy efficiency dataset from UCI repository, predicting heating load (Y1), normalized with the wall area
    feature (X3).
    """
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "EF1"
        # self.name = "energyefficiency1normalized3"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("EnergyEfficiency", data_id=242)
            self.y = self.y['Y1'].squeeze()
            self.other_params['contextual_transform_feature'] = 'X3'

class EnergyEfficiency1Normalized4(Dataset):
    """
    Energy efficiency dataset from UCI repository, predicting heating load (Y1), normalized with the roof area
    feature (X4).
    """
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "EF1"
        # self.name = "energyefficiency1normalized4"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("EnergyEfficiency", data_id=242)
            self.y = self.y['Y1'].squeeze()
            self.other_params['contextual_transform_feature'] = 'X4'

class EnergyEfficiency1Normalized5(Dataset):
    """
    Energy efficiency dataset from UCI repository, predicting heating load (Y1), normalized with the overall height
    feature (X5).
    """
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "EF1"
        # self.name = "energyefficiency1normalized5"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("EnergyEfficiency", data_id=242)
            self.y = self.y['Y1'].squeeze()
            self.other_params['contextual_transform_feature'] = 'X5'


class EnergyEfficiency2(Dataset):
    """
    Energy efficiency dataset from UCI repository, predicting cooling load (Y2).
    """
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "EF2"
        # self.name = "energyefficiency2"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("EnergyEfficiency", data_id=242)
            self.y = self.y['Y2'].squeeze()
            # self.minmax_normalize()


class HeartFailure(Dataset):
    def __init__(self):
        super().__init__(Task.CLASSIFICATION)
        self.acronym = "HF"
        # self.name = "heartfailure"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("HeartFailure", data_id=519)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class Iris(MulticlassDataset):
    def __init__(self):
        super().__init__(Task.MULTI_CLASSIFICATION)
        self.acronym = "IR"
        # self.name = "iris"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Iris", data_id=53)
            self.y = self.y.squeeze()
            self.encode_y()
            # self.minmax_normalize()


class LiverDisorder(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "LD"
        # self.name = "liverdisorder"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("LiverDisorder", data_id=60)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class Obesity(Dataset):
    def __init__(self):
        super().__init__(Task.MULTI_CLASSIFICATION)
        self.acronym = "OB"
        # self.name = "obesity"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Obesity", data_id=544)
            self.y = self.y.squeeze()
            self.encode_object_types()
            self.encode_y()
            # self.minmax_normalize()


class OnlineNewsPopularity(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "ONP"
        # self.name = "onlinenewspopularity"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("OnlineNewsPopularity", data_id=332)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class OnlineNewsPopularityFull(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "ONP"
        # self.name = "onlinenewspopularityfull"

    def load_dataset(self):
        if self.X is None or self.y is None:
            full_dataset = fetch_ucirepo(id=332)
            self.X = full_dataset.data.original.drop(columns=['url', ' shares'])
            self.y = full_dataset.data.targets.squeeze()
            # self.minmax_normalize()


class OnlineNewsPopularityNormalized(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "ONP"
        # self.name = "onlinenewspopularitynormalized"

    def load_dataset(self):
        if self.X is None or self.y is None:
            full_dataset = fetch_ucirepo(id=332)
            self.X = full_dataset.data.original.drop(columns=['url', ' shares'])
            self.y = full_dataset.data.targets.squeeze()
            self.other_params['contextual_transform_feature'] = ' timedelta'
            # self.y = y/self.X[' timedelta']
            # self.X = self.X.drop(columns=[' timedelta'])
            # self.minmax_normalize()


class Parkinsons1(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "PK1"
        # self.name = "parkinsons1"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Parkinsons", data_id=189)
            self.y = self.y['motor_UPDRS'].squeeze()
            # self.minmax_normalize()


class Parkinsons2(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "PK2"
        # self.name = "parkinsons2"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Parkinsons", data_id=189)
            self.y = self.y['total_UPDRS'].squeeze()
            # self.minmax_normalize()


class RealEstateValuation(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "REV"
        # self.name = "realestatevaluation"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("RealEstateValuation", data_id=477)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class Servo(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "SRV"
        # self.name = "servo"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Servo", data_id=87)
            self.y = self.y.squeeze()
            self.encode_object_types()
            # self.minmax_normalize()


class WineQuality(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "WQ"
        # self.name = "winequality"

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("WineQuality", data_id=186)
            self.y = self.y.squeeze()
            # self.minmax_normalize()


class YouTube(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "YT"
        # self.name = "youtube"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading youtube h5 file")
            df = pd.read_hdf(f"{self.data_dir}/youtube.h5", key='dataset')

            # noinspection PyUnresolvedReferences
            self.y = df.pop('views')
            self.X = df
            # self.minmax_normalize()


class YouTubeNormalized(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "YT"
        # self.name = "youtubenormalized"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading youtube h5 file")
            df = pd.read_hdf(f"{self.data_dir}/youtube.h5", key='dataset')

            # noinspection PyUnresolvedReferences
            df_publish = df[['publish_year', 'publish_month', 'publish_day']]
            df_publish = df_publish.rename(columns=
                                           {'publish_year': 'year', 'publish_month': 'month', 'publish_day': 'day'})
            df_publish = df_publish.astype(int)
            publish_dates = pd.to_datetime(df_publish, format='%Y%m%d')
            max_date = publish_dates.max()
            time_online = max_date - publish_dates

            # noinspection PyUnresolvedReferences
            self.y = df.pop('views')
            # self.y = df.pop('views') / (time_online.dt.days + 1)
            # noinspection PyUnresolvedReferences
            self.X = df.drop(['publish_year', 'publish_month', 'publish_day'], axis=1)
            self.X['time_online'] = time_online.dt.days + 1
            self.other_params['contextual_transform_feature'] = 'time_online'
            # self.minmax_normalize()


class YouTubeLg(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "YT"
        # self.name = "youtubelg"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading youtube h5 file")
            df = pd.read_hdf(f"{self.data_dir}/youtubelg.h5", key='dataset')

            # noinspection PyUnresolvedReferences
            self.y = df.pop('views')
            self.X = df
            # self.minmax_normalize()


class YouTubePlus(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "YT"
        self.missing_values = True
        # self.name = "youtubeplus"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading youtube h5 file")
            df = pd.read_hdf(f"{self.data_dir}/youtube+.h5", key='dataset')

            # noinspection PyUnresolvedReferences
            self.y = df.pop('views')
            self.X = df
            # self.minmax_normalize()


class YouTubeLgMin(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.acronym = "YT"
        # self.name = "youtubelgmin"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading youtube h5 file")
            df = pd.read_hdf(f"{self.data_dir}/youtubelg-.h5", key='dataset')

            # noinspection PyUnresolvedReferences
            self.y = df.pop('views')
            self.X = df
            # self.minmax_normalize()


class CoffeeSalesDaily(Dataset):
    def __init__(self):
        super().__init__(Task.FORECASTING)
        self.acronym = "CS"
        # self.name = "coffeesalesdaily"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading CoffeeSales h5 file")
            df = pd.read_hdf(f"{self.data_dir}/CoffeeSales.h5", key='daily_sales')

            self.y = None
            # noinspection PyUnresolvedReferences
            self.X = df[['Total']]


class CoffeeSalesMonthly(Dataset):
    def __init__(self):
        super().__init__(Task.FORECASTING)
        self.acronym = "CS"
        self.forecasting_horizon = 1
        # self.name = "coffeesalesmonthly"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading CoffeeSales h5 file")
            df = pd.read_hdf(f"{self.data_dir}/CoffeeSales.h5", key='monthly_sales')

            self.y = None
            # noinspection PyUnresolvedReferences
            self.X = df[['Total']]


class CoffeeSalesMonthlyNormalized(Dataset):
    def __init__(self):
        super().__init__(Task.FORECASTING)
        self.acronym = "CS"
        self.forecasting_horizon = 1
        # self.name = "coffeesalesmonthlynormalized"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading CoffeeSales h5 file")
            df = pd.read_hdf(f"{self.data_dir}/CoffeeSales.h5", key='monthly_sales')

            self.y = None
            # noinspection PyUnresolvedReferences
            self.X = df[['Total']]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # noinspection PyUnresolvedReferences
                self.X['Nb_days'] = df.index.days_in_month
                # self.X.reset_index(drop=True, inplace=True)
            self.other_params['contextual_transform_feature'] = 'Nb_days'

class SolarEnergyProductionDaily(Dataset):
    def __init__(self):
        super().__init__(Task.FORECASTING)
        self.acronym = "SEP"
        self.forecasting_horizon = 52
        # self.name = "solarenergyproductiondaily"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading SolarEnergyProduction h5 file")
            df = pd.read_hdf(f"{self.data_dir}/SolarEnergyProduction.h5", key='daily')

            self.y = None
            # noinspection PyUnresolvedReferences
            self.X = df[['kWh']]

    def model_params(self):
        return {'seasonal': 'add'}

class SolarEnergyProductionDailyNormalized(Dataset):
    def __init__(self):
        super().__init__(Task.FORECASTING)
        self.acronym = "SEP"
        self.forecasting_horizon = 52
        # self.name = "solarenergyproductiondailynormalized"

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading SolarEnergyProduction h5 file")
            df = pd.read_hdf(f"{self.data_dir}/SolarEnergyProduction.h5", key='daily')

            self.y = None
            # noinspection PyUnresolvedReferences
            self.X = df[['kWh', 'daylight_hours']]
            # # Find the instance with the most time steps
            # counts = self.X.groupby('id').count()
            # random_column_name = self.X.columns[0]
            # max_id = counts[counts[random_column_name] == counts[random_column_name].max()].index[0]
            # self.X = self.X.loc[max_id]
            self.other_params['contextual_transform_feature'] = 'daylight_hours'

    def model_params(self):
        return {'seasonal': 'add'}

class SunspotsMonthly(Dataset):
    def __init__(self):
        super().__init__(Task.FORECASTING)
        self.acronym = "SS"
        self.forecasting_horizon = 12

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X = self._load_pmdarima("SunSpots", pm.datasets.load_sunspots, index2date=True, force=True)
            # noinspection PyUnresolvedReferences
            self.X.index  = self.X.index.to_period('M')
            self.y = None

    def model_params(self):
        return {'seasonal': 'add'}
        # return {}

class SunspotsMonthlyNormalized(Dataset):
    def __init__(self):
        super().__init__(Task.FORECASTING)
        self.acronym = "CS"
        self.forecasting_horizon = 12

    def load_dataset(self):
        if self.X is None or self.y is None:
            # noinspection PyUnresolvedReferences
            self.X: pd.DataFrame = self._load_pmdarima("SunSpots", pm.datasets.load_sunspots, index2date=True, force=True).to_frame()

            self.X.index = self.X.index.to_period('M')
            self.y = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.X['Nb_days'] = self.X.index.days_in_month
                # self.X.reset_index(drop=True, inplace=True)
            self.other_params['contextual_transform_feature'] = 'Nb_days'

imbalanced_distribution_datasets = {
            "autompg": AutoMPG,  # Missing values
            "bikesharing": BikeSharing,  # Does not converge
            "powerplant": CombinedCyclePowerPlant,
            "concrete": ConcreteCompressingStrength,
            "energyefficiency1": EnergyEfficiency1,
            "energyefficiency2": EnergyEfficiency2,
            "liverdisorder": LiverDisorder,
            "onlinenewspopularity": OnlineNewsPopularity,  # Does not converge
            "realestatevaluation": RealEstateValuation,
            "servo": Servo,
            }

forecasting_datasets = {
    "coffeesalesdaily": CoffeeSalesDaily,
    "coffeesalesmonthly": CoffeeSalesMonthly,
    "coffeesalesmonthlynormalized": CoffeeSalesMonthlyNormalized,
    "solarenergyproductiondaily": SolarEnergyProductionDaily,
    "solarenergyproductiondailynormalized": SolarEnergyProductionDailyNormalized,
    "sunspotsmonthly": SunspotsMonthly,
    "sunspotsmonthlynormalized": SunspotsMonthlyNormalized,
}

datasets = {"abalone": Abalone,
            "autompg": AutoMPG,  # Missing values
            "bikesharing": BikeSharing,  # Does not converge
            "powerplant": CombinedCyclePowerPlant,
            # "challenger": Challenger, # Does not converge
            "coffeesalesdaily": CoffeeSalesDaily,
            "coffeesalesmonthly": CoffeeSalesMonthly,
            "coffeesalesmonthlynormalized": CoffeeSalesMonthlyNormalized,
            # "computerhardware": ComputerHardware, # What is the target?
            "concrete": ConcreteCompressingStrength,
            "energyefficiency1": EnergyEfficiency1,
            "energyefficiency1normalized2": EnergyEfficiency1Normalized2,
            "energyefficiency1normalized3": EnergyEfficiency1Normalized3,
            "energyefficiency1normalized4": EnergyEfficiency1Normalized4,
            "energyefficiency1normalized5": EnergyEfficiency1Normalized5,
            "energyefficiency2": EnergyEfficiency2,
            # "heartfailure": HeartFailure, # Classification
            # "iris": Iris, # Classification
            "liverdisorder": LiverDisorder,
            # "obesity": Obesity(), # Classification
            # "parkinsons1": Parkinsons1, # Does not converge
            # "parkinsons2": Parkinsons2, # Does not converge
            "onlinenewspopularity": OnlineNewsPopularity,  # Does not converge
            "onlinenewspopularityfull": OnlineNewsPopularityFull,
            "onlinenewspopularitynormalized": OnlineNewsPopularityNormalized,
            "realestatevaluation": RealEstateValuation,
            "servo": Servo,
            "solarenergyproductiondaily": SolarEnergyProductionDaily,
            "solarenergyproductiondailynormalized": SolarEnergyProductionDailyNormalized,
            "sunspotsmonthly": SunspotsMonthly,
            "sunspotsmonthlynormalized": SunspotsMonthlyNormalized,
            "winequality": WineQuality,
            "youtube": YouTube,
            "youtubenormalized": YouTubeNormalized,
            "youtubelg": YouTubeLg,
            "youtube+": YouTubePlus,
            "youtubelg-": YouTubeLgMin,
            }

