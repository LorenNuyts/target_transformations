import os
from typing import Optional

import joblib
import enum
import numpy as np
import pandas as pd
import multiprocessing
import time

import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import QuantileTransformer
# from sklearn.datasets import fetch_openml
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
# from sklearn.neighbors import KDTree
# from sklearn.preprocessing import LabelEncoder
# from functools import partial

# import xgboost as xgb
from ucimlrepo import fetch_ucirepo

from data.utils import SEED

# MODEL_DIR = os.environ["MODEL_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
# DATA_DIR = "/home/loren/Code/Evaluation/data/datasets/"
NTHREADS = os.cpu_count()
base_path = os.path.dirname(os.path.abspath(__file__))


class Task(enum.Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTI_CLASSIFICATION = 3


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

    def name(self):
        return type(self).__name__

    def xgb_params(self, task, custom_params=None):
        if custom_params is None:
            custom_params = {}
        if task == Task.REGRESSION:
            params = {  # defaults
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "seed": SEED,
                "nthread": self.nthreads,
            }
        elif task == Task.CLASSIFICATION:
            params = {  # defaults
                "objective": "binary:logistic",
                "eval_metric": "error",
                "tree_method": "hist",
                "seed": SEED,
                "nthread": self.nthreads,
            }
        elif task == Task.MULTI_CLASSIFICATION:
            params = {
                "num_class": 0,
                "objective": "multi:softmax",
                "tree_method": "hist",
                "eval_metric": "merror",
                "seed": SEED,
                "nthread": self.nthreads,
            }
        else:
            raise RuntimeError("unknown task")
        params.update(custom_params)
        return params

    def rf_params(self, custom_params):
        params = custom_params.copy()
        params["n_jobs"] = self.nthreads
        return params

    def extra_trees_params(self, custom_params):
        params = custom_params.copy()
        params["n_jobs"] = self.nthreads
        return params

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

    def cross_validation(self, Itrain, Itest, force=False):
        if self.X is None or self.y is None:
            raise RuntimeError("data not loaded")

        if self.Itrain is None or self.Itest is None or force:
            self.Itrain = Itrain
            self.Itest = Itest

        if self.Xtrain is None or self.ytrain is None or force:
            self.Xtrain = self.X.iloc[self.Itrain]
            self.ytrain = self.y.iloc[self.Itrain]

        if self.Xtest is None or self.ytest is None or force:
            self.Xtest = self.X.iloc[self.Itest]
            self.ytest = self.y.iloc[self.Itest]

    def split_validation_set(self, split_fraction=0.5, seed=SEED):
        if self.Xtrain is None or self.ytrain is None:
            raise RuntimeError("training data not loaded")

        np.random.seed(seed)
        indices = np.random.permutation(self.Xtrain.shape[0])

        m = int(self.Xtrain.shape[0] * split_fraction)
        self.Itrain = indices[0:m]
        self.Ival = indices[m:]

        # First validation set, because otherwise the training set has already changed!
        self.Xval = self.Xtrain.iloc[self.Ival]
        self.yval = self.ytrain.iloc[self.Ival]

        self.Xtrain = self.Xtrain.iloc[self.Itrain]
        self.ytrain = self.ytrain.iloc[self.Itrain]

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

    def minmax_normalize(self):
        if self.X is None:
            raise RuntimeError("data not loaded")

        X = self.X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df

    def encode_object_types(self):
        cat_columns = self.X.select_dtypes(['object']).columns
        self.X.loc[:, cat_columns] = self.X[cat_columns].astype('category').apply(lambda x: x.cat.codes)

    def impute_missing_values(self):
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=10, random_state=SEED)
        if self.Xtrain is not None and self.Xtest is not None:
            imp.fit(self.Xtrain)
            self.Xtrain = imp.transform(self.Xtrain)
            if self.Xval is not None:
                self.Xval = imp.transform(self.Xval)
            self.Xtest = imp.transform(self.Xtest)
        else:
            self.X = imp.fit_transform(self.X)

    def encode_y(self):
        if self.y is None:
            raise RuntimeError("data not loaded")

        le = preprocessing.LabelEncoder()
        self.y = pd.Series(le.fit_transform(self.y.values.ravel()), index=self.y.index)

    # def normalize_y(self):
    #     if self.y is None:
    #         raise RuntimeError("data not loaded")
    #     if self.ytrain is None or self.ytest is None:
    #         raise RuntimeError("train and test sets not loaded")
    #
    #     ytrain_mean = self.ytrain.mean()
    #     ytrain_std = self.ytrain.std()
    #
    #     self.ytrain = (self.ytrain - ytrain_mean) / ytrain_std
    #     if self.yval is not None:
    #         self.yval = (self.yval - ytrain_mean) / ytrain_std
    #     self.ytest = (self.ytest - ytrain_mean) / ytrain_std
    #
    #     self.other_params["ytrain_mean"] = ytrain_mean
    #     self.other_params["ytrain_std"] = ytrain_std

    # def log_transform_target(self):
    #     if self.y is None:
    #         raise RuntimeError("data not loaded")
    #     if self.ytrain is None or self.ytest is None:
    #         raise RuntimeError("train and test sets not loaded")
    #
    #     self.ytrain = np.log(self.ytrain)
    #     if self.yval is not None:
    #         self.yval = np.log(self.yval)
    #     # self.ytest = np.log(self.ytest)

    def transform_target_custom(self, transformer):
        if self.y is None:
            raise RuntimeError("data not loaded")
        if self.ytrain is None or self.ytest is None:
            raise RuntimeError("train and test sets not loaded")

        self.ytrain = transformer.fit_transform(self.ytrain.values.reshape(-1, 1)).ravel()
        if self.yval is not None:
            self.yval = transformer.transform(self.yval.values.reshape(-1, 1)).ravel()
        # self.ytest = transformer.transform(self.ytest.values.reshape(-1, 1)).ravel()
        self.other_params["target_transformer"] = transformer

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
        return f"{super().name()}{self.class1}v{self.class2}"


# class Calhouse(Dataset):
#     def __init__(self):
#         super().__init__(Task.REGRESSION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("calhouse", data_id=537)
#             self.y = np.log(self.y)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_rmse_metric, self), "rmse", custom_params, naming_args=naming_args)
#
#
# class Allstate(Dataset):
#     dataset_name = "allstate.h5"
#
#     def __init__(self):
#         super().__init__(Task.REGRESSION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             allstate_data_path = os.path.join(self.data_dir, Allstate.dataset_name)
#             data = pd.read_hdf(allstate_data_path)
#             self.X = data.drop(columns=["loss"])
#             self.y = data.loss
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_rmse_metric, self), "rmse", custom_params, naming_args=naming_args)
#
#
# class Covtype(Dataset):
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("covtype", data_id=1596)
#             self.y = (self.y == 2)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class CovtypeNormalized(Covtype):
#     def __init__(self):
#         super().__init__()
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             super().load_dataset()
#             self.minmax_normalize()
#
#
# class Higgs(Dataset):
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             higgs_data_path = os.path.join(self.data_dir, "higgs.h5")
#             self.X = pd.read_hdf(higgs_data_path, "X")
#             self.y = pd.read_hdf(higgs_data_path, "y")
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# # class LargeHiggs(Dataset):
# #     def __init__(self):
# #         super().__init__()
# #
# #     def load_dataset(self):
# #         if self.X is None or self.y is None:
# #             higgs_data_path = os.path.join(self.data_dir, "higgs_large.h5")
# #             data = pd.read_hdf(higgs_data_path)
# #             self.y = data[0]
# #             self.X = data.drop(columns=[0])
# #             columns = [f"a{i}" for i in range(self.X.shape[1])]
# #             self.X.columns = columns
# #             self.minmax_normalize()
# #
# #     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
# #         if naming_args is None:
# #             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
# #         custom_params = {}
# #         return super()._get_xgb_model(num_trees, tree_depth,
# #                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class Mnist(MulticlassDataset):
#     def __init__(self):
#         super().__init__(num_classes=10)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("mnist", data_id=554)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {
#             "num_class": self.num_classes,
#         }
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_multi_acc_metric, self), "macc", custom_params,
#                                       naming_args)
#
#
# # class MnistNormalized(Mnist):
# #    def __init__(self):
# #        super().__init__()
# #
# #    def load_dataset(self):
# #        if self.X is None or self.y is None:
# #            super().load_dataset()
# #            self.minmax_normalize()
#
# class MnistBinClass(MultiBinClassDataset):
#     def __init__(self, class1, class2):
#         super().__init__(Mnist(), class1, class2)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {
#             "subsample": 0.5,
#             "colsample_bytree": 0.8,
#         }
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# # 0  T-shirt/top
# # 1  Trouser
# # 2  Pullover
# # 3  Dress
# # 4  Coat
# # 5  Sandal
# # 6  Shirt
# # 7  Sneaker
# # 8  Bag
# # 9  Ankle boot
# class FashionMnist(MulticlassDataset):
#     def __init__(self):
#         super().__init__(num_classes=10)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("fashion_mnist", data_id=40996)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {
#             "num_class": self.num_classes,
#         }
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_multi_acc_metric, self), "macc", custom_params,
#                                       naming_args)
#
#
# class FashionMnistBinClass(MultiBinClassDataset):
#     def __init__(self, class1, class2):
#         super().__init__(FashionMnist(), class1, class2)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {
#             "subsample": 0.5,
#             "colsample_bytree": 0.8,
#         }
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class Ijcnn1(Dataset):
#     dataset_name = "ijcnn1.h5"
#
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             ijcnn1_data_path = os.path.join(self.data_dir, Ijcnn1.dataset_name)
#
#             # we choose new train/test subsets in 'train_and_test_set'
#             Xtrain = pd.read_hdf(ijcnn1_data_path, "Xtrain")
#             Xtest = pd.read_hdf(ijcnn1_data_path, "Xtest")
#             ytrain = pd.read_hdf(ijcnn1_data_path, "ytrain")
#             ytest = pd.read_hdf(ijcnn1_data_path, "ytest")
#
#             self.X = pd.concat((Xtrain, Xtest), axis=0, ignore_index=True)
#             self.y = pd.concat((ytrain, ytest), axis=0, ignore_index=True)
#             self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
#
#             self.minmax_normalize()
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class Webspam(Dataset):
#     dataset_name = "webspam_wc_normalized_unigram.h5"
#
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             data_path = os.path.join(self.data_dir, Webspam.dataset_name)
#             self.X = pd.read_hdf(data_path, "X")
#             self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
#             self.y = pd.read_hdf(data_path, "y")
#             self.minmax_normalize()
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class BreastCancer(Dataset):
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("breast-w", data_id=15, y_type=str)
#             self.y = (self.y == 'malignant')
#             self.X.fillna(self.X.mean(), inplace=True)
#             self.y = self.y.astype(np.float32)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class BreastCancerNormalized(BreastCancer):
#     def __init__(self):
#         super().__init__()
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             super().load_dataset()
#             self.minmax_normalize()
#
#
# class GinaAgnostic(Dataset):
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("gina_agnostic", data_id=1038)
#             self.y = (self.y == 1)
#             self.y = self.y.astype(np.float32)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class GinaAgnosticNormalized(GinaAgnostic):
#     def __init__(self):
#         super().__init__()
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             super().load_dataset()
#             self.minmax_normalize()
#
#
# class Scene(Dataset):
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("scene", data_id=312)
#             self.y = (self.y == 1)
#             self.y = self.y.astype(np.float32)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class SceneNormalized(Scene):
#     def __init__(self):
#         super().__init__()
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             super().load_dataset()
#             self.minmax_normalize()
#
#
# class MonksProblem2(Dataset):
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("monks-problem-2", data_id=334)
#             self.y = (self.y == 1)
#             self.y = self.y.astype(np.float32)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class MonksProblem2Normalized(MonksProblem2):
#     def __init__(self):
#         super().__init__()
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             super().load_dataset()
#             self.minmax_normalize()
#
#
# class TicTacToe(Dataset):
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("tic-tac-toe", data_id=50, y_type=str, X_type=str)
#             self.y = (self.y == 'positive')
#             self.y = self.y.astype(np.float32)
#             encoder = LabelEncoder()
#             for i in range(self.X.shape[1]):
#                 self.X.iloc[:, i] = encoder.fit_transform(self.X.iloc[:, i])
#             self.X = self.X.astype(np.float32)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class TicTacToeNormalized(TicTacToe):
#     def __init__(self):
#         super().__init__()
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             super().load_dataset()
#             self.minmax_normalize()
#
#
# class Yeast(MulticlassDataset):
#     """
#     The target classes are (the first one occurs the most often): CYT, NUC, MIT, ME3, ME2, ME1, EXC, VAC, POX, ERL.
#     """
#
#     def __init__(self):
#         super().__init__(num_classes=10)
#         self._classes = {"CYT": 0, "NUC": 1, "MIT": 2, "ME3": 3, "ME2": 4, "ME1": 5, "EXC": 6, "VAC": 7, "POX": 8,
#                          "ERL": 9}
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("yeast", data_id=181, y_type=str)
#             self.y.replace(self._classes, inplace=True)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {
#             "num_class": self.num_classes,
#         }
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_multi_acc_metric, self), "macc", custom_params,
#                                       naming_args)
#
#
# class YeastMnistBinClass(MultiBinClassDataset):
#     def __init__(self, class1, class2):
#         super().__init__(Yeast(), class1, class2)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {
#             "subsample": 0.5,
#             "colsample_bytree": 0.8,
#         }
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class YeastNormalizedMnistBinClass(YeastMnistBinClass):
#     def __init__(self, class1, class2):
#         super().__init__(class1, class2)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             super().load_dataset()
#             self.minmax_normalize()
#
#
# class Adult(Dataset):
#     def __init__(self):
#         super().__init__(Task.CLASSIFICATION)
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             self.X, self.y = self._load_openml("adult", data_id=179, y_type=str, X_type=str)
#             self.y = (self.y == 'class')
#             self.X.fillna(self.X.mean(), inplace=True)
#             self.y = self.y.astype(np.float32)
#             encoder = LabelEncoder()
#             for i in range(self.X.shape[1]):
#                 if not self.X.iloc[:, i].name in ['fnlwgt', 'education-num']:
#                     self.X.iloc[:, i] = encoder.fit_transform(self.X.iloc[:, i])
#             self.X = self.X.astype(np.float32)
#
#     def get_xgb_model(self, num_trees, tree_depth, naming_args=None):
#         if naming_args is None:
#             naming_args = {Keys.current_fold_nb: 0, Keys.max_fold: 0}
#         custom_params = {}
#         return super()._get_xgb_model(num_trees, tree_depth,
#                                       partial(_acc_metric, self), "acc", custom_params, naming_args=naming_args)
#
#
# class AdultNormalized(Adult):
#     def __init__(self):
#         super().__init__()
#
#     def load_dataset(self):
#         if self.X is None or self.y is None:
#             super().load_dataset()
#             self.minmax_normalize()


class Abalone(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("abalone", data_id=1)
            self.encode_object_types()
            self.minmax_normalize()

            self.y = self.y.squeeze()


class AutoMPG(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)
        self.missing_values = True

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("AutoMPG", data_id=9)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class BikeSharing(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("BikeSharing", data_id=275)
            self.y = self.y.squeeze()
            self.encode_object_types()
            self.minmax_normalize()


class Challenger(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Challenger", data_id=92)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class CombinedCyclePowerPlant(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("CombinedCyclePowerPlant", data_id=294)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class ComputerHardware(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("ComputerHardware", data_id=29)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class ConcreteCompressingStrength(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("ConcreteCompressingStrength", data_id=165)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class EnergyEfficiency1(Dataset):
    """
    Energy efficiency dataset from UCI repository, predicting heating load (Y1).
    """
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("EnergyEfficiency", data_id=242)
            self.y = self.y['Y1'].squeeze()
            self.minmax_normalize()


class EnergyEfficiency2(Dataset):
    """
    Energy efficiency dataset from UCI repository, predicting cooling load (Y2).
    """
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("EnergyEfficiency", data_id=242)
            self.y = self.y['Y2'].squeeze()
            self.minmax_normalize()


class HeartFailure(Dataset):
    def __init__(self):
        super().__init__(Task.CLASSIFICATION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("HeartFailure", data_id=519)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class Iris(MulticlassDataset):
    def __init__(self):
        super().__init__(Task.MULTI_CLASSIFICATION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Iris", data_id=53)
            self.y = self.y.squeeze()
            self.encode_y()
            self.minmax_normalize()


class LiverDisorder(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("LiverDisorder", data_id=60)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class Obesity(Dataset):
    def __init__(self):
        super().__init__(Task.MULTI_CLASSIFICATION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Obesity", data_id=544)
            self.y = self.y.squeeze()
            self.encode_object_types()
            self.encode_y()
            self.minmax_normalize()


class OnlineNewsPopularity(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("OnlineNewsPopularity", data_id=332)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class Parkinsons1(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Parkinsons", data_id=189)
            self.y = self.y['motor_UPDRS'].squeeze()
            self.minmax_normalize()


class Parkinsons2(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Parkinsons", data_id=189)
            self.y = self.y['total_UPDRS'].squeeze()
            self.minmax_normalize()


class RealEstateValuation(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("RealEstateValuation", data_id=477)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class Servo(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("Servo", data_id=87)
            self.y = self.y.squeeze()
            self.encode_object_types()
            self.minmax_normalize()


class WineQuality(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_ucirepo("WineQuality", data_id=186)
            self.y = self.y.squeeze()
            self.minmax_normalize()


class YouTubeViewCount(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading youtube h5 file")
            df = pd.read_hdf(f"{self.data_dir}/youtube.h5", key='dataset')

            # noinspection PyUnresolvedReferences
            self.y = df.pop('viewslg')
            self.X = df
            self.minmax_normalize()
