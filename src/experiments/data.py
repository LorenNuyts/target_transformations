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
from sklearn.experimental import enable_iterative_imputer  # Don't remove this one, it is used!

from ucimlrepo import fetch_ucirepo

from .utils.constants import SEED

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

    def name(self) -> str:
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
        if self.y is None:
            raise RuntimeError("data not loaded")
        if self.ytrain is None or self.ytest is None:
            raise RuntimeError("train and test sets not loaded")

        self.ytrain = transformer.fit_transform(self.ytrain.values.reshape(-1, 1)).ravel()
        if self.yval is not None:
            self.yval = transformer.transform(self.yval.values.reshape(-1, 1)).ravel()
        # self.ytest = transformer.transform(self.ytest.values.reshape(-1, 1)).ravel()
        self.other_params["target_transformer"] = transformer

    def transform_features_custom(self, transformer):
        if self.X is None:
            raise RuntimeError("data not loaded")
        if self.Xtrain is None or self.Xtest is None:
            raise RuntimeError("train and test sets not loaded")

        self.Xtrain = transformer.fit_transform(self.Xtrain.values)
        if self.Xval is not None:
            self.Xval = transformer.transform(self.Xval.values)
        self.Xtest = transformer.transform(self.Xtest.values)
        self.other_params["feature_transformer"] = transformer

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


class YouTube(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading youtube h5 file")
            df = pd.read_hdf(f"{self.data_dir}/youtube.h5", key='dataset')

            # noinspection PyUnresolvedReferences
            self.y = df.pop('views')
            self.X = df
            self.minmax_normalize()


class YouTubeLg(Dataset):
    def __init__(self):
        super().__init__(Task.REGRESSION)

    def load_dataset(self):
        if self.X is None or self.y is None:
            print(f"loading youtube h5 file")
            df = pd.read_hdf(f"{self.data_dir}/youtubelg.h5", key='dataset')

            # noinspection PyUnresolvedReferences
            self.y = df.pop('views')
            self.X = df
            self.minmax_normalize()


datasets = {"abalone": Abalone,
            "autompg": AutoMPG,  # Missing values
            "bikesharing": BikeSharing,  # Does not converge
            "powerplant": CombinedCyclePowerPlant,
            # "challenger": Challenger, # Does not converge
            # "computerhardware": ComputerHardware, # What is the target?
            "concrete": ConcreteCompressingStrength,
            "energyefficiency1": EnergyEfficiency1,
            "energyefficiency2": EnergyEfficiency2,
            # "heartfailure": HeartFailure, # Classification
            # "iris": Iris, # Classification
            "liverdisorder": LiverDisorder,
            # "obesity": Obesity(), # Classification
            # "parkinsons1": Parkinsons1, # Does not converge
            # "parkinsons2": Parkinsons2, # Does not converge
            "onlinenewspopularity": OnlineNewsPopularity,  # Does not converge
            "realestatevaluation": RealEstateValuation,
            "servo": Servo,
            "winequality": WineQuality,
            "youtube": YouTube,
            # "youtubelg": YouTubeLg,
            }
