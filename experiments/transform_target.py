import argparse
import copy
import warnings

import scipy
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.preprocessing import RobustScaler, PowerTransformer

from data import *
from experiments.utils import load_results, save_results, print_all_results_excel
from experiments.utils.constants import *
from experiments.utils.alpha_search import AlphaSearch
from experiments.utils.classifiers import *

base = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CLFS = [
    LassoTuned(SEED),
    RidgeRegressionTuned(SEED),
    GradientBoostingRegressorWrapper(SEED)
]


def run(data: Dataset, normalize_y=False, quantile=False, custom=False, clf=DEFAULT_CLFS[0]):
    clf_name = (f"{clf.name}{'__normalized' if normalize_y else ''}"
                f"{'__quantile_uniform' if quantile else ''}"
                # f"{'__quantile_normal' if custom else ''}"
                # f"{'__robust_scaler' if custom else ''}"
                f"{'__power_transformer' if custom else ''}"
                )
    results = load_results(base, dataset_, suffix=suffix, reset=False)
    if clf_name not in results:
        results[clf_name] = {}

    data.load_dataset()

    if data.task == Task.REGRESSION:
        rskf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=SEED)
    else:
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=SEED)
    all_rmse = []
    all_nrmse = []

    for i, (train_index, test_index) in enumerate(rskf.split(data.X, data.y)):
        if i in results[clf_name].keys():
            print(f"Fold {i} already in results, skipping...")
            continue
        else:
            results[clf_name][i] = {}
        # print(f"Fold {i}:")
        data.cross_validation(train_index, test_index, force=True)
        data.split_validation_set()
        if data.missing_values:
            data.impute_missing_values()
        if normalize_y:
            data.normalize_y()
        elif custom or quantile:
            if quantile:
                transformer = QuantileTransformer(n_quantiles=10, random_state=0)
            else:
                # transformer = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
                # transformer = RobustScaler()
                transformer = PowerTransformer()
            data.transform_target_custom(transformer)

        clf = copy.deepcopy(clf)

        if "LassoTuned" in clf_name or "RidgeRegressionTuned" in clf_name:
            do_alpha_search(clf, data)
        else:
            no_alpha_search(clf, data)
        predictions = clf.predict(data.Xtest)
        if normalize_y:
            # transformed_pred = data.other_params["ytrain_mean"] + predictions * data.other_params["ytrain_std"]
            # score *= (data.y.max() - data.y.min())
            # score = root_mean_squared_error(data.ytest, transformed_pred)
            score = root_mean_squared_error(data.ytest, predictions)
            score *= data.other_params["ytrain_std"]
        elif custom or quantile:
            try:
                transformed_pred = data.other_params['target_transformer'].inverse_transform(
                    predictions.reshape(-1, 1)).ravel()
                score = root_mean_squared_error(data.ytest, transformed_pred)
            except ValueError:
                score = np.nan

        else:
            score = root_mean_squared_error(data.ytest, predictions)
        all_rmse.append(score)

        if custom or quantile:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ytest = data.other_params['target_transformer'].transform(data.ytest.to_frame()).ravel()
        else:
            ytest = data.ytest
        nrmse = root_mean_squared_error(ytest, predictions) / (ytest.max() - ytest.min())
        # nrmse = scipy.stats.variation(ytest)
        all_nrmse.append(nrmse)

        results[clf_name][i] = {Keys.clf: clf, Keys.rmse: score, Keys.nrmse: nrmse}

    if len(all_rmse) == 10:
        print(f"Average RMSE {'normalized' if normalize_y else ''}:", np.mean(all_rmse))
        results[clf_name].update({Keys.average_rmse: np.mean(all_rmse),
                                  Keys.std_rmse: np.std(all_rmse)})
        results[clf_name].update({Keys.average_nrmse: np.mean(all_nrmse),
                                  Keys.std_nrmse: np.std(all_nrmse)})
        save_results(results, base, dataset_, suffix=suffix)
    # print_results(results)


def do_alpha_search(clf, data):
    alpha_search = AlphaSearch()
    for alpha_record in alpha_search:
        clf.update_lin_clf_alpha(alpha_record.alpha)
        clf.fit_coefficients(data, alpha_record)

    best_record = alpha_search.get_best_record()
    clf.clf.intercept_ = best_record.intercept
    clf.clf.coef_ = best_record.coefs


def no_alpha_search(clf, data):
    clf.fit(data.Xtrain, data.ytrain)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    # parser.add_argument('--series_fusion', action='store_true')
    # parser.add_argument("--auc", type=float, nargs="?", default=default_auc_percentage)
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    args = parser.parse_args()
    dataset_ = args.dataset
    suffix = args.suffix

    datasets = {"abalone": Abalone,
                "autompg": AutoMPG,  # Missing values
                # "bikesharing": BikeSharing, # Does not converge
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
                # "onlinenewspopularity": OnlineNewsPopularity, # Does not converge
                "realestatevaluation": RealEstateValuation,
                "servo": Servo,
                "winequality": WineQuality,
                }

    all_datasets = list(datasets.keys())

    for clf_ in DEFAULT_CLFS:
        if dataset_ == 'all':
            for dataset_ in all_datasets:
                print(f"Running {dataset_}...")
                # run(datasets[dataset_](), normalize_y=False, clf=clf_)
                # run(datasets[dataset_](), normalize_y=True, clf=clf_)
                # run(datasets[dataset_](), quantile=True, clf=clf_)
                run(datasets[dataset_](), custom=True, clf=clf_)
            dataset_ = 'all'
        else:
            run(datasets[dataset_.lower()](), normalize_y=False, clf=clf_)
            run(datasets[dataset_.lower()](), normalize_y=True, clf=clf_)
            run(datasets[dataset_.lower()](), quantile=True, clf=clf_)
            run(datasets[dataset_.lower()](), custom=True, clf=clf_)

    if dataset_ == 'all':
        print("RMSE:")
        print_all_results_excel(all_datasets, Keys.average_rmse, base, suffix=suffix)
        print("###########################################################################")
        print("NRMSE:")
        print_all_results_excel(all_datasets, Keys.average_nrmse, base, suffix=suffix)
