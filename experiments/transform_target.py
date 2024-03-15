import argparse
import copy

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from data import *
from experiments.utils import load_results, save_results, print_all_results_excel
from experiments.utils.constants import *
from experiments.utils.alpha_search import AlphaSearch
from experiments.utils.classifiers import *

base = os.path.dirname(os.path.realpath(__file__))

# DEFAULT_CLF = LassoTuned(SEED)
# DEFAULT_CLF = RidgeRegressionTuned(SEED)
DEFAULT_CLF = GradientBoostingRegressorWrapper(SEED)


def run(data: Dataset, normalize_y=False, clf=DEFAULT_CLF):
    clf_name = f"{clf.name}{'__normalized' if normalize_y else ''}"
    results = load_results(base, dataset_, suffix=suffix, reset=False)
    if clf_name not in results:
        results[clf_name] = {}

    data.load_dataset()

    if data.task == Task.REGRESSION:
        rskf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=SEED)
    else:
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=SEED)
    all_scores = []
    for i, (train_index, test_index) in enumerate(rskf.split(data.X, data.y)):
        # if i in results[clf_name].keys():
        #     print(f"Fold {i} already in results, skipping...")
            # continue
        # else:
        results[clf_name][i] = {}
        # print(f"Fold {i}:")
        data.cross_validation(train_index, test_index, force=True)
        data.split_validation_set()
        if data.missing_values:
            data.impute_missing_values()
        if normalize_y:
            data.normalize_y()

        clf = copy.deepcopy(clf)

        if clf_name == "LassoTuned" or clf_name == "RidgeRegressionTuned":
            do_alpha_search(clf, data)
        else:
            no_alpha_search(clf, data)
        score = root_mean_squared_error(data.ytest, clf.predict(data.Xtest))
        if normalize_y:
            score *= data.other_params["ytrain_std"]
        all_scores.append(score)
        results[clf_name][i] = {Keys.clf: clf, Keys.rmse: score}

    if len(all_scores) == 10:
        print(f"Average RMSE {'normalized' if normalize_y else ''}:", np.mean(all_scores))
        results[clf_name].update({Keys.average_rmse: np.mean(all_scores),
                                  Keys.std_rmse: np.std(all_scores)})
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

    datasets = {"abalone": Abalone(),
                "autompg": AutoMPG(),  # Missing values
                # "bikesharing": BikeSharing(), # Does not converge
                "powerplant": CombinedCyclePowerPlant(),
                # "challenger": Challenger(), # Does not converge
                # "computerhardware": ComputerHardware(), # What is the target?
                "concrete": ConcreteCompressingStrength(),
                "energyefficiency1": EnergyEfficiency1(),
                "energyefficiency2": EnergyEfficiency2(),
                # "heartfailure": HeartFailure(), # Classification
                # "iris": Iris(), # Classification
                "liverdisorder": LiverDisorder(),
                # "obesity": Obesity(), # Classification
                # "parkinsons1": Parkinsons1(), # Does not converge
                # "parkinsons2": Parkinsons2(), # Does not converge
                # "onlinenewspopularity": OnlineNewsPopularity(), # Does not converge
                "realestatevaluation": RealEstateValuation(),
                "servo": Servo(),
                "winequality": WineQuality(),
                }

    all_datasets = list(datasets.keys())

    if dataset_ == 'all':
        for dataset_ in all_datasets:
            print(f"Running {dataset_}...")
            run(datasets[dataset_], normalize_y=False)
            run(datasets[dataset_], normalize_y=True)
        print_all_results_excel(all_datasets, Keys.average_rmse, base, suffix=suffix)
    else:
        run(datasets[dataset_.lower()], normalize_y=False)
        run(datasets[dataset_.lower()], normalize_y=True)
