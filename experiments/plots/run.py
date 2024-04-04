import argparse
import copy

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from data import *
from experiments.plots import plot_distribution_y
from experiments.utils import load_results, save_results, print_all_results_excel
from experiments.utils.constants import *
from experiments.utils.alpha_search import AlphaSearch
from experiments.utils.classifiers import *

base = os.path.dirname(os.path.realpath(__file__))


def run(data: Dataset, normalize_y=False, quantile=False):
    data.load_dataset()
    results_dir = os.path.join(base, "results")
    if normalize_y and not quantile:
        y_mean = data.y.mean()
        y_std = data.y.std()

        y = (data.y - y_mean) / y_std
        plot_distribution_y(y, f"{data.name()}: Entire dataset (normalized)",
                            save_path=os.path.join(results_dir, f"{data.name()}_normalized_distribution.png"))
    elif quantile:
        qt = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')

        y = qt.fit_transform(data.y.values.reshape(-1, 1)).ravel()
        plot_distribution_y(y, f"{data.name()}: Entire dataset (quantile, normal distribution)",
                            save_path=os.path.join(results_dir, f"{data.name()}_quantile_normal_distribution.png"))

    else:
        plot_distribution_y(data.y, f"{data.name()}: Entire dataset",
                            save_path=os.path.join(results_dir, f"{data.name()}_distribution.png"))

    # if data.task == Task.REGRESSION:
    #     rskf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=SEED)
    # else:
    #     rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=SEED)
    # for i, (train_index, test_index) in enumerate(rskf.split(data.X, data.y)):
    #     data.cross_validation(train_index, test_index, force=True)
    #
    #     plot_distribution_y(data.ytrain, f"{data.name()}: Train fold {i}",
    #                         save_path=os.path.join(results_dir, f"{data.name()}_train_fold_{i}_distribution.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
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
            # run(datasets[dataset_], normalize_y=False)
            # run(datasets[dataset_], normalize_y=True)
            run(datasets[dataset_], normalize_y=True, quantile=True)
    else:
        # run(datasets[dataset_.lower()], normalize_y=False)
        run(datasets[dataset_.lower()], normalize_y=True)
        run(datasets[dataset_.lower()], normalize_y=True, quantile=True)