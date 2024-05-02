import argparse
import sys

from data import *
from experiments.plots import plot_distribution_y, plot_error_bars
from experiments.utils.constants import *
from experiments.utils.classifiers import *

base = os.path.dirname(os.path.realpath(__file__))


def plot_target_distribution(data: Dataset, target_transformer_name=None):
    data.load_dataset()
    results_dir = os.path.join(base, "results")
    if target_transformer_name is not None:
        transformer = get_transformer(target_transformer_name)
        y = transformer.fit_transform(data.y.values.reshape(-1, 1)).ravel()
        plot_distribution_y(y, f"{data.name()}: {target_transformer_name}",
                            save_path=os.path.join(results_dir, dataset_,
                                                   f"{target_transformer_name}_distribution.png"))
    # if transformer == Keys.transformer_normalized:
    #     y_mean = data.y.mean()
    #     y_std = data.y.std()
    #
    #     y = (data.y - y_mean) / y_std
    #     plot_distribution_y(y, f"{data.name()}: Entire dataset (normalized)",
    #                         save_path=os.path.join(results_dir, f"{data.name()}_normalized_distribution.png"))
    # elif transformer == Keys.transformer_quantile_uniform:
    #     qt = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='uniform')
    #
    #     y = qt.fit_transform(data.y.values.reshape(-1, 1)).ravel()
    #     plot_distribution_y(y, f"{data.name()}: Entire dataset (quantile, normal distribution)",
    #                         save_path=os.path.join(results_dir, f"{data.name()}_quantile_normal_distribution.png"))

    else:
        plot_distribution_y(data.y, f"{data.name()}: Entire dataset",
                            save_path=os.path.join(results_dir, f"{data.name()}_distribution.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plot', type=str)
    parser.add_argument('dataset', type=str)
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
                # "iris": Iris(), # Classification
                "liverdisorder": LiverDisorder,
                # "obesity": Obesity, # Classification
                # "parkinsons1": Parkinsons1, # Does not converge
                # "parkinsons2": Parkinsons2, # Does not converge
                # "onlinenewspopularity": OnlineNewsPopularity, # Does not converge
                "realestatevaluation": RealEstateValuation,
                "servo": Servo,
                "winequality": WineQuality,
                }

    all_datasets = list(datasets.keys())

    if args.plot == 'target_distribution':
        if dataset_ == 'all':
            for dataset_ in all_datasets:
                print(f"Running {dataset_}...")
                plot_target_distribution(datasets[dataset_]())
                plot_target_distribution(datasets[dataset_](), target_transformer_name=Keys.transformer_normalized)
                plot_target_distribution(datasets[dataset_](),
                                         target_transformer_name=Keys.transformer_quantile_uniform)
                plot_target_distribution(datasets[dataset_](), target_transformer_name=Keys.transformer_quantile_normal)
                plot_target_distribution(datasets[dataset_](), target_transformer_name=Keys.transformer_robustscaler)
                plot_target_distribution(datasets[dataset_](),
                                         target_transformer_name=Keys.transformer_powertransformer)
                plot_target_distribution(datasets[dataset_](), target_transformer_name=Keys.transformer_logtransformer)
                plot_target_distribution(datasets[dataset_](), target_transformer_name=Keys.transformer_lntransformer)
        else:
            plot_target_distribution(datasets[dataset_.lower()]())
            plot_target_distribution(datasets[dataset_.lower()](), target_transformer_name=Keys.transformer_normalized)
            plot_target_distribution(datasets[dataset_.lower()](),
                                     target_transformer_name=Keys.transformer_quantile_uniform)
            plot_target_distribution(datasets[dataset_.lower()](),
                                     target_transformer_name=Keys.transformer_quantile_normal)
            plot_target_distribution(datasets[dataset_.lower()](),
                                     target_transformer_name=Keys.transformer_robustscaler)
            plot_target_distribution(datasets[dataset_.lower()](),
                                     target_transformer_name=Keys.transformer_powertransformer)
            plot_target_distribution(datasets[dataset_.lower()](),
                                     target_transformer_name=Keys.transformer_logtransformer)
            plot_target_distribution(datasets[dataset_.lower()](),
                                     target_transformer_name=Keys.transformer_lntransformer)

    elif args.plot == 'error_distribution':
        if dataset_ == 'all':
            path_console = os.path.join(base, 'results', 'error_distribution_all')
            sys.stdout = open(f'{path_console}_console.txt', 'w')
            for dataset_ in all_datasets:
                print(f"Running {dataset_}...")
                data = datasets[dataset_.lower()]()
                data.load_dataset()
                plot_error_bars(data.y, dataset_,
                                included_transformers=[None,
                                                       Keys.transformer_quantile_uniform,
                                                       Keys.transformer_quantile_normal,
                                                       # Keys.transformer_robustscaler,
                                                       Keys.transformer_powertransformer,
                                                       # Keys.transformer_logtransformer,
                                                       Keys.transformer_lntransformer])
            sys.stdout.close()
        else:
            data = datasets[dataset_.lower()]()
            data.load_dataset()
            plot_error_bars(data.y, dataset_,
                            included_transformers=[None,
                                                   Keys.transformer_quantile_uniform,
                                                   Keys.transformer_quantile_normal,
                                                   # Keys.transformer_robustscaler,
                                                   Keys.transformer_powertransformer,
                                                   # Keys.transformer_logtransformer,
                                                   Keys.transformer_lntransformer])
