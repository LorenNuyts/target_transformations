import argparse
import numpy as np
from scipy.optimize._optimize import BracketError
from sklearn.model_selection import TimeSeriesSplit

from src.experiments.classifiers import ExponentialSmoothingWrapper, AutoArimaWrapper, GBForecaster
from src.experiments.data import Dataset, datasets, Task
from src.experiments.utils import load_results, get_clf_full_name, save_results

from src.experiments.utils.constants import get_transformer, Keys, SEED
from src.experiments.utils.evaluation import compute_metrics

# clf_name = "ExponentialSmoothing"
# clf_name = "AutoArima"
forecasting_clfs = {
    # "ExponentialSmoothing": lambda d: ExponentialSmoothingWrapper(**d.model_params()),
    # "AutoArima": lambda _: AutoArimaWrapper(),
    "GBForecaster": lambda d: GBForecaster(window_length=d.forecasting_horizon, strategy='recursive')
                    }


NAME = "forecasting"

# MAX_NB_FOLDS = 2
MAX_NB_FOLDS = float('inf')


def run(data: Dataset, clf_name, target_transformer_name=None, suffix="", force=False):
    results = load_results(NAME, data.name, suffix=suffix, reset=False)
    clf_full_name = get_clf_full_name(clf_name, target_transformer_name)
    if clf_full_name not in results:
        results[clf_full_name] = {}

    data.load_dataset()
    assert data.task == Task.FORECASTING

    nb_splits = 2 if data.X.shape[0] < 15 else 10
    if nb_splits - 1 in results[clf_full_name].keys() and not force:
        print("All folds already in results, skipping...")
        return

    # tscv = TimeSeriesSplit(n_splits=nb_splits)

    all_transformed_rse = []
    all_transformed_mape = []
    all_transformed_smape = []
    all_transformed_error = []
    all_rse = []
    all_mape = []
    all_smape = []
    all_error = []
    for i, (train_index, test_index) in enumerate(data.generate_cross_validation_splits(nb_splits, seed=SEED)):
        if i >= MAX_NB_FOLDS:
            break
        if i in results[clf_full_name].keys() and not force:
            print(f"Fold {i} already in results, skipping...")
            continue
        else:
            results[clf_full_name][i] = {}
        print(f"Fold {i}:")
        data.cross_validation(train_index, test_index, force=True)

        if data.missing_values:
            data.impute_missing_values()
        if 'contextual_transform_feature' in data.other_params.keys():
            data.transform_contextual()
        if target_transformer_name is not None:
            transformer = get_transformer(target_transformer_name)
            data.transform_target_custom(transformer)

        clf = forecasting_clfs[clf_name](data)
        # clf = ExponentialSmoothingWrapper(**data.model_params())
        # clf = AutoArimaWrapper()
        # clf = GBForecaster(window_length=52, strategy='recursive')

        clf.fit(data)
        predictions = clf.forecast(data)
#
        transformed_rse, transformed_mape, transformed_smape, transformed_error, rse, mape, smape, error = compute_metrics(data, predictions, target_transformer_name)
        all_transformed_rse.append(transformed_rse)
        all_transformed_mape.append(transformed_mape)
        all_transformed_smape.append(transformed_smape)
        all_transformed_error.append(transformed_error)
        all_rse.append(rse)
        all_mape.append(mape)
        all_smape.append(smape)
        all_error.append(error)

        results[clf_full_name][i] = {Keys.clf: clf,
                                Keys.predictions: predictions,
                                Keys.error: error,
                                Keys.rse: rse,
                                Keys.mape: mape,
                                Keys.smape: smape,
                                Keys.transformed_rse: transformed_rse,
                                Keys.transformed_mape: transformed_mape,
                                Keys.transformed_smape: transformed_smape}
    # if len(all_rse) == nb_splits or len(all_rse) >= MAX_NB_FOLDS:
    results[clf_full_name].update({Keys.average_rse: np.nanmean(all_rse),
                              Keys.std_rse: np.nanstd(all_rse)})
    results[clf_full_name].update({Keys.average_mape: np.nanmean(all_mape),
                                Keys.std_mape: np.nanstd(all_mape)})
    results[clf_full_name].update({Keys.average_smape: np.nanmean(all_smape),
                                Keys.std_smape: np.nanstd(all_smape)})
    results[clf_full_name].update({Keys.average_transformed_rse: np.nanmean(all_transformed_rse),
                                Keys.std_transformed_rse: np.nanstd(all_transformed_rse)})
    results[clf_full_name].update({Keys.average_transformed_mape: np.nanmean(all_transformed_mape),
                                Keys.std_transformed_mape: np.nanstd(all_transformed_mape)})
    results[clf_full_name].update({Keys.average_transformed_smape: np.nanmean(all_transformed_smape),
                                Keys.std_transformed_smape: np.nanstd(all_transformed_smape)})

    save_results(results, NAME, dataset_, suffix=suffix)


def run_all_target_transformers(dataset: Dataset, suffix, force=False):
    for clf_name in forecasting_clfs.keys():
        run(dataset, clf_name, suffix=suffix, force=force)
    # run(dataset, target_transformer_name=Keys.transformer_normalized, suffix=suffix)
    # run(dataset, target_transformer_name=Keys.transformer_quantile_uniform, suffix=suffix)
    # run(dataset, target_transformer_name=Keys.transformer_quantile_normal, suffix=suffix)
    # run(dataset, target_transformer_name=Keys.transformer_robustscaler, suffix=suffix)
    # try:
    #     run(dataset,  target_transformer_name=Keys.transformer_powertransformer, suffix=suffix)
    # except (ValueError, BracketError) as e:
    #     print(f"PowerTransformer failed for {dataset.name}")
    #
    # run(dataset, target_transformer_name=Keys.transformer_logtransformer, suffix=suffix)
    # run(dataset, target_transformer_name=Keys.transformer_lntransformer, suffix=suffix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    # parser.add_argument('--plot_error', action='store_true')
    # parser.add_argument("--auc", type=float, nargs="?", default=default_auc_percentage)
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    parser.add_argument("--force", action='store_true')
    args = parser.parse_args()
    dataset_ = args.dataset
    suffix_ = args.suffix
    force_ = args.force

    all_datasets = list(datasets.keys())
    # run(datasets[dataset_](), suffix=suffix_)

    if dataset_ == 'all':
        for dataset_ in all_datasets:
            print(f"Running {dataset_}...")
            run_all_target_transformers(datasets[dataset_](), suffix_, force_)
    else:
        run_all_target_transformers(datasets[dataset_.lower()](), suffix_, force_)