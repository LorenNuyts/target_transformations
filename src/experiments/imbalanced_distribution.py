import argparse
import copy
import os

import numpy as np
from scipy.optimize._optimize import BracketError
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from src.experiments.data import Dataset, Task, datasets, imbalanced_distribution_datasets
from src.experiments.utils import load_results, get_clf_full_name, save_results
from src.experiments.alpha_search import AlphaSearch
from src.experiments.classifiers import LassoTuned, RidgeRegressionTuned, GradientBoostingRegressorWrapper, \
    SupportVectorRegressorWrapper
from src.experiments.utils.constants import SEED, get_transformer, Keys
from src.experiments.utils.evaluation import compute_metrics

base = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CLFS = [
    LassoTuned(SEED),
    RidgeRegressionTuned(SEED),
    GradientBoostingRegressorWrapper(SEED),
    SupportVectorRegressorWrapper(SEED)
]

NAME = "imbalanced_distribution"

# feature_transform_condition_default = skewed_columns
feature_transform_condition_default = None


def run(data: Dataset, clf=DEFAULT_CLFS[1], target_transformer_name=None, feature_transformer_name=None,
        feature_transform_condition=feature_transform_condition_default, suffix=""):
    results = load_results(NAME, dataset_, suffix=suffix, reset=False)
    clf_name = get_clf_full_name(clf.name, target_transformer_name, feature_transformer_name,
                                 feature_transform_condition is not None)
    if clf_name not in results:
        results[clf_name] = {}

    nb_splits = 2
    nb_repeats = 5
    if data.task == Task.REGRESSION:
        rskf = RepeatedKFold(n_splits=nb_splits, n_repeats=nb_repeats, random_state=SEED)
    else:
        rskf = RepeatedStratifiedKFold(n_splits=nb_splits, n_repeats=nb_repeats, random_state=SEED)
    all_transformed_rse = []
    all_transformed_mape = []
    all_transformed_smape = []
    all_transformed_error = []
    all_rse = []
    all_mape = []
    all_smape = []
    all_error = []

    # if nb_splits*nb_repeats - 1 in results[clf_name].keys():
    #     print("All folds already in results, skipping...")
    #     # save_results(results, NAME, dataset_, suffix=suffix)
    #     return
    data.load_dataset()

    for i, (train_index, test_index) in enumerate(rskf.split(data.X, data.y)):
        if i in results[clf_name].keys():
            print(f"Fold {i} already in results, skipping...")
            continue
        else:
            results[clf_name][i] = {}
        # print(f"Fold {i}:")
        data.cross_validation(train_index, test_index, force=True)
        data.split_validation_set()
        data.minmax_normalize_after_split()
        if data.missing_values:
            data.impute_missing_values()
        if 'contextual_transform_feature' in data.other_params.keys():
            data.transform_contextual()
        if target_transformer_name is not None:
            transformer = get_transformer(target_transformer_name)
            data.transform_target_custom(transformer)
        if feature_transformer_name is not None:
            feature_transformer = get_transformer(feature_transformer_name)
            data.transform_features_custom(feature_transformer, feature_transform_condition)

        clf = copy.deepcopy(clf)

        if "LassoTuned" in clf_name or "RidgeRegressionTuned" in clf_name:
            do_alpha_search(clf, data)
        else:
            no_alpha_search(clf, data)
        predictions = clf.predict(data.Xtest)
        # error_bars.append(data.ytest - predictions)

        transformed_rse, transformed_mape, transformed_smape, transformed_error, rse, mape, smape, error = compute_metrics(data, predictions, target_transformer_name)
        all_transformed_rse.append(transformed_rse)
        all_transformed_mape.append(transformed_mape)
        all_transformed_smape.append(transformed_smape)
        all_transformed_error.append(transformed_error)
        all_rse.append(rse)
        all_mape.append(mape)
        all_smape.append(smape)
        all_error.append(error)

        results[clf_name][i] = {Keys.clf: clf,
                                Keys.predictions: predictions,
                                Keys.error: error,
                                Keys.rse: rse,
                                Keys.mape: mape,
                                Keys.smape: smape,
                                Keys.transformed_rse: transformed_rse,
                                Keys.transformed_mape: transformed_mape,
                                Keys.transformed_smape: transformed_smape}
    if len(all_rse) == 10:
        # print(f"Average RMSE {'normalized' if normalize_y else ''}:", np.mean(all_rmse))

        results[clf_name].update({Keys.average_rse: np.nanmean(all_rse),
                                  Keys.std_rse: np.nanstd(all_rse)})
        results[clf_name].update({Keys.average_mape: np.nanmean(all_mape),
                                    Keys.std_mape: np.nanstd(all_mape)})
        results[clf_name].update({Keys.average_smape: np.nanmean(all_smape),
                                    Keys.std_smape: np.nanstd(all_smape)})
        results[clf_name].update({Keys.average_transformed_rse: np.nanmean(all_transformed_rse),
                                    Keys.std_transformed_rse: np.nanstd(all_transformed_rse)})
        results[clf_name].update({Keys.average_transformed_mape: np.nanmean(all_transformed_mape),
                                    Keys.std_transformed_mape: np.nanstd(all_transformed_mape)})
        results[clf_name].update({Keys.average_transformed_smape: np.nanmean(all_transformed_smape),
                                    Keys.std_transformed_smape: np.nanstd(all_transformed_smape)})

        save_results(results, NAME, dataset_, suffix=suffix)
    # elif len(all_rse) == 10:
    #     results[clf_name].update({Keys.average_rse: np.mean(all_rse),
    #                               Keys.std_rse: np.std(all_rse)})
    #     save_results(results, NAME, dataset_, suffix=suffix)
    # elif len(all_transformed_rse) == 10:
    #     results[clf_name].update({Keys.average_transformed_rse: np.mean(all_transformed_rse),
    #                               Keys.std_transformed_rse: np.std(all_transformed_rse)})
    #     save_results(results, NAME, dataset_, suffix=suffix)
    # if Keys.average_rse not in results[clf_name].keys():
    #     print("Average RSE was not added to the results :(")
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


def run_all_target_transformers(dataset: Dataset, clf, feature_transformer, suffix):
    run(dataset, clf=clf, feature_transformer_name=feature_transformer, suffix=suffix)
    run(dataset, clf=clf, target_transformer_name=Keys.transformer_normalized,
        feature_transformer_name=feature_transformer, suffix=suffix)
    run(dataset, clf=clf, target_transformer_name=Keys.transformer_quantile_uniform,
        feature_transformer_name=feature_transformer, suffix=suffix)
    run(dataset, clf=clf, target_transformer_name=Keys.transformer_quantile_normal,
        feature_transformer_name=feature_transformer, suffix=suffix)
    run(dataset, clf=clf, target_transformer_name=Keys.transformer_robustscaler,
        feature_transformer_name=feature_transformer, suffix=suffix)
    try:
        run(dataset, clf=clf, target_transformer_name=Keys.transformer_powertransformer,
            feature_transformer_name=feature_transformer, suffix=suffix)
    except (ValueError, BracketError) as e:
        print(f"PowerTransformer failed for {dataset.name()}")

    run(dataset, clf=clf, target_transformer_name=Keys.transformer_logtransformer,
        feature_transformer_name=feature_transformer, suffix=suffix)
    run(dataset, clf=clf, target_transformer_name=Keys.transformer_lntransformer,
        feature_transformer_name=feature_transformer, suffix=suffix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--feature_transformer', type=str, nargs="?", default=None)
    # parser.add_argument('--plot_error', action='store_true')
    # parser.add_argument("--auc", type=float, nargs="?", default=default_auc_percentage)
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    args = parser.parse_args()
    dataset_ = args.dataset
    feature_transformer_ = args.feature_transformer
    suffix_ = args.suffix

    all_datasets = list(imbalanced_distribution_datasets.keys())

    if dataset_ == 'all':
        for dataset_ in all_datasets:
            print(f"Running {dataset_}...")
            if feature_transformer_ == "PowerTransformer" and dataset_ == "onlinenewspopularity":
                continue
            for clf_ in DEFAULT_CLFS:
                print(f"Running regressor {clf_.name}...")
                run_all_target_transformers(datasets[dataset_](), clf_, feature_transformer_, suffix_)
    else:
        for clf_ in DEFAULT_CLFS:
            print(f"Running regressor {clf_.name}...")
            run_all_target_transformers(datasets[dataset_.lower()](), clf_, feature_transformer_, suffix_)

    # if dataset_ == 'all':
    #     print("RMSE:")
    #     print_all_results_excel(all_datasets, Keys.average_rmse, NAME, suffix=suffix_)
    #     print("###########################################################################")
    #     print("NRMSE:")
    #     print_all_results_excel(all_datasets, Keys.average_nrmse, NAME, suffix=suffix_)
