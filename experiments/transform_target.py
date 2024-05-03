import argparse
import copy
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from data import *
from data import datasets
from experiments.utils import load_results, save_results, print_all_results_excel, get_clf_full_name
from experiments.utils.constants import *
from experiments.utils.alpha_search import AlphaSearch
from experiments.utils.classifiers import *

base = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CLFS = [
    LassoTuned(SEED),
    RidgeRegressionTuned(SEED),
    GradientBoostingRegressorWrapper(SEED)
]


def run(data: Dataset, clf=DEFAULT_CLFS[1], target_transformer_name=None):
    results = load_results(base, dataset_, suffix=suffix, reset=False)
    clf_name = get_clf_full_name(clf, target_transformer_name)
    if clf_name not in results:
        results[clf_name] = {}

    data.load_dataset()

    if data.task == Task.REGRESSION:
        rskf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=SEED)
    else:
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=SEED)
    all_rmse = []
    all_nrmse = []
    # error_bars = []

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
        if target_transformer_name is not None:
            transformer = get_transformer(target_transformer_name)
            data.transform_target_custom(transformer)

            # data.normalize_y()
        # elif custom or quantile:
        #     if quantile:
        #         transformer = QuantileTransformer(n_quantiles=10, random_state=0)
        #     else:
        #         # transformer = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
        #         # transformer = RobustScaler()
        #         # transformer = PowerTransformer()
        #         # transformer = LogTransformer(base=10)
        #         transformer = LogTransformer(base=np.e)

        clf = copy.deepcopy(clf)

        if "LassoTuned" in clf_name or "RidgeRegressionTuned" in clf_name:
            do_alpha_search(clf, data)
        else:
            no_alpha_search(clf, data)
        predictions = clf.predict(data.Xtest)
        # error_bars.append(data.ytest - predictions)
        if target_transformer_name is not None:
            error = None
            try:
                transformed_pred = data.other_params['target_transformer'].inverse_transform(
                    predictions.reshape(-1, 1)).ravel()
                error = data.ytest - transformed_pred
                score = root_mean_squared_error(data.ytest, transformed_pred)
            except ValueError:
                score = np.nan
                if error is None:  # The transformation failed
                    error = np.nan
            # transformed_pred = data.other_params["ytrain_mean"] + predictions * data.other_params["ytrain_std"]
            # score *= (data.y.max() - data.y.min())
            # score = root_mean_squared_error(data.ytest, transformed_pred)
        #     score = root_mean_squared_error(data.ytest, predictions)
        #     score *= data.other_params["ytrain_std"]
        # elif custom or quantile:
        #     try:
        #         transformed_pred = data.other_params['target_transformer'].inverse_transform(
        #             predictions.reshape(-1, 1)).ravel()
        #         score = root_mean_squared_error(data.ytest, transformed_pred)
        #     except ValueError:
        #         score = np.nan

        else:
            score = root_mean_squared_error(data.ytest, predictions)
            error = data.ytest - predictions
        all_rmse.append(score)

        if target_transformer_name is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if target_transformer_name == Keys.transformer_normalized:
                    ytest = data.other_params['target_transformer'].inverse_transform(data.ytest.to_frame()).squeeze()
                else:
                    ytest = data.other_params['target_transformer'].transform(data.ytest.to_frame()).ravel()
        else:
            ytest = data.ytest
        nrmse = root_mean_squared_error(ytest, predictions) / (ytest.max() - ytest.min())
        # nrmse = scipy.stats.variation(ytest)
        all_nrmse.append(nrmse)

        results[clf_name][i] = {Keys.clf: clf,
                                Keys.predictions: predictions,
                                Keys.error: error,
                                Keys.rmse: score,
                                Keys.nrmse: nrmse}

    if len(all_rmse) == 10:
        # print(f"Average RMSE {'normalized' if normalize_y else ''}:", np.mean(all_rmse))
        results[clf_name].update({Keys.average_rmse: np.mean(all_rmse),
                                  Keys.std_rmse: np.std(all_rmse)})
        results[clf_name].update({Keys.average_nrmse: np.mean(all_nrmse),
                                  Keys.std_nrmse: np.std(all_nrmse)})
        # if plot_error_bars:
        #     absolute_errors = [np.abs(error_bars[i]) for i in range(len(error_bars))]
        #     df_error = pd.DataFrame(absolute_errors).transpose()
        #     average_error = df_error.mean(axis=1)
        #     # df_error_bars = pd.DataFrame(index=data.y.index, columns=range(len(error_bars)//2))
        #     # for i in data.y.index:
        #     #     not_nans = df.loc[i].dropna()
        #     #     df_error_bars.loc[i] = not_nans.values
        #     title = f"Error bars for {data.name()} - {clf_name}"
        #     x_label = "Absolute error"
        #     save_path = os.path.join(base, f"plots/results/{dataset_}/{clf_name}_error_bars.png")
        #
        #     plot_distribution_y(average_error.values, title, save_path, x_label)
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
    # parser.add_argument('--plot_error', action='store_true')
    # parser.add_argument("--auc", type=float, nargs="?", default=default_auc_percentage)
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    args = parser.parse_args()
    dataset_ = args.dataset
    suffix = args.suffix

    all_datasets = list(datasets.keys())

    for clf_ in DEFAULT_CLFS:
        if dataset_ == 'all':
            for dataset_ in all_datasets:
                print(f"Running {dataset_}...")
                run(datasets[dataset_](), clf=clf_)
                run(datasets[dataset_](), clf=clf_, target_transformer_name=Keys.transformer_normalized)
                run(datasets[dataset_](), clf=clf_, target_transformer_name=Keys.transformer_quantile_uniform)
                run(datasets[dataset_](), clf=clf_, target_transformer_name=Keys.transformer_quantile_normal)
                run(datasets[dataset_](), clf=clf_, target_transformer_name=Keys.transformer_robustscaler)
                run(datasets[dataset_](), clf=clf_, target_transformer_name=Keys.transformer_powertransformer)
                run(datasets[dataset_](), clf=clf_, target_transformer_name=Keys.transformer_logtransformer)
                run(datasets[dataset_](), clf=clf_, target_transformer_name=Keys.transformer_lntransformer)

            dataset_ = 'all'
        else:
            run(datasets[dataset_.lower()](), clf=clf_)
            run(datasets[dataset_.lower()](), clf=clf_, target_transformer_name=Keys.transformer_normalized)
            run(datasets[dataset_.lower()](), clf=clf_, target_transformer_name=Keys.transformer_quantile_uniform)
            run(datasets[dataset_.lower()](), clf=clf_, target_transformer_name=Keys.transformer_quantile_normal)
            run(datasets[dataset_.lower()](), clf=clf_, target_transformer_name=Keys.transformer_robustscaler)
            # run(datasets[dataset_.lower()](), clf=clf_, target_transformer_name=Keys.transformer_powertransformer)
            run(datasets[dataset_.lower()](), clf=clf_, target_transformer_name=Keys.transformer_logtransformer)
            run(datasets[dataset_.lower()](), clf=clf_, target_transformer_name=Keys.transformer_lntransformer)

    if dataset_ == 'all':
        print("RMSE:")
        print_all_results_excel(all_datasets, Keys.average_rmse, base, suffix=suffix)
        print("###########################################################################")
        print("NRMSE:")
        print_all_results_excel(all_datasets, Keys.average_nrmse, base, suffix=suffix)
