import argparse

from src.experiments.data import datasets, imbalanced_distribution_datasets, forecasting_datasets
from src.experiments.print_results import print_all_results_excel
from src.experiments.print_results.latex_tables import big_latex_table_from_txt, latex_table_from_txt
from src.experiments.utils import default_suffix, Keys, get_clf_full_name, get_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', type=str)
    parser.add_argument('experiment', type=str)
    # parser.add_argument('metric', type=str)
    parser.add_argument('-d', '--datasets', nargs='+', help='<Required> Add datasets', required=True)
    # parser.add_argument("--method", type=str, nargs="?", default=Keys.xgboost_default)
    # parser.add_argument("--split", type=str, nargs="?", default=Keys.cv5x2)
    parser.add_argument("--suffix", type=str, nargs="?", default=default_suffix)
    parser.add_argument("--from_pkl", action='store_true')
    parser.add_argument("--latex", action='store_true')
    # parser.add_argument("--latex_metric", action='store_true')
    # parser.add_argument("--rel_diff", action='store_true')
    # parser.add_argument("--bold", action='store_true')

    # dataset_ = parser.parse_args().dataset
    experiment_ = parser.parse_args().experiment
    # metric_ = parser.parse_args().metric
    datasets_ = parser.parse_args().datasets

    # split_ = parser.parse_args().split
    suffix_ = parser.parse_args().suffix
    from_pkl_ = parser.parse_args().from_pkl
    latex_ = parser.parse_args().latex
    # latex_metric = parser.parse_args().latex_metric
    # rel_diff = parser.parse_args().rel_diff
    # bold_ = parser.parse_args().bold

    if 'all' in datasets_:
        if experiment_ == 'imbalanced_distribution':
            datasets_ = list(imbalanced_distribution_datasets.keys())
        elif experiment_ == 'forecasting':
            datasets_ = list(forecasting_datasets.keys())
        else:
            datasets_ = list(datasets.keys())

    all_metrics = {'rse': Keys.average_rse, 'transformed_rse': Keys.average_transformed_rse, 'mape': Keys.average_mape,
                   'transformed_mape': Keys.average_transformed_mape, 'smape': Keys.average_smape,
                   'transformed_smape': Keys.average_transformed_smape,}
    feature_transformer_name = None

    if experiment_ == 'imbalanced_distribution':
        from src.experiments.imbalanced_distribution import DEFAULT_CLFS
        # Classifiers: 0. Lasso, 1. Ridge, 2. GBTR,  3. SVR
        # clfs = DEFAULT_CLFS[0:2]
        clfs = DEFAULT_CLFS
        # clfs = [DEFAULT_CLFS[i] for i in (0, 3)]
        column_order = [get_clf_full_name(clf.name, transformer,
                                                                feature_transformer_name).replace(' ', '')
                                              for clf in clfs for transformer in [None] + Keys.all_transformers]
        datasets_ = list(imbalanced_distribution_datasets.keys())
        incl_transformers = [None, Keys.transformer_quantile_normal, Keys.transformer_quantile_uniform,
                             Keys.transformer_powertransformer, Keys.transformer_lntransformer]
        # column_names = {clf.acronym: [Keys.transformer_acronyms[t] if t is not None else "Base" for t in incl_transformers] for clf in clfs}
        column_names = {clf.acronym: [Keys.transformer_acronyms[t] if t is not None else "Base" for t in incl_transformers] for clf in clfs}
        # row_names = {m: [d().acronym for d in imbalanced_distribution_datasets.values()] for m in ["RSE", "SMAPE"]}
        row_names = {clf.acronym: [d().acronym for d in imbalanced_distribution_datasets.values()] for clf in clfs}
        relevant_configs = [get_clf_full_name(clf.name, transformer,
                                              feature_transformer_name).replace(' ', '')
                            for clf in clfs
                            for transformer in incl_transformers]
    elif experiment_ == 'forecasting':
        from src.experiments.forecasting import forecasting_clfs
        incl_transformers = [None, "Normalized"]
        column_order = [get_clf_full_name(clf, None).replace(' ', '') for clf in forecasting_clfs.keys()]
        # column_order = [get_clf_full_name("ExponentialSmoothing", transformer).replace(' ', '')
        #                                       for transformer in [None] + Keys.all_transformers]
        relevant_configs = column_order
        column_names = {clf: ["Base"] for clf in forecasting_clfs}
        row_names = {m: [d().acronym for d in forecasting_datasets.values()] for m in ["RSE", "MAPE", "SMAPE"]}
    elif experiment_ == "skewness":
        from scipy.stats import skew, skewtest
        for d in imbalanced_distribution_datasets.values():
            dataset = d()
            print(f"Dataset: {dataset.acronym}")
            dataset.load_dataset()
            print(f"Skewness: {skew(dataset.y.values)}")
            print(f"Skewness test: {skewtest(dataset.y.values)}")

    else:
        column_order = None
        relevant_configs = [] # TODO: Add relevant configs
        column_names = [] # TODO: Add column names
        row_names = [] # TODO: Add row names
        incl_transformers = [] # TODO: Add transformers

    if latex_:
        files = [get_paths(experiment_, dataset, suffix=suffix_)[1] for dataset in datasets_]
        metrics = [Keys.average_rse.replace(' ', ''), Keys.average_mape.replace(' ', ''),
                   Keys.average_smape.replace(' ', '')]
        # row_names = {m: datasets_ for m in metrics}
        # big_latex_table_from_txt(files, relevant_configs, metrics, row_names, column_names, len(incl_transformers),
        #                          bold=True)
        for metric in metrics:
            print(f"Metric: {metric}")
            relevant_configs = {clf.acronym: [get_clf_full_name(clf.name, transformer,
                                                    feature_transformer_name).replace(' ', '') for transformer in incl_transformers]
                                    for clf in clfs}
            latex_table_from_txt(files, relevant_configs, metric, row_names,
                                 [Keys.transformer_acronyms[t] if t is not None else "Base" for t in incl_transformers],
                                 bold=True)
            print("###########################################################################")
        exit()

    for metric_ in all_metrics.keys():
        print(f"Metric: {metric_}")
        print_all_results_excel(datasets_, all_metrics[metric_].replace(' ', ''), experiment_,
                                # present_substring=f"__f_{feature_transformer_name}".replace(' ', ''),
                                # absent_substring='__f_',
                                suffix=suffix_, from_text=not from_pkl_,
                                column_order=column_order)
        print("###########################################################################")
