import math
import os
import re
from typing import List

import numpy as np
import pandas as pd

from src.experiments.print_results.TextParser import TextParser

from src.experiments.utils import get_paths, Keys, default_suffix, load_results

base_ = os.path.dirname(os.path.realpath(__file__))


def find_most_recent_file(experiment_name, dataset_name):
    suffices = ['']
    while len(suffices) > 0:
        path, path_txt = get_paths(experiment_name, dataset_name, suffix=suffices[0])
        if os.path.exists(path_txt):
            return path, path_txt
        else:
            suffices.pop(0)
    return None, None


# def latex_table(datasets, experiment_name, configs, metrics, method=Keys.xgboost_default, name=Keys.cv5x2,
#                 suffix=default_suffix, bold=True):
#     table = np.empty((len(datasets), len(configs) * len(metrics.keys())), dtype=object)
#
#     base = os.path.join(base_, algorithm)
#
#     for j, config in enumerate(configs):
#         for i, dataset in enumerate(datasets):
#             path, path_txt = find_most_recent_file(str(base), dataset, config)
#             if path is None:
#                 print("No valid file could be found")
#                 for k in range(len(metrics.keys())):
#                     table[i, j + k * len(configs)] = "{missing}"
#                 continue
#             try:
#                 results = dill.load(open(path, 'rb'))
#             except (EOFError, UnpicklingError):
#                 print("File is corrupt:", path)
#                 table[i, j + len(configs):j + len(configs) * len(metrics.keys())] = "{missing}"
#                 continue
#             except ModuleNotFoundError as e:
#                 print("Module not found:", path)
#                 print(e)
#                 table[i, j + len(configs):j + len(configs) * len(metrics.keys())] = "{missing}"
#                 continue
#             metrics_values = extract_metrics(results, metrics, method, name)
#             fill_metrics(metrics_values, table, configs, i, j)
#     if bold:
#         table = best_in_bold(table, list(metrics.keys()), len(configs), len(datasets))
#     df_index = [dataset_acronyms[all_datasets.index(dataset)] for dataset in datasets]
#     df_table = pd.DataFrame(table, index=df_index,
#                             columns=[f"{m}_{i}" for m in metrics.keys()
#                                      for i in range(len(configs))])
#     result_table = df_table.style.to_latex()
#     print(result_table)
#     print("End of table\n\n")
#
#
# def latex_table_from_txt(datasets, algorithm, configs, metrics, method=Keys.xgboost_default, name=Keys.cv5x2,
#                          suffix=default_suffix, bold=True):
#     if suffix != default_suffix:
#         print("Warning: suffix is ignored when using txt files.")
#     table = np.empty((len(datasets), len(configs) * len(metrics.keys())), dtype=object)
#
#     base = os.path.join(base_, "..", algorithm)
#
#     for j, config in enumerate(configs.values()):
#         for i, dataset in enumerate(datasets):
#             path, path_txt = find_most_recent_file(str(base), dataset, config)
#             if path_txt is None:
#                 print("No valid file could be found")
#                 for k in range(len(metrics.keys())):
#                     table[i, j + k * len(configs)] = "{missing}"
#                 continue
#             results = parse_txt_file(path_txt)
#             metrics_values = extract_metrics(results, metrics, method, name)
#             fill_metrics(metrics_values, table, configs, i, j)
#     if bold:
#         table = best_in_bold(table, list(metrics.keys()), len(configs), len(datasets))
#     df_index = [dataset_acronyms[all_datasets.index(dataset)] for dataset in datasets]
#     df_table = pd.DataFrame(table, index=df_index,
#                             columns=[f"{m}_{i}" for m in metrics.keys()
#                                      for i in range(len(configs))])
#     result_table = df_table.style.to_latex()
#     print(result_table)
#     print("End of table\n\n")
#
#
# def latex_table_per_metric(datasets, algorithms, configs, metric, name=Keys.cv5x2,
#                            suffix=default_suffix, bold=False):
#     metric_name = list(metric.keys())[0]
#     round_value = 0 if 'time' in metric_name and 'filtering' not in metric_name else 3
#     round_std = 0 if 'time' in metric_name and 'filtering' not in metric_name else 2
#     nb_configs = len(configs[list(algorithms.keys())[0]])
#     if suffix != default_suffix:
#         print("Warning: suffix is ignored when using txt files.")
#     table = np.empty((len(datasets) + 1, len(algorithms.keys()) * nb_configs), dtype="f,f")
#
#     for a, (algorithm, method) in enumerate(algorithms.items()):
#         base = os.path.join(base_, "..", algorithm)
#         base_config = None
#
#         for j, config in enumerate(configs[algorithm].values()):
#             if (not (config['series_fusion'] or config['series_filtering'] or config['ECP']) or
#                     (algorithm == 'tsfuse_full' and not (config['series_filtering'] or config['ECP']))):
#                 base_config = j
#
#             for i, dataset in enumerate(datasets):
#                 path, path_txt = find_most_recent_file(str(base), dataset, config)
#                 if path_txt is None:
#                     print("No valid file could be found")
#                     table[i, j + nb_configs * a] = (np.nan, np.nan)
#                     continue
#                 results = parse_txt_file(path_txt)
#                 mean, std = extract_metrics(results, metric, method, name)[metric_name]
#                 if mean is None or std is None:
#                     table[i, j + a * nb_configs] = (np.nan, np.nan)
#                     continue
#                 table[i, j + a * nb_configs] = (mean, std)
#             table[len(datasets), a * nb_configs: (a + 1) * nb_configs] = (
#                 average_rel_diff_metrics(table[:len(datasets), a * nb_configs: (a + 1) * nb_configs], base_config, metric_name))
#     result = format_table(table, round_value, round_std)
#     if bold:
#         time_related = 'time' in metric_name
#         result = best_in_bold(result, list(algorithms.keys()), nb_configs, len(datasets), time_related=time_related)
#     df_index = [dataset_acronyms[all_datasets.index(dataset)] for dataset in datasets] + ['Average']
#     df_table = pd.DataFrame(result, index=df_index,
#                             columns=[f"{m}_{i}" for m in algorithms.keys()
#                                      for i in range(nb_configs)])
#     result_table = df_table.style.to_latex()
#     print(result_table)
#     print("End of table\n\n")
#
#
# def latex_rel_diff_per_metric(datasets, algorithms, configs, metric, name=Keys.cv5x2,
#                               suffix=default_suffix, ):
#     metric_name = list(metric.keys())[0]
#     random_algo_name = list(algorithms.keys())[0]
#     if suffix != default_suffix:
#         print("Warning: suffix is ignored when using txt files.")
#     table = np.empty((len(algorithms.keys()), len(configs[random_algo_name])), dtype=object)
#
#     for a, (algorithm, method) in enumerate(algorithms.items()):
#         scores, base_config = get_values_per_metric(datasets, algorithm, configs, metric, method, name)
#
#         rel_diff = average_rel_diff_metrics(scores, base_config, metric_name)
#         round_value = 2 if 'time' in metric and 'filtering' not in metric else 3
#         round_std = 1 if 'time' in metric and 'filtering' not in metric else 2
#         rel_diff = format_table(rel_diff, round_value, round_std)
#         table[a, :] = rel_diff
#
#     df_index = list(algorithms.keys())
#     df_table = pd.DataFrame(table, index=df_index)
#     result_table = df_table.style.to_latex()
#     print(result_table)
#     print("End of table\n\n")


# def get_values_per_metric(datasets, algorithm, configs, metric, method, name):
#     metric_name = list(metric.keys())[0]
#     base = os.path.join(base_, "..", algorithm)
#     base_config = None
#     scores = np.empty((len(datasets), len(configs[algorithm])), dtype="f,f")
#     for j, config in enumerate(configs[algorithm].values()):
#         if (not (config['series_fusion'] or config['series_filtering'] or config['ECP']) or
#                 (algorithm == 'tsfuse_full' and not (config['series_filtering'] or config['ECP']))):
#             base_config = j
#
#         for i, dataset in enumerate(datasets):
#             path, path_txt = find_most_recent_file(str(base), dataset, config)
#             if path_txt is None:
#                 print("No valid file could be found")
#                 scores[i, j] = np.nan
#                 continue
#             results = parse_txt_file(path_txt)
#             metrics_values = extract_metrics(results, metric, method, name)
#             scores[i, j] = metrics_values[metric_name]
#
#     return scores, base_config


def extract_metrics(results, metrics, method, name):
    values = {}
    for nb_metric, (metric_name, get_metric) in enumerate(metrics.items()):
        try:
            round_i = 0 if 'time' in metric_name and 'filtering' not in metric_name else 3
            value = round(get_metric(results, method, name), round_i)
            if round_i == 0:
                value = int(value)
            std = round(all_metrics[metric_name.replace('average', 'std')](results, method, name), round_i)
            if round_i == 0:
                if not math.isnan(std):
                    std = int(std)
            values[metric_name] = value, std
        except KeyError:
            values[metric_name] = None, None
    return values


# def fill_metrics(metric_values, table, configs, i, j):
#     for nb_metric, (metric_name, values) in enumerate(metric_values.items()):
#         (value, std) = values
#         if value is None or std is None or math.isnan(value):  # or math.isnan(std):
#             table[i, j + nb_metric * len(configs)] = "{missing}"
#             continue
#         round_value = 0 if 'time' in metric_name and 'filtering' not in metric_name else 3
#         round_std = 0 if 'time' in metric_name and 'filtering' not in metric_name else 2
#
#         table[i, j + nb_metric * len(
#             configs)] = format_metric(value, std, round_value, round_std)


def format_metric(value, std, round_value=3, round_std=2):
    """
    Format a metric value and its standard deviation for LaTeX. If one of the values is None, return "{missing}". The
    standard deviation is displayed in gray and in a smaller font. The value and std are rounded to 3 decimal places,
    except for time related metrics, which are rounded to 0 decimal places.
    Parameters
    ----------
    value:
        The value of the metric
    std:
        The standard deviation of the metric
    round_value:
        The number of decimal places to round the value to
    round_std:
        The number of decimal places to round the standard deviation to

    Returns
    -------
    str
        The formatted metric value and standard deviation
    """
    value = round(value, round_value)
    std = round(std, round_std)
    if round_std == 0 and not math.isnan(std):
        std = int(std)

    if value is None or std is None:
        return "{missing}"
    return f'\\num{{{value}}} {{\\tiny \\textcolor{{gray}}{{$\\pm$ \\num{{{std}}}}}}}'


def format_table(table, round_value=3, round_std=2):
    """
    Format a table for LaTeX. The standard deviation is displayed in gray and in a smaller font.

    Parameters
    ----------
    table: np.ndarray
        The table to format
    round_value: int, default=3
        The number of decimal places to round the value to
    round_std: int, default=2
        The number of decimal places to round the standard deviation to

    Returns
    -------
    np.ndarray
        The formatted table

    """
    f = np.vectorize(lambda x:
                     format_metric(x[0], x[1], round_value, round_std) if x[0] is not None and not math.isnan(x[0])
                     else "{missing}")
    return f(table).astype("object")


def rel_diff_metrics(scores: np.ndarray, base_config, metric):
    """
    Calculate the relative difference between the base configuration and the other configurations for all
    datasets. The relative difference is calculated as (value - base_value), except if the metric is time related, then
    it is calculated as (value - base_value) / base_value.
    Parameters
    ----------
    scores: np.ndarray
        2D array containing the scores for each dataset (rows) and configuration (columns)
    base_config: int
        Index of the base configuration
    metric: str
        The name of the metric to calculate the relative difference for

    Returns
    -------
    List[str]
        List of formatted relative differences and standard deviations for each configuration and dataset.
    """
    if "time" in metric.lower():
        time_related = True
    else:
        time_related = False
    rel_diff = np.empty(scores.shape[0:2])
    for i in range(scores.shape[0]):
        base_score = scores[i, base_config][0]
        if base_score is np.nan or base_score is None:
            rel_diff[i, :] = np.nan
            continue
        rel_diff[i, 0] = base_score
        for j in range(0, scores.shape[1]):
            if j == base_config:
                continue

            index = j if base_config == 0 else j + 1
            score = scores[i, j][0]
            if score is np.nan or score is None:
                rel_diff[i, index] = np.nan
                continue

            if time_related:
                rel_diff[i, index] = (score - base_score) / base_score * 100
            else:
                rel_diff[i, index] = score - base_score

    return rel_diff


def average_rel_diff_metrics(scores: np.ndarray, base_config, metric):
    """
    Calculate the average relative difference between the base configuration and the other configurations over all
    datasets. The relative difference is calculated as (value - base_value), except if the metric is time related, then
    it is calculated as (value - base_value) / base_value.
    Parameters
    ----------
    scores: np.ndarray
        2D array containing the scores for each dataset (rows) and configuration (columns)
    base_config: int
        Index of the base configuration
    metric: str
        The name of the metric to calculate the relative difference for

    Returns
    -------
    List[str]
        List of formatted average relative differences and standard deviations for each configuration. The average is
        over the datasets.
    """
    result = []
    rel_diff = rel_diff_metrics(scores, base_config, metric)

    # Per column average and std of rel_diff
    for j in range(0, scores.shape[1]):
        avg = np.nanmean(rel_diff[:, j])
        std = np.nanstd(rel_diff[:, j])
        result.append((avg, std))
    return np.array(result, dtype="f,f")


def best_in_bold(table, columns, nb_configs, nb_datasets, time_related=False):
    for i in range(nb_datasets):
        try:
            for j in range(len(columns)):
                str_table_values = table[i, j * nb_configs:(j + 1) * nb_configs]
                table_values = [float(re.findall(r"[-+]?\d*\.\d+|\d+", value)[0]) if value != '{missing}' else -1 for
                                value in str_table_values]
                # table_values = [float(value.split(" $\pm$ ")[0]) if value != '{missing}' else -1 for value in
                #                 str_table_values]
                if time_related or 'time' in columns[j]:
                    best_value = np.min(table_values)
                else:
                    best_value = np.max(table_values)
                for k, value in enumerate(table_values):
                    if value == best_value:
                        table[i, j * nb_configs + k] = f"\\bfseries {str_table_values[k]}"
        except (AttributeError, AttributeError):
            pass
    return table


def average_rmse(results, method, name):
    return results[method][name][Keys.average_rmse]


def std_rmse(results, method, name):
    return results[method][name][Keys.std_rmse]


def average_nrmse(results, method, name):
    return results[method][name][Keys.average_nrmse]


def std_nrmse(results, method, name):
    return results[method][name][Keys.std_nrmse]


all_metrics = {
    'average_rmse': average_rmse,
    'std_rmse': std_rmse,
    'average_nrmse': average_nrmse,
    'std_nrmse': std_nrmse,
}


def print_results_excel(results: dict, metric: str):
    """
    Print results in Excel format

    Parameters
    ----------
    results: dict
        Results dictionary
    metric: str
        Metric to print
    """
    # Start header with 1 empty cell
    header = "\t"
    values = f"{results['dataset']}\t"
    for method in results.keys():
        header += f"{method}\t"
        values += f"{results[method][metric]}\t"
    print(header)
    print(values)


def print_all_results_excel(datasets: List[str], metric: str,  experiment_name, substring: str = None,
                            suffix: str = default_suffix, from_text=True):
    """
    Print results in Excel format

    Parameters
    ----------
    datasets
        List of datasets to print the results of
    metric: str
        Metric to print
    experiment_name: str
        The name of the experiments whose results are being printed
    substring: str
        Substring that the method name should contain
    suffix: str
        Suffix to add to the file name
    from_text: bool
        Whether to load the results from a txt file. If False, the results are loaded from a pickle file
    """
    # all_results = {}
    values = pd.DataFrame(index=datasets, columns=[])
    for dataset in datasets:
        if from_text:
            results = load_results_txt(experiment_name, dataset, suffix=suffix)
        else:
            results = load_results(experiment_name, dataset, suffix=suffix)
        for method in results.keys():
            if method == 'dataset' or (substring is not None and substring not in method):
                continue
            # if method not in all_results:
            #     all_results[method] = []
            # all_results[method].append(results[method][metric])
            values.at[dataset, method] = results[method][metric]
        # values.loc[dataset] = [results[method][metric] for method in results.keys() if method != 'dataset' and (
        #         substring is None or substring in method)]

    # values = pd.DataFrame(index=datasets, columns=list(all_results.keys()))
    # for i, method in enumerate(all_results.keys()):
    #     values[method] = all_results[method]

    # Transform values to excel format such that I can copy paste them in excel
    print(values.to_csv(sep='\t'))


def load_results_txt(experiment_name, dataset_name, suffix=None):
    _, path_txt = get_paths(experiment_name, dataset_name, suffix=suffix)
    return parse_txt_file(path_txt)


def parse_txt_file(path: str):
    with open(path, 'r') as file:
        lines = file.readlines()
        parser = TextParser(lines, length_indentation=5, current_indentation=0)
        parser.parse()
        result = parser.parsed_text
    return result
