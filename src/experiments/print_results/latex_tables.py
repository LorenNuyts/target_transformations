import math
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from src.experiments.print_results import parse_txt_file
from src.experiments.utils import replace_last_occurences


def latex_table_from_txt(files: List[str], relevant_configs: List[str], metrics: List[str], row_names: Dict[str, List[str]],
                         column_names: Dict[str, List[str]], nb_configs_groups: int,
                         # algorithm, configs, metrics, method=Keys.xgboost_default, name=Keys.cv5x2,
                         # suffix=default_suffix,
                         bold=False):
    """
    Print a latex table from txt files

    Parameters
    ----------
    files: List[str]
        List of paths to the txt files
    relevant_configs: List[str]

    metrics: List[str]
        List of metrics to extract from the txt files
    row_names: Dict[str, List[str]]
        Names of the rows, possibly nested
    column_names: Dict[str, List[str]]
        Names of the columns, possibly nested
    """
    # if suffix != default_suffix:
    #     print("Warning: suffix is ignored when using txt files.")
    # nb_configs_per_key = [len(configs) for configs in relevant_configs.values()]
    table = np.empty((len(files) * len(metrics), len(relevant_configs)), dtype=object)

    for i, file in enumerate(files):
        if file is None:
            for j in range(len(relevant_configs) * len(metrics)):
                table[i, j] = "{missing}"
            continue
        results = parse_txt_file(file)
        for j, config in enumerate(relevant_configs):
            metric_values = extract_metrics(results[config], metrics)
            fill_metrics_big_table(metric_values, table, i, j, len(files))

    if bold:
        table = best_in_bold(table, relevant_configs, nb_configs_groups, lower_is_better=True)
    df_index = [f"{k}_{v}" for k in row_names.keys() for v in row_names[k]]
    df_columns = [x for xs in column_names.values() for x in xs]
    df_table = pd.DataFrame(table, index=df_index, columns=df_columns)
    result_table = df_table.style.to_latex(hrules=True)
    result_table = make_multi_column(result_table, column_names, df_columns)
    result_table = make_multi_row(result_table, row_names, df_index)

    print(result_table)
    print("End of table\n\n")


def extract_metrics(results: Dict[str, float], metrics: List[str]) -> Dict[str, Tuple[float, float]]:
    values = {}
    for metric in metrics:
        try:
            round_i = 0 if 'time' in metric.lower() else 3
            value = round(results[metric], round_i)
            if round_i == 0:
                value = int(value)
            std = round(results[metric.replace('average', 'std')], round_i)
            if round_i == 0:
                if not math.isnan(std):
                    std = int(std)
            values[metric] = value, std
        except KeyError:
            values[metric] = None, None
    return values

def fill_metrics_big_table(metric_values, table, i_dataset, j_config, nb_datasets):
    """
    Fill the table with the metric values

    Parameters
    ----------
    metric_values: dict
        Dictionary with the metric values
    table: np.ndarray
        Table to fill, with dimensions (len(datasets)*len(metrics), len(configs))
    i_dataset: int
        Index of the dataset
    j_config: int
        Index of the config
    nb_datasets: int
        Number of datasets

    Returns
    -------
    None, modifies the table in place

    """
    for nb_metric, (metric_name, values) in enumerate(metric_values.items()):
        (value, std) = values
        indices = [i_dataset + nb_metric*nb_datasets, j_config]
        if value is None or std is None or math.isnan(value):  # or math.isnan(std):
            table[indices] = "{missing}"
            continue
        round_value = 0 if ('time' in metric_name.lower() or 'ies' in metric_name.lower()) else 3
        round_std = 0 if ('time' in metric_name.lower() or 'ies' in metric_name.lower()) else 2

        table[indices[0], indices[1]] = format_metric(value, std, round_value, round_std)

def format_metric(value, std, round_value=3, round_std=2, max_value:float =1e5):
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

    if round_value == 0 and not math.isnan(value):
        value = int(value)
    if round_std == 0 and not math.isnan(std):
        std = int(std)

    if value is None or std is None:
        return "{missing}"

    if value > max_value:
        return f'{{$\\infty$}}'

    if std < 0.01:
        return f'\\num{{{value}}} {{\\tiny \\textcolor{{gray}}{{$\\pm$ <0.01}}}}'
    else:
        return f'\\num{{{value}}} {{\\tiny \\textcolor{{gray}}{{$\\pm$ \\num{{{std}}}}}}}'

def best_in_bold(table:np.ndarray[str], columns, nb_configs, lower_is_better=True):
    dummy = np.inf if lower_is_better else -np.inf
    for i in range(table.shape[0]):
        try:
            for j in range(0, len(columns), nb_configs):
                str_table_values = table[i, j :j + nb_configs]
                table_values = [float(re.findall(r"[-+]?\d*\.\d+|\d+", value)[0]) if value != '{missing}' and 'inf' not in value else dummy for
                                value in str_table_values]
                # table_values = [float(value.split(" $\pm$ ")[0]) if value != '{missing}' else -1 for value in
                #                 str_table_values]
                if lower_is_better:
                    best_value = np.min(table_values)
                else:
                    best_value = np.max(table_values)
                for k, value in enumerate(table_values):
                    if value == best_value:
                        li = str_table_values[k].split("{")
                        if "num" in li[0]:
                            li[0] = li[0].replace("num", "textbf")
                        else:
                            li[0] = "\\textbf" + li[0]
                        table[i, j + k] = "{".join(li)
                        # table[i, j + k] = f"\\textbf{str_table_values[k][4:]}"
        except (AttributeError, AttributeError):
            pass
    return table

def make_multi_column(table: str, columns: Dict[str, List[str]], current_column_names: List[str], alignment="|c"):
    # Example:
    # \multicolumn{3}{l}{Lasso} & \multicolumn{3}{l}{Ridge} & \multicolumn{3}{l}{GBR} \\
    # Qn & Qu & Ln & Qn & Qu & Ln & Qn & Qu & Ln \\
    multicol_str = ""
    subcol_str = ""
    for col, subcols in columns.items():
        multicol_str += f" & \\multicolumn{{{len(subcols)}}}{{{alignment}}}{{{col}}}"
        subcol_str += " & \\multicolumn{1}{c}{" + "} & \\multicolumn{1}{c}{".join(subcols[:-1]) + "}" + f" & \\multicolumn{1}{{c|}}{{{subcols[-1]}}}"
    # Replace last {c|} with {c}
    replace_last_occurences(subcol_str, "{c|}", "{c}", 1)
    # li = subcol_str.rsplit("{c|}", 1)
    # subcol_str = "{c}".join(li)

    multicol_str += "\\\\"
    subcol_str += "\\\\"
    new_col_names = f"{multicol_str}\n{subcol_str}"
    old_col_names = " & " + " & ".join(current_column_names) + " \\\\"
    return table.replace(old_col_names, new_col_names)

def make_multi_row(table: str, rows: Dict[str, List[str]], current_row_names: List[str]):
    # Example:
    # \multirow{3}{*}{\rotatebox{90}{\textbf{RSE}}} & ....

    table = table.replace("\\begin{tabular}{", "\\begin{tabular}{p{0.5cm}")
    rows_keys = list(rows.keys())
    current_values = None
    current_row_names = current_row_names.copy()
    new_table = ""

    # for every line in table
    for line in table.splitlines(keepends=True):
        if line.startswith(" & ") or line.startswith("\\multicolumn"):
            new_table += " & " + line   # Add extra column for the multirow
        elif line.startswith("\\"):
            new_table += line
        else:
            if current_values is None or len(current_values) == 0:
                current_key = rows_keys.pop(0)
                current_values = rows[current_key].copy()
                multi_row_header = (f"\\multirow{{{len(current_values)}}}{{*}}"
                                    f"{{\\rotatebox{{90}}{{\\textbf{{{current_key}}}}}}}")
                new_table += line.replace(current_row_names.pop(0), multi_row_header + " & " + current_values.pop(0))
            else:
                new_table += line.replace(current_row_names.pop(0), " & " + current_values.pop(0))
                if len(current_values) == 0:
                    current_values = None
                    new_table += "\\hline \n"

    # Remove last \hline
    new_table = replace_last_occurences(new_table, "\\hline \n", "", 1)
    return new_table


