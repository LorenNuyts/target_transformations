import operator
import os.path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from experiments.transform_target import DEFAULT_CLFS
from experiments.utils import get_clf_full_name, load_results, Keys


def pretty_print(x, y):
    return ' '.join([str(i) for i in zip(x, y)])


def plot_distribution_y(y: pd.Series, title: str, save_path: str = None, x_label: str = None):
    if x_label is None:
        x_label = "Value of target variable"
    fig, ax = plt.subplots()
    sns.histplot(y, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    if save_path is not None:
        # if path does not exist yet, create it
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    # plt.show()


def plot_multiple_y(x, ys: list, x_label, ylabels: list, title="", save_name=None):
    """
    Plot two y-axis in the same plot

    Parameters
    ----------
    x: list
        x-axis values
    y1: list
        y-axis values for the first plot
    y2: list
        y-axis values for the second plot
    x_label: str
        x-axis label
    y1_label: str
        y-axis label for the first plot
    y2_label: str
        y-axis label for the second plot
    title: str
        title of the plot
    save_name: str
        name of the file to save the plot. If None, the plot is shown instead of being saved
    """
    fig, ax1 = plt.subplots()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', ]
    linestyles = ['-', '--', '-.', ':']
    ax1.set_xlabel(x_label)

    for i, y in enumerate(ys):
        ax1.plot(x, y, color=colors[i], marker='o', linestyle=linestyles[i], label=ylabels[i])
    # ax1.set_ylabel(y1_label, color=color)
    # ax1.plot(x, y1, color=color, marker='o')
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:blue'
    # ax2.set_ylabel(y2_label, color=color)  # we already handled the x-label with ax1
    # ax2.plot(x, y2, color=color, marker='o')
    # ax2.tick_params(axis='y', labelcolor=color)
    plt.suptitle(title)
    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save_name is not None:
        fig.savefig(f'{save_name}.jpg')
    else:
        plt.show()


def plot_error_bars(y, dataset_name: str, included_transformers: list, clf=DEFAULT_CLFS[1],
                    suffix=""):
    base = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    results = load_results(base, dataset_name, suffix=suffix, reset=False)

    for tr in included_transformers:
        clf_name = get_clf_full_name(clf, tr)
        errors = []
        for fold in results[clf_name].keys():
            if isinstance(fold, int):
                errors.append(results[clf_name][fold][Keys.error])
        absolute_errors = [np.abs(errors[i]) for i in range(len(errors))]
        df_error = pd.DataFrame(absolute_errors).transpose()
        # average_error = df_error.mean(axis=1)
        # df_error_short = pd.DataFrame(index=y.index, columns=range(len(errors)//2))
        # for i in y.index:
        #     not_nans = df_error.loc[i].dropna()
        #     df_error_short.loc[i] = not_nans.values

        error_distribution = {y[index]: (np.nanmean(error_folds), np.nanquantile(error_folds, 0.25),
                                         np.nanquantile(error_folds, 0.75))
                              for index, error_folds in df_error.iterrows()}
        error_distr_asc = sorted(error_distribution.items(), key=operator.itemgetter(0))

        means = []
        lower = []
        upper = []
        xrange = []
        for x, values in error_distr_asc:
            means.append(values[0])
            lower.append(values[1])
            upper.append(values[2])
            xrange.append(x)

        print("Transformation: ", tr)
        print("means: \n", pretty_print(xrange, means), "\n")
        print("upper: \n", pretty_print(xrange, upper), "\n")
        print("lower: \n", pretty_print(xrange, lower), "\n")
    print("\n\n")
