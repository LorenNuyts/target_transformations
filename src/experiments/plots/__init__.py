import operator
import os.path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src.experiments.transform_target import DEFAULT_CLFS
from src.experiments.utils import get_clf_full_name, load_results, Keys


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
        fig.savefig(f'{save_name}.png')
    else:
        plt.show()


def plot_multiple_with_fill(x, means, lower, upper, labels, x_label=None, y_label=None, title=None, save_name=None):
    # ('solid', 'solid'),  # Same as (0, ()) or '-'
    # ('dotted', 'dotted'),  # Same as (0, (1, 1)) or ':'
    # ('dashed', 'dashed'),  # Same as '--'
    # ('dashdot', 'dashdot')]  # Same as '-.'
    #
    # linestyle_tuple = [
    # ('loosely dotted', (0, (1, 10))),
    # ('dotted', (0, (1, 1))),
    # ('densely dotted', (0, (1, 1))),
    # ('long dash with offset', (5, (10, 3))),
    # ('loosely dashed', (0, (5, 10))),
    # ('dashed', (0, (5, 5))),
    # ('densely dashed', (0, (5, 1))),
    #
    # ('loosely dashdotted', (0, (3, 10, 1, 10))),
    # ('dashdotted', (0, (3, 5, 1, 5))),
    # ('densely dashdotted', (0, (3, 1, 1, 1))),
    #
    # ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    # ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    linestyles = ['-', ':', '--', '-.',  (0, (3, 5, 1, 5, 1, 5)), (0, (5, 3)), (0, (5, 10)), (0, (5, 1)),
                  (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1)), (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]

    fig, ax = plt.subplots()

    for i in range(len(means)):
        ax.plot(x, means[i], label=labels[i], linestyle=linestyles[i % len(linestyles)])
        ax.fill_between(x, lower[i], upper[i], alpha=0.2)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    plt.legend()
    if save_name is not None:
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        fig.savefig(f'{save_name}.png')
    else:
        plt.show()
    plt.close()


def plot_error_bars(y, dataset_name: str, included_transformers: list, clf=DEFAULT_CLFS[1],
                    suffix="", latex=False, nb_tr_per_plot=6):
    plots_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results/error_distribution")
    results = load_results("error_distribution", dataset_name, suffix=suffix, reset=False)
    xrange = None
    all_means = []
    all_upper = []
    all_lower = []
    labels = []

    for tr_i, tr in enumerate(included_transformers):
        clf_name = get_clf_full_name(clf, tr)
        errors = []
        for fold in results[clf_name].keys():
            if isinstance(fold, int):
                errors.append(results[clf_name][fold][Keys.error])
        absolute_errors = [np.abs(errors[i]) for i in range(len(errors))]
        df_error = pd.DataFrame(absolute_errors).transpose()

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

        if latex:
            print("Transformation: ", tr)
            print("means: \n", pretty_print(xrange, means), "\n")
            print("upper: \n", pretty_print(xrange, upper), "\n")
            print("lower: \n", pretty_print(xrange, lower), "\n")
        else:
            all_means.append(means)
            all_upper.append(upper)
            all_lower.append(lower)
            labels.append(tr if tr is not None else "No transformation")

            # if tr_i % nb_tr_per_plot == 0 or tr_i == len(included_transformers) - 1:
            #     plot_multiple_with_fill(xrange, all_means, all_lower, all_upper, labels,
            #                             "Target value", "Absolute error",
            #                             save_name=os.path.join(plots_dir, f"{dataset_name}_{math.ceil(tr_i / nb_tr_per_plot)}"))
            #     all_means = []
            #     all_upper = []
            #     all_lower = []
            #     labels = []

    if latex:
        print("\n\n")

    plot_multiple_with_fill(xrange, all_means, all_lower, all_upper, labels,
                            "Target value", "Absolute error",
                            save_name=os.path.join(plots_dir, f"{dataset_name}"),
                            title=f"Absolute error distribution for {dataset_name}")
