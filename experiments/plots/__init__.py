import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_distribution_y(y: pd.Series, title: str, save_path: str = None, x_label: str = None):
    if x_label is None:
        x_label = "Value of target variable"
    fig, ax = plt.subplots()
    sns.histplot(y, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
