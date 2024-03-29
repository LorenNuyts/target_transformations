import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_distribution_y(y: pd.Series, title: str, save: bool = False, path: str = None):
    fig, ax = plt.subplots()
    sns.histplot(y, kde=True, ax=ax)
    ax.set_title(title)
    if save:
        plt.savefig(path)
    plt.show()