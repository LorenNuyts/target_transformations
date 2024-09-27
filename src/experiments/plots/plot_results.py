import os

import numpy as np
from matplotlib import pyplot as plt

base = os.path.dirname(os.path.realpath(__file__))

def plot_rel_diff():
    path = os.path.join(base, "..", "..", "..", "results", "plots", "transformation_results", "results_rel_diff")
    algorithms = ["Lasso", "RidgeRegression", "GBRegressor"]
    transformations = ["Quantile Uniform", "Quantile Normal", "PowerTransformer", "Ln"]
    rse = {"values": np.array([[5.93, -9.09, -2.33, -3.24],
                     [6.46, -10.8, -4, -3.7],
                     [18.62, 19.36, 5.6, 1.13]]),
           "significance": np.array([[False, False, False, False],
                            [False, False, False, False],
                            [False, False, False, False]]),
           "y_lower": -17,
           "y_upper": 27,
           "nb_digits": 2,
           "suffix": "%",
           "higher is better": False,
           "title": 'RSE w.r.t. no target transformation',
           "ylabel": 'Relative RSE difference (%) w.r.t. no target transformation',
           # "legend_position": (0.61, 0.39),
           "legend_position": None,
           "save_name": path + "_rse"}
    mape = {"values": np.array([[-6.28, -8.02, -6.84, -11.42],
                    [-6.48, -7.87, -9.91, -9.5],
                    [-3.14, 0.23, -2.88, -4.39]]),
            "significance": np.array([[False, True, False, False],
                            [False, True, False, False],
                            [False, False, False, False]]),
            "y_lower": -15,
            "y_upper": 10,
            "nb_digits": 2,
            "suffix": "%",
            "higher is better": False,
            "title": 'MAPE w.r.t. no channel selection w.r.t no target transformation',
           "legend_position": None,
            "ylabel": 'Relative MAPE difference (%) w.r.t. no target transformation',
            "save_name": path + "_mape"}
    smape = {"values": np.array([[-4.8, -7.79, -6.78, -6.06],
                                [-4.95, -5.08, -6.53, -6.15],
                                [-0.02, 2.62, -1.92, -1.63]]),
            "significance": np.array([[False, False, False, False],
                                      [False, False, False, False],
                                      [False, False, False, False]]),
            "y_lower": -10,
            "y_upper": 6,
            "nb_digits": 2,
           "suffix": "%",
           "higher is better": False,
           "title": 'SMAPE w.r.t. no target transformation',
           "ylabel": 'Relative SMAPE difference (%) w.r.t. no target transformation',
           "legend_position": None,
           "save_name": path + "_smape"}
    metric = smape
    save_fig = True
    relative_improvement = metric["values"]

    # Mask matrix: True means not masked, False means masked
    # mask_matrix = metric["significance"]
    y_lower = metric["y_lower"]
    y_upper = metric["y_upper"]
    nb_digits = metric["nb_digits"]

    # Custom colors (normalized RGB values)
    colors = [(78 / 255, 167 / 255, 46 / 255),  # RGB(78, 167, 46)
              (15 / 255, 158 / 255, 213 / 255),  # RGB(15, 158, 213)
              (112 / 255, 48 / 255, 160 / 255),  # RGB(112, 48, 160)
              (163 / 255, 33 / 255, 52 / 255)]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # X positions for the bars
    bar_height = 0.2
    x = np.arange(len(algorithms))

    # Plotting bars for each algorithm
    for i, (cs, color) in enumerate(zip(transformations, colors)):
        bars = ax.barh(x + i * bar_height, relative_improvement[:, i], bar_height, label=cs, color=color,
                       # xerr=metric["std"][:, i]
                       )

        # Adding values on top of each bar or below if the value is negative, with padding
        for j, bar in enumerate(bars):
            # yval = bar.get_height()
            xval = bar.get_width()
            padding = (y_upper-y_lower)/100 if xval > 0 else -(y_upper-y_lower)/100*1.3
            # if mask_matrix[j,i]:
            if metric["significance"][j, i]:
                ystr = f'{xval:.{nb_digits}f}{metric["suffix"]}*'
            else:
                ystr = f'{xval:.{nb_digits}f}{metric["suffix"]}'
            ax.text(xval + padding, bar.get_y() + bar.get_height() / 2, ystr,
                    ha='left' if xval > 0 else 'right', va='center', fontsize=15)

    # Invert the y-axis so the first algorithm appears at the top
    ax.invert_yaxis()

    # Customizing x-axis with an arrow
    ax.spines['bottom'].set_position(('outward', 0))  # Moves the x-axis slightly outward
    ax.spines['bottom'].set_color('none')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_capstyle('round')

    # Removing the top and right spines to eliminate the border
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # Adding the x-axis arrow
    ax.annotate('', xy=(y_upper*1.1, 0), xycoords=('data', 'axes fraction'), xytext=(y_lower*1.1, 0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

    # Adding labels "Better" and "Worse" next to the arrow
    higher = "Better" if metric["higher is better"] else "Worse"
    lower = "Worse" if metric["higher is better"] else "Better"
    ax.text(y_upper*0.94, 2.7, higher, ha='center', va='center', fontsize=18, color='grey')
    ax.text(y_lower*0.94, 2.7, lower, ha='center', va='center', fontsize=18, color='grey')

    # Removing the y-axis spine
    ax.spines['left'].set_color('none')

    # Removing x-axis label
    ax.set_xlabel('')

    # Customizing x-axis limit
    ax.set_xlim(y_lower, y_upper)  # Adjust x-axis limit to allow space for the *
    ax.axvline(0, color='black', linewidth=1)

    # Title and labels
    plt.xlabel(metric['ylabel'], fontsize=20)

    # Custom y-axis tick labels
    ax.set_yticks(x + bar_height)
    ax.set_yticklabels(algorithms, fontsize=20)

    # Hide the xticks but keep the tick labels (text)
    ax.tick_params(axis='x', which='both', length=0)  # Set xtick length to 0 to hide them
    ax.tick_params(axis='y', which='both', length=0)
    # ax.set_yticklabels([])  # Hide the yticks
    ax.set_xticklabels([])  # Hide the yticks

    # Adding a legend
    plt.legend(title="Transformations", fontsize=18, title_fontsize=20,
               bbox_to_anchor=metric["legend_position"]
               # , loc='upper right'
               )

    # Add the text in the bottom left of the figure
    fig.text(0.1, 0.95, '* represents a significant difference', ha='left', va='center', fontsize=15, color='grey')

    # Display the plot
    plt.tight_layout()
    if save_fig:
        plt.savefig(metric["save_name"] + ".png")
    else:
        plt.show()