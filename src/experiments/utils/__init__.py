import os
from typing import Union, TextIO, List

import dill
import numpy as np
import pandas as pd

from .constants import Keys, default_suffix


# \Section: Project general
def round_sign_fig(x: float, i: int = 5) -> float:
    """
    Rounds a float to a specified number of significant figures.

    Parameters
    ----------
    x : float
        The float to round.
    i : int, optional
        The number of significant figures to round to, by default 5.

    Returns
    -------
    float
        The rounded float
    """
    return float('%.{}g'.format(i) % x)


def isworse(metric, reference):  # higher is better
    return metric < reference


# def is_almost_eq(metric, reference, relerr=1e-5):
#     eps = abs((metric - reference) / reference)
#     return eps < relerr
#
#
# def is_not_almost_eq(metric, reference, relerr=1e-5):
#     return not is_almost_eq(metric, reference, relerr)


def encode_onehot(y: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Encodes the given labels using one-hot encoding.
    """
    classes = sorted(np.unique(y))
    y_onehot = np.zeros((len(y), len(classes)))
    for i, c in enumerate(classes):
        y_onehot[y == c, i] = 1
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_onehot = pd.DataFrame(data=y_onehot, index=y.index, columns=classes)
    return y_onehot


def load_results(experiment_name, dataset_name, suffix=None, reset=False):
    path, path_txt = get_paths(experiment_name, dataset_name, suffix=suffix)
    if not os.path.exists(path): dill.dump({
        'dataset': dataset_name,
    }, open(path, 'wb'))
    if reset:
        results = {}
    else:
        results = dill.load(open(path, 'rb'))
    return results


def save_results(results, experiment_name, dataset_name, suffix=None):
    path, path_txt = get_paths(experiment_name, dataset_name, suffix=suffix)
    write_to_txt(results, path_txt, exclude=exclude_txt)
    dill.dump(results, open(path, 'wb'))


def write_to_txt(results: dict, path: str, exclude: List[str] = None) -> None:
    """
    Write results to txt file

    Parameters
    ----------
    results: dict
        Results dictionary
    path: str
        Path to txt file
    exclude: List[str]
        List of keys to exclude from the txt file
    """
    with open(path, 'w') as f:
        write_to_textio(results, f, "", exclude=exclude)


def write_to_textio(results: dict, f: TextIO, prefix: str = "", exclude: List[str] = None) -> None:
    """
    Write results to txt file

    Parameters
    ----------
    results: dict
        Results dictionary
    f: `TextIO`
        `TextIO` object to write to
    prefix: str
        Prefix to add to each line
    exclude: List[str]
        List of keys to exclude from the txt file
    """
    if exclude is None:
        exclude = []

    for key, value in results.items():
        if key in exclude or isinstance(key, int):
            continue
        if isinstance(value, dict):
            f.write(prefix + '%s:\n' % str(key))
            write_to_textio(value, f, "     " + prefix, exclude=exclude)
            f.write('\n')
        else:
            # Round floats for readability
            value = round_sign_fig(value) if isinstance(value, float) else value
            if (isinstance(value, list) or isinstance(value, np.ndarray)) \
                    and all(isinstance(item, float) for item in value):
                value = [round_sign_fig(x) for x in value]

            # Cut too long strings for readability
            value = value[:20] if isinstance(value, str) and len(value) > 20 else value
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                value = [x[:20] for x in value]

            f.write(prefix + '%s: %s\n' % (key, value))


# \Section: Project specific
# Exclude uninformative items from txt file
exclude_txt = [Keys.clf, Keys.average_error, Keys.std_error]


def get_file_name_base(dataset_name, suffix=None):
    """
    Get file name base for results file

    Parameters
    ----------
    dataset_name: str
        Name of the dataset
    suffix: str, optional, default None
        Suffix to add to the file name

    Returns
    -------
    str
        File name base
    """
    return f'{dataset_name}' \
           f'{suffix}'


def get_clf_full_name(clf, target_transformer_name=None, feature_transformer_name=None):
    """
    Get the full name of a classifier

    Parameters
    ----------
    clf
        Classifier
    target_transformer_name: str, optional, default None
        Name of the target transformer
    feature_transformer_name: str, optional, default None
        Name of the feature transformer

    Returns
    -------
    str
        Full name of the classifier
    """
    return (f"{clf.name}{'__f_' + feature_transformer_name if feature_transformer_name is not None else ''}"
            f"{'__' + target_transformer_name if target_transformer_name is not None else ''}")


def get_paths(experiment_name, dataset_name, suffix=None):
    """
    Get paths for results file and txt file

    Parameters
    ----------
    experiment_name: str
        The name of the experiments whose results are being saved
    dataset_name: str
        Name of the dataset
    suffix: str, optional, default None
        Suffix to add to the file name

    Returns
    -------
    str
        Path to results file
    str
        Path to txt file
    """
    results_dir = get_results_dir()
    # if "dtai" in base:
    #     # config = base.split("/")[-1]
    #     base = "/cw/dtaiarch/ml/2021-LorenNuyts/evaluation"
    #     # if not os.path.exists(base):
    #     #     os.mkdir(base)
    #         # os.mkdir(os.path.join(base, 'results'))
    # else:
    #     splitted = base.split("/")
    #     pos = splitted.index("src")
    #     base = "/".join(splitted[:pos])
    file_name = get_file_name_base(dataset_name, suffix)
    path_start = os.path.join(results_dir, experiment_name, file_name)

    if not os.path.exists(os.path.dirname(path_start)):
        os.makedirs(os.path.dirname(path_start))
    path = path_start + '.pkl'
    path_txt = path_start + '.txt'
    return path, path_txt


def get_results_dir():
    """
    Get the directory where results are stored. If the environment variable RESULTS_DIR is set, it is used. Otherwise,
    the default path is used.
    :return: str
        Path to the results directory
    """
    try:
        return os.environ["RESULTS_DIR"]
    except KeyError:
        base = os.path.dirname(os.path.realpath(__file__))
        if "dtai" in base:
            return "/cw/dtaiarch/ml/2021-LorenNuyts/evaluation/results"
        else:
            splitted = base.split("/")
            pos = splitted.index("src")
            path = "/".join(splitted[:pos])
            return os.path.join(path, "results")

