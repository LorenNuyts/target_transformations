import os

import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, RobustScaler

from src.algorithms.transform_target import NormalizeTransformer, LogTransformer

SEED = 0
default_suffix = ''

class Keys:
    """
    A class containing strings that are used as dictionary keys throughout the package.
    """

    rmse = "RMSE"
    average_rmse = "average RMSE"
    std_rmse = "std RMSE"

    nrmse = "NRMSE"
    average_nrmse = "average NRMSE"
    std_nrmse = "std NRMSE"

    clf = "clf"
    predictions = "predictions"
    average_predictions = "average predictions"
    std_predictions = "std predictions"
    error = "error"
    average_error = "average error"
    std_error = "std error"

    transformer_normalized = "Normalized"
    transformer_quantile_uniform = "Quantile (uniform)"
    transformer_quantile_normal = "Quantile (normal)"
    transformer_powertransformer = "PowerTransformer"
    transformer_logtransformer = "LogTransformer (log10)"
    transformer_lntransformer = "LogTransformer (ln)"
    transformer_robustscaler = "RobustScaler"


def get_transformer(transformer_name):
    if transformer_name == Keys.transformer_normalized:
        return NormalizeTransformer()
    elif transformer_name == Keys.transformer_quantile_uniform:
        return QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='uniform')
    elif transformer_name == Keys.transformer_quantile_normal:
        return QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
    elif transformer_name == Keys.transformer_powertransformer:
        return PowerTransformer()
    elif transformer_name == Keys.transformer_logtransformer:
        return LogTransformer(base=10)
    elif transformer_name == Keys.transformer_lntransformer:
        return LogTransformer(base=np.e)
    elif transformer_name == Keys.transformer_robustscaler:
        return RobustScaler()
    else:
        raise ValueError("Invalid transformer name: {}".format(transformer_name))

