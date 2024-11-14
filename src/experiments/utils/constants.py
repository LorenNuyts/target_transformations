import os

import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, RobustScaler

from src.algorithms.transformers import NormalizeTransformer, LogTransformer

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

    rse = "RSE"
    average_rse = "average RSE"
    std_rse = "std RSE"
    transformed_rse = "Transformed RSE"
    average_transformed_rse = "average Transformed RSE"
    std_transformed_rse = "std Transformed RSE"

    mape = "MAPE"
    average_mape = "average MAPE"
    std_mape = "std MAPE"
    transformed_mape = "Transformed MAPE"
    average_transformed_mape = "average Transformed MAPE"
    std_transformed_mape = "std Transformed MAPE"

    smape = "SMAPE"
    average_smape = "average SMAPE"
    std_smape = "std SMAPE"
    transformed_smape = "Transformed SMAPE"
    average_transformed_smape = "average Transformed SMAPE"
    std_transformed_smape = "std Transformed SMAPE"

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
    all_transformers = [transformer_normalized, transformer_quantile_uniform, transformer_quantile_normal,
                        transformer_robustscaler, transformer_powertransformer, transformer_logtransformer,
                        transformer_lntransformer]
    transformer_acronyms = {transformer_normalized: "Norm.", transformer_quantile_uniform: "Uniform",
                            transformer_quantile_normal: "Normal", transformer_powertransformer: "Yeo-Johnson",
                            transformer_logtransformer: "Log10", transformer_lntransformer: "Ln",
                            transformer_robustscaler: "Robust"}


def get_transformer(transformer_name):
    if transformer_name == Keys.transformer_normalized or transformer_name == "normalized":
        return NormalizeTransformer()
    elif transformer_name == Keys.transformer_quantile_uniform or transformer_name == "quantile_uniform":
        return QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='uniform')
    elif transformer_name == Keys.transformer_quantile_normal or transformer_name == "quantile_normal":
        return QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
    elif transformer_name == Keys.transformer_powertransformer or transformer_name == "powertransformer":
        return PowerTransformer()
    elif transformer_name == Keys.transformer_logtransformer or transformer_name == "logtransformer":
        return LogTransformer(base=10)
    elif transformer_name == Keys.transformer_lntransformer or transformer_name == "lntransformer":
        return LogTransformer(base=np.e)
    elif transformer_name == Keys.transformer_robustscaler or transformer_name == "robustscaler":
        return RobustScaler()
    else:
        raise ValueError("Invalid transformer name: {}".format(transformer_name))

