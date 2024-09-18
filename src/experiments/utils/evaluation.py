import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def relative_squared_error(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / np.sum(np.square(np.mean(y_true) - y_pred))

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100


def compute_metrics(data, predictions, target_transformer_name):
    # Compute RSE, MAPE, and SMAPE
    predictions = predictions.values
    ytest = data.ytest.values if isinstance(data.ytest, pd.Series) else data.ytest
    # ytest = data.ytest.reset_index(drop=True)
    transformed_rse = relative_squared_error(ytest, predictions)
    transformed_mape = mean_absolute_percentage_error(ytest, predictions)
    transformed_smape = symmetric_mean_absolute_percentage_error(ytest, predictions)
    transformed_error = ytest - predictions

    # Compute backtransformed RSE, MAPE, and SMAPE
    back_transformed_pred = predictions
    back_transformed_y = ytest
    transformation_failed = False
    if target_transformer_name is not None:
        try:
            back_transformed_pred = data.other_params['target_transformer'].inverse_transform(
                back_transformed_pred.reshape(-1, 1)).ravel()
            back_transformed_y = data.other_params['target_transformer'].inverse_transform(back_transformed_y.reshape(-1, 1)).ravel()
        except ValueError:
            transformation_failed = True
    if 'contextual_transform_feature' in data.other_params.keys():
        try:
            back_transformed_pred = data.inverse_contextual_transform(back_transformed_pred)
            back_transformed_y = data.inverse_contextual_transform(back_transformed_pred)
        except ValueError:
            transformation_failed = True
    if transformation_failed:
        back_transformed_rse = np.nan
        back_transformed_mape = np.nan
        back_transformed_smape = np.nan
        back_transformed_error = np.nan
    else:
        back_transformed_rse = relative_squared_error(back_transformed_y, back_transformed_pred)
        back_transformed_smape = symmetric_mean_absolute_percentage_error(back_transformed_y, back_transformed_pred)
        back_transformed_error = back_transformed_y - back_transformed_pred
        pred_nan: np.ndarray = np.isnan(back_transformed_pred)
        y_nan: np.ndarray = np.isnan(back_transformed_y)
        union_nan = np.union1d(np.where(pred_nan == True), np.where(y_nan == True))
        if len(union_nan) > 0 and len(back_transformed_pred[union_nan]) < 0.1 * len(back_transformed_pred):
            back_transformed_pred_cleaned = np.delete(back_transformed_pred, union_nan)
            back_transformed_y_cleaned = np.delete(back_transformed_y, union_nan)
            back_transformed_mape = mean_absolute_percentage_error(back_transformed_y_cleaned, back_transformed_pred_cleaned)
        else:
            back_transformed_mape = mean_absolute_percentage_error(back_transformed_y, back_transformed_pred)

    return (transformed_rse, transformed_mape, transformed_smape, transformed_error,
            back_transformed_rse, back_transformed_mape, back_transformed_smape, back_transformed_error)




    # # Target transformation with/without contextual transformation
    # if target_transformer_name is not None:
    #     error = None
    #     try:
    #         backtransformed_pred = data.other_params['target_transformer'].inverse_transform(
    #             predictions.reshape(-1, 1)).ravel()
    #         backtransformed_y = data.other_params['target_transformer'].inverse_transform(data.ytest.to_frame()).squeeze()
    #         if 'contextual_transform_feature' in data.other_params.keys():
    #             backtransformed_pred = data.inverse_contextual_transform(backtransformed_pred)
    #             backtransformed_y = data.inverse_contextual_transform(backtransformed_pred)
    #         else:
    #             backtransformed_y = data.ytest
    #         error = transformed_y - transformed_pred
    #         score = root_mean_squared_error(transformed_y, transformed_pred)
    #         transformed_rse = relative_squared_error(transformed_y, transformed_pred)
    #     except ValueError:
    #         score = np.nan
    #         transformed_rse = np.nan
    #         if error is None:  # The transformation failed
    #             error = np.nan
    #
    # # No target transformation, but there is a contextual transformation
    # elif 'contextual_transform_feature' in data.other_params.keys():
    #     transformed_pred = data.inverse_contextual_transform(predictions)
    #     transformed_ytest = data.inverse_contextual_transform(data.ytest)
    #     score = root_mean_squared_error(transformed_ytest, transformed_pred)
    #     transformed_rse = relative_squared_error(transformed_ytest, transformed_pred)
    #     error = transformed_ytest - transformed_pred
    #
    # # No target transformation, no contextual transformation
    # else:
    #     score = root_mean_squared_error(data.ytest, predictions)
    #     transformed_rse = relative_squared_error(data.ytest, predictions)
    #     error = data.ytest - predictions
    # if target_transformer_name is not None:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         if target_transformer_name == Keys.transformer_normalized:
    #             ytest = data.other_params['target_transformer'].inverse_transform(data.ytest.to_frame()).squeeze()
    #         else:
    #             ytest = data.other_params['target_transformer'].transform(data.ytest.to_frame()).ravel()
    # else:
    #     ytest = data.ytest
    # nrmse = root_mean_squared_error(ytest, predictions) / (ytest.max() - ytest.min())
    # rse = relative_squared_error(ytest, predictions)
    # return error, nrmse, rse, score, transformed_rse
