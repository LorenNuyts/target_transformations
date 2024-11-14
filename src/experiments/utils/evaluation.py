import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from src.experiments.utils import elementwise_mean


def relative_squared_error(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / np.sum(np.square(np.mean(y_true) - y_pred))

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100


def compute_metrics(data, predictions, target_transformer_name):
    if isinstance(predictions, pd.DataFrame) and predictions.index.nlevels > 1:
        return compute_metrics_multiple_instances(data, predictions, target_transformer_name)
    else:
        return compute_metrics_single_instance(data, predictions, target_transformer_name)

def compute_metrics_multiple_instances(data, predictions, target_transformer_name):
    transformed_rse = []
    transformed_mape = []
    transformed_smape = []
    transformed_error = []
    back_transformed_rse = []
    back_transformed_mape = []
    back_transformed_smape = []
    back_transformed_error = []
    for i in predictions.index.get_level_values(0).unique():
        all_test_data = data.ytest
        all_Itest = data.Itest
        data.ytest = data.ytest.loc[i]
        data.Itest = data.Itest[all_test_data.index.get_loc(i)]

        predictions_i = predictions.loc[i].squeeze()

        transformed_rse_i, transformed_mape_i, transformed_smape_i, transformed_error_i, back_transformed_rse_i, back_transformed_mape_i, back_transformed_smape_i, back_transformed_error_i = compute_metrics_single_instance(data, predictions_i, target_transformer_name)
        transformed_rse.append(transformed_rse_i)
        transformed_mape.append(transformed_mape_i)
        transformed_smape.append(transformed_smape_i)
        transformed_error.append(transformed_error_i)
        back_transformed_rse.append(back_transformed_rse_i)
        back_transformed_mape.append(back_transformed_mape_i)
        back_transformed_smape.append(back_transformed_smape_i)
        back_transformed_error.append(back_transformed_error_i)

        data.ytest = all_test_data
        data.Itest = all_Itest
    average_transformed_rse = np.nanmean(transformed_rse)
    average_transformed_mape = np.nanmean(transformed_mape)
    average_transformed_smape = np.nanmean(transformed_smape)
    average_transformed_error = elementwise_mean(transformed_error)
    average_back_transformed_rse = np.nanmean(back_transformed_rse)
    average_back_transformed_mape = np.nanmean(back_transformed_mape)
    average_back_transformed_smape = np.nanmean(back_transformed_smape)
    average_back_transformed_error = elementwise_mean(back_transformed_error)
    return (average_transformed_rse, average_transformed_mape, average_transformed_smape, average_transformed_error,
            average_back_transformed_rse, average_back_transformed_mape, average_back_transformed_smape,
            average_back_transformed_error)

def compute_metrics_single_instance(data, predictions, target_transformer_name):
    # Compute RSE, MAPE, and SMAPE
    predictions_values = predictions.values if not isinstance(predictions, np.ndarray) else predictions
    ytest = data.ytest.values if not isinstance(predictions, np.ndarray) else data.ytest
    # ytest = data.ytest.reset_index(drop=True)
    transformed_rse = relative_squared_error(ytest, predictions_values)
    transformed_mape = mean_absolute_percentage_error(ytest, predictions_values)
    transformed_smape = symmetric_mean_absolute_percentage_error(ytest, predictions_values)
    transformed_error = ytest - predictions_values

    # Compute backtransformed RSE, MAPE, and SMAPE
    back_transformed_pred = predictions_values
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
