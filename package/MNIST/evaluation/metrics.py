import numpy as np


def mean_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6))


def median_percentage_error(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred) / (y_true + 1e-6))


def within_error(y_true, y_pred, error=0.1):
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6) <= error)


def root_mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** 0.5


def mean_absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()


def prediction_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]

