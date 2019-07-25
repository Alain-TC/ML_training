from MNIST.evaluation.metrics import mean_absolute_error, \
    mean_percentage_error, \
    prediction_correlation, \
    within_error, \
    root_mean_squared_error, \
    median_percentage_error
from sklearn.metrics import classification_report, confusion_matrix
import os
import pandas as pd
import numpy as np
import json

def evaluate_performance(y_test, y_pred):
    performance = {}
    performance['MAPE'] = float(mean_percentage_error(y_test, y_pred))
    performance['MDAPE'] = float(median_percentage_error(y_test, y_pred))
    performance['RMSE'] = float(root_mean_squared_error(y_test, y_pred))
    performance['MAE'] = float(mean_absolute_error(y_test, y_pred))
    performance['confusion'] = confusion_matrix(y_test, y_pred)
    performance['TP'] = float(sum(list(x == y for x, y in zip(y_test.values, y_pred.values)))) / len(y_test.values)
    #performance['classification_report'] = classification_report(y_test, y_pred)

    return performance

if __name__ == '__main__':

    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = "/".join(dir_path.split('/')[:-1])

    df_test = pd.read_csv("{}/data/learning_test.csv".format(main_path))
    target_test_prediction_df = pd.read_csv("{}/data/target_test_prediction_df.csv".format(main_path))

    performances = evaluate_performance(df_test[["label"]], target_test_prediction_df)

    with open('data/performances.json', 'w') as json_file:
        json_file.write(str(performances))


