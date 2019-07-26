from MNIST.evaluation.metrics import mean_absolute_error, \
    mean_percentage_error, \
    prediction_correlation, \
    within_error, \
    root_mean_squared_error, \
    median_percentage_error, evaluate_performance
import os
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.pipeline import make_pipeline
from MNIST.preprocessing.transformers.feature_engineering import DataframeToMatrix, ColumnSelector



if __name__ == '__main__':

    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = "/".join(dir_path.split('/')[:-1])

    #target_test_prediction_df = pd.read_csv("{}/data/target_test_prediction_df.csv".format(main_path))
    preprocessing_pipe = make_pipeline(DataframeToMatrix())


    df_test = pd.read_csv("{}/data/learning_test.csv".format(main_path))
    target_test = df_test[['label']]
    features_test = df_test.drop('label', axis=1, inplace=False)
    preprocessing_pipe.fit(features_test, target_test)
    X_test = preprocessing_pipe.transform(features_test)

    filename = "{}/model/finalized_model.csv".format(main_path)
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(X_test)

    y_pred = pd.DataFrame(np.array([list(x).index(max(x)) for x in y_pred]), columns=["label"])



    performances = evaluate_performance(y_pred, target_test)
    print(performances)

    with open('data/performances.json', 'w') as json_file:
        json_file.write(str(performances))

