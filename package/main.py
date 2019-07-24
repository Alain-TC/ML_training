from MNIST.utils.argparser import *
from MNIST.connection.io import *
from MNIST.preprocessing.undersampling import *
from MNIST.pipelines import pipe_feature_engineering
from MNIST.preprocessing.transformers.feature_engineering import DataframeToMatrix, ColumnSelector
import numpy as np
from sklearn import *
import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import pandas as pd
import os


if __name__ == '__main__':
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = "/".join(dir_path.split('/')[:-1])

    df_train = pd.read_csv("{}/data/learning_train.csv".format(main_path))

    target_train = df_train[['label']]
    features_train = df_train.drop('label', axis=1, inplace=False)

    preprocessing_pipe = make_pipeline(DataframeToMatrix())
    preprocessing_pipe.fit(features_train, target_train)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

    clf.fit(features_train, target_train)

    df_test = pd.read_csv('/Users/Alain/GitHub/MNIST_kaggle/data/learning_test.csv')

    target_test = df_test[['label']]
    features_test = df_test.drop('label', axis=1, inplace=False)




    df_test = pd.read_csv("{}/data/learning_test.csv".format(main_path))

    target_test_prediction = clf.predict(features_test)
    target_test_prediction_df = pd.DataFrame(target_test_prediction, columns=["label"])
    result = pd.concat([target_test_prediction_df, features_test], axis=1, sort=False)

    target_test_prediction_df.to_csv("{}/data/target_test_prediction_df.csv".format(main_path), index=False)