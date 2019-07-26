from MNIST.utils.argparser import *
from MNIST.connection.io import *
from MNIST.preprocessing.undersampling import *
from MNIST.pipelines import pipe_feature_engineering
from MNIST.preprocessing.transformers.feature_engineering import DataframeToMatrix, ColumnSelector
from MNIST.preprocessing.transformers.image_format_transformer import ReformatImage
from MNIST.prediction.CNN import CNN
import numpy as np
from sklearn import *
import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from keras.utils import np_utils
from MNIST.evaluation.metrics import evaluate_performance

import pandas as pd
import os
import pickle


if __name__ == '__main__':
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = "/".join(dir_path.split('/')[:-1])


    #result = pd.concat([target_test_prediction_df, features_test], axis=1, sort=False)

    #target_test_prediction_df.to_csv("{}/data/target_test_prediction_df.csv".format(main_path), index=False)



    preprocessing_pipe = make_pipeline(DataframeToMatrix())


    df_train = pd.read_csv("{}/data/learning_train.csv".format(main_path))
    target_train = df_train[['label']]
    features_train = df_train.drop('label', axis=1, inplace=False)
    preprocessing_pipe.fit(features_train, target_train)
    X_train = preprocessing_pipe.transform(features_train)
    y_train = preprocessing_pipe.transform(target_train)

    df_test = pd.read_csv("{}/data/learning_test.csv".format(main_path))
    target_test = df_test[['label']]
    features_test = df_test.drop('label', axis=1, inplace=False)
    preprocessing_pipe.fit(features_test, target_test)
    X_test = preprocessing_pipe.transform(features_test)
    y_test = preprocessing_pipe.transform(target_test)


    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    pipeline = make_pipeline(ReformatImage(),CNN())

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_train)
    y_pred = pd.DataFrame(np.array([list(x).index(max(x)) for x in y_pred]),columns=["label"])


    performances = evaluate_performance(y_pred, target_train)
    print(performances)

    with open('data/train_performances.json', 'w') as json_file:
        json_file.write(str(performances))

    filename = "{}/model/finalized_model.sav".format(main_path)

    pickle.dump(pipeline, open(filename, 'wb'))
