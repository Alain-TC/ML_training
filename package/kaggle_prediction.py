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

    preprocessing_pipe = make_pipeline(DataframeToMatrix())


    df_test = pd.read_csv("{}/data/test.csv".format(main_path))
    features_test = df_test
    X_test = preprocessing_pipe.transform(features_test)

    filename = "{}/model/finalized_model.sav".format(main_path)
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(X_test)

    y_pred = pd.DataFrame(list(zip([x+1for x in range(28000)],np.array([list(x).index(max(x)) for x in y_pred]))), columns=["ImageId","Label"])

    y_pred.to_csv("{}/data/test_prediction.csv".format(main_path), sep=",", index=False)
