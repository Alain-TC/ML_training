from MNIST.utils.argparser import *
from MNIST.connection.io import *
from MNIST.prediction.pipeline import *
from MNIST.preprocessing.undersampling import *
from sklearn import *

import sklearn

if __name__ == '__main__':
    args = argparser.parse_args()
    input_path = args.input_path

    # Prediction
    select = sklearn.feature_selection.SelectKBest(k=100)

    clf = sklearn.ensemble.RandomForestClassifier()

    parameters = dict(feature_selection__k=[40, 60, 80]
                      ,random_forest__n_estimators=[100, 200, 1000]
                      #,random_forest__min_samples_split = [2, 3]
                      )

    steps = [("feature_selection",select),
             ("random_forest", clf)]

    # Pipeline creation
    pipeline = pipe_feature_engineering(steps)
