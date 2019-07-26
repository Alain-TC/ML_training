import numpy as np

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin




class ReformatImage(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = np.array([x.reshape(28, 28) for x in df])
        df = df.reshape(df.shape[0], 1, 28, 28)
        df = df.astype('float32')
        df /= 255

        return df

        # 6. Preprocess class labels
        #y_train = np_utils.to_categorical(y_train, 10)
        #y_test = np_utils.to_categorical(y_test, 10)
