import numpy as np

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin




class ReformatImage_28_28(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = np.array([x.reshape(28, 28) for x in df])
        return df

class ReformatImage_1_28_28(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.reshape(df.shape[0], 1, 28, 28)
        return df


class Normalize_Image(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.astype('float32')
        df /= 255

        return df