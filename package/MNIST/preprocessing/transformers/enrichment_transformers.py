import logging
from scipy import ndimage, misc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def f(x):
    l= []
    for i in [15, 30]:
        l.append(ndimage.rotate(x, i, reshape=False))
        l.append(ndimage.rotate(x, -i, reshape=False))
    l.append(x)
    return l

class Enrich_Images(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = np.array([y for x in df for y in f(x)])
        df = np.array([x.reshape(28, 28) for x in df])
        return df

    def transform_target(self, y):
        new_y = np.array([w for x in y for w in (x, x, x, x, x)])
        return new_y