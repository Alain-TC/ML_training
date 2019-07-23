import pandas as pd
from sklearn.base import TransformerMixin
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Undersampler(TransformerMixin):
    def __init__(self, new_ratio):
        self.new_ratio = new_ratio

    def fit(self, df=None):
        return self

    def transform(self, df):
        negative_target_df = df.loc[df.convert == 0]
        positive_target_df = df.loc[df.convert == 1]

        ratio_pos_neg = float(positive_target_df.shape[0])/negative_target_df.shape[0]

        new_negative_target_df = negative_target_df.sample(frac=ratio_pos_neg/self.new_ratio)
        new_dataframe = positive_target_df.append(new_negative_target_df)

        return new_dataframe.sample(frac=1)