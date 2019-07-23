import pandas as pd


def undersample(df, new_ratio):
    negative_target_df = df.loc[df.convert == 0]
    positive_target_df = df.loc[df.convert == 1]

    ratio_pos_neg = float(positive_target_df.shape[0])/negative_target_df.shape[0]

    new_negative_target_df = negative_target_df.sample(frac=ratio_pos_neg/new_ratio)
    new_dataframe = positive_target_df.append(new_negative_target_df)

    return new_dataframe.sample(frac=1)
