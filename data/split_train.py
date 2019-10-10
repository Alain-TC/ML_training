import pandas as pd
import os
cwd = os.getcwd()

train = pd.read_csv("{}/data/train.csv".format(cwd))
train = train.sample(frac=1, random_state=1)
shape = int(train.shape[0] * 0.8)

learning_train = train.iloc[:shape, :]
learning_test = train.iloc[shape:, :]

learning_train.to_csv("{}/data/learning_train.csv".format(cwd), index=False)
learning_test.to_csv("{}/data/learning_test.csv".format(cwd), index=False)
