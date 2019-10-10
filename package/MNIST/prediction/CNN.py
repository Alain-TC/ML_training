import numpy as np

np.random.seed(123)  # for reproducibility
from sklearn.base import BaseEstimator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

class CNN(BaseEstimator):
    def __init__(self):
        pass

    def _create_model(self):
        self.model = Sequential()

        self.model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))
        self.model.add(Convolution2D(32, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

    def _compile_model(self):
        # 8. Compile model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def fit(self, df, y=None):
        self._create_model()
        self._compile_model()
        # 9. Fit model on training data
        self.model.fit(df, y,
                  batch_size=32, nb_epoch=1, verbose=1)

    def predict(self, df):
        return self.model.predict(df)

