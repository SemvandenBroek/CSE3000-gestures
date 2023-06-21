import keras
import numpy as np

from keras import layers
from utils import Gestures


def build_simple_model(input_shape, num_classes=len(Gestures)):
    model = keras.Sequential(name="Simple")
    model.add(layers.Input(shape=input_shape, dtype=np.float32, name="input"))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))

    return model


def build_lstm_stateless_model(input_shape, lstm_units=64, dense_1_units=64, dense_1=False, dropout=True,
                               bidirectional=False,
                               num_classes=len(Gestures)):
    # https://towardsdatascience.com/time-series-classification-for-human-activity-recognition-with-lstms-using-tensorflow-2-and-keras-b816431afdff
    model = keras.Sequential(name="LSTM_Stateless")

    model.add(layers.Input(shape=input_shape, dtype=np.float32, name="sensor_image"))

    if bidirectional:
        model.add(layers.Bidirectional(
            layers.LSTM(units=lstm_units, name="lstm_bidirectional", time_major=False, return_sequences=True)))
    else:
        model.add(layers.LSTM(units=lstm_units, name="lstm", time_major=False, return_sequences=True))

    # Extra
    if dropout:
        model.add(layers.Dropout(rate=0.5, name="dropout"))

    model.add(layers.Flatten())

    if dense_1:
        model.add(layers.Dense(units=dense_1_units, activation='relu', name="dense_1"))

        if dropout:
            model.add(layers.Dropout(rate=0.5, name="dropout_2"))

    # Output stage
    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))
    model.summary()
    return model


def build_conv_lstm_model(input_shape=(100, 3), num_classes=len(Gestures), lstm_units=64) -> keras.Model:
    model = keras.Sequential(name="Conv_LSTM")

    model.add(layers.Input(shape=input_shape, dtype=np.float32))
    model.add(layers.Reshape((20, 15)))
    model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu')))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu')))
    model.add(keras.layers.Dropout(0.25))
    # model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu')))
    # model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(64, kernel_size=(3, 1), activation='relu')))
    model.add(keras.layers.Reshape((20, 64)))
    model.add(keras.layers.LSTM(lstm_units))

    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))

    return model
