# import keras as ks
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
# from keras.layers.core import Dense
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM


def create_network():
    seq = Sequential()

    seq.add(LSTM(128, input_shape=(5, 55), return_sequences=True))
    seq.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dense(3, activation='softmax'))
    seq.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
    return seq


def preprocess(df: pd.DataFrame, threshold: float=0.0001) -> (pd.DataFrame, pd.DataFrame):
    label = pd.DataFrame(index=df.index)
    for idx, name in enumerate(df.columns):
        "Creat training labels"
        if re.match('Y_M_.', name):
            idx = name.split('_')[2]
            label.loc[:, 'Label_{}'.format(idx)] = np.ones([df.shape[0],])
            label.loc[df[name] < -threshold, 'Label_{}'.format(idx)] = 2
            label.loc[abs(df[name]) <= threshold , 'Label_{}'.format(idx)] = 0

        if int(idx) >= 8:
            "Normalize the dataframe"
            df.loc[:, name] = (df[name] - df[name].mean()) / (df[name].max() - df[name].min())

    "Create a new feature - LOB imbalance"
    df['imbalance'] = df['SP1'] * df['SV1'] - df['BV1'] * df['BP1']
    df['imbalance'] = (df['imbalance'] - df['imbalance'].mean()) / (df['imbalance'].max() - df['imbalance'].min())
    return df, label


if __name__ == '__main__':
    df = pd.read_csv('data/step1_0050.csv', index_col=0).dropna()
    processed_df, label = preprocess(df, 0.0001)
    a = processed_df.values[317:, 8:]
    cnn = create_network()

    cnn.fit(processed_df.values[317:, 8:].reshape((-1, 5, 55)), np_utils.to_categorical(label['Label_1'].values[317::5], num_classes=3),
            epochs=50, validation_split=0.05)
    cnn.model.summary()
