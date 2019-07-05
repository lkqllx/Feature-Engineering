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
import os
import h5py

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
            label.loc[abs(df[name]) <= threshold, 'Label_{}'.format(idx)] = 0

        if int(idx) >= 8:
            "Normalize the dataframe"
            df.loc[:, name] = (df[name] - df[name].mean()) / (df[name].max() - df[name].min())

    "Create a new feature - LOB imbalance"
    df['imbalance'] = df['SP1'] * df['SV1'] - df['BV1'] * df['BP1']
    df['imbalance'] = (df['imbalance'] - df['imbalance'].mean()) / (df['imbalance'].max() - df['imbalance'].min())
    return df, label

def concat_data(path='data/'):
    files = os.listdir(path)
    files = [file for file in files if file != '.DS_Store']
    save_to = 'data/all_data.h5'
    for idx, file in enumerate(files):
        curr_df = pd.read_csv(path+file)
        if idx == 0:
            curr_df.to_hdf(save_to, 'data', mode='w', format='table')
            del curr_df
        else:
            curr_df.to_hdf(save_to, 'data', append=True)
            del curr_df
        if idx % 5 == 0:
            print(f'Processing {idx}')

def comp_BM(file='data/step1_2492.csv'):
    try:
        curr_df = pd.read_csv(file)
    except:
        curr_df = pd.read_hdf(file)
    processed_df, label = preprocess(curr_df)
    shifted_label = label.shift(1)
    shifted_label, label = shifted_label[1:], label[1:]
    BM = {}
    for col in label.columns:
        correct = sum(shifted_label[col] == label[col])
        total = shifted_label.shape[0]
        BM[col] = round(correct / total, 2)
    print(BM)

if __name__ == '__main__':
    comp_BM()
    # concat_data()
    # df = pd.read_csv('data/step1_0050.csv', index_col=0).dropna()
    # processed_df, label = preprocess(df, 0.0001)
    # a = processed_df.values[317:, 8:]
    # cnn = create_network()
    # cnn.fit(processed_df.values[317:, 8:].reshape((-1, 5, 55)), np_utils.to_categorical(label['Label_1'].values[317::5], num_classes=3),
    #         epochs=50, validation_split=0.05)
    # cnn.model.summary()
