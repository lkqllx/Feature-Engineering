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
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

def comp_BM(file='data/step1_0050.csv'):
    try:
        curr_df = pd.read_csv(file, index_col=0)
    except:
        curr_df = pd.read_hdf(file)
    nn = NeuralNet(curr_df)
    _, label, _ = nn.preprocess(curr_df)
    shifted_label = label.shift(1)
    shifted_label, label = shifted_label[1:], label[1:]
    BM = {}
    for col in label.columns:
        correct = sum(shifted_label[col] == label[col])
        total = shifted_label.shape[0]
        BM[col] = round(correct / total, 2)
    print(BM)


class Preprocess:
    def preprocess(self, df: pd.DataFrame, threshold: float = 0.0001, time_step=5) -> (pd.DataFrame, pd.DataFrame):
        label = pd.DataFrame(index=df.index)
        for idx, name in enumerate(df.columns):
            "Creat training labels"
            if re.match('Y_M_.', name):
                idx = name.split('_')[2]
                label.loc[:, 'Label_{}'.format(idx)] = np.ones([df.shape[0], ])
                label.loc[df[name] < -threshold, 'Label_{}'.format(idx)] = 2
                label.loc[abs(df[name]) <= threshold, 'Label_{}'.format(idx)] = 0

            if int(idx) >= 8:
                "Normalize the dataframe"
                df.loc[:, name] = (df[name] - df[name].mean()) / (df[name].max() - df[name].min())

        "Create a new feature - LOB imbalance"
        df['imbalance'] = df['SP1'] * df['SV1'] - df['BV1'] * df['BP1']
        df['imbalance'] = (df['imbalance'] - df['imbalance'].mean()) / (df['imbalance'].max() - df['imbalance'].min())


        y_reg = df.iloc[:, 2:7]
        df = df.iloc[:, 8:]
        for idx in range(0, 10 - 1):
            if (df.size - 55 * idx) % (time_step * 55) == 0:
                possible_reshape_row = idx
                break

        return df[possible_reshape_row:], label[possible_reshape_row::time_step], y_reg[possible_reshape_row::time_step]

class NeuralNet(Preprocess):
    def __init__(self, df, time_step = 5):
        self.df = df
        self.time_step = time_step

    def create_network(self):
        seq = Sequential()
        seq.add(LSTM(128, input_shape=(self.time_step, 55), return_sequences=True))
        seq.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        seq.add(Flatten())
        seq.add(Dense(128, activation='relu'))
        seq.add(Dense(3, activation='softmax'))
        seq.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
        return seq

    def run(self):
        self.processed_df, self.label, _ = self.preprocess(self.df, 0.0001)
        nn = self.create_network()
        nn.fit(self.processed_df.values.reshape((-1, self.time_step, 55)),
                np_utils.to_categorical(self.label['Label_1'], num_classes=3),
                epochs=20, validation_split=0.1)

class Regressor(Preprocess):
    """
    TODO: Handle weekend
    """
    def __init__(self, df, train_interval=10, test_interval=1, test_period=100):
        """
        Regressor for linear regression
        :param df: input data
        :param train_interval: length of training period
        :param test_period: how many days to be tested
        """
        self.df = df
        self.train_interval = train_interval
        self.test_period = test_period
        self.test_interval = test_interval
        self.curr_start = df.date[0]
        self.date = df.date.reset_index(drop=True)
        self.curr_pointer = 0 # pointed to the index of current date
        self.preprocess_df, _, self.label = self.preprocess(df=df, time_step=1)


    def fetch_data(self):
        train_start = dt.datetime.strptime(self.curr_start, '%Y-%m-%d')
        train_end = train_start + dt.timedelta(days=self.train_interval)
        test_end = train_end + dt.timedelta(days=self.test_interval)

        train_start_idx = self.curr_pointer
        updated = False
        for train_idx in range(train_start_idx, self.date.shape[0]):
            if ((dt.datetime.strptime(self.date[train_idx], '%Y-%m-%d') - train_start).days >= self.test_interval) and not updated:
                self.curr_pointer = train_idx
                self.curr_start = self.date[self.curr_pointer]
                updated = True

            if (train_end - dt.datetime.strptime(self.date[train_idx], '%Y-%m-%d')).days <= -1:
                train_end_idx = train_idx - 1
                test_start_idx = train_idx
                break

        for test_idx in range(test_start_idx, self.date.shape[0]):
            if (test_end - dt.datetime.strptime(self.date[test_idx], '%Y-%m-%d')).days <= -1:
                test_end_idx = test_idx - 1
                break


        # print(f'Train start  - {self.date[train_start_idx]}')
        # print(f'Train end  - {self.date[train_end_idx]}')
        # print(f'Test start  - {self.date[test_start_idx]}')
        # print(f'Test end  - {self.date[test_end_idx]}')

        return self.preprocess_df[train_start_idx:train_end_idx], self.label[train_start_idx:train_end_idx],\
               self.preprocess_df[test_start_idx:test_end_idx], self.label[test_start_idx:test_end_idx]

if __name__ == '__main__':
    comp_BM(file='data/step1_0050.csv')
    # concat_data()

    """
    Preprocess dataframe and reshape the data fed into neural networks
    """
    df = pd.read_csv('data/step1_0050.csv', index_col=0).dropna()
    nn = NeuralNet(df, time_step=5)
    nn.run()


    """
    1.Preprocess dataframe regarding to different time scale
    2.Build the regression model
    """
    # df = pd.read_csv('data/step1_0050.csv', index_col=0).dropna()
    # reg = Regressor(df=df)
    # for _ in range(100):
    #     try:
    #         train_x, train_y, test_x, test_y = reg.fetch_data()
    #         regr = LinearRegression().fit(train_x, train_y)
    #         y_pred = regr.predict(test_x)
    #         # print(f'Mean error is {mean_squared_error(test_y, y_pred)}')
    #         print(f'R-square is {r2_score(test_y, y_pred)}')
    #     except:
    #         pass






