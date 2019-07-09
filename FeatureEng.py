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
from statsmodels.api import OLS, add_constant
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def parse_date(date):
    return dt.datetime.strptime(date, '%Y-%m-%d')

def next_weekday(date):
    date += dt.timedelta(days=1)
    for _ in range(7):
        if date.weekday() < 5:
            break
        else:
            date += dt.timedelta(days=1)
    return date

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

def split_dataset(path='data/'):
    files = os.listdir(path)
    test_file = 'step1_0050.csv'
    ticker = test_file.split('.')[0].split('_')[1]
    if not os.path.exists(f'data/test/{ticker}'):
        os.mkdir(f'data/test/{ticker}')

    test_df = pd.read_csv('data/'+test_file, index_col=0).dropna()
    test_df['date_str'] = test_df['date'].values
    test_df.date = test_df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

    date_set = set(test_df['date_str'].values.tolist())
    date_key = {date: dt.datetime.strptime(date, '%Y-%m-%d') for date in date_set}
    date_key = sorted(date_key.items(), key=lambda x:x[1])

    """Decide how many training and testing sample"""
    sorted_date_set = [date for date, _ in date_key][3:33]


    TRAIN_DURATION = 28


    """
    Initialize directory and test file
    """
    for idx, curr_date in enumerate(sorted_date_set):
        if idx % 1 == 0:
            print(f'Test Done - {idx}')

        training = fetch_training(test_df, training_start=curr_date, duration=TRAIN_DURATION)
        test_date = dt.datetime.strptime(curr_date, '%Y-%m-%d') + dt.timedelta(days=TRAIN_DURATION)
        for _ in range(100):
            if test_date.strftime('%Y-%m-%d') in date_set:
                break
            else:
                test_date += dt.timedelta(days=1)
        testing = fetch_testing(test_df, testing_start=test_date.strftime('%Y-%m-%d'), duration=1, curr_date_set=date_set)
        training.to_csv(f'data/training/training_{idx}.csv')
        testing.to_csv(f'data/test/{ticker}/test_{idx}.csv')


    """Append training data to different test period"""
    for count, file in enumerate(files):
        print('-'*20, f'{file} - Count {count}', '-'*20)
        if file.split('.')[1] == 'csv' and file != test_file:
            df = pd.read_csv(path+file, index_col=0)
            df.date = df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            for idx, curr_date in enumerate(sorted_date_set):
                if idx % 10 == 0:
                    print(f'{file} Done - {idx}')
                training = fetch_training(df, training_start=curr_date, duration=TRAIN_DURATION)
                prev_training = pd.read_csv(f'data/training/training_{idx}.csv', index_col=0)
                prev_training = prev_training.append(training, sort=False)
                prev_training.to_csv(f'data/training/training_{idx}.csv')


def fetch_training(df, training_start, duration):
    """
    Fetch the training data for a fixed period
    :return: output - in that duration
    """
    # if not dt.datetime.strptime(training_start, '%Y-%m-%d').weekday() < 5:
    #     print('Start date should be weekday!')
    #     return False

    dt_training_start = dt.datetime.strptime(training_start, '%Y-%m-%d')
    dt_training_end = dt_training_start + dt.timedelta(days=duration)
    try:
        return df.loc[(df.date < dt_training_end) & (df.date >= dt_training_start)]

    except:
        print('Error fetch_training')
        return False

def fetch_testing(df, testing_start, curr_date_set, duration=1):
    """
    Fetch the testing data for a fixed period but the testing start may not in the dataframe
    So we need to look for the next available start date
    :return: output - in that duration
    """
    dt_testing_start = dt.datetime.strptime(testing_start, '%Y-%m-%d')
    for _ in range(200):
        if testing_start in curr_date_set:
            dt_testing_end = dt_testing_start + dt.timedelta(days=duration)
            break
        else:
            dt_testing_start = next_weekday(dt_testing_start)
    try:
        return df.loc[(df.date < dt_testing_end) & (df.date >= dt_testing_start)]

    except:
        print('Error fectch_testing')
        return False

class Preprocess:
    def preprocess(self, df: pd.DataFrame, threshold: float = 0.0001, time_step=5, num_feature=8, normalize=True) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        label = pd.DataFrame(index=df.index)
        for idx, name in enumerate(df.columns):
            "Creat training labels"
            if re.match('Y_M_.', name):
                idx = name.split('_')[2]
                label.loc[:, 'Label_{}'.format(idx)] = np.ones([df.shape[0], ])
                label.loc[df[name] < -threshold, 'Label_{}'.format(idx)] = 2
                label.loc[abs(df[name]) <= threshold, 'Label_{}'.format(idx)] = 0

            if int(idx) >= num_feature and normalize:
                "Normalize the dataframe"
                df.loc[:, name] = (df[name] - df[name].mean()) / (df[name].max() - df[name].min())

        "Create a new feature - LOB imbalance"
        df['imbalance'] = df['SP1'] * df['SV1'] - df['BV1'] * df['BP1']
        df['imbalance'] = (df['imbalance'] - df['imbalance'].mean()) / (df['imbalance'].max() - df['imbalance'].min())


        y_reg = df.iloc[:, 2:7]
        df = df.iloc[:, num_feature:]
        for idx in range(0, 10 - 1):
            if (df.size - (df.shape[1]) * idx) % (time_step * df.shape[1]) == 0:
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
        self.nn = self.create_network()
        self.nn.fit(self.processed_df.values.reshape((-1, self.time_step, 55)),
                np_utils.to_categorical(self.label['Label_1'], num_classes=3),
                epochs=20, validation_split=0.05, shuffle=False)

    def predict(self):
        self.nn.predict()


class Regressor(Preprocess):
    def __init__(self, df, train_interval=14, test_interval=1, test_period=100, num_feature=16):
        """
        Regressor for linear regression
        :param df: input data
        :param train_interval: length of training period
        :param test_period: how many days to be tested
        """
        super().__init__()
        self.df = df
        self.train_interval = train_interval
        self.test_period = test_period
        self.test_interval = test_interval
        self.curr_start = df.date[0]
        self.date = df.date.reset_index(drop=True)
        self.curr_pointer = 0 # pointed to the index of current date
        self.preprocess_df, _, self.label = self.preprocess(df=df, time_step=1, num_feature=num_feature, normalize=False)


    def fetch_data(self):
        """
        Fetch data for a fixed duration.

        If train_interval = 14, test_interval = 1
        The function will fetch the training data of next two weeks' available data regardless the weekends
        :return: training and testing data
        """
        train_start = dt.datetime.strptime(self.curr_start, '%Y-%m-%d')
        train_end = train_start + dt.timedelta(days=self.train_interval)

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
                test_end = dt.datetime.strptime(self.date[test_start_idx], '%Y-%m-%d') # only one day test
                break

        for test_idx in range(test_start_idx, self.date.shape[0]):
            if (test_end - dt.datetime.strptime(self.date[test_idx], '%Y-%m-%d')).days <= -1:
                test_end_idx = test_idx - 1
                break
        print(f'Train start  - {self.date[train_start_idx]}')
        print(f'Train end  - {self.date[train_end_idx]}')
        print(f'Test start  - {self.date[test_start_idx]}')
        print(f'Test end  - {self.date[test_end_idx+1]}')

        return self.preprocess_df[train_start_idx:train_end_idx], self.label[train_start_idx:train_end_idx],\
               self.preprocess_df[test_start_idx:test_end_idx], self.label[test_start_idx:test_end_idx]

    def run_regr(self):
        preprocess_df = add_constant(self.preprocess_df)
        regr = OLS(self.label['Y_M_1'], preprocess_df).fit()
        print(regr.summary())
        for _ in range(50):
            try:
                train_x, train_y, test_x, test_y = self.fetch_data()
                # regr = OLS(train_y['Y_M_1'], train_x).fit()
                # print(regr.summary())
                # y_pred = regr.predict(test_x)
                # # print(f'Mean error is {mean_squared_error(test_y, y_pred)}')
                # print(f'R-square is {r2_score(test_y, y_pred)}')
            except:
                pass


if __name__ == '__main__':
    # comp_BM(file='data/step1_1101.csv')
    # concat_data()

    """
    Splite the data
    """
    split_dataset()

    """
    Preprocess dataframe and reshape the data fed into neural networks
    """
    # df = pd.read_csv('data/step1_0050.csv', index_col=0).dropna()
    # nn = NeuralNet(df, time_step=30)
    # nn.run()


    """
    1.Preprocess dataframe regarding to different time scale
    2.Build the regression model
    """
    # df = pd.read_csv('data/step1_0050.csv', index_col=0).dropna()
    # df['date_str'] = df['date'].values
    # df.date = df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    # # training = fetch_training(df, training_start='2018-01-22', duration=14)
    # testing = fetch_testing(df, testing_start='2018-01-22', duration=1)
    # df.drop('date_str', axis=1)
    # reg = Regressor(df=df)







