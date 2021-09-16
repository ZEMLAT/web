import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'

hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)

hist.head(5)


def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


train, test = train_test_split(hist, test_size=0.2)


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)


# line_plot(train[target_col], test[target_col], 'training', 'test', title='')


def line_plot_args(title='', lw=2, **kwargs):
    line1 = kwargs.pop('line1') if 'line1' in kwargs else []
    line2 = kwargs.pop('line2') if 'line2' in kwargs else []
    line3 = kwargs.pop('line3') if 'line3' in kwargs else []
    line4 = kwargs.pop('line4') if 'line4' in kwargs else []
    line5 = kwargs.pop('line5') if 'line5' in kwargs else []
    line6 = kwargs.pop('line6') if 'line6' in kwargs else []
    line7 = kwargs.pop('line7') if 'line7' in kwargs else []
    line8 = kwargs.pop('line8') if 'line8' in kwargs else []
    line9 = kwargs.pop('line9') if 'line9' in kwargs else []

    label1 = kwargs.pop('label1') if 'label1' in kwargs else []
    label2 = kwargs.pop('label2') if 'label2' in kwargs else []
    label3 = kwargs.pop('label3') if 'label3' in kwargs else []
    label4 = kwargs.pop('label4') if 'label4' in kwargs else []
    label5 = kwargs.pop('label5') if 'label5' in kwargs else []
    label6 = kwargs.pop('label6') if 'label6' in kwargs else []
    label7 = kwargs.pop('label7') if 'label7' in kwargs else []
    label8 = kwargs.pop('label8') if 'label8' in kwargs else []
    label9 = kwargs.pop('label9') if 'label9' in kwargs else []

    fig, ax = plt.subplots(1, figsize=(13, 7))
    if (len(line1) > 0):
        ax.plot(line1, label=label1, linewidth=lw, color='red')
    if (len(line2) > 0):
        ax.plot(line2, label=label2, linewidth=lw, color='green')
    if (len(line3) > 0):
        ax.plot(line3, label=label3, linewidth=lw, color='blue')
    if (len(line4) > 0):
        ax.plot(line4, label=label4, linewidth=lw, color='gold')
    if (len(line5) > 0):
        ax.plot(line5, label=label5, linewidth=lw, color='deepskyblue')
    if (len(line6) > 0):
        ax.plot(line6, label=label6, linewidth=lw, color='purple')
    if (len(line7) > 0):
        ax.plot(line7, label=label7, linewidth=lw, color='orange')
    if (len(line8) > 0):
        ax.plot(line8, label=label8, linewidth=lw, color='teal')
    if (len(line9) > 0):
        ax.plot(line9, label=label9, linewidth=lw, color='crimson')
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)


def normalise_zero_base(df):
    return df / df.iloc[0] - 1


def normalise_min_max(df):
    return (df - df.min()) / (data.max() - df.min())


def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx:(idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data,
                     output_size,
                     neurons=100,
                     activ_func='linear',
                     dropout=0.2,
                     loss='mse',
                     optimizer='adam'):
    model = Sequential()
    model.add(
        LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'

train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist,
    target_col,
    window_len=window_len,
    zero_base=zero_base,
    test_size=test_size)

model = build_lstm_model(X_train,
                         output_size=1,
                         neurons=lstm_neurons,
                         dropout=dropout,
                         loss=loss,
                         optimizer=optimizer)
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    shuffle=True)

import matplotlib.pyplot as plt

targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)

from sklearn.metrics import mean_squared_error

MAE = mean_squared_error(preds, y_test)
MAE

from sklearn.metrics import r2_score

R2 = r2_score(y_test, preds)
R2

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
# line_plot(targets, preds, 'actual', 'prediction', lw=3)

# this only makes theplot going wrong
# plt.plot(history.history['loss'], 'r', linewidth=2, label='Train loss')
# plt.plot(history.history['val_loss'],
#          'g',
#          linewidth=2,
#          label='Validation loss')

# plt.plot(preds,
#         #  'g',
#          linewidth=2,
#          label='Predictions')

# line_plot(train[target_col], test[target_col], 'training', 'test', title='')

line_plot_args('',
               2,
               line1=train[target_col], label1='training',
               line2=test[target_col], label2='test',
            #    line3=hist[target_col], label3='real',
               line4=preds, label4='predictions')

plt.title('LSTM')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()