import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import time

#Importing stock data
import yfinance as yf
from datetime import date,datetime,timedelta
ticker = '^GSPC'
first_day = datetime(2000, 1, 3)
last_day = datetime(2019, 7, 1)
data = yf.Ticker(ticker).history(interval = '1d', start=first_day, end=last_day)
data.reset_index(inplace=True)


from models import *
from dataUtils import *

close_data = data.Close.to_numpy()

unroll_length = 50
X_train, X_test, y_train, y_test = train_test_split_lstm(close_data, 5, int(close_data.shape[0] * 0.1))
X_train = np.expand_dims(unroll(X_train, unroll_length), axis = 2)
y_train = np.expand_dims(unroll(y_train, unroll_length), axis = 2)
print(int(close_data.shape[0] * 0.1))
print(X_train.shape)
print(y_train.shape)

#model = build_basic_lstm_model(lstm_input_dim = X_train.shape[-1], lstm_output_dim = unroll_length, dense_output_dim = y_train.shape[-1], return_sequences=True)
model = build_att_lstm_model(lstm_input_dim = X_train.shape[-1], lstm_output_dim = unroll_length, dense_output_dim = y_train.shape[-1], return_sequences=True)

# Compile the model
start = time.time()
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(loss='mean_squared_error', optimizer = opt)
print('compilation time : ', time.time() - start)

model.fit(X_train, y_train, epochs = 5, validation_split = 0.05)
