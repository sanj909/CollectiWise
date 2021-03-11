import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import time
import os
import tensorflow as tf


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
from waveletDenoising import optDenoise, normalise


close_data = data.Close.to_numpy()
close_data = optDenoise(close_data) 
close_data = normalise(close_data, 0, 1) #Normalise to N(0, 1)
print(close_data.shape)

unroll_length = 50
X_train, X_test, y_train, y_test = train_test_split_lstm(close_data, 5, int(close_data.shape[0] * 0.1))
X_train = np.expand_dims(unroll(X_train, unroll_length), axis = 2)
y_train = np.expand_dims(unroll(y_train, unroll_length), axis = 2)
X_test = np.expand_dims(unroll(X_test, unroll_length), axis = 2)
y_test = np.expand_dims(unroll(y_test, unroll_length), axis = 2)
print(int(close_data.shape[0] * 0.1))
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#model = build_basic_lstm_model(lstm_input_dim = X_train.shape[-1], lstm_output_dim = unroll_length, dense_output_dim = y_train.shape[-1], return_sequences=True)
model = build_att_lstm_model(lstm_input_dim = X_train.shape[-1], lstm_output_dim = unroll_length, dense_output_dim = y_train.shape[-1], return_sequences=True)

# Compile the model
start = time.time()
opt = tf.keras.optimizers.Adam(learning_rate = 0.1)
model.compile(loss='mean_squared_error', optimizer = opt, metrics=['accuracy']) #metrics argument is necessary for model.evaluate to return accuracy 
print('compilation time : ', time.time() - start)


# Create a callback that saves the model's weights
checkpoint_path = "GHRepos/CollectiWise/model_checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

#model.fit(X_train, y_train, epochs = 5, validation_split = 0.05, callbacks=[cp_callback])

# Load saved weights
model.load_weights(checkpoint_path) #All we have to do before this line is to create and compile the model
results = model.evaluate(X_test, y_test, verbose=1)
print("test loss, test acc:", results)
predictions = model.predict(X_test[(len(X_test)-1):]) #Predict using the last row of X_test (afaik, the last row is the 50 most recent prices)
print("predictions shape:", predictions.shape) #Model prediction of the 50 next prices
print(predictions)