import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import time
import os
import tensorflow as tf

from dataUtils import *
from waveletDenoising import *
from models import *

"""
RESULTS WE NEED:
1. single asset result: lstm vs wlstm vs wlstm+a
2. one model trained on all asset results(to test transfer learning): lstm vs wlstm vs wlstm+a

Metrics: mse mae, rmse, R^2
"""

DATA_PATH = "formatted_features.csv"

df = pd.read_csv(DATA_PATH)
asset_dfs = split_df_by_asset(df)
print("Number of assets: ", len(asset_dfs))

#dataframe for a single asset. Here, XBTUSD
test_df = asset_dfs[0] 
print("Shape of test_df: ", test_df.shape)

#dataframe of standardised data
cleaned_data, scalers = standardise_df(test_df) 
#dataframe of standardised and denoised data
cleaned_data = denoise_df(cleaned_data) 

#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––

#The model will use this many of the most recent rows to make a prediction
unroll_length = 50
#The prediction will be this many timesteps in the future. If horizon=1, we're predicting data from the next timestep.
horizon = 1

#percentage of total data to be set aside for testing
train_test_split = 0.1 
X_train, X_test, y_train, y_test = train_test_split_lstm(cleaned_data, horizon, int(cleaned_data.shape[0] * train_test_split))

#If X is rows 0 to 1000 of cleaned_data, then y is rows horizon to 1000+horizon of cleaned_data.
#We want to use unroll_length rows to predict the average price, volume and standard deviation in the next row (since horizon=1). So:
#Shape of X data should be in the form (samples, unroll_length, features)
#Shape of y data should be in the form (samples, features)?
X_train = unroll(X_train, unroll_length)
X_test = unroll(X_test, unroll_length)
#y_train = y_train[unroll_length:]
#y_test = y_test[unroll_length:]
y_train = unroll(y_train, unroll_length)
y_test = unroll(y_test, unroll_length)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––

print("lstm_input_dim: ",X_train.shape[-1])
print("lstm_output_dim: ", unroll_length)
print("dense_output_dim :", y_train.shape[-1])

model = build_basic_lstm_model(lstm_input_dim = X_train.shape[-1], lstm_output_dim = unroll_length, dense_output_dim = y_train.shape[-1], return_sequences=True)
#model = build_att_lstm_model(lstm_input_dim = X_train.shape[-1], lstm_output_dim = unroll_length, dense_output_dim = y_train.shape[-1], return_sequences=True)
#model = lstm(0.01, X_train.shape[1], X_train.shape[2]) #learning rate, input dimension 1, input dimension 2 

# Compile the model
start = time.time()
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(loss='mean_squared_error', optimizer = opt, metrics=['mse', 'mae']) 
print('compilation time : ', time.time() - start)

# Create a callback that saves the model's weights
checkpoint_path = "/Users/Sanjit/Repos/CollectiWise/model_checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

#model.fit(X_train, y_train, epochs = 5, validation_split = 0.05, callbacks=[cp_callback], verbose=1)

# Load saved weights
model.load_weights(checkpoint_path) #All we have to do before this line is to create and compile the model
results = model.evaluate(X_test, y_test, verbose=1)
print("test loss, mse, mae:", results)
predictions = model.predict(X_test) 
print("predictions shape:", predictions.shape) 

#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––

#Convert predictions and target back to 2D arrays, i.e. undo the effect of unroll()
predictions = reroll(predictions, unroll_length)
target = reroll(y_test, unroll_length)

#Compute metrics
mse = np.power((predictions - target), 2).sum(axis=0) / len(predictions)
mae = np.abs(predictions - target).sum(axis=0)/len(predictions)
rmse = np.sqrt(mse)

ybar = np.tile(target.sum(axis=0)/len(target), (len(target), 1))
tss = np.power(target - ybar, 2).sum(axis=0)
r2 = 1 - (len(predictions)*mse / tss)

print("MSE: ", mse)
print("MAE: ", mae)
print("RMSE: ", rmse)
print("R-squared: ", r2)

'''
1.
Why is our calculated values of mse, mae different from what we get by running model.evaluate?

2.
For each 2hr period, i.e. each row in the dataframe:
How many standard deviations above the average price is the high price? 
How many standard deviations below the average price is the low price?
Estimate this from OHLC data. 
Then we can give estimate high and low prices in the next 2hr period using predictions from the model.
'''