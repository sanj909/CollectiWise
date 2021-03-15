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
print(len(asset_dfs))

test_df = asset_dfs[0]
# Temporary line - because denoising only handles one feature now
test_df = test_df[test_df.columns[0]]
cleaned_data = normalise(test_df, 0, 1)
cleaned_data = optDenoise(cleaned_data)

unroll_length = 50
X_train, X_test, y_train, y_test = train_test_split_lstm(cleaned_data, 5, int(cleaned_data.shape[0] * 0.1))
X_train = np.expand_dims(unroll(X_train, unroll_length), axis = 2)
y_train = np.expand_dims(unroll(y_train, unroll_length), axis = 2)
X_test = np.expand_dims(unroll(X_test, unroll_length), axis = 2)
y_test = np.expand_dims(unroll(y_test, unroll_length), axis = 2)
print(int(cleaned_data.shape[0] * 0.1))
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#model = build_basic_lstm_model(lstm_input_dim = X_train.shape[-1], lstm_output_dim = unroll_length, dense_output_dim = y_train.shape[-1], return_sequences=True)
model = build_att_lstm_model(lstm_input_dim = X_train.shape[-1], lstm_output_dim = unroll_length, dense_output_dim = y_train.shape[-1], return_sequences=True)

# Compile the model
start = time.time()
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.compile(loss='mean_squared_error', optimizer = opt, metrics=['accuracy']) #metrics argument is necessary for model.evaluate to return accuracy
print('compilation time : ', time.time() - start)


# Create a callback that saves the model's weights
checkpoint_path = "GHRepos/CollectiWise/model_checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

model.fit(X_train, y_train, epochs = 100, validation_split = 0.05, callbacks=[cp_callback])
exit()
# Load saved weights
model.load_weights(checkpoint_path) #All we have to do before this line is to create and compile the model
results = model.evaluate(X_test, y_test, verbose=1)
print("test loss, test acc:", results)
predictions = model.predict(X_test[(len(X_test)-1):]) #Predict using the last row of X_test (afaik, the last row is the 50 most recent prices)
print("predictions shape:", predictions.shape) #Model prediction of the 50 next prices
print(predictions)
