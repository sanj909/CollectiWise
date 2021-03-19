import tensorflow as tf
import tensorflow.keras.layers as layers
import keras as og_keras
from keras_self_attention import SeqSelfAttention
# pip install keras_self_attention

def build_basic_lstm_model(lstm_input_dim, lstm_output_dim, dense_output_dim, return_sequences):
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(input_shape=(None, lstm_input_dim), units = lstm_output_dim, return_sequences = return_sequences))
    #model.add(layers.LSTM(100, return_sequences = False))
    model.add(layers.Dense(units = dense_output_dim))
    #model.add(Activation('softmax'))
    model.add(layers.Activation('linear'))
    return model


def build_att_lstm_model(lstm_input_dim, lstm_output_dim, dense_output_dim, return_sequences):
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(input_shape = (None, lstm_input_dim), units = lstm_output_dim, return_sequences = return_sequences))
    #model.add(layers.LSTM(100, return_sequences = False))
    #model.add(layers.Attention())
    model.add(SeqSelfAttention(attention_activation= 'tanh'))
    model.add(layers.Dense(units = dense_output_dim))
    #model.add(Activation('softmax'))
    model.add(layers.Activation('linear'))
    return model


def lstm(learning_rate, window_length, n_features):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units = 25, activation='relu', input_shape=(window_length, n_features)))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units = n_features, activation='linear')) #We want the model to output a single number, it's prediction of bid1
    return model
