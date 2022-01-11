import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

def lstm_model(input_shape):

    # construct layers
    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]))
    x = LSTM(128, activation='tanh', return_sequences=True)(inputs)
    x = LSTM(128, activation='tanh')(x)
    # x = Dropout(0.1)(x)
    x = Dense(64, activation='tanh')(x)
    # x = Dropout(0.1)(x)
    x = Dense(16, activation='tanh')(x)
    # x = Dropout(0.1)(x)
    outputs = Dense(1, activation='tanh')(x)
    model = tf.keras.Model(inputs, outputs)

    # show summary
    model.summary()

    return model

