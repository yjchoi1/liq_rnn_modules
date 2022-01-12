import tensorflow as tf
import numpy as np
import os
import json
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

import cssdata
import preparedata
import plotresult
import model

# %% import data and dataframe

cwd = os.getcwd()  # get current working directory
data_dir = os.path.join(cwd, "rawdata")  # define the path for the rawdata

expNumList1 = cssdata.exp_num_list()  # get exp_num_list that you want to consider (by default, [7, 8, 9, 10])
Drs = cssdata.relative_density()  # get relative density (Dr) data for each trial

# get dataframe for all trials
df_all = cssdata.to_dataframe(data_dir=data_dir, exp_num_list=expNumList1, Drs=Drs)

# # get some useful information about data
# cssdata.look_into_data(dataframe=df_all, timeIndex=1000)
#
# # plot a trial
# cssdata.plot_trial(dataframe=df_all, expIndex=2, trialIndex=34)

# %% load selected a dataframe at a exp-trial.

inputfile = "input.json"
inputfilepath = os.path.join(cwd, inputfile)

with open(inputfilepath, "r") as f:
    json_data = json.load(f)

exps_train = json_data["exps_train"]
trials_train = json_data["trials_train"]
exps_test = json_data["exps_test"]
trials_test = json_data["trials_test"]
cols = json_data["cols"]

# get a list of 2-D shaped array-like data with shape=(data_points, columns), which corresponds to each trial.
data_arrays_train, train_indices = preparedata.select_data(
    df_list=df_all, exps=exps_train, trials=trials_train, cols=cols)
data_arrays_test, test_indices = preparedata.select_data(
    df_list=df_all, exps=exps_test, trials=trials_test, cols=cols)

#%% Normalize

# get conf pressure that will be used for the normalization
conf_pressures_train = preparedata.get_conf_pressure(dfList=df_all, exps=exps_train, trials=trials_train)
conf_pressures_test = preparedata.get_conf_pressure(dfList=df_all, exps=exps_test, trials=trials_test)

# get max values that will be used for the normalization
max_col_value_train = preparedata.get_max_col_value(data_arrays=data_arrays_train, cols=[0])  # cols=[0] is 'time [sec]'
max_col_value_test = preparedata.get_max_col_value(data_arrays=data_arrays_test, cols=[0])

# normalize train data array
data_arrays_train_normalized = preparedata.normalize(
    data_arrays=data_arrays_train,
    max_col_values=max_col_value_train, conf_pressures=conf_pressures_train,
    cols_to_max_normalize=[0], col_to_stress_normalize=[2]
)

# normalize test data array
data_arrays_test_normalized = preparedata.normalize(
    data_arrays=data_arrays_test,
    max_col_values=max_col_value_test, conf_pressures=conf_pressures_test,
    cols_to_max_normalize=[0], col_to_stress_normalize=[2]
)

# %% RNN inputs

# each index specified below corresponds to the index of the variable `cols`
features = [0, 1, 2]
targets = [3]
length = 100

# obtain data for RNN inputs and a few other useful variables in `dict` format
data_dict_train = preparedata.rnn_inputs(
    data_arrays=data_arrays_train_normalized, features=features, targets=targets, length=length)
data_dict_test = preparedata.rnn_inputs(
    data_arrays=data_arrays_test_normalized, features=features, targets=targets, length=length)

# input datasets for RNN model (shape=(samples, window_length, features))
x_rnn_train = data_dict_train["x_rnn_concat"]
y_rnn_train = data_dict_train["y_rnn_concat"]
x_rnn_test = data_dict_test["x_rnn_concat"]
y_rnn_test = data_dict_test["y_rnn_concat"]

# %% shuffle

shuffler = np.random.permutation(len(x_rnn_train))  # get indices for shuffling
x_rnn_train_sf = x_rnn_train[shuffler]
y_rnn_train_sf = y_rnn_train[shuffler]

# %% build a model based on tf2.0 API

model = model.lstm_model(input_shape=x_rnn_train.shape)

# monitor validation progress
early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)
callbacks_list = [early]

# set optimizer
opt = optimizers.Adam(learning_rate=0.001)  # default learning rate=0.001

# compile
loss = ['mean_squared_error', 'mean_absolute_percentage_error', 'mean_absolute_error']
metrics = ['mse', 'mape', 'mae']
choose_loose = 0
model.compile(loss=loss[choose_loose],
              optimizer=opt,
              metrics=metrics[choose_loose])

# %% train

# train with batch
history = model.fit(x_rnn_train_sf, y_rnn_train_sf,
                    epochs=10, batch_size=32, verbose=2,
                    validation_split=0.20,
                    callbacks=callbacks_list,
                    shuffle=True)

# %% plot model training history

plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel(f"Loss ({metrics[choose_loose]})")

# %% get prediction result for each trial

# inputs
Xs_rnn_train = data_dict_train['x_rnns']
Xs_rnn_test = data_dict_test['x_rnns']

# containers
Ys_rnn_train_pred = []
Ys_rnn_test_pred = []

# get prediction with trial datasets
num_trials = len(Xs_rnn_train)  # total num of trials in train sets
for i in range(num_trials):
    Y_rnn_train_pred = model.predict(Xs_rnn_train[i])
    Ys_rnn_train_pred.append(Y_rnn_train_pred)

# get prediction with test datasets
num_trials = len(Xs_rnn_test)  # total num of trials in test sets
for i in range(num_trials):
    Y_rnn_test_pred = model.predict(Xs_rnn_test[i])
    Ys_rnn_test_pred.append(Y_rnn_test_pred)

# %% plot datasets

x_arrays_train = data_dict_train["x_arrays"]  # features selected to be sampled by window later
y_arrays_train = data_dict_train["y_arrays"]  # labels selected to be sampled by window later
x_arrays_test = data_dict_test["x_arrays"]  # features selected to be sampled by window later
y_arrays_test = data_dict_test["y_arrays"]  # labels selected to be sampled by window later

# plot data used for train sets
plotresult.plot_dataset(
    x_arrays=x_arrays_train, y_arrays=y_arrays_train,
    title_index=train_indices, legends=cols, subplot_ncols=3
)

# plot data used for test sets
plotresult.plot_dataset(
    x_arrays=x_arrays_test, y_arrays=y_arrays_test,
    title_index=test_indices, legends=cols, subplot_ncols=3
)

# %% plot prediction

# get target (true) Y data
Ys_rnn_train = data_dict_train['y_rnns']
Ys_rnn_test = data_dict_test['y_rnns']

# plot prediction result for train sets
plotresult.plot_predictionResult_v2(Ys_rnn=Ys_rnn_train, Ys_rnn_pred=Ys_rnn_train_pred,
                                    title_index=train_indices, subplot_ncols=3
                                    )
# plot prediction result for test sets
plotresult.plot_predictionResult_v2(Ys_rnn=Ys_rnn_test, Ys_rnn_pred=Ys_rnn_test_pred,
                                    title_index=test_indices, subplot_ncols=3
                                    )

# %% [not yet finished] Use ru as an input and also use it to prediction

Xs_rnn_train = data_dict_train['x_rnns']
ru_preds_list = []

for X_rnn_train in Xs_rnn_train:

    # containers
    ru_preds = []
    ru_pred_tmp = X_rnn_train[0, :, 2]
    x_rnn_train2 = np.copy(x_rnn_train)

    # predict
    for i in range(len(x_rnn_train2)):
        # for i in range(2):

        ru_pred = model.predict(x_rnn_train2[i:i + 1, :, :])
        ru_preds.append(ru_pred)
        if i == len(x_rnn_train) - 1:
            break

        tmpIndex = i % len(ru_pred_tmp)
        ru_pred_tmp[-tmpIndex - 1] = ru_pred
        x_rnn_train2[i + 1, :, 2] = ru_pred_tmp

    ru_preds_list.append(ru_preds)
