import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import  EarlyStopping

import cssData
import prepareData
import plotResult

from matplotlib import pyplot as plt

# %% import data and dataframe

data_dir = "C:/Users/baage/choi_gr16/2_Research/202109_MLliq/liq_data2"  # get the data folder path
expNumList1 = cssData.expNumList()  # get expNumList that you want to consider
Drs = cssData.relativeDensity()  # get relative density (Dr) data for each trial

# get dataframe for all trials
df_all = cssData.to_dataframe(data_dir=data_dir, expNumList=expNumList1, Drs=Drs)

# # look into data
# cssData.LookIntoData(dataframe=df_all, timeIndex=1000)
#
# # plot trial
# cssData.plotTrial(dataframe=df_all, expIndex=2, trialIndex=34)

# %% select a dataframe at a exp-trial

# input
exps_trian = [0]
trials_trian = [[0, 1]]
exps_test = [1]
trials_test = [[0, 2]]
cols = ['Time [sec]', 'Shear Stress [kPa]', 'ru']

data_arrays_train, trian_indices = prepareData.selectData(
    dfList=df_all, exps=exps_trian, trials=trials_trian, cols=cols)
data_arrays_test, test_indices = prepareData.selectData(
    dfList=df_all, exps=exps_test, trials=trials_test, cols=cols)
#%% Look at data

# plot trial
cssData.plotTrial(dataframe=df_all, expIndex=1, trialIndex=2)
#%% Normalize

# get conf pressure
confPressures_train = prepareData.getConfPressure(dfList=df_all, exps=exps_trian, trials=trials_trian)
confPressures_test = prepareData.getConfPressure(dfList=df_all, exps=exps_test, trials=trials_test)

# get max pressure
maxColValue_train = prepareData.getMaxColValue(data_arrays=data_arrays_train, cols=[0])
maxColValue_test = prepareData.getMaxColValue(data_arrays=data_arrays_test, cols=[0])

# normalize train data array
data_arrays_train_normalized = prepareData.normalize(
    data_arrays=data_arrays_train,
    maxColValues=maxColValue_train, confPressures=confPressures_train,
    colsToMaxNormalize=[0], colToStressNormalize=[1]
)

# normalize test data array
data_arrays_test_normalized = prepareData.normalize(
    data_arrays=data_arrays_test,
    maxColValues=maxColValue_test, confPressures=confPressures_test,
    colsToMaxNormalize=[0], colToStressNormalize=[1]
)

#%% RNN inputs

features = [0, 1, 2]
targets = [2]
length = 50

data_dict_train = prepareData.RNN_inputs(
    data_arrays=data_arrays_train_normalized, features=features, targets=targets, length=50)
data_dict_test = prepareData.RNN_inputs(
    data_arrays=data_arrays_test_normalized, features=features, targets=targets, length=50)

x_rnn_train = data_dict_train["x_rnn"]
y_rnn_train = data_dict_train["y_rnn"]
x_rnn_test = data_dict_test["x_rnn"]
y_rnn_test = data_dict_test["y_rnn"]

#%% shuffle

shuffler = np.random.permutation(len(x_rnn_train))  # get indices for shuffling
x_rnn_train_sf = x_rnn_train[shuffler]  # shuffle the dataset
y_rnn_train_sf = y_rnn_train[shuffler]  # shuffle the dataset

#%% build a model

# now build the RNN
model = Sequential()
model.add(LSTM(128, input_shape=(x_rnn_train.shape[1], x_rnn_train.shape[2]), activation='tanh'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='tanh'))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='tanh'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh'))

# monitor validation progress
early = EarlyStopping(monitor="val_loss", mode="min", patience=1000)
callbacks_list = [early]

# set optimizer
opt = optimizers.Adam(learning_rate=0.001)  # default learning rate=0.001

# compile
loss = ['mean_squared_error', 'mean_absolute_percentage_error', 'mean_absolute_error']
metrics = ['mse','mape','mae']
choose_loose = 0
model.compile(loss=loss[choose_loose],
              optimizer=opt,
              metrics=metrics[choose_loose])

# Show a model summary table
model.summary()

# %% train

# train with batch
history = model.fit(x_rnn_train_sf, y_rnn_train_sf,
                    epochs=20, batch_size=32, verbose=2,
                    validation_split=0.20,
                    callbacks=callbacks_list,
                    shuffle=True)

# %% plot model training history
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("loss (MAPE)")

# %% get predictions

# Note that the model is trained with shuffled data, ...
# while the prediction is done with unshuffled data.
y_rnn_train_pred = model.predict(x_rnn_train)
y_rnn_test_pred = model.predict(x_rnn_test)

# %% plot datasets

x_arrays_train = data_dict_train["x_arrays"]  # features before sampling by window
y_arrays_train = data_dict_train["y_arrays"]  # labels before sampling by window
x_arrays_test = data_dict_test["x_arrays"]  # features before sampling by window
y_arrays_test = data_dict_test["y_arrays"]  # labels before sampling by window

plotResult.plot_dataset(
    x_arrays=x_arrays_train, y_arrays=y_arrays_train,
    title_index=trian_indices, legends=cols, subplot_ncols=3
)

plotResult.plot_dataset(
    x_arrays=x_arrays_test, y_arrays=y_arrays_test,
    title_index=test_indices, legends=cols, subplot_ncols=3
)

# %% plot prediction

dps_train = data_dict_train["dataPoints_list"]
dps_test = data_dict_test["dataPoints_list"]

plotResult.plot_predictionResult(
    x_arrays=x_arrays_train, y_arrays=y_arrays_train, y_rnn_pred=y_rnn_train_pred,
    window_length=length, dps=dps_train, title_index=trian_indices, subplot_ncols=3
)

plotResult.plot_predictionResult(
    x_arrays=x_arrays_test, y_arrays=y_arrays_test, y_rnn_pred=y_rnn_test_pred,
    window_length=length, dps=dps_test, title_index=test_indices, subplot_ncols=3
)

# %%
