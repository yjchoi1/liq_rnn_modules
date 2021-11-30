import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Flatten
from keras import optimizers
from keras.callbacks import EarlyStopping

import cssData
import prepareData
from matplotlib import pyplot as plt
# %% import data ad dataframe

data_dir = "C:/Users/baage/Desktop/Choi_MSI/02_개인_공동연구/Liq/liq_data"  # get the data folder path
expNumList1 = cssData.expNumList()  # get expNumList that you want to consider
Drs = cssData.relativeDensity()  # get relative density (Dr) data for each trial

# get dataframe for all trials
df_all = cssData.to_dataframe(expNumList=expNumList1, data_dir=data_dir, Drs=Drs)

# look into data
cssData.LookIntoData(timeIndex=1000)

# plot trial
cssData.plotTrial(expIndex=2, trialIndex=34)
# %% reshape dataframe

# see columns in the raw dataframe
colNames = list(df_all[0][0].columns)

# select columns and length and cut the dataframe to the form you what to get
selectCols = ['Time [sec]', 'Shear Stress [kPa]', 'ru']
time_steps = 8600
df_cut = prepareData.cutDataFrame(time_steps=8600, cols=selectCols, dataframe=df_all)

# %% Train and test split

# choose your exp and trial index to use.
# for i in expIndex; for j in trialIndex, i and j iterate.
# for example, expIndex=[0, 1], trialIndex=[[0,1], [0, 1, 3]],
# the result indices are (0,0) (0, 1), (1, 0), (1, 1), (1, 3).
expIndex_train = [0]
trialIndex_train = [[0, 1]]
expIndex_test = [0]
trialIndex_test = [[11]]

# The method `prepareData.data_array` also returns the index of (exp, trial).
# This will be used to define the title of result plots.
data_array_train, train_trial_index = prepareData.data_array(
    expIndex=expIndex_train, trialIndex=trialIndex_train, df_cut=df_cut)
data_array_test, test_trial_index = prepareData.data_array(
    expIndex=expIndex_test, trialIndex=trialIndex_test, df_cut=df_cut)

# %% normalize the data (only selected cols)

# The best practice is to normalize train & test separately to prevent data leak
# The method `prepareData.normalize_cols` also returns the normalization factors.
# These might be used to denormalize the data back to the original values.
colsToConsider = [0, 1]  # 0: time steps, 1: stress, 2: ru
data_array_train_normalized, normalizingFactors_train = prepareData.normalize_cols(
    data_array=data_array_train, cols_to_normalize=colsToConsider)
data_array_test_normalized, normalizingFactors_test = prepareData.normalize_cols(
    data_array=data_array_test, cols_to_normalize=colsToConsider)

# %% set x and y

# inputs
features = [0, 1]  # 0: timestep, 1: stress
target = [2]  # 2: ru

# split x and y for train and test, from the data_array
x_array_train = data_array_train_normalized[:, :, features]
y_array_train = data_array_train_normalized[:, :, target]
x_array_test = data_array_test_normalized[:, :, features]
y_array_test = data_array_test_normalized[:, :, target]

# %% make data for time-series input for RNN layers (time window method)
length = 50

x_rnn_train, y_rnn_train, x_rnn_test, y_rnn_test = prepareData.windowDataset(
    x_array_train, y_array_train, x_array_test, y_array_test, length, time_steps
)

# %% shuffle

shuffler = np.random.permutation(len(x_rnn_train))  # get indices for shuffling
x_rnn_train_sf = x_rnn_train[shuffler]  # shuffle the dataset
y_rnn_train_sf = y_rnn_train[shuffler]  # shuffle the dataset

# %% build a model

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
model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['mape'])

# Show a model summary table
model.summary()

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
plt.ylabel("loss (MAPE)")
# plt.ylim(3, 46)
# plt.show()
# plt.savefig(save_eval_dir)

# %% get predictions

# Note that the model is trained with shuffled data, ...
# while the prediction is done with unshuffled data.
y_rnn_train_pred = model.predict(x_rnn_train)
y_rnn_test_pred = model.predict(x_rnn_test)

#%% plot x_array and y_array

import plotResult

inputData = plotResult.plot_dataset(
    x_array=x_array_train, y_array=y_array_train,
    title_index=train_trial_index, legends=selectCols)

#%% plot training and prediction result
trainResult = plotResult.plot_predictionResult(
    x_array_train=x_array_train, y_array_train=y_array_train, y_rnn_pred=y_rnn_train_pred,
    time_steps=time_steps, window_length=length,
    title_index=train_trial_index)

trainResult = plotResult.plot_predictionResult(
    x_array_train=x_array_train, y_array_train=y_array_train, y_rnn_pred=y_rnn_test_pred,
    time_steps=time_steps, window_length=length,
    title_index=test_trial_index)


# %% Plot train result

nc = 3  # num of columns in subplot

# plot train result
num_trials_train = len(x_array_train)  # total num of trials in train set
nr = int(np.ceil(num_trials_train / nc))  # num of rows in subplot

fig1, axs1 = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 5, nr * 3))
axs_unroll = axs1.flatten()
for i in range(num_trials_train):
    axs_unroll[i].plot(x_array_train[i, :, 0],
                       y_array_train[i, :, 0],
                       c='goldenrod', label='train data')
    axs_unroll[i].plot(x_array_train[i, length:time_steps, 0],
                       y_rnn_train_pred[i * (time_steps - length):(i + 1) * (time_steps - length)],
                       c='navy', label='train result')
    axs_unroll[i].legend(loc='best')
    axs_unroll[i].set_xlabel("time (sec)")
    axs_unroll[i].set_ylabel("ru")
    axs_unroll[i].set_title(f"Experiment-{train_trial_index[i][0]}, Trial-{train_trial_index[i][1]}")
fig1.tight_layout()

# plot test result
num_trials_test = len(x_array_test)  # total num of trials in test set
nr = int(np.ceil(num_trials_test / nc))  # num of rows in subplot

fig2, axs2 = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 5, nr * 3))
axs_unroll = axs2.flatten()
for i in range(num_trials_test):
    axs_unroll[i].plot(x_array_test[i, :, 0],
                       y_array_test[i, :, 0],
                       c='goldenrod', label='test data')
    axs_unroll[i].plot(x_array_test[i, length:time_steps, 0],
                       y_rnn_test_pred[i * (time_steps - length):(i + 1) * (time_steps - length)],
                       c='navy', label='test result')
    axs_unroll[i].legend(loc='best')
    axs_unroll[i].set_xlabel("time (sec)")
    axs_unroll[i].set_ylabel("ru")
    axs_unroll[i].set_title(f"Experiment-{test_trial_index[i][0]}, Trial-{test_trial_index[i][1]}")
fig2.tight_layout()
