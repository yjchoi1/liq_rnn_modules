import numpy as np
from matplotlib import pyplot as plt

def plot_dataset(x_array, y_array, title_index, legends, subplot_cols=3):
    """plot input datasets for train"""

    nc = subplot_cols  # num of columns in subplot
    num_trials = len(x_array)  # total num of trials in train set
    nr = int(np.ceil(num_trials / nc))  # num of rows in subplot

    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 5, nr * 3))
    axs_unroll = axs.flatten()
    for i in range(num_trials):
        axs_unroll[i].plot(x_array[i, :, :])
        axs_unroll[i].plot(y_array[i, :, :])
        axs_unroll[i].legend(legends, loc='best')
        axs_unroll[i].set_xlabel("Data points")
        axs_unroll[i].set_ylabel("Normalized values")
        axs_unroll[i].set_title(f"Experiment-{title_index[i][0]}, Trial-{title_index[i][1]}")
    fig.tight_layout()

    return fig

def plot_predictionResult(
        x_array_sample, y_array_sample, y_rnn_pred, time_steps, window_length,
        title_index, subplot_cols=3):
    """plot train and prediction result along with the inputs of the model"""

    nc = subplot_cols  # num of columns in subplot
    num_trials_train = len(x_array_sample)  # total num of trials in train set
    nr = int(np.ceil(num_trials_train / nc))  # num of rows in subplot

    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 5, nr * 3))
    axs_unroll = axs.flatten()
    for i in range(num_trials_train):
        # plot ru over time (sec) used for prediction
        axs_unroll[i].plot(x_array_sample[i, :, 0],
                           y_array_sample[i, :, 0],
                           c='goldenrod', label='train data')
        # plot ru over time (sec) that the trained model produce (prediction result)
        axs_unroll[i].plot(x_array_sample[i, window_length:time_steps, 0],
                           y_rnn_pred[i * (time_steps - window_length):(i + 1) * (time_steps - window_length)],
                           c='navy', label='train result')
        axs_unroll[i].legend(loc='best')
        axs_unroll[i].set_xlabel("time (sec)")
        axs_unroll[i].set_ylabel("ru")
        axs_unroll[i].set_title(f"Experiment-{title_index[i][0]}, Trial-{title_index[i][1]}")
    fig.tight_layout()

    return fig