import numpy as np
from matplotlib import pyplot as plt


def plot_dataset(x_arrays, y_arrays, title_index, legends, subplot_ncols=3):
    # plot
    num_trials = len(x_arrays)  # total num of trials in train set
    nr = int(np.ceil(num_trials / subplot_ncols))  # num of rows in subplot

    fig, axs = plt.subplots(nrows=nr, ncols=subplot_ncols, figsize=(subplot_ncols * 5, nr * 3))
    axs_unroll = axs.flatten()
    for i in range(num_trials):
        axs_unroll[i].plot(x_arrays[i][:, :])
        axs_unroll[i].plot(y_arrays[i][:, :])
        axs_unroll[i].legend(legends, loc='best')
        axs_unroll[i].set_xlabel("Data points")
        axs_unroll[i].set_ylabel("Normalized values")
        axs_unroll[i].set_title(f"Experiment-{title_index[i][0]}, Trial-{title_index[i][1]}")
    fig.tight_layout()

    return fig


def plot_predictionResult(x_arrays, y_arrays, y_rnn_pred, window_length, dps,
                          title_index, subplot_ncols=3):
    # calculate datapoint indices in `y_rnn_pred` to plot
    dp_index = 0
    dp_indices = [0]
    for dp in dps:
        dp_index += dp - window_length
        dp_indices.append(dp_index)

    # plot
    num_trials = len(x_arrays)  # total num of trials in train set
    nr = int(np.ceil(num_trials / subplot_ncols))  # num of rows in subplot

    fig, axs = plt.subplots(nrows=nr, ncols=subplot_ncols, figsize=(subplot_ncols * 5, nr * 3))
    axs_unroll = axs.flatten()
    for i in range(num_trials):
        # plot ru over time (sec) used for prediction
        axs_unroll[i].plot(x_arrays[i][:, 0],
                           y_arrays[i][:, 0],
                           c='goldenrod', label='train data')
        # plot ru over time (sec) that the trained model produce (prediction result)
        axs_unroll[i].plot(x_arrays[i][window_length:, 0],
                           y_rnn_pred[dp_indices[i]:dp_indices[i + 1]],
                           c='navy', label='train result')
        axs_unroll[i].legend(loc='best')
        axs_unroll[i].set_xlabel("Data points")
        axs_unroll[i].set_ylabel("Normalized values")
        axs_unroll[i].set_title(f"Experiment-{title_index[i][0]}, Trial-{title_index[i][1]}")
    fig.tight_layout()

    return fig
