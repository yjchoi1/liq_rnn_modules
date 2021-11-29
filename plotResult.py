import numpy as np
from matplotlib import pyplot as plt

def plot_dataset(x_array, y_array, title_index, legends, subplot_cols=3):

    # plot data for train
    nc = subplot_cols
    num_trials = len(x_array)  # total num of trials in train set
    nr = int(np.ceil(num_trials / nc))  # num of rows in subplot

    fig1, axs1 = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 5, nr * 3))
    axs_unroll = axs1.flatten()
    for i in range(num_trials):
        axs_unroll[i].plot(x_array[i, :, :])
        axs_unroll[i].plot(y_array[i, :, :])
        axs_unroll[i].legend(legends, loc='best')
        axs_unroll[i].set_xlabel("Data points")
        axs_unroll[i].set_ylabel("Normalized values")
        axs_unroll[i].set_title(f"Experiment-{title_index[i][0]}, Trial-{title_index[i][1]}")
    fig1.tight_layout()

    return fig1
