import numpy as np


def select_data(df_list, exps, trials, cols):
    """
    choose which exp-trials and its columns to use from the list of dataframes.
    :param df_list: a list of dataframes
    :param exps: choose a list of experiment indices
    :param trials: choose a list of trial indices
    for example, when you set exps=[0, 1] and trials=[[0,1], [5,7]],
    you are selecting exps-trials of 0-0, 0-1, 1-5, 1-7.
    :param cols: choose a list of headers of the dataframe to use
    :return:
    a list of 2D-arrays (shape=(data_points, columns)) that corresponds to the selected exps-trials
    """
    # containers
    exp_trial_index = []
    data_arrays = []

    for i, exp in enumerate(exps):
        for trial in trials[i]:

            # save each exp-trial index
            exp_trial_index.append([exp, trial])

            # select a dataframe at a exp-trial and make it as an array
            df_selected = df_list[exp][trial][cols]  # select a dataframe
            data_array = np.asarray(df_selected)  # make it as an array
            data_arrays.append(data_array)  # save each data_array at each exp-trial

    return data_arrays, exp_trial_index

def get_conf_pressure(dfList, exps, trials):
    """
    get confining pressures for each exp-trial
    :param dfList: a list of dataframe to consider
    :param exps: experiment indices to consider
    :param trials: trial indices to consider
    :return: a list of confining pressure
    """

    conf_pressures = []

    for i, exp in enumerate(exps):
        for trial in trials[i]:

            conf_pressure = \
                dfList[exp][trial]["Effective Vertical Stress [kPa]"][0]  # get confining pressure of the trial
            conf_pressures.append(conf_pressure)  # save confpressure at each exp-trial

    return conf_pressures


def get_max_col_value(data_arrays, cols):
    """
    find a maximum value of the `data_arrays` for specified `cols` for all trials
    :param data_arrays: 3D shaped array-like data. shape=(trials, data_points, columns)
    :param cols: list of indices of columns that you want to find the max value
    :return:
    maximum values of specified `cols`
    """

    # container
    max_col_values = np.zeros((len(cols), 1))  # array to contain normalization factor

    # find max vals and save
    data_arrays_train_unroll = np.concatenate(data_arrays, axis=0)
    for i in cols:
        max_col_values[i, 0] = max(abs(data_arrays_train_unroll[:, i]))  # save the max vals of each col

    return max_col_values


def normalize(data_arrays, max_col_values, conf_pressures, cols_to_max_normalize, col_to_stress_normalize=1):
    """
    Normalize a list of 2-D shaped array-like data.
    It is intended to normalize the shear stress by the confining pressure of each trial,
    and time [sec] by the maximum time of whole trials.
    :param data_arrays: a list of 2-D shaped array-like data. shape=(data_points, columns) corresponding to each trial
    :param max_col_values: maximum column values for all trials obtained by `get_max_col_value` function
    :param conf_pressures: confining pressure of each trial obtained by `get_conf_pressure` function
    :param cols_to_max_normalize: column indices of `data_arrays` that you want to normalize with the max value
    :param col_to_stress_normalize: column indices of `data_arrays` that you want to normalize with confining pressure
    :return:
    normalized data_arrays
    """

    # container
    data_arrays_normalized = data_arrays.copy()

    # normalize data
    for i in range(len(data_arrays)):
        # normalize shear stress by confining pressure
        data_arrays_normalized[i][:, col_to_stress_normalize] = data_arrays[i][:, col_to_stress_normalize] / conf_pressures[i]
        # normalize selected columns by the maximum value of the whole selected data
        for j in cols_to_max_normalize:
            data_arrays_normalized[i][:, j] = data_arrays[i][:, j] / max_col_values[j, 0]

    return data_arrays_normalized


def rnn_inputs(data_arrays, features, targets, length):
    """
    make data for RNN inputs and a few other useful variables in `dict` format.
    :param data_arrays: a list of normalized 2-D shaped array-like data (shape=(data_points, columns))
    :param features: a list of indices for features
    :param targets: a list of indices for target
    :param length: length of sampling window
    :return:
    returns dict that is described at the end of this function.
    """
    # containers
    x_arrays = []
    y_arrays = []
    data_points_list = []

    x_rnns = []
    y_rnns = []

    # number of data_arrays
    num_data_arrays = len(data_arrays)

    for i in range(num_data_arrays):

        # split features and targets
        x_array = data_arrays[i][:, features]  # x and y array for data_array[i]
        y_array = data_arrays[i][:, targets]
        x_arrays.append(x_array)  # save x and y array for data_array[i]
        y_arrays.append(y_array)

        # make inputs for RNN layers (time window method)
        data_points = len(x_array)  # get num of the data points for data_array[i]
        data_points_list.append(data_points)  # save dataPoints for data_array[i]

        x_rnn = list()  # container for x_rnn
        y_rnn = list()  # container for y_rnn

        for j in range(data_points - length):  # grab samples from j to j + length
            sample_train = x_array[j:j+length, :]
            outcome_train = y_array[j+length, 0]
            x_rnn.append(sample_train)
            y_rnn.append(outcome_train)

        x_rnn = np.asarray(x_rnn)
        y_rnn = np.asarray(y_rnn)
        x_rnns.append(x_rnn)
        y_rnns.append(y_rnn)

    x_rnn_concat = np.concatenate(x_rnns, axis=0)
    y_rnn_concat = np.concatenate(y_rnns, axis=0)

    data_dict = {
        "x_arrays": x_arrays,  # a list of data to be used as features
        "y_arrays": y_arrays,  # a list of data to be used as a target
        "data_points_list": data_points_list,  # number of datapoint for each trial
        "x_rnn_concat": x_rnn_concat,  # x_rnn concatenated for feeding model
        "y_rnn_concat": y_rnn_concat,  # y_rnn concatenated for feeding model
        "x_rnns": x_rnns,  # x_rnn before concatenating
        "y_rnns": y_rnns  # y_rnn before concatenating
    }

    return data_dict

