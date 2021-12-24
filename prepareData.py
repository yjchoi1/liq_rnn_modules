import numpy as np


def selectData(dfList, exps, trials, cols):
    """
    choose which exp-trials and its columns to use from the list of dataframes.
    :param dfList: a list of dataframes
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
            df_selected = dfList[exp][trial][cols]  # select a dataframe
            data_array = np.asarray(df_selected)  # make it as an array
            data_arrays.append(data_array)  # save each data_array at each exp-trial

    return data_arrays, exp_trial_index

def getConfPressure(dfList, exps, trials):
    """
    get confining pressures for each exp-trial
    :param dfList: a list of dataframe to consider
    :param exps: experiment indices to consider
    :param trials: trial indices to consider
    :return: a list of confining pressure
    """

    confPressures = []

    for i, exp in enumerate(exps):
        for trial in trials[i]:

            confPressure = \
                dfList[exp][trial]["Effective Vertical Stress [kPa]"][0]  # get confining pressure of the trial
            confPressures.append(confPressure)  # save confpressure at each exp-trial

    return confPressures


def getMaxColValue(data_arrays, cols):
    """
    find a maximum value of the `data_arrays` for specified `cols` for all trials
    :param data_arrays: 3D shaped array-like data. shape=(trials, data_points, columns)
    :param cols: list of indices of columns that you want to find the max value
    :return:
    maximum values of specified `cols`
    """

    # container
    maxColValues = np.zeros((len(cols), 1))  # array to contain normalization factor

    # find max vals and save
    data_arrays_train_unroll = np.concatenate(data_arrays, axis=0)
    for i in cols:
        maxColValues[i, 0] = max(abs(data_arrays_train_unroll[:, i]))  # save the max vals of each col

    return maxColValues


def normalize(data_arrays, maxColValues, confPressures, colsToMaxNormalize, colToStressNormalize=1):
    """
    Normalize a list of 2-D shaped array-like data.
    It is intended to normalize the shear stress by the confining pressure of each trial,
    and time [sec] by the maximum time of whole trials.
    :param data_arrays: a list of 2-D shaped array-like data. shape=(data_points, columns) corresponding to each trial
    :param maxColValues: maximum column values for all trials obtained by `getMaxColValue` function
    :param confPressures: confining pressure of each trial obtained by `getConfPressure` function
    :param colsToMaxNormalize: column indices of `data_arrays` that you want to normalize with the max value
    :param colToStressNormalize: column indices of `data_arrays` that you want to normalize with confining pressure
    :return:
    normalized data_arrays
    """

    # container
    data_arrays_normalized = data_arrays.copy()

    # normalize data
    for i in range(len(data_arrays)):
        # normalize shear stress by confining pressure
        data_arrays_normalized[i][:, colToStressNormalize] = data_arrays[i][:, colToStressNormalize] / confPressures[i]
        # normalize selected columns by the maximum value of the whole selected data
        for j in colsToMaxNormalize:
            data_arrays_normalized[i][:, j] = data_arrays[i][:, j] / maxColValues[j, 0]

    return data_arrays_normalized


def RNN_inputs(data_arrays, features, targets, length):
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
    dataPoints_list = []

    Xs_rnn = []
    Ys_rnn = []

    # number of data_arrays
    num_data_arrays = len(data_arrays)

    for i in range(num_data_arrays):

        # split features and targets
        x_array = data_arrays[i][:, features]  # x and y array for data_array[i]
        y_array = data_arrays[i][:, targets]
        x_arrays.append(x_array)  # save x and y array for data_array[i]
        y_arrays.append(y_array)

        # make inputs for RNN layers (time window method)
        dataPoints = len(x_array)  # get num of the data points for data_array[i]
        dataPoints_list.append(dataPoints)  # save dataPoints for data_array[i]

        x_rnn = list()  # container for x_rnn
        y_rnn = list()  # container for y_rnn

        for j in range(dataPoints - length):  # grab samples from j to j + length
            sample_train = x_array[j:j+length, :]
            outcome_train = y_array[j+length, 0]
            x_rnn.append(sample_train)
            y_rnn.append(outcome_train)

        x_rnn = np.asarray(x_rnn)
        y_rnn = np.asarray(y_rnn)
        Xs_rnn.append(x_rnn)
        Ys_rnn.append(y_rnn)

    x_rnn = np.concatenate(Xs_rnn, axis=0)
    y_rnn = np.concatenate(Ys_rnn, axis=0)

    dataDict = {
        "x_arrays": x_arrays,  # a list of data to be used as features
        "y_arrays": y_arrays,  # a list of data to be used as a target
        "dataPoints_list": dataPoints_list,  # number of datapoint for each trial
        "x_rnn": x_rnn,  # x_rnn (concatenated)
        "y_rnn": y_rnn,  # y_rnn (concatenated)
        "Xs_rnn": Xs_rnn,  # x_rnn before concatenating
        "Ys_rnn": Ys_rnn  # y_rnn before concatenating
    }

    return dataDict

