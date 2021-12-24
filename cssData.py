# import from existing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import from my modules
from data_importing_functions import getDataDirList


def data_dir():
    """"
    define directory where data exists
    """

    data_dir = "C:/Users/baage/Desktop/Choi_MSI/02_개인_공동연구/Liq/liq_data"
    return data_dir


def expNumList():
    """"
    define experiment numbers that you want to include from the whole data
    """

    expNumList = [7, 8, 9, 10]
    return expNumList


def relativeDensity():
    """
    define relative densities for each experiment and trials
    """

    Dr_exp7 = np.array([55, 54, 47, 56, 52, 52, 39, 52, 41, 43,
                        39, 39, 33, 35, 43, 35, 41, 44, 53]) / 100
    Dr_exp8 = np.array([76, 94, 89, 77, 72, 73, 85, 88, 90, 74]) / 100
    Dr_exp9 = np.array([41, 47, 47, 40, 39, 43, 41, 49, 41, 37,
                        44, 47, 48, 44, 50, 47, 51, 48, 50, 50, 40, 44,
                        44, 45, 41, 44, 43, 41, 43, 37, 42, 43, 36, 50,
                        42, 45, 49, 43, 39, 43, 47, 48, 55, 44, 49, 41,
                        55]) / 100
    Dr_exp10 = np.array([78, 72, 84, 79, 82, 86, 89, 74, 70, 74, 80, 78, 89,
                         84, 89, 84, 81, 83, 91, 86, 78, 73, 72, 71, 76, 80, 73, 74, 73]) / 100
    Drs = [Dr_exp7, Dr_exp8, Dr_exp9, Dr_exp10]

    return Drs


def to_dataframe(data_dir, expNumList=expNumList(), Drs=relativeDensity()):
    """make a list to contain dataframe for the whole data"""

    df_all = []  # make an empty list to contain dataframes

    # get the number of trials for each experiment
    for exp in expNumList:
        dataDirList = getDataDirList(exp, basedir=data_dir)  # get dir list of trial files in each exp
        num_exps = len(expNumList)  # get num of experiments
        num_trials = len(dataDirList)  # get num of trials in each exp

        # make a list to contain a dataframe for each trial in the exp
        df_trial = []

        # get dataframe for a single trial and append it to `df_trial`
        for trial in range(num_trials):

            # make a dataframe for each trial
            dataDir = dataDirList[trial]  # get directory of each `trial.csv`
            df_single = pd.read_csv(dataDir, header=5)  # make `.csv` to pandas `df`

            # insert Dr values at the first column of the dataframe
            df_single.insert(0, "Dr [%]", Drs[exp-7][trial])

            # compute and insert ru at the 4th column of the dataframe
            # confining pressure of the test is the first (starting) effective vertical stress
            confining_pressure = df_single.iloc[0, 4]  # get conf pressure of the trial
            ru = df_single['Excess Pore Pressure [kPa]']/confining_pressure
            df_single.insert(5, "ru", ru)

            # Append this dataframe for the trial to `df_trial`
            df_trial.append(df_single)

        # Append `df_trial` to `df_all`
        df_all.append(df_trial)

    return df_all


def plotTrial(dataframe, expIndex, trialIndex):
    """plot all the columns in the dataframe at expIndex and trialIndex"""

    dataColNames = dataframe[expIndex][trialIndex].columns  # get data column names

    # plot for each data columns
    fig, axs = plt.subplots(nrows=len(dataColNames), ncols=1, figsize=(13, 15))
    axs_unroll = axs.flatten()
    for i, axi in enumerate(axs_unroll):
        axi.plot(dataframe[expIndex][trialIndex][dataColNames[i]])
        axi.set(xlabel='Data point')
        axi.set(ylabel=dataColNames[i])


def LookIntoData(dataframe, timeIndex, expNumList=expNumList()):
    """
    Some of data has a irregular time interval and different data points.
    This function look into those.
    """
    for k in range(len(expNumList)):
        len_dataList = len(dataframe[k])  # return num of trials at `k`th experiment
        print(f"Experiment{expNumList[k]}----------------------------------------")

        for i in range(len_dataList):
            print(f"*Trial-{i+1}")
            df = dataframe[k][i]  # load dataframe for the specified exp-trial
            index = df.index  # get the index of the dataframe
            index_last = index[-1]  # get the last index
            time = df['Time [sec]']  # get the time steps (sec)
            time_last = time[index_last]  # get the last time step
            time_choose = time[timeIndex]  # get the time (sec) at the specified index
            time_interval = time_last/index_last  # get the time interval (sec) between each data point

            print(f"last index and time:{index_last}, {time_last}; interval: {time_interval:.4f}\n{timeIndex} time step is: {time_choose} sec")


