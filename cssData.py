# import from existing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import from my modules
from data_importing_functions import getDataDirList


def data_dir():
    # define directory where data exists
    data_dir = "C:/Users/baage/Desktop/Choi_MSI/02_개인_공동연구/Liq/liq_data"
    return data_dir


def expNumList():
    # define experiment numbers that you want to include from the whole data
    expNumList = [7, 8, 9, 10]
    return expNumList


def relativeDensity():
    # define relative densities for each experiment and trials
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


def to_dataframe(expNumList=expNumList(), data_dir=data_dir(), Drs=relativeDensity()):

    # make a list to contain dataframe for the whole data
    df_all = []

    # get the number of trials for each experiment
    for exp in expNumList:
        dataDirList = getDataDirList(exp, basedir=data_dir)
        num_exps = len(expNumList) # get the number of experiments
        num_trials = len(dataDirList) #

        # make a list to contain dataframe for each trial
        df_trial = []

        # get data frame for a single trial and append it to `df_trial`
        for trial in range(num_trials):

            # make a dataframe for the specified `trialNum`
            dataDir = dataDirList[trial] # get directory of each `trial.csv`
            df_single = pd.read_csv(dataDir, header=5) # make `.csv` to pandas `df`

            # insert Dr values as a first column
            df_single.insert(0, "Dr [%]", Drs[exp-7][trial])

            # insert ru
            # confining pressure of the test is the first (starting) effective vertical stress
            confining_pressure = df_single.iloc[0, 4] # get conf pressure of the trial
            ru = df_single['Excess Pore Pressure [kPa]']/confining_pressure
            df_single.insert(5, "ru", ru)

            # Append this dataframe for a single trial to `df_trial`
            df_trial.append(df_single)

        # Append `df_trial`
        df_all.append(df_trial)

    return df_all


def plotTrial(expIndex, trialIndex, dataframe=to_dataframe()):

    dataColNames = dataframe[expIndex][trialIndex].columns # get data column names

    # plot for each data columns
    fig, axs = plt.subplots(nrows=len(dataColNames), ncols=1, figsize=(13, 15))
    axs_unroll = axs.flatten()
    for i, axi in enumerate(axs_unroll):
        axi.plot(dataframe[expIndex][trialIndex][dataColNames[i]])
        axi.set(xlabel='Data point')
        axi.set(ylabel=dataColNames[i])


def LookIntoData(timeIndex, expNumList=expNumList(), dataframe=to_dataframe()):

    for k in range(len(expNumList)):
        len_dataList = len(dataframe[k])
        print(f"Experiment{expNumList[k]}----------------------------------------")

        for i in range(len_dataList):
            print(f"*Trial-{i+1}")
            df = dataframe[k][i]  # choose your experiment number
            index = df.index
            index_last = index[-1]
            time = df['Time [sec]']
            time_last = time[index_last]
            index_choose = timeIndex  # choose an index for the time you want to get
            time_choose = time[index_choose]

            time_interval = time_last/index_last

            print(f"last index and time:{index_last}, {time_last}; interval: {time_interval:.4f}\n{timeIndex} time step is: {time_choose} sec")


