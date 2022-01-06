import os
from natsort import natsorted # Sort 1, 10, 2, 3 to 1, 2, 3, 10

def get_data_dirlist(expNum, basedir):
    """
    get `.txt` data directory list for specified experiments
    For example, when expNum=1, it returns ["MyData/trial-1", "MyData/trial-2", ..., "MyData/trial-n"]
    """
    
    experimentList = os.listdir(basedir)  # get a list of file names in `basedir`
    experimentList = [ex for ex in experimentList if 'Experiment' in ex]  # only get `Experiment` folders
    experimentList = natsorted(experimentList)  # `natsorted` enables to sort "1, 10, 2, ..." to "1, 2, ..., 10, ..."
    expFolderName = experimentList[expNum-1]  # get the file name (e.g, "Trial-5_accel_corr_Motion12_300mv.txt")
    expDir = os.path.join(basedir, expFolderName)  # get the data file dir with `join`
    dataList = os.listdir(expDir)  # get the data file list
    
    # get the data file directory lists for a specified `expNum`
    dataDirList = []  # make an empty list to contain directories
    for data in dataList:
        dataDir = os.path.join(expDir, data)  # dir for a data file
        dataDirList.append(dataDir)  # append
    
    # sort in a numerical order
    dataDirList = natsorted(dataDirList)
    
    return dataDirList
