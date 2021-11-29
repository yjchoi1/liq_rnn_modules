import os
from natsort import natsorted # Sort 1, 10, 2, 3 to 1, 2, 3, 10


def getDataDirList(expNum, basedir="/home/jupyter/MyData/liq_data_2"):
    '''
    get `.txt` data directory list for a certian exhttps://jupyter.designsafe-ci.org/user/baagee/edit/MyData/liq2/getDataDirList.py#periment
    For example, when expNum=1, it returns ["trial-1", "trial-2", ..., "trial-n"]
    '''
    
    # get the data file list for a certain `expNum`
    basedir = basedir # change dir depending on where you are
    experimentList = os.listdir(basedir) # get a list of file names in `basedir`
    experimentList = [ex for ex in experimentList if 'Experiment' in ex] # only get `Experiment` folders
    experimentList = natsorted(experimentList) # `natsorted` is a method to sort "1, 10, 2, ..."" to "1, 2, ..., 10, ...""
    expFolderName = experimentList[expNum-1] # get the file name (e.g, "Trial-5_accel_corr_Motion12_300mv.txt")
    expDir = os.path.join(basedir, expFolderName) # get the data file dir with `join`
    dataList = os.listdir(expDir) # get the data file list
    
    # get the data file direction list for a certain `expNum`
    dataDirList = [] # make an empty list
    for data in dataList:
        dataDir = os.path.join(expDir, data) # dir for a data file
        dataDirList.append(dataDir) # append
    
    # sort in a numerical order
    dataDirList = natsorted(dataDirList)
    
    return dataDirList
