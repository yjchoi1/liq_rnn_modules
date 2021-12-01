# liq_rnn_modules
Code to run RNN with cyclic simple shear (CSS) test data with modules.
## Data
Experiment data download link: https://drive.google.com/file/d/1G1zMdk4n9qdUHg7BexmXGwWnRZ5etxrL/view?usp=sharing Reference: https://doi.org/10.1193/093016EQS167DP
## Code Structure
`data_importing_functions.py`: Get the directory and file names of the data.\
`cssData.py`: Script for converting CSS data (.txt files) to dataframe, and add relative density and ru to it. 
Script for plotting the trial.\
`prepareData.py`: Script for preprocessing of the prepared dataframe\
`train.py`: RNN training\

