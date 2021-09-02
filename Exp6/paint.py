#!/usr/bin/env python
"""
File Name: paint.py
Description: paint SNR plot
Author: Orange Killer
Date: 2021/9/2
Log:
"""

import numpy as np
from helper import *

if __name__ == '__main__':

    ## EOG

    # EOGmset_list = np.loadtxt("EOGmset matrix")
    # EOGmses_list = np.loadtxt("EOGmses matrix")
    # EOGcc_list = np.loadtxt("EOGcc matrix")
    # EOGmset_list_CNN = np.loadtxt("EOGmset matrix_CNN")
    # EOGmses_list_CNN = np.loadtxt("EOGmses matrix_CNN")
    # EOGcc_list_CNN = np.loadtxt("EOGcc matrix_CNN")
    # plotSNRhighCNN("EOG",EOGmset_list,EOGmses_list,EOGcc_list, EOGmset_list_CNN, EOGmses_list_CNN, EOGcc_list_CNN)
    #

    # EMG
    EMGmset_list = np.loadtxt("EMGmset matrix")
    EMGmses_list = np.loadtxt("EMGmses matrix")
    EMGcc_list = np.loadtxt("EMGcc matrix")
    EMGmset_list_CNN = np.loadtxt("EMGmset matrix_CNN")
    EMGmses_list_CNN = np.loadtxt("EMGmses matrix_CNN")
    EMGcc_list_CNN = np.loadtxt("EMGcc matrix_CNN")
    plotSNRhighCNN("EMG",EMGmset_list,EMGmses_list,EMGcc_list, EMGmset_list_CNN, EMGmses_list_CNN, EMGcc_list_CNN)




