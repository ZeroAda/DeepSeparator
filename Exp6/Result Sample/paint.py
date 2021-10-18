#!/usr/bin/env python
"""
File Name: paint.py
Description: paint SNR plot and Count average
Author: Orange Killer
Date: 2021/9/2
Log:
"""

import numpy as np
from helper import *
import matplotlib.pyplot as plt
if __name__ == '__main__':

    ## EOG
    #
    EOGmset_list = np.loadtxt("EOGrrmset matrix")
    EOGmses_list = np.loadtxt("EOGrrmses matrix")
    EOGcc_list = np.loadtxt("EOGcc matrix")
    EOGmset_list_CNN = np.loadtxt("EOGmset matrix_CNN_test")
    EOGmses_list_CNN = np.loadtxt("EOGmses matrix_CNN_test")
    EOGcc_list_CNN = np.loadtxt("EOGcc matrix_CNN_test")
    print("---EOG---")
    EOGmset_avg = np.mean(EOGmset_list, axis=0)
    EOGmses_avg = np.mean(EOGmses_list, axis=0)
    EOGcc_avg = np.mean(EOGcc_list, axis=0)
    print(EOGmset_avg)
    print(EOGmses_avg)
    print(EOGcc_avg)
    plotSNRhighCNN("(a) Ocular artifact removal",EOGmset_list,EOGmses_list,EOGcc_list, EOGmset_list_CNN, EOGmses_list_CNN, EOGcc_list_CNN)


    # EMG
    EMGmset_list = np.loadtxt("EMGrrmset matrix")
    EMGmses_list = np.loadtxt("EMGrrmses matrix")
    EMGcc_list = np.loadtxt("EMGcc matrix")
    EMGmset_list_CNN = np.loadtxt("EMGmset matrix_CNN_test")
    EMGmses_list_CNN = np.loadtxt("EMGmses matrix_CNN_test")
    EMGcc_list_CNN = np.loadtxt("EMGcc matrix_CNN_test")
    plotSNRhighCNN("(b) Myogenic artifact removal",EMGmset_list,EMGmses_list,EMGcc_list, EMGmset_list_CNN, EMGmses_list_CNN, EMGcc_list_CNN)
    print("---EMG---")
    EMGmset_avg = np.mean(EMGmset_list, axis=0)
    EMGmses_avg = np.mean(EMGmses_list, axis=0)
    EMGcc_avg = np.mean(EMGcc_list, axis=0)
    print(EMGmses_avg)
    print(EMGmset_avg)
    print(EMGcc_avg)

    # EMG_EOG
    print("overall")
    all_cc = np.loadtxt("allrrcc matrix")
    all_mses = np.loadtxt("allrrmses matrix")
    all_mset = np.loadtxt("allrrmset matrix")
    avg_cc = np.mean(all_cc, axis=0)
    avg_mses = np.mean(all_mses, axis=0)
    avg_mset = np.mean(all_mset, axis=0)
    print(avg_mset)
    print(avg_mses)
    print(avg_cc)
    # plotSNRhigh(all_mset, all_mses, all_cc, "all")







