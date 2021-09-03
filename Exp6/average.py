#!/usr/bin/env python
"""
File Name: 
Description: Count Average performance of traditional method
Author: Orange Killer
Date: 
Log:
"""

import numpy as np

if __name__ == '__main__':
    EOGmset_list = np.loadtxt("EOGmset matrix")
    EOGmses_list = np.loadtxt("EOGmses matrix")
    EOGcc_list = np.loadtxt("EOGcc matrix")
    EOGmset_list_CNN = np.loadtxt("EOGmset matrix_CNN")
    EOGmses_list_CNN = np.loadtxt("EOGmses matrix_CNN")
    EOGcc_list_CNN = np.loadtxt("EOGcc matrix_CNN")
    EOGmset_av = np.mean(EOGmset_list,axis=0)
    EOGmses_av = np.mean(EOGmses_list,axis=0)
    EOGcc_av = np.mean(EOGcc_list,axis=0)
    EOGmset_av_CNN = np.mean(EOGmset_list_CNN, axis=0)
    EOGmses_av_CNN = np.mean(EOGmses_list_CNN, axis=0)
    EOGcc_av_CNN = np.mean(EOGcc_list_CNN, axis=0)
    np.savetxt("EOGmset_av", EOGmset_av)
    np.savetxt("EOGmses_av", EOGmses_av)
    np.savetxt("EOGcc_av", EOGcc_av)
    # np.savetxt("EOGmset_av_CNN",EOGmset_av_CNN)
    # np.savetxt("EOGmses_av_CNN",EOGmses_av_CNN)
    # np.savetxt("EOGcc_av_CNN",EOGcc_av_CNN)
    print(EOGmset_av_CNN)
    print(EOGmses_av_CNN)
    print(EOGcc_av_CNN)





    # EMG
    EMGmset_list = np.loadtxt("EMGmset matrix")
    EMGmses_list = np.loadtxt("EMGmses matrix")
    EMGcc_list = np.loadtxt("EMGcc matrix")
    EMGmset_list_CNN = np.loadtxt("EMGmset matrix_CNN")
    EMGmses_list_CNN = np.loadtxt("EMGmses matrix_CNN")
    EMGcc_list_CNN = np.loadtxt("EMGcc matrix_CNN")
    EMGmset_av = np.mean(EMGmset_list, axis=0)
    EMGmses_av = np.mean(EMGmses_list, axis=0)
    EMGcc_av = np.mean(EMGcc_list, axis=0)
    EMGmset_av_CNN = np.mean(EMGmset_list_CNN, axis=0)
    EMGmses_av_CNN = np.mean(EMGmses_list_CNN, axis=0)
    EMGcc_av_CNN = np.mean(EMGcc_list_CNN, axis=0)
    np.savetxt("EMGmset_av", EMGmset_av)
    np.savetxt("EMGmses_av", EMGmses_av)
    np.savetxt("EMGcc_av", EMGcc_av)
    # np.savetxt("EMGmset_av_CNN",EMGmset_av_CNN)
    # np.savetxt("EMGmses_av_CNN",EMGmses_av_CNN)
    # np.savetxt("EMGcc_av_CNN",EMGcc_av_CNN)
    print(EMGmset_av_CNN)
    print(EMGmses_av_CNN)
    print(EMGcc_av_CNN)
