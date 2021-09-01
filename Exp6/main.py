#!/usr/bin/env python
"""
File Name: main.py
Description: Apply traditional method to denoise single-channel EEG
Author: Chenyi Li
Date: 7/31/2021
"""
import time
from adaptive_filter import *
from hht import *
from cca import *
from ica import *
from helper import *



if __name__ == '__main__':
    start = time.time()
    # Data acquisition
    test_output = np.load('../data/test_output.npy')
    test_input = np.load('../data/test_input.npy')

    # test_output = np.load('data/EMG_EEG_test_output.npy')
    # test_input = np.load('data/EMG_EEG_test_input.npy')
    #
    # test_output = np.load('data/EOG_EEG_test_output.npy')
    # test_input = np.load('data/EOG_EEG_test_input.npy')

    EMG = np.load('../data/EMG_all_epochs.npy',allow_pickle=True)
    EOG = np.load('../data/EOG_all_epochs.npy',allow_pickle=True)
    # (5598, 512) EMG
    # (3400, 512) EOG

    # # sample number
    sample = 1
    # MSE temporal matrix
    mset_list = np.zeros((10, 4))
    # MSE spectral matrix
    mses_list = np.zeros((10, 4))
    # Correlation coefficient matrix
    cc_list = np.zeros((10, 4))
    # adaptive filter parameter
    L = 300
    ld = 0.001
    for i in range(sample):
        print("------- sample ", i, "----------")
        for j in range(10):
            print("--------- SNR", j-7, "-----------")


            # standardization of EMG, EOG
            stdEMG = EMG[j] / np.std(EMG[j])
            stdEOG = EOG[j] / np.std(EOG[j])

            # 1. adaptive filter
            print("------ adaptive filter ------")
            retain_eeg = adafilter(test_input[i+400*j], test_output[i+400*j], stdEMG, stdEOG, L, ld)
            mse_s, mse_t, cc = metric(test_input[i+400*j], test_output[i+400*j], retain_eeg)
            mset_list[j,0] += mse_t
            mses_list[j,0] += mse_s
            cc_list[j,0] += cc

            # 2. HHT
            print("------ HHT ------")
            retain_eeg = HHTFilter(test_input[i+400*j],test_output[i+400*j],2, 'threshold')
            mse_s, mse_t, cc = metric(test_input[i+400*j], test_output[i+400*j], retain_eeg)
            mset_list[j, 1] += mse_t
            mses_list[j, 1] += mse_s
            cc_list[j, 1] += cc

            # 3. EEMD-ICA
            print("------ EEMD-ICA ------")
            IMF, residue = EEMDanalysis(test_input[i+400*j], test_output[i+400*j])
            retain_eeg = ICAanalysis(test_input[i+400*j], test_output[i+400*j], IMF, residue)
            mse_s, mse_t, cc = metric(test_input[i+400*j], test_output[i+400*j], retain_eeg)
            mset_list[j, 2] += mse_t
            mses_list[j, 2] += mse_s
            cc_list[j, 2] += cc

            # 4. EEMD-CCA
            print("------ EEMD-CCA ------")
            IMF, residue = EEMDanalysis(test_input[i], test_output[i])
            retain_eeg = CCAanalysis(test_input[i+400*j], test_output[i+400*j], IMF, residue)
            mse_s, mse_t, cc = metric(test_input[i+400*j], test_output[i+400*j], retain_eeg)
            mset_list[j, 3] += mse_t
            mses_list[j, 3] += mse_s
            cc_list[j, 3] += cc

    mset_list /= sample
    mses_list /= sample
    cc_list /= sample

    np.savetxt("mset matrix", mset_list)
    np.savetxt("mses matrix", mses_list)
    np.savetxt("cc matrix", cc_list)

    # plotSNRhigh(mset_list, mses_list, cc_list)
    print("-------final result----------")
    for i in range(4):
        print(np.min(mset_list[:,i]), np.min(mses_list[:,i]), np.max(cc_list[:,i]))

    end = time.time()
    interval = end - start
    print("Time consumpition: ",interval)

