#!/usr/bin/env python
"""
File Name: datainput.py
Description: Check data input
Author: Orange Killer
Date: 2021/8/10
Log:
"""
import numpy as np


train_x = np.load("data/noise_eeg.npy")
train_y = np.load("data/clean_eeg.npy")
EMG = np.load('data/EMG_all_epochs.npy',allow_pickle=True)
EOG = np.load('data/EOG_all_epochs.npy',allow_pickle=True)
test_x = np.load("data/test_input.npy")
test_y = np.load("data/test_output.npy")

print("train_x",train_x.shape)
print("train_y",train_y.shape)
print("test_x",test_x.shape)
print("test_y",test_y.shape)
print("EMG",EMG.shape)
print("EOG",EOG.shape)