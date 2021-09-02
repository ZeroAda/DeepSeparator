#!/usr/bin/env python
"""
File Name: psd.py
Description: Power Spectral Density TEST
Author: Orange Killer
Date: 
Log:
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

if __name__ == '__main__':
    clean_eeg = np.load('data/clean_eeg.npy')
    EMG = np.load('data/EMG_all_epochs.npy', allow_pickle=True)
    clean_eeg = clean_eeg[0]
    EMG = EMG[0]
    clean_eeg_ap = fft(clean_eeg)
    EMG_ap = fft(EMG)


    clean_eeg_s = 2*clean_eeg_ap * np.conj(clean_eeg_ap) / (512*256)
    EMG_s = 2*EMG_ap * np.conj(EMG_ap) / (512*256)
    # plt.figure()
    # plt.plot(clean_eeg_s)
    # plt.plot(EMG_s)
    # plt.show()

    plt.figure()
    # plt.subplot(3,1,1)
    f = np.linspace(0,512/4,256)
    plt.plot(f,10*np.log10(EMG_s[:512//2]), label = 'Noise EEG')
    plt.plot(f,10*np.log10(clean_eeg_s[:512//2]), label = 'Clean EEG')
    plt.title("PSD by hand")



    # plt.subplot(3, 1, 2)

    noise_pxx, freq = plt.psd(EMG,NFFT=512,Fs=256, pad_to=1024,
    scale_by_freq=True)
    clean_pxx, freq = plt.psd(clean_eeg,NFFT=512,Fs=256, pad_to=1024,
    scale_by_freq=True)
    # plt.legend()
    plt.title("PSD by plt")

    print("psd: ",noise_pxx)
    print("maximum psd", np.max(noise_pxx),"minimum",np.min(noise_pxx))
    # plt.subplot(3, 1, 3)
    # plt.plot(np.arange(0,128.25,0.25),10*np.log10(noise_pxx), label='Noisy EEG')
    # plt.plot(np.arange(0,128.25,0.25),10*np.log10(clean_pxx), label='Clean EEG')
    # plt.title("PSD by plt transfer")
    # plt.show()

    # plt.figure()
    # plt.plot(EMG_s[:512//2]-noise_pxx)
    plt.show()

