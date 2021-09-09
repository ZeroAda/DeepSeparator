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
import math

def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def rrmsef(eeg,deeg,eog):
    RRMSEF=[]
    for j in range (0,eeg.shape[0]):
        eegj=eeg[j].reshape(eeg.shape[1])
        deegj=deeg[j].reshape(eeg.shape[1])
        #noisej= deegj-eegj

        num_fft=eeg.shape[1]
        Y = fft(deegj, eeg.shape[1])
        Y = np.abs(Y)
        psy = Y**2 / num_fft
        psdY=psy[:num_fft//2]

        X = fft(eegj, eeg.shape[1])
        X = np.abs(X)
        psx = X**2 / num_fft
        psdX=psx[:num_fft//2]

        coe = get_rms(psdY-psdX) / get_rms(psdX)
        RRMSEF.append(coe)

    test_mean_f = np.mean(RRMSEF)
    test_std_f = np.std(RRMSEF)
    return test_mean_f , test_std_f

if __name__ == '__main__':
    # clean_eeg = np.load('../data/clean_eeg.npy')
    EMG = np.load('../data/EMG_all_epochs.npy', allow_pickle=True)
    # clean_eeg = clean_eeg[0]
    EMG = EMG[0]
    # clean_eeg_ap = fft(clean_eeg)
    EMG_ap = fft(EMG)


    # clean_eeg_s = 2*clean_eeg_ap * np.conj(clean_eeg_ap) / (512*256)
    EMG_s = 2*EMG_ap * np.conj(EMG_ap) / (512*256)
    # plt.figure()
    # plt.plot(clean_eeg_s)
    # plt.plot(EMG_s)
    # plt.show()





    # plt.subplot(3, 1, 2)

    noise_pxx, freq = plt.psd(EMG,NFFT=512,Fs=256, pad_to=1024,
    scale_by_freq=True)
    # clean_pxx, freq = plt.psd(clean_eeg,NFFT=512,Fs=256, pad_to=1024,
    # scale_by_freq=True)


    # print("psd: ",noise_pxx)
    # print("maximum psd", np.max(noise_pxx),"minimum",np.min(noise_pxx))
    # plt.subplot(3, 1, 3)
    # plt.plot(np.arange(0,128.25,0.25),10*np.log10(noise_pxx), label='PSD by plt transfer')
    # plt.plot(np.arange(0,128.25,0.25),10*np.log10(clean_pxx), label='Clean EEG')
    # plt.title("PSD by plt transfer")

    plt.figure()
    # plt.subplot(3,1,1)
    f = np.linspace(0,512/4,256)
    oo = 10*np.log10(EMG_s[:512//2])
    oo_std = oo
    plt.plot(f,oo_std, label = 'PSD by hand')
    # plt.plot(f,10*np.log10(clean_eeg_s[:512//2]), label = 'Clean EEG')
    # plt.title("PSD by hand")

    # plt.figure()
    # plt.plot(EMG_s[:512//2]-noise_pxx)
    origin = 10*np.log10(noise_pxx)
    origin_std = origin
    plt.plot(np.arange(0,128.25,0.25),origin_std, label='PSD by plt transfer')


    num_fft = EMG.shape[0]
    signal = EMG.reshape(num_fft)
    # noisej= deegj-eegj

    num_fft = signal.shape[0]
    Y = fft(signal, num_fft)
    Y = np.abs(Y)
    psy = Y ** 2 / num_fft
    psdY = psy[:120]
    psdY = 10 * (np.log10(psdY))
    psdY_std = psdY
    plt.plot(psdY_std,label='psd by HaoMing')
    plt.legend()
    plt.show()



