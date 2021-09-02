"""
File Name: helper.py
Description: Helper functions to plot figures and measure performance
Author: Chenyi Li
Date: 7/31/2021
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft


def metric(origin_noisy,clean_eeg,retain_eeg):
    # problem: psd xx 与 手算的不大一样
    """ Metric
    MSE,CC
    :param origin_noisy:
    :param clean_eeg:
    :param retain_eeg:
    :return:
    """
    # sample point
    # temporal mse
    mse_t = np.mean(np.power((retain_eeg-clean_eeg),2))
    # spectral mse
    # fft
    num_sample = 512
    # retain_eeg_ap = fft(retain_eeg, num_sample)
    # clean_eeg_ap = fft(clean_eeg, num_sample)
    # noise_eeg_ap = fft(origin_noisy, num_sample)

    # psd


    retain_eeg_s,freq = plt.psd(retain_eeg, NFFT=512, Fs=256, pad_to=1024,
                                 scale_by_freq=True)
    clean_eeg_s,freq = plt.psd(clean_eeg, NFFT=512, Fs=256, pad_to=1024,
                                 scale_by_freq=True)
    noise_eeg_s = plt.psd(origin_noisy, NFFT=512, Fs=256, pad_to=1024,
                                 scale_by_freq=True)
    # print(np.double(np.mean(np.power(retain_eeg_s-clean_eeg_s,2))))
    # plt.subplot(3,1,1)
    # f = np.linspace(0,num_sample/4,256)
    # plt.plot(f,10*np.log10(noise_eeg_s[:num_sample//2]), label = 'Noise EEG')
    # plt.plot(f,10*np.log10(clean_eeg_s[:num_sample//2]), label = 'Clean EEG')
    # plt.plot(f,10*np.log10(retain_eeg_s[:num_sample//2]),label = 'Retain EEG')
    # plt.title("PSD by hand")

    # plt.subplot(3, 1, 2)
    # noise_pxx, freq = plt.psd(origin_noisy,NFFT=512,Fs=256,window=mlab.window_none, pad_to=1024,
    # scale_by_freq=True)
    # clean_pxx, freq = plt.psd(clean_eeg,NFFT=512,Fs=256,window=mlab.window_none, pad_to=1024,
    # scale_by_freq=True)
    # retain_pxx, freq = plt.psd(retain_eeg,NFFT=512,Fs=256,window=mlab.window_none, pad_to=1024,
    # scale_by_freq=True)
    # # plt.legend()
    # plt.title("PSD by plt")
    #
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(10*np.log10(noise_pxx), label='Noisy EEG')
    # plt.plot(10*np.log10(clean_pxx), label='Clean EEG')
    # plt.plot(10*np.log10(retain_pxx), label='Retain EEG')
    # plt.title("PSD by plt transfer")

    # plt.figure()
    # plt.plot(noise_eeg_ap,label='Noisy EEG')
    # plt.plot(clean_eeg_ap,label='Clean EEG')
    # plt.plot(retain_eeg_ap,label='Retain EEG')
    # plt.legend()
    # plt.title("FFT")


    # plt.show()

    mse_s = np.mean(np.power(retain_eeg_s-clean_eeg_s, 2))
    # correlation coefficient
    cc = np.corrcoef(retain_eeg, clean_eeg)[0,1]
    # print("mse t",mse_t)
    # print("mse s",np.double(mse_s))
    # print("cc",cc)
    plt.close()

    return mse_s, mse_t, cc

def plotSNRhigh(mset_list, mses_list, cc_list):
    plt.figure()
    plt.plot(range(-7, 3), mset_list[:,0].T,'.-.',label='Adaptive Filter')
    plt.plot(range(-7, 3), mset_list[:,1].T,'.-.',label='HHT')
    plt.plot(range(-7, 3), mset_list[:,2].T,'.-.',label='EEMD-ICA')
    plt.plot(range(-7, 3), mset_list[:,3].T,'.-.',label='EEMD-CCA')
    plt.xlabel("SNR(db)")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("MSE temporal")

    plt.figure()
    plt.plot(range(-7, 3), mses_list[:, 0].T, '.-.',label='Adaptive Filter')
    plt.plot(range(-7, 3), mses_list[:, 1].T, '.-.',label='HHT')
    plt.plot(range(-7, 3), mses_list[:, 2].T, '.-.',label='EEMD-ICA')
    plt.plot(range(-7, 3), mses_list[:, 3].T, '.-.',label='EEMD-CCA')
    plt.xlabel("SNR(db)")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("MSE spectral")

    plt.figure()
    plt.plot(range(-7, 3), cc_list[:, 0].T, '.-.',label='Adaptive Filter')
    plt.plot(range(-7, 3), cc_list[:, 1].T, '.-.',label='HHT')
    plt.plot(range(-7, 3), cc_list[:, 2].T, '.-.',label='EEMD-ICA')
    plt.plot(range(-7, 3), cc_list[:, 3].T, '.-.',label='EEMD-CCA')
    plt.xlabel("SNR(db)")
    plt.ylabel("CC")
    plt.title("CC")
    plt.legend()
    plt.show()


def plotSNRhighCNN(name,mset_list, mses_list, cc_list, mset_list_CNN, mses_list_CNN, cc_list_CNN):
    plt.figure()
    plt.plot(range(-7, 3), mset_list[:,0].T,'.-.',label='Adaptive Filter')
    plt.plot(range(-7, 3), mset_list[:,1].T,'.-.',label='HHT')
    plt.plot(range(-7, 3), mset_list[:,2].T,'.-.',label='EEMD-ICA')
    plt.plot(range(-7, 3), mset_list[:,3].T,'.-.',label='EEMD-CCA')
    plt.plot(range(-7, 3), mset_list_CNN.T,'.-.',label='CNN-CNN')


    plt.xlabel("SNR(db)")
    plt.ylabel("MSE")
    plt.legend()
    title = name + " MSE temporal"
    plt.title(title)
    fig = title + ".png"
    plt.savefig(fig)


    plt.figure()
    plt.plot(range(-7, 3), mses_list[:, 0].T, '.-.',label='Adaptive Filter')
    plt.plot(range(-7, 3), mses_list[:, 1].T, '.-.',label='HHT')
    plt.plot(range(-7, 3), mses_list[:, 2].T, '.-.',label='EEMD-ICA')
    plt.plot(range(-7, 3), mses_list[:, 3].T, '.-.',label='EEMD-CCA')
    plt.plot(range(-7, 3), mses_list_CNN.T,'.-.',label='CNN-CNN')

    plt.xlabel("SNR(db)")
    plt.ylabel("MSE")
    plt.legend()
    title = name + " MSE spectral"
    plt.title(title)
    fig = title + ".png"
    plt.savefig(fig)

    plt.figure()
    plt.plot(range(-7, 3), cc_list[:, 0].T, '.-.',label='Adaptive Filter')
    plt.plot(range(-7, 3), cc_list[:, 1].T, '.-.',label='HHT')
    plt.plot(range(-7, 3), cc_list[:, 2].T, '.-.',label='EEMD-ICA')
    plt.plot(range(-7, 3), cc_list[:, 3].T, '.-.',label='EEMD-CCA')
    plt.plot(range(-7, 3), cc_list_CNN.T,'.-.',label='CNN-CNN')

    plt.xlabel("SNR(db)")
    plt.ylabel("CC")
    title = name + " CC"
    plt.title(title)
    fig = title + ".png"
    plt.savefig(fig)

    plt.legend()
    plt.show()

def plotSNR(mset_list, mses_list, cc_list):
    plt.figure()
    plt.plot(range(-7, 3), mset_list, '--r')
    plt.xlabel("SNR(db)")
    plt.ylabel("MSE")
    plt.title("MSE temporal")

    plt.figure()
    plt.plot(range(-7, 3), mses_list, '--g')
    plt.xlabel("SNR(db)")
    plt.ylabel("MSE")
    plt.title("MSE spectral")

    plt.figure()

    plt.plot(range(-7, 3), cc_list, '--b')
    plt.xlabel("SNR(db)")
    plt.ylabel("CC")
    plt.title("CC")
    plt.show()


def plotSignal(noise_eeg, clean_eeg, retain_eeg, name):
    """
    Plot signal
    :param noise_eeg:
    :param clean_eeg:
    :param retain_eeg:
    :return:
    """
    plt.figure(figsize=[10, 7])
    time = np.linspace(0,2,512)
    plt.plot(time,noise_eeg, label='Noise EEG')
    plt.plot(time,clean_eeg, label='Clean EEG')
    plt.plot(time,retain_eeg, label='Retain EEG')

    name = name

    plt.title(name)
    plt.legend()
    plt.show()

def plotIMF(IMF, residue):
    """
    Plot IMF and its frequency
    :param IMF:
    :param residue:
    :return:
    """
    N = IMF.shape[0]
    # print(IMF)
    for n, imf in enumerate(IMF):
        plt.subplot(N+1, 1, n + 1)
        plt.plot(imf, 'g')
        plt.title("IMF " + str(n + 1))
        plt.xlabel("Time [s]")
    plt.subplot(N+1, 1, N+1)
    plt.plot(residue, 'lightblue')
    plt.title("Residue")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.title("EEMD")
    plt.show()

    for n, imf in enumerate(IMF):
        imf_fft = fft(imf)
        t_s = 2 / (512 - 1)
        f_s = 1 / t_s
        f_x = np.arange(0, f_s + 0.5, 1 / 2)
        # print(f_x)
        imf_fft_plt = np.abs(imf_fft[:imf_fft.size // 2])
        plt.subplot(N+1, 1, n + 2)
        plt.plot(f_x[:len(f_x)//2],imf_fft_plt, 'g')
        plt.title("IMF frequency " + str(n + 1))
        plt.xlabel("Frequency")
        print("IMF",n,"Max frequency",f_x[np.argmax(imf_fft_plt)])
    plt.show()

def plotIC(IC):
    N = IC.T.shape[0]
    print(N)
    dis = []
    index = []
    H = IC.T

    for n, ic in enumerate(IC.T):
        # print(n)
        # distance = np.sum(noise_eeg-ic.T)
        # dis.append(distance)

        plt.subplot(N+1, 1, n+1)
        plt.plot(ic, 'g')
        plt.title("IC " + str(n + 1))
        plt.xlabel("Time")
    plt.show()


def Frequencyanalysis(noise_eeg, clean_eeg):
    # fft analysis

    noise_fft = fft(noise_eeg)
    t_s = 2 / (512 - 1)
    f_s = 1 / t_s
    f_x = np.arange(0, f_s + 0.5, 1 / 2)
    noise_fft_plt = noise_fft[:noise_fft.size // 2]

    clean_fft = fft(clean_eeg)

    clean_fft_plt = clean_fft[:clean_fft.size // 2]

    dif = abs(abs(noise_fft_plt) - abs(clean_fft_plt))

    sorted_freq = np.argsort(dif)
    # a = np.array([12,3,4])
    max_dif = f_x[sorted_freq]

    # visualization
    plt.figure(figsize=[10, 7])
    plt.subplot(211)

    plt.plot(f_x[:len(f_x) // 2], np.abs(noise_fft_plt))
    plt.plot(f_x[:len(f_x) // 2], np.abs(clean_fft_plt))
    plt.legend(['noise', 'clean'])
    plt.subplot(212)
    plt.plot(f_x[:len(f_x) // 2], dif)

    plt.title("FFT")
    plt.show()