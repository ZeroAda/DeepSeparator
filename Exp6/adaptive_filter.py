"""
File Name: adaptive_filter.py
Description: Apply adaptive filter to denoise single-channel EEG
Author: Chenyi Li
Date: 7/31/2021
"""
import warnings
warnings.filterwarnings("ignore")
from helper import *
def adafilter(noise_eeg, clean_eeg, EMG, EOG, L, ld):
    """ Adaptive filter.

    :param noise_eeg: an array of noisy EEG, 1 * Time lag
    :param clean_eeg: an array of clean EEG, 1 * Time lag
    :param EOG: an array of EOG, 1 * Time lag
    :param EMG: an array of EMG, 1 * Time lag
    :param L: integer of Filter order
    :param ld: integer of step size
    :return retain_eeg: an array of reconstructed EEG signal
    """
    # 1st Filter: denoise EMG
    # Prepare for processing
    origin_noisy = noise_eeg
    concate = np.zeros(L-1)
    noise_eeg = np.concatenate((concate,noise_eeg),axis=0)
    EMG = np.concatenate((concate,EMG))
    # Initialization
    # length of signal
    N =  noise_eeg.shape[0] - L + 1
    # x1: filter input
    x1 = EMG
    # d1: noisy eeg input
    d1 = noise_eeg
    # y1: filter output
    y1 = np.zeros(N)
    # e1: error
    e1 = np.zeros(N)
    # w1: paramter
    w1 = np.zeros(L)
    # Filter
    # for signal in each time lag, denoise with LMS algorithm
    for i in range(N):
        x = np.flipud(x1[i:i+L])

        y1[i] = w1 @ x
        e1[i] =  d1[i+L-1] - y1[i]
        w1 = w1 + ld * e1[i] * x
        y1[i] = w1 @ x

    # 2st Filter: denoise EOG
    # Prepare for processing
    concate = np.zeros(L - 1)
    noise_eeg2 = np.concatenate((concate, e1), axis=0)
    EOG = np.concatenate((concate, EOG))
    # Initialization
    # length of signal
    N = noise_eeg2.shape[0] - L + 1
    # x1: filter input
    x2 = EOG
    # d1: noisy eeg input
    d2 = noise_eeg2
    # y1: filter output
    y2 = np.zeros(N)
    # e1: error
    e2 = np.zeros(N)
    # w1: paramter
    w2 = np.zeros(L)
    # Filter
    # for signal in each time lag, denoise with LMS algorithm
    for i in range(N):
        x = np.flipud(x2[i:i + L])

        y2[i] = w2 @ x
        e2[i] = d2[i + L - 1] - y2[i]
        w2 = w2 + ld * e2[i] * x
        y2[i] = w2 @ x

    retain_eeg = e2
    return retain_eeg




