"""
File Name: cca.py
Description: Apply Canonical Correlation Analysis to denoise single-channel EEG
Author: Chenyi Li
Date: 7/31/2021
"""
from PyEMD import EEMD, EMD, Visualisation
from helper import *


def EEMDanalysis(noise_eeg, clean_eeg):
    """ Ensemble Empirical Mode Decomposition.

    :param noise_eeg: an array of noisy EEG, 1 * Time lag.
    :param clean_eeg: an array of clean EEG, 1 * Time lag.
    :return IMF: a multidimension array of Intrinsic Mode Function,
                 n * Time lag, where n is the number of decomposed IMF.
    :return residue: an array of residue of EMD, 1 * Time lag.
    """
    eemd = EEMD()
    IMF = eemd.eemd(noise_eeg)
    IMF,residue= eemd.get_imfs_and_residue()
    # plotIMF(IMF,residue)
    return IMF,residue

def estimated_autocorrelation(x):
    """Estimate autocorrelation.

    :param x: an array of signal.
    :return result: autocorrelation of the given signal.
    """
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result


def cca(X,Y,n):
    """Canonical Correlation Analysis.

    :param X: a multi-dimension array of IMF, n * Time lags.
    :param Y: a multi-dimension array of delayed IMF, n * Time lags.
    :param n: integer, number of target component.
    :return S1: cca result of X.
    :return S2: cca result of Y.
    """
    # covariance matrix
    Sxx = np.cov(X)
    Syy = np.cov(Y)
    Mixy = X @ Y.T
    Miyx = Y @ X.T
    Sxy = np.cov(Mixy)
    Syx = np.cov(Miyx)


    M = np.linalg.inv(Sxx) @ Sxy @ np.linalg.inv(Syy) @ Syx

    a_val, a = np.linalg.eig(M)
    S1 = a @ X

    N = np.linalg.inv(Syy) @ Syx @ np.linalg.inv(Sxx) @ Sxy
    b_val, b = np.linalg.eig(N)
    S2 = b @ Y

    return S1, S2, a

def CCAanalysis(noise_eeg, clean_eeg, IMF, residue, threshold1, threshold2):
    """Apply CCA decompose IMF. Utilize ACF to select artifact component. Retain EEG signal
    Users could adjust the threshold and threshold2 to tune the model.

    :param noise_eeg: an array of noisy EEG, 1 * Time lag.
    :param clean_eeg: an array of clean EEG, 1 * Time lag.
    :param IMF: a multi-dimension array of Intrinsic Mode Function,
                 n * Time lag, where n is the number of decomposed IMF.
    :param residue: an array of residue of EMD, 1 * Time lag.
    :return: error: integer, error between clean EEG and retained EEG.
    """
    # Select artifact IMF with autocorrelation
    # print("####### IMF selection ###############")

    acfimf_list = []
    for imf in IMF:
        acf = estimated_autocorrelation(imf)
        acfimf_list.append(np.abs(acf[1]))
    # threshold = np.mean(acfimf_list)
    threshold1 = 1
    super_threshold_indices = np.where(np.array(acfimf_list) < threshold1)
    mask = np.zeros(IMF.shape[0],dtype=bool)
    mask[super_threshold_indices] = True
    IMF_arti = IMF[mask]
    IMF_pure = IMF[~mask]

    # standardization of IMF
    mu = np.mean(IMF_arti)
    stdimf = np.std(IMF_arti)
    IMF_arti = (IMF_arti - mu)/stdimf

    # delay signal to be IMF2
    IMF2 = IMF_arti[:,1:]
    n = IMF_arti.shape[0]
    comple = np.zeros((n,1))
    IMF2 = np.concatenate((IMF2,comple),axis=1)
    # cca
    # print("####### CCA ###############")

    s1, s2, a = cca(IMF_arti,IMF2,8)  # Get the estimated sources

    # print("####### CCA selection###############")
    # Select artifact component to be 0
    acf_list = []
    for cc in s1:
        acf = estimated_autocorrelation(cc)
        acf_list.append(np.abs(acf[1]))
    threshold2 = np.mean(acf_list)
    super_threshold_indices = np.where(np.array(acf_list) < threshold2)
    s1[super_threshold_indices] = np.zeros(s1.shape[1])

    # retain IMF
    # print("####### IMF retain###############")
    retainIMF = np.linalg.inv(a) @ s1
    retainIMF = (retainIMF * stdimf) + mu
    retainIMF = np.append(retainIMF,IMF_pure,axis=0)
    retainIMFs = np.append(retainIMF,residue.reshape(1,-1),axis=0)

    # retain EEG
    retainEEG = np.sum(retainIMFs, axis=0)
    return retainEEG

def entropy(component):
    """Compute entropy of the component.

    :param component
    :return entropy
    """
    maxval = np.max(component)
    minval = np.min(component)
    bin = 10
    interval = (maxval - minval) / 10
    bag = []
    a = minval
    for i in range(bin):
        a = minval + i * interval
        b = a + interval
        c = component[np.where((component >= a) & (component <= b))]
        bag.append(c.shape[0])
    bag = np.array(bag)
    bag = bag / 512
    entropy = -1 * bag @ np.log(bag)
    return entropy


if __name__ == '__main__':
    # data acquisition
    noise_eeg = np.load('../data/test_input.npy')
    clean_eeg = np.load('../data/test_output.npy')
    # sample number
    sample = noise_eeg.shape[0]+1
    errorlist = []
    mse_sa = 0
    mse_ta = 0
    ccaa = 0
    for i in range(4000,4100):
        print("-------", i, "----------")
        IMF,residue = EEMDanalysis(noise_eeg[i],clean_eeg[i])
        retainEEG = CCAanalysis(noise_eeg[i], clean_eeg[i],IMF, residue, 0.9,0.9)
        plotSignal(noise_eeg[i],clean_eeg[i],retainEEG,"CCA")
        mse_s, mse_t, cc = metric(noise_eeg[i],clean_eeg[i],noise_eeg[i])
        mse_sa += mse_s
        mse_ta += mse_t
        ccaa += cc
    print(mse_sa," ",mse_ta," ",ccaa)



