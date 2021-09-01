"""
File Name: ica.py
Description: Apply Independent Component Analysis to denoise single-channel EEG
Author: Chenyi Li
Date: 7/31/2021
"""
from PyEMD import EEMD, EMD, Visualisation
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA, PCA
from nolds import lyap_r,hurst_rs
from helper import *
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



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
    # plotIMF(IMF, residue)
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

def entropy(component):
    """Compute Entropy of component.

    :param component
    :return: entropy
    """
    # print(component.shape)
    maxval = np.max(component)
    minval = np.min(component)
    # print(maxval,minval)
    bin = 10
    interval = (maxval-minval) / 10
    bag = []
    a = minval
    for i in range(bin):
        a = minval + i*interval
        b = a + interval
        c = component[np.where((component >= a) & (component <= b))]
        # print("There are component",c)
        bag.append(c.shape[0])
    # prob = np.array(prob)
    bag = np.array(bag)
    # print("sum",np.sum(bag))
    bag = bag/512
    entropy = -1 * bag @ np.log(bag)
    return entropy


def ICAanalysis(noise_eeg, clean_eeg, IMF, residue):
    """Apply ICA to decompose IMF into independent components. Then utilize exponent analysis
    to select artifact. User could adjust threshold parameter

    :param noise_eeg: an array of noisy EEG, 1 * Time lag.
    :param clean_eeg: an array of clean EEG, 1 * Time lag.
    :param IMF: a multi-dimension array of Intrinsic Mode Function,
                 n * Time lag, where n is the number of decomposed IMF.
    :param residue: an array of residue of EMD, 1 * Time lag.
    :return: error: integer, error between clean EEG and retained EEG.
    """
    # print("####### ICA decomposition ###############")
    ica = FastICA(n_components=7, random_state=1, max_iter=1000)
    IC = ica.fit_transform(IMF.T)
    # plotIC(IC)
    # print("####### ICA selection: exponent analysis ###############")

    # Largest Lyapunov exponent: >0 chaos, <0 periodic signal
    lelist = np.array([lyap_r(ic) for ic in IC.T])
    # Kurtosis: >>0 sparse and peak, = 0 Gaussian, <0 periodic
    ktlist = np.array([kurtosis(ic) for ic in IC.T])
    # Hurst Exponent: human phenomena 0.64-0.69
    helist = np.array([hurst_rs(ic) for ic in IC.T])

    learti = np.where(lelist > 0)

    ktmu = np.mean(ktlist)
    deviation = (ktlist - ktmu)/ktmu
    ktarti = np.where(deviation > 1)

    hearti = np.where((helist > 0.7) & (helist < 0.9))

    artifact = np.union1d(np.intersect1d(learti,ktarti),hearti)
    # select artifact and turn it to 0
    IC.T[artifact] = np.zeros((artifact.shape[0],IC.T.shape[1]))

    # print("####### IMF retain ###############")
    retainIMF = ica.inverse_transform(IC)
    retainIMFs = np.append(retainIMF, residue.reshape(-1,1), axis=1)

    # print("########### EEG retain ###############")
    retainEEG = np.sum(retainIMFs.T, axis=0)
    # plotSignal(noise_eeg, clean_eeg, retainEEG,'ica')
    # plt.plot(noise_eeg)
    # plt.show()

    error = np.mean(np.power((retainEEG - clean_eeg), 2))
    origin = np.mean(np.power((noise_eeg - clean_eeg), 2))
    print("error: ",error)

    return retainEEG



if __name__ == '__main__':
    # data acquisition
    noise_eeg = np.load('data/noise_eeg.npy')
    clean_eeg = np.load('data/clean_eeg.npy')
    # sample number
    sample = noise_eeg.shape[0]+1
    # fs sampling frequency
    fs = 512

    # ICA + exponent analysis
    errorlist = []
    for i in range(10):
        print("-------", i, "----------")
        # plot frequency
        # Frequencyanalysis(noise_eeg[i], clean_eeg[i])
        # band filter
        # filtered = butter_bandpass_filter(noise_eeg[i], 0.1, 50.0, fs, order=5)
        IMF,residue = EEMDanalysis(noise_eeg[i],clean_eeg[i])
        retainEEG = ICAanalysis(noise_eeg[i],clean_eeg[i],IMF, residue)

