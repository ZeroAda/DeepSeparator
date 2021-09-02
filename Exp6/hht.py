"""
File Name: hht.py
Description: Apply Hilbert-Huang Transfer and hierarchical clustering to denoise single-channel EEG
Author: Chenyi Li
Date: 7/31/2021
"""
from pyhht import EMD
from scipy.signal import hilbert
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import tftb.processing
from scipy.cluster.hierarchy import fcluster
import operator
from helper import *



def HHTFilter(noise_eeg, clean_eeg, threshold, mode):
    """Apply HHT to decompose noise signal and utilize hierarchical clustering to select artifact
    whose instant frequency has large distance with others.

    :param noise_eeg: an array of noisy EEG, 1 * Time lag.
    :param clean_eeg: an array of clean EEG, 1 * Time lag.
    :param threshold: number. distance for 'filter' mode; number for 'threshold' mode.
    :param mode: 'filter' or 'threshold'.
    :return: integer, error between clean EEG and retained EEG.
    """
    # EMD
    decomposer = EMD(noise_eeg)
    imfs = decomposer.decompose()
    n_components = imfs.shape[0]
    # plotIMF(imfs, np.zeros(imfs.shape[1]))
    instf_ls = []
    for i in range(n_components):
        # HHT
        imfsHT = hilbert(imfs[i])
        # instant frequency
        instf, timestamps = tftb.processing.inst_freq(imfsHT)
        instf_ls.append(instf)

    # distance matrix
    instf_ls = np.mat(instf_ls)
    dismatrix = pdist(instf_ls)

    # normalized matrix
    maxm = max(dismatrix)
    minm = min(dismatrix)
    k = (0.1 - 0.01) / (maxm - minm)
    nordismatrix = 0.01 + k * (dismatrix - minm)

    # hierachical cluster
    tree = linkage(nordismatrix, method='single', metric='euclidean')
    # fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(tree)
    # plt.axhline(threshold)
    # plt.show()


    # IMF selection
    ## filter mode:
    ## select the distance of a point larger than the threshold as artifact
    if mode == 'filter':
        newcluster = fcluster(tree,threshold,criterion='distance')
        count = {}
        for point in newcluster:
            count[point] = count.get(point,0) + 1
        # find the cluster index with largest number
        retain = max(count.items(), key=operator.itemgetter(1))[0]
        componentsRetain = np.where(newcluster == retain)
        # print("Maintain component: ",componentsRetain)

    ## threshold mode:
    ## select two component with largest distance from other clusters
    elif mode == 'threshold':
        forbid = set()
        for i in range(tree.shape[0],0,-1):
            if tree[i-1,1] < n_components:
                forbid.add(tree[i-1,1])
                if len(forbid) >= threshold:
                    break
            if tree[i-1,0] < n_components:
                forbid.add(tree[i-1,0])
                if len(forbid) >= threshold:
                    break
        origin = np.array(range(n_components))
        mask = [False if i in forbid else True for i in range(n_components)]
        componentsRetain = origin[mask]
        # print("Maintain component: ",componentsRetain)

    retain_eeg = np.sum(imfs[componentsRetain], axis=0)
    # plot
    # plotSignal(noise_eeg,clean_eeg,retain_eeg,'hht')

    # MSE for each signal
    error = np.mean(np.power((retain_eeg - clean_eeg),2))
    # print("error:",error)
    return retain_eeg


if __name__ == '__main__':
    # load data
    noise_eeg = np.load('../data/test_input.npy')
    clean_eeg = np.load('../data/test_output.npy')
    sample = noise_eeg.shape[0]
    error = 0
    errorlist = []
    # HHT filter
    mse_sa = 0
    mse_ta = 0
    ccaa = 0
    for i in range(4001,4010):
        print("-------",i,"----------")
        retain_eeg = HHTFilter(noise_eeg[i], clean_eeg[i], threshold=2, mode='threshold')
        # plotSignal(noise_eeg[i],clean_eeg[i],retain_eeg,"hht")
        mse_s, mse_t, cc = metric(noise_eeg[i], clean_eeg[i], retain_eeg)
        mse_sa += mse_s
        mse_ta += mse_t
        ccaa += cc
    print(mse_sa, " ", mse_ta, " ", ccaa)


