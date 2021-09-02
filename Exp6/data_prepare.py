import math
import numpy as np


def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

def random_signal(signal):
    random_num = np.random.permutation(signal.shape[0])
    shuffled_dataset = signal[random_num, :]
    shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
    return shuffled_dataset

def data_prepare(noise, EEG):
    # SNR
    SNR_test_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_test = 10 ** (0.1 * (SNR_test_dB))

    eeg_test = np.array(EEG)
    noise_test = np.array(noise)

    EEG_test = []
    noise_EEG_test = []
    noise_list = []
    # for each SNR
    for i in range(eeg_test.shape[0]):
        noise_eeg_test = []
        noise_sublist = []
        eegsublist = []
        # for each EEG signal
        for j in range(10):
            eeg = eeg_test[i]
            # plt.plot(eeg)
            noise = noise_test[i]
            coe = get_rms(eeg) / (get_rms(noise) * SNR_test[j])
            noise = noise * coe
            neeg = noise + eeg
            # plotSignal(neeg,eeg,noise)
            noise_eeg_test.append(neeg)
            noise_sublist.append(noise)
            eegsublist.append(eeg)

        EEG_test.extend(eegsublist)
        noise_EEG_test.extend(noise_eeg_test)
        noise_list.extend(noise_sublist)
    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)
    noise_list = np.array(noise_list)

    # standardization
    EEG_test_end_standard = []
    noiseEEG_test_end_standard = []
    noise_list_end_standard = []
    std_VALUE = []
    print(noise_EEG_test.shape[0])
    for i in range(noise_EEG_test.shape[0]):
        noise_std_value = np.std(noise_EEG_test[i])
        std_VALUE.append(noise_std_value)
        noiseeeg_test_end_standard = noise_EEG_test[i] / noise_std_value
        noiseEEG_test_end_standard.append(noiseeeg_test_end_standard)
        # store std value to restore EEG signal
        std_value = np.std(EEG_test[i])
        eeg_test_all_std = EEG_test[i] / noise_std_value
        EEG_test_end_standard.append(eeg_test_all_std)
        # plt.plot(EEG_test[i])
        # plt.show()

        # noise_std_value = np.std(noise_list[i])
        noise_eeg_test_all_std = noise_list[i] / noise_std_value
        noise_list_end_standard.append(noise_eeg_test_all_std)

        # plotSignal(noiseeeg_test_end_standard,eeg_test_all_std,noise_eeg_test_all_std)

    std_VALUE = np.array(std_VALUE)
    noiseEEG_test_end_standard = np.array(noiseEEG_test_end_standard)
    EEG_test_end_standard = np.array(EEG_test_end_standard)
    noise_list_end_standard = np.array(noise_list_end_standard)

    print('test data prepared, test data shape: ', noiseEEG_test_end_standard.shape, EEG_test_end_standard.shape)
    return noiseEEG_test_end_standard, EEG_test_end_standard, noise_list_end_standard
