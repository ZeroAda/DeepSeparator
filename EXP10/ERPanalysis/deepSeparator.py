import os
import mne
import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn
import os
import torch
import torch.utils.data as Data

from network import *

def standardization(input):
    mu = np.mean(input)
    sigma = np.std(input)
    return (input - mu) / sigma, sigma


def normalization(input):
    _range = np.max(input) - np.min(input)
    return (data - np.min(input)) / _range


sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)

sample_data_events_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw-eve.fif')
events = mne.read_events(sample_data_events_file)

raw.crop(tmax=90)  # in seconds; happens in-place
events = events[events[:, 0] <= raw.last_samp]

raw.pick(['eeg']).load_data()
# raw.pick_channels(['EEG 026', 'EEG 025']).load_data()

data = raw.get_data()

'''   把数据抽出来进行处理    '''


data_len = data.shape[1]
data = normalization(data)

test_input = torch.from_numpy(data)
test_input = test_input.float()

model = CNN_CNN()

if os.path.exists('checkpoint/CNN_CNN.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/CNN_CNN.pkl'))


for i in range(data_len//512):

    test_preds, atte_x = model(test_input[:, i*512:i*512+512], 0)

    test_preds = test_preds.cpu()
    test_preds = test_preds.detach().numpy()

    if i == 0:
        denoised_result = test_preds
    else:
        denoised_result = np.append(denoised_result, test_preds, axis=1)


new_raw = mne.io.RawArray(denoised_result, raw.info, first_samp=raw.first_samp).load_data()


event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}

epochs = mne.Epochs(new_raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7,
                    preload=True)

l_aud = epochs['auditory/left'].average()
l_aud.plot_joint(picks='eeg')


evokeds = dict(visual=list(epochs['auditory/left'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, combine='mean', title='denoised ERP')
