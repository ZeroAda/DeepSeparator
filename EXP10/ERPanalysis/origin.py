import mne
import numpy as np
import os

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)

sample_data_events_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_filt-0-40_raw-eve.fif')
events = mne.read_events(sample_data_events_file)

raw.crop(tmax=90)  # in seconds; happens in-place
events = events[events[:, 0] <= raw.last_samp]

raw.pick(['eeg']).load_data()
# raw.pick_channels(['EEG 026', 'EEG 025']).load_data()



event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}

epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7,
                    preload=True)

l_aud = epochs['auditory/left'].average()

l_aud.plot_joint(picks='eeg')

evokeds = dict(visual=list(epochs['auditory/left'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, combine='mean', title='denoised ERP')
