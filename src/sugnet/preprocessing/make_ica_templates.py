# imports
from xarray_creator import _cut_noisy
from preprocessing import _make_montage
import numpy as np
import mne_bids
import mne

# # ecg template from subject 1 that has a typical ecg component
# bids_path = mne_bids.BIDSPath(subject='01', session='01', task='baseline1', root='data/BIDS_data')
# raw1 = mne_bids.read_raw_bids(bids_path, verbose=False)
# raw1 = _cut_noisy(raw1, 'baseline1', 'hun')
# raw1.info['bads'] = []

# raw1.load_data().set_eeg_reference(['M1', 'M2'])
# raw1.filter(1, 30)
# pos = _make_montage()
# raw1.set_montage(pos)

# ica1 = mne.preprocessing.ICA(30, max_iter='auto', random_state=42)
# ica1.fit(raw1)
# ecg_inds = ica1.find_bads_ecg(raw1, ch_name='ECG')
# # save
# ica1.save('data/ica/templates/ecg-ica.fif')
# np.save('data/ica/templates/ecg_template.npy', np.array(ecg_inds))

# eog template from subject 2 that has two eog component
bids_path = mne_bids.BIDSPath(subject='02', session='01', task='baseline1', root='data/BIDS_data')
raw2 = mne_bids.read_raw_bids(bids_path, verbose=False)
raw2 = _cut_noisy(raw2, 'baseline1', 'hun')

raw2.filter(1, 30)
pos = _make_montage()
raw2.set_montage(pos)

ica2 = mne.preprocessing.ICA(40, max_iter='auto', random_state=97)
ica2.fit(raw2)
eog_inds = ica2.find_bads_eog(raw2, ch_name=['EOG1', 'EOG2', 'Fp1', 'Fp2', 'Fpz'])
# save
ica2.save('data/ica/templates/eog-ica.fif')
np.save('data/ica/templates/eog_template.npy', np.array(eog_inds))

# eog template from subject 4 that has eog component
bids_path = mne_bids.BIDSPath(subject='02', session='01', task='baseline1', root='data/BIDS_data')
raw3 = mne_bids.read_raw_bids(bids_path, verbose=False)
raw3 = _cut_noisy(raw3, 'baseline1', 'hun')
raw3.info['bads'] = []

raw3.load_data().set_eeg_reference(['M1', 'M2'])
raw3.filter(1, 30)
pos = _make_montage()
raw3.set_montage(pos)

ica3 = mne.preprocessing.ICA(40, max_iter='auto', random_state=42)
ica3.fit(raw3)
eog_inds = [7]
# save
ica3.save('data/ica/templates/eog_3rd-ica.fif')

# visualization
# print(ecg_inds)
print(eog_inds)
# ica1.plot_components()
ica2.plot_components()
