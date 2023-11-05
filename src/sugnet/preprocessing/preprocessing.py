"""
=============
Preprocessing
=============
"""

# imports
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr  # noqa
import mne
from mne_bids import BIDSPath, read_raw_bids
from autoreject import AutoReject
from autoreject import get_rejection_threshold
from calculateDispersion import amplitude_vector, dispersion_report
from run_ica import run_ica
from xarray_creator import _cut_noisy
from tqdm import tqdm


def preprocessing(path,
                  subjects,
                  tasks,
                  ids_map,
                  resampling_frq=None,
                  ref_chs=None,
                  filter_bounds=None,
                  ica_n_components=None,
                  autoreject='global',
                  ica_report=True,
                  n_job=-1):

    """
    Parameters
    ----------
    path : str
        path to xarray dataset

    subjects : list of str or int

    task : str

    fname_ica_template : string
        path to a instance of mne.preprocessing.ica. this ica and its eog and ecg components
        will be used as a template for biophysiological ICs detection

    resampling_frq : int
        resampling frequency

    ref_chs : list of str

    filter_bounds : tuple of float
        first item in tuple is the lower pass-band edge and second item is the upper
        pass-band edge

    ica : boolean

    autoreject : str
        can be 'global' or 'local'. if none, it won't calculate rejection threshold to
        remove bad epochs.

    ica_report : boolean
        whether output a report on ica components removal or not

    Returns
    -------
    xarray dataset of preprocessed data

    """

# open dataset and read it into a mne raw object.
    # initialize empty containers for storing amplitude vectors, rejection threshold, etc.
    ampVectors = []
    ampVectors_after_ica = []
    ampVectors_after_autoreject = []
    rejects = {}
    pos = _make_montage()

    for sub in tqdm(subjects):
        for task in tqdm(tasks):
            # open bids
            bids_path = BIDSPath(subject=sub,
                                 session='01',
                                 task=task,
                                 root=path)
            raw = read_raw_bids(bids_path, extra_params={'preload': True}, verbose=False)

            # cut noisy parts for experiences and baselines
            if task[:-1] in ['baseline', 'experience']:
                language = ids_map.loc[int(sub), 'language'][:3].lower()
                raw = _cut_noisy(raw, task, language)

            # set channels positions
            raw.set_montage(pos)

            # interpolate bad channels
            if raw.info['bads'] != []:
                raw.interpolate_bads()

            # resampling
            if resampling_frq is not None:
                raw.resample(resampling_frq)

            # filtering
            if filter_bounds is not None:
                raw.filter(
                    l_freq=filter_bounds[0],
                    h_freq=filter_bounds[1],
                    n_jobs=n_job
                    )

            # create amplitude vector before ICA
            ch_names = raw.ch_names
            ampVector = amplitude_vector(raw, ch_names=ch_names)
            ampVectors.append(ampVector)

            # apply ICA, remove eog and ecg ICs using templates, and save the report
            raw = run_ica(raw, sub, task, n_components=ica_n_components, report=ica_report)

            # calculate amplidute vector after ica
            ampVector = amplitude_vector(raw, ch_names=ch_names)
            ampVectors_after_ica.append(ampVector)

            # epoching (note: for creating epochs with mne.epochs, tmin and tmax should be specified!)
            epochs = mne.make_fixed_length_epochs(raw, duration=1, preload=True)
            del raw

            # autoreject
            if autoreject == 'global':
                # pick only eeg channels for getting rejection threshold:
                reject = get_rejection_threshold(epochs.copy().pick_types(eeg=True))
                rejects[sub] = reject
                epochs.drop_bad(reject=reject)
                # reject_log = np.array([
                #         i for i in range(len(epochs.drop_log)) if epochs.drop_log[i] != ()
                #         ])
                # np.save(f'data/rejectlog/{sub}-{task}.npy', reject_log)

            if autoreject == 'local':
                ar = AutoReject()  # TODO consider setting random state
                epochs = ar.fit_transform(epochs)  # TODO check if this is ok to save the
                # fit_transformed data to the save object

            # rereferencing
            if ref_chs is not None:
                epochs.add_reference_channels(ref_channels='FCz')  # adding reference channel to the data
                epochs.set_eeg_reference(ref_channels=ref_chs)

            # epochs to continous data and calculate amplitude vector after autoreject and rereferencing
            # filter for calculating the ampvector
            # TODO epochs_filt = epochs.copy().filter(1, 20) this line gives and error because of the filter length
            # I should consider using a method to filter epoched data without changing the length.
            continuous_data = np.hstack(epochs.get_data())
            ampVector = amplitude_vector(continuous_data, ch_names=ch_names, thisIsNotNumpy=False)
            # delete FCz to have the same position across ampVectors and then append
            ampVector = np.delete(ampVector, -1, 0)  # we know that FCz is the last item in the list
            ampVectors_after_autoreject.append(ampVector)

            # save clean epochs
            epochs.save(f'data/clean_data/sub-{sub}_ses-01_task-{task}_proc-clean_epo.fif', overwrite=True)

            # just for safety!! should be deleted later!!
            ampVectors_collection = [ampVectors,
                                     ampVectors_after_ica,
                                     ampVectors_after_autoreject]
            np.save('docs/dispersionVectors.npy', np.array(ampVectors_collection))
            del ampVectors_collection

    # calculate dispersion vector from each stage and save a report
    # first concatenate DVs from each stage and then input it into the custom function
    ampVectors_collection = [ampVectors,
                             ampVectors_after_ica,
                             ampVectors_after_autoreject]
    # calculate unnormalized and normalized DVs with mean for each DV in the collection,
    # make and save a report
    dispersion_report(ampVectors_collection,
                      pos,
                      fname='docs/dispersionVectors.npy',
                      fname_report='DV-report.html',
                      save=True,
                      normalization=False)
    dispersion_report(ampVectors_collection,
                      pos,
                      fname='docs/dispersionVectors_normalized.npy',
                      fname_report='DV-report_normalized.html',
                      save=True,
                      normalization=True)

    # cleans-up and savings
    df = pd.DataFrame.from_dict(rejects)
    df.to_csv('data/rejectlog/rejection_thresholds.csv')
    del ampVectors_collection, ampVectors, ampVectors_after_ica, ampVectors_after_autoreject,

    # clean dataset
    # read epochss from file
    # mne.read_epochs()
    # epoch to continuous (maybe I won't do this, depending on how I calculate psd, bacuase near the rejected epochs
    # I would see edge effects and if I retuen this to continuos data I cannot track them!)
    # use to_xarray to create xarray dataset


def _make_montage(path='data/raw/plb-hyp-live2131111.vhdr'):
    """
    Create a montage from barin vision raw data

    Parameters
    ----------
    path : str
        path to barinvision data header file

    """
    raw = mne.io.read_raw_brainvision(path, verbose=False, misc=['ECG'])
    raw.crop(1, 10)  # crop a small segment of the data to speed up data loading in next lines
    raw.load_data().set_channel_types({'ECG': 'ecg'})
    # raw.add_reference_channels('FCz')  # FCz was used as online reference
    ch_names = copy.deepcopy(raw.info['ch_names'])  # make a deep copy of the lists of the
    # channel names otherwise ecg channel will be removed in the raw object!!
    ch_names.remove('ECG')

    pos_array = raw._get_channel_positions()

    # # add FCz position based on channels CPz (Their positions are the same, only y-axis value is different)
    # pos_fcz = pos_array[ch_names.index('CPz')] * np.array([1, -1, 1])
    # pos_array = np.insert(pos_array, 60, pos_fcz, axis=0)
    # pos_array = np.delete(pos_array, -1, axis=0)

    pos_dict = dict(zip(ch_names, pos_array))
    pos = mne.channels.make_dig_montage(pos_dict)

    return pos


def plot_psd_topomap_raw(raw, bands=None):

    """
    plot psd topomap for all frequency bands using welch methods.
    raw : mne.io.Raw
    bands : dict
    """
    # Set montage
    pos = raw.get_montage()  # because of eog channels, we should re-get the digitized
    # point after we set the new montage

    # pos is an object of DigMontage type, instead we need a 2D array of XY channel postions,
    # the following will do that
    ch_pos = np.array(list(pos.get_positions()['ch_pos'].values()))
    ch_pos = ch_pos[:, 0:2]

    # creat a dictionary of the frequency bands with their lower and upper bounds
    if bands is None:  # if user did not define special frequency bands, use this as defualt
        keys = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        bounds = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
        bands = dict(zip(keys, bounds))

    # psd topomap across different frequency bands
    fig, axes = plt.subplots(1, len(bands), sharex=True, figsize=(18, 3.5))
    for i, key in enumerate(bands.keys()):
        psds = mne.time_frequency.psd_welch(
            raw, fmin=bands[key][0], fmax=bands[key][1], picks=['eeg'], average='mean', verbose=False)
        psds_mean = psds[0].mean(axis=1)
        mne.viz.plot_topomap(
            psds_mean, ch_pos, cmap='RdBu_r', axes=axes[i], outlines='head',
            show_names=True, names=raw.ch_names, show=False)
        axes[i].set_title(key, fontsize=18)

    plt.suptitle('PSD topomap (to detect bad channels)', y=1.12, fontsize=24)
    plt.show()
