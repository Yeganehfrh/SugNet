"""
=================
spectral analysis
=================
"""

# imports
import mne
import numpy as np
import pandas as pd
import pickle
import os.path as op
from mne.time_frequency import psd_welch


def calculatePSD(path,
                 subjects,
                 tasks,
                 freqs,
                 n_overlap,
                 n_fft=1000,
                 n_job=1):

    """
    run spectral analysis using Welch’s method with a Hanning window of 1s with 50% overlap.

    Paremeters
    ----------
    path : str
        path to clean data

    freqs : dict
    """

    psd_unaggregated = {}
    psd_total = pd.DataFrame()

    for n_sub in subjects:
        psd_aggregated = {}
        for task in tasks:
            epo_name = f'sub-{n_sub}_ses-01_task-{task}_proc-clean_epo.fif'
            dir = op.join(path, epo_name)
            # open clean epochs
            epochs = mne.read_epochs(dir)

            # calculate psd for broadbands and all channels
            psds, _ = psd_welch(epochs,
                                fmin=1,
                                fmax=40,
                                picks='all',
                                n_fft=n_fft,
                                n_overlap=n_overlap,
                                n_jobs=n_job)
            psd_unaggregated[f'{n_sub}-{task}'] = psds
            # freq_dict[f'{n_sub}-{task}'] = freqs
            # transform
            psd_transformed = 10. * np.log10(psds)
            # aggregate over the epoch dimention
            psd_transformed = psd_transformed.mean(0)
            # calculate psds for different frequency bands across different brain areas
            ch_nam = epochs.ch_names
            ba_list = _patch_brain()
            for key in ba_list.keys():
                channels = ba_list[key]
                temp1 = [psd_transformed[ch_nam.index(i)] for i in channels]  # sift psd of relevant channels out
                # aggregate over different frequency bands
                for k, v in freqs.items():
                    temp2 = [temp1[i][v[0]:v[1]] for i in range(len(temp1))]  # TODO change this code: depending on the
                    # parameters of psd_welch it would malfunction! I should use something like this:
                    # temp2 = temp1[:, np.where((freqs[k][0] <= psd_freq) & (psd_freq <= freqs[k][1]) == True)[0]]
                    # where psd freq is the frequency vector from psd_welch
                    temp3 = np.array(temp2)
                    psd_aggregated[f'{key}-{k}'] = temp3.mean(0).mean(0)

            psd_df = pd.DataFrame(psd_aggregated, index=[f'{n_sub}-{task}'])
            psd_total = psd_total.append(psd_df)

    # save
    psd_total.to_csv('docs/psds_2nd_analysis.csv')
    with open('psd_unaggragated_2nd_analysis.pkl', 'wb') as handle:
        pickle.dump(psd_unaggregated, handle)


def _patch_brain():
    brain_areas = {
                'LF': ['Fp1', 'F3', 'F7', 'AF3', 'F1', 'F5', 'FT7'],
                'LC': ['C3', 'T7', 'FC1', 'FC3', 'FC5', 'C1', 'C5'],
                'LP': ['P3', 'P7', 'CP1', 'CP3', 'CP5', 'TP7', 'P1', 'P5'],
                'LO': ['O1', 'PO3'],
                'RF': ['Fp2', 'F4', 'F8', 'AF4', 'F2', 'F6', 'FT8'],
                'RC': ['C4', 'T8', 'FC2', 'FC4', 'FC6', 'C2', 'C6'],
                'RP': ['P4', 'P8', 'CP2', 'CP4', 'CP6', 'TP8', 'P2', 'P6'],
                'RO': ['O2', 'PO4'],
                'FZ': ['Fpz', 'Fz'],
                'CZ': ['Cz', 'CPz', 'FCz'],
                'PZ': ['Pz', 'POz'],
                'OZ': ['Oz', 'Iz'],
                'all': ['Fp1',
                        'Fp2',
                        'F3',
                        'F4',
                        'C3',
                        'C4',
                        'P3',
                        'P4',
                        'O1',
                        'O2',
                        'F7',
                        'F8',
                        'T7',
                        'T8',
                        'P7',
                        'P8',
                        'Fpz',
                        'Fz',
                        'Cz',
                        'CPz',
                        'Pz',
                        'POz',
                        'Oz',
                        'Iz',
                        'AF3',
                        'AF4',
                        'F1',
                        'F2',
                        'F5',
                        'F6',
                        'FC1',
                        'FC2',
                        'FC3',
                        'FC4',
                        'FC5',
                        'FC6',
                        'FT7',
                        'FT8',
                        'C1',
                        'C2',
                        'C5',
                        'C6',
                        'CP1',
                        'CP2',
                        'CP3',
                        'CP4',
                        'CP5',
                        'CP6',
                        'TP7',
                        'TP8',
                        'P1',
                        'P2',
                        'P5',
                        'P6',
                        'PO3',
                        'PO4']}
    return brain_areas


def extract_psds_freatures(path='docs/1.psd_unaggragated_2nd_analysis.pkl',
                           n_epochs=60):
    with open(path, 'rb') as handle:
        psds_unagg = pickle.load(handle)
    features = {}
    for k, v in psds_unagg.items():
        features[k+'_start_allbroadband'] = v[:n_epochs].mean()
        features[k+'_end_allbroadband'] = v[-n_epochs:].mean()
        # features[k] = v.mean(0)

    features_csv = pd.DataFrame.from_dict(features)
    features_csv.to_csv('start_end_features.csv')

    return features
