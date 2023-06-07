"""calculate and visualize dispersion vector (using mean for normalization)"""

# Import necessary modules
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path # noqa
from scipy.stats import median_abs_deviation


# transalted from Vislab's MATLAB code to python
def dispersion_with_mean(ampVectors, normalization=True):
    """
    calculate dispersion vector [1]
    this function use mean for normalization

    Parameters
    ----------
    ampVectors : list of np.array
    visualize : boolean
    matrix : array-like
        amplitude matrix. rows are channels and columns are recordings

    Reference
    ---------
    [1] Bigdely-Shamlo, N., Touryan, J., Ojeda, A., Kothe, C., Mullen, T., & Robbins, K. (2020).
    Automated EEG mega-analysis I: Spectral and amplitude characteristics across studies. NeuroImage, 207, 116361.

    """

    amplitude_matrix = np.array(ampVectors).T
    if normalization:
        mean_recording = amplitude_matrix.mean(0)
        amplitude_matrix = np.divide(amplitude_matrix, mean_recording)
    stdFromMad = 1.4826 * median_abs_deviation(amplitude_matrix, 1)
    dispersion = stdFromMad / np.median(amplitude_matrix)

    return dispersion


def dispersion_report(ampVectors_collection,
                      pos,
                      fname_report,
                      fname,
                      caption=[
                            'DV before ICA and Autoreject',
                            'DV after ICA and before Autoreject',
                            'DV after ICA and Autoreject'],
                      save=True,
                      normalization=True):
    """
    calculate dispersion with mean for a collecation of amplitude matrix and create a
    report containing topomaps
    """
    dispersions = []
    ranges = range(len(ampVectors_collection))
    [dispersions.append(dispersion_with_mean(ampVectors_collection[i], normalization)) for i in ranges]

    figs = []
    for i in ranges:
        fig, _ = plt.subplots()
        _visualize_dispersion(dispersions[i], pos)
        figs.append(fig)

    report = mne.Report(title='QC: Dispersion Vectors')
    report.add_figure(
        fig=figs,
        title='topomap of Dispersion vectors',
        caption=caption,
    )
    report.save(fname_report, overwrite=True)

    if save:
        np.save(fname, np.array(ampVectors_collection))


def amplitude_vector(raw, ch_names, thisIsNotNumpy=True):
    # Calculate robust std from MAD
    if thisIsNotNumpy:
        raw_filt = raw.copy().filter(1, 20)
        raw = raw_filt.get_data()
    ampVector = 1.4826 * median_abs_deviation(raw, 1)
    # delete ecg and eog channels as they are not of interest but will impact dispersion vector calculation
    ampVector = np.delete(ampVector, [ch_names.index(i) for i in ['EOG1', 'EOG2', 'ECG']])

    return ampVector


def _visualize_dispersion(dispVector, pos):
    ch_pos = np.array(list(pos.get_positions()['ch_pos'].values()))
    ch_pos = ch_pos[:, 0:2]

    # remove position of EOG1 and EOG2 channels that are not included in the dispersion vector computation
    ch_pos = np.delete(ch_pos, [pos.ch_names.index(i) for i in ['EOG1', 'EOG2']], 0)

    im, _ = mne.viz.plot_topomap(dispVector, ch_pos, cmap='RdBu_r', show=False)
    plt.colorbar(im)

    return im
