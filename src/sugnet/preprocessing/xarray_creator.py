"""
=====================
create xarray dataset
=====================
Warning:
This code only works when (RealHyp == True and subjects == 'all')
TODO: modify the script to include other conditions
In order to do that, the code should save the dataset for each subject and then combine them all.
Having dataArray from each subject is also efficient when merging them.

"""

# imports
import xarray as xr
import pandas as pd  # noqa
import numpy as np
import mne_bids


def xarray_creator(bids_root, subjects, tasks, dir, ids_map, realHyp=True, sfreq=1000):
    """
    bids_root : str
        path to the folder that contains bids data

    subjects : list of str
        subjects' bids id. It can also take up 'all'.

    dir : str
        path in which save the dataset

    sfreq : int
        sampling frequency rate

    ids_map : pandas dataframe
        pandas dataframe of maping between bids ids and other information

    """
    if subjects == 'all':
        subjects = [f"{num:02d}" for num in range(1, 51)]  # TODO retrieve 51 from number of files in bids_root
        # idea: get either subjects ranges (first, and last ids in an orde) or their ids,
        # and with that we can create this list

    _to_ds(bids_root, subjects, tasks, ids_map, dir, realHyp)

    # open and merge two dataset
    ds0 = xr.open_dataset('data/dataset/ds0.nc')
    ds1 = xr.open_dataset('data/dataset/ds1.nc')
    dataset = ds0.merge(ds1)
    ds2 = xr.open_dataset('data/dataset/ds2.nc')
    dataset = dataset.merge(ds2)

    # add attributes
    bads = [__bad_channel_matcher(ids_map, sub) for sub in subjects]
    sfreq = sfreq
    attrs = {
        "bad_channels": bads,
        "sfreq": sfreq
    }
    # montage = make_montage()
    dataset = dataset.assign_attrs(attrs)

    # save new dataset
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in dataset.data_vars}
    dataset.to_netcdf(f'{dir}/dataset.nc', engine="h5netcdf", encoding=encoding)


def _to_ds(bids_root, subjects, tasks, ids_map, dir, realHyp=True):
    """
    open files and create a dictionary from them

    Parameters
    ----------
    bids_root : str
        path to bids data

    subjects : list of str
        subjects' bids id. It can also take up 'all'.

    tasks : list of str
        tasks names. It can also take up 'all'.

    realHyp : boolean
        If True, it read a csv file and will only include the expereince from those trials
        where hypnosis was introduced as hypnosis.

    """

    # Running this script will become unnecessarily slow for sub>25
    # so if len(subject) > 25, we will create xarray in two batches and then merge:
    ind = int(len(subjects)/3)
    subjects_b1 = subjects[:ind]
    subjects_b2 = subjects[ind:2*ind]
    subjects_b3 = subjects[2*ind:]
    subjects_list = [subjects_b1, subjects_b2, subjects_b3]

    for i, batch in enumerate(subjects_list):
        subjects = batch
        tasks = ['baseline1', 'experience1']  # this is when RealHyp == True
        allSubjectsTasks = {}

        for task in tasks:

            # timpoints and initialize
            timepoints = 300001  # change this if resampling data
            allSubjects = np.empty((1, 61, timepoints))

            # Open data of a specific task for all subjects one by one, reshape and append them
            for sub in subjects:
                # find hypnosis-as-hypnosis trial index (only for experience)
                if task[:-1] == 'experience':
                    ind = __task_ind_finder(ids_map, sub)
                    task = task[:-1] + str(ind)

                bids_path = mne_bids.BIDSPath(subject=sub, session='01', task=task, root=bids_root)
                raw = mne_bids.read_raw_bids(bids_path)
                # Cut noisy parts only for experience and baseline task (it's always True when realHyp=True)
                language = ids_map.loc[int(sub), 'language'][:3].lower()
                raw = _cut_noisy(raw, task, language)

                # Get eeg data as np.array and add subject dimension
                oneSubject = np.expand_dims(raw.get_data(), 0)

                allSubjects = np.append(allSubjects, oneSubject, axis=0)

            # remove empty array
            allSubjects = np.delete(allSubjects, 0, axis=0)

            # unify task indexes across participants
            task = task[:-1]

            # mega data
            allSubjectsTasks[task] = allSubjects

        channels = raw.ch_names
        _, time = raw.get_data(return_times=True)

        # create xarray from dictionary
        # data variables
        baseline = allSubjectsTasks['baseline']  # TODO find a way to create these without knowing the variable names
        experience = allSubjectsTasks['experience']

        # coordinates
        subjects = subjects
        channels = channels
        time = time

        dataset = xr.Dataset(
            {
                "baseline": (["subject", "channel", "time"], baseline),
                "experience": (["subject", "channel", "time"], experience)
            },
            coords={
                "subject": (["subject"], subjects),
                "channel": (["channel"], channels),
                "time": (["time"], time)
            }
        )

        # save dataset in the given directory
        # TODO this takes for a while to run this code (maybe we don't need to compress dataset at this stage)
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in dataset.data_vars}
        dataset.to_netcdf(f'{dir}/ds{i}.nc', engine="h5netcdf", encoding=encoding)
        del dataset


# cut noisy parts
def _cut_noisy(raw, task, language):
    """
    First step of preprocessing in my pipeline!
    it's important to get rid of noisy parts in baseline and experinece segments: informed from the audio files.

    Parameters
    ----------
    raw : mne.io.Raw
        eeg raw data
    task : str
        can be experience1, etc or baseline1 or baseline2
    language : str
        language of the experiment ['eng' or 'hun']
    """
    # validate task and language name
    import re
    from collections import namedtuple

    if not re.fullmatch('(baseline[12])|(experience[1-4])', task):
        raise Exception('Invalid task!')

    tasklang = task + language

    # helper named tuple to make the code more readable
    Interval = namedtuple('Interval', ['tmin', 'tmax'])

    cut_intervals = {
        '(experience[1-4](eng|hun))|baseline1hun': Interval(tmin=20, tmax=320),
        'baseline1eng': Interval(tmin=25, tmax=325),
        'baseline2(eng|hun)': Interval(tmin=30, tmax=330),
        }
    interval = [value for key, value in cut_intervals.items() if re.fullmatch(key, tasklang)][0]

    raw.crop(tmin=interval.tmin, tmax=interval.tmax)

    return raw


# real hypnosis index finder
def __task_ind_finder(ids_map, sub):
    exp_ind = ids_map.loc[int(sub), 'true_hyp_ind']
    return exp_ind


# bad channel finder
def __bad_channel_matcher(ids_map, sub):
    """
    find the name of channels marked as bad for each
    """
    bads = ids_map.loc[int(sub), 'bad_channels']
    return bads
