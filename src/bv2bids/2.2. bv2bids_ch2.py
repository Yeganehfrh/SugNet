# In this group, we'll create BIDS data for all the data segments except the first baseline
# and we'll take the last trigger as the reference point for cropping the data

# %%
# imports
import mne
import os.path as op
import pandas as pd
from mne_bids import write_raw_bids, BIDSPath

# create BIDS dataset
study_name = 'Main-study'
data_dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA/Live Sessions'
bids_root = f'/Users/yeganeh/Codes/otka-preprocessing/data/{study_name}'
docs_dir = '/Users/yeganeh/Codes/otka-preprocessing/docs'


# map between eeg and bids ids
elig_par = pd.read_excel(f'{docs_dir}/eeg_behavioral_bids_idsmap.xlsx', header=1)
chunck2 = elig_par.iloc[25:40]
chunck2 = chunck2[['eeg_id', 'bids_id']]
subject_ids = chunck2['eeg_id']  # eeg ids
[subject_ids.remove(i) for i in [21042214, 215711, 2151114]]  # these data have a redundant triggers
chunck2.set_index('eeg_id', drop=True, inplace=True)
ids = chunck2.to_dict()
ids_map = ids['bids_id']


iterators = range(10)
runs = [0, 2, 3, 6, 7, 10, 11, 14, 15, 18]
run_map = dict(zip(iterators, runs))

raw_list = list()
bids_list = list()
bads = list()
new_annot = dict(onset=[], duration=[], description=[])

for subject_id in subject_ids:
    fname = op.join(data_dir, '{}.vhdr'.format(subject_id))
    raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc=['ECG'])
    raw.set_channel_types({'ECG': 'ecg'})
    raw.info['line_freq'] = 50  # specify power line frequency
    tmax_index = raw.annotations.description.tolist().index('Stimulus/S  9')

    onset_orig = raw.annotations.onset

    for iterator in iterators:
        # crop
        tmax = onset_orig[tmax_index - run_map[iterator]]
        tmin = onset_orig[tmax_index - run_map[iterator] - 1]
        if run_map[iterator] in [0, 18]:
            description = f'baseline{int(1/(iterator+1)+1)}'
        elif run_map[iterator] % 2:
            description = f'induction{int((20 - run_map[iterator] - 1) /4)}'
        else:
            description = f'experience{int((20 - run_map[iterator] - 2) /4)}'

        raw_segment = raw.copy().crop(tmin=tmin, tmax=tmax)

        # define a more meaningfull annotations
        onset = raw_segment.annotations.onset
        new_annot = mne.Annotations(onset=[0],
                                    duration=[onset[1]-onset[0]],
                                    description=[description])

        raw_segment.set_annotations(new_annot)

        # save
        dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA'
        fname_new = op.join(dir, f'fif_files/{subject_id}-{description}-raw.fif')
        raw_segment.save(fname_new)
        raw_segment = mne.io.read_raw_fif(fname_new)

        # creat lists
        raw_list.append(raw_segment)
        bids_path = BIDSPath(subject=f'{ids_map[subject_id]:02}',
                             session='01',
                             task=description,
                             # run=f'{iterator+1}',
                             root=bids_root)
        bids_list.append(bids_path)

# create bids from lists
for raw, bids_path in zip(raw_list, bids_list):
    write_raw_bids(raw, bids_path, overwrite=True)
