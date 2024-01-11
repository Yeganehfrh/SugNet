
#%%
# imports
import mne
from mne_bids import write_raw_bids, BIDSPath
from pathlib import Path

# create BIDS dataset
bids_root = 'data/BIDS_data_test'

# create ids maps between subject ids and their future BIDS ids
subject_ids = []
[subject_ids.append(subject_path.stem)
 for subject_path in sorted(Path('data/raw_test').glob('*.vhdr'))]
subject_ids = subject_ids[:-3]
ids_map = dict(zip(subject_ids, range(53, len(subject_ids)+53)))

# number of eeg segments defined as different runs
iterators = range(10)
runs = [0, 3, 4, 7, 8, 11, 12, 15, 16, 18]
run_map = dict(zip(iterators, runs))

raw_list = list()
bids_list = list()
bads = list()
new_annot = dict(onset=[],
                 duration=[],
                 description=[])

for subject_path in Path('data/raw_test').glob('*.vhdr'):
    subject_id = subject_path.stem

    raw = mne.io.read_raw_brainvision(subject_path, eog=('EOG1', 'EOG2'), misc=['ECG'])
    raw.set_channel_types({'ECG': 'ecg'})
    raw.info['line_freq'] = 50  # specify power line frequency
    tmin_index = raw.annotations.description.tolist().index('Stimulus/S  2')

    for iterator in iterators:

        # crop
        tmin = raw.annotations.onset[tmin_index + run_map[iterator]]
        tmax = raw.annotations.onset[tmin_index + run_map[iterator] + 1]
        if run_map[iterator] in [0, 18]:
            description = f'baseline{int(iterator/9+1)}'
        elif run_map[iterator] % 2:
            description = f'induction{int((run_map[iterator] - 3)/4 + 1)}'
        else:
            description = f'experience{int(run_map[iterator]/4)}'

        raw_segment = raw.copy().crop(tmin=tmin, tmax=tmax)

        # define a more meaningful annotations
        onset = raw_segment.annotations.onset
        new_annot = mne.Annotations(onset=[0],
                                    duration=[onset[1]-onset[0]],
                                    description=[description])

        raw_segment.set_annotations(new_annot)

        # save
        dir = Path('data/raw_test/temp')
        fname_new = dir / f'{subject_id}-{description}-raw.fif'
        raw_segment.save(fname_new)
        raw_segment = mne.io.read_raw_fif(fname_new)

        #creat lists
        raw_list.append(raw_segment)
        bids_path = BIDSPath(subject=f'{ids_map[subject_id]:02}',
                             session='01',
                             task=description,
                             root=bids_root)
        bids_list.append(bids_path)


# create bids from lists
for raw, bids_path in zip(raw_list, bids_list):
    write_raw_bids(raw, bids_path)
