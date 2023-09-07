# %%
# This code groups EEG data based on the condition of their markers.
# This is a necessary step before bidsifying.

# Setup
import numpy as np
import mne
import os.path as op
import pandas as pd

eeg_dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA/Live Sessions'
docs_dir = '/Users/yeganeh/Codes/otka-preprocessing/docs'

# open ids map between eeg, behavioral and bids files
fname = op.join(docs_dir, 'eeg_bh_bids_idmap.csv')
total_ids_map = pd.read_csv(fname)
total_ids_map.rename(columns={'Unnamed: 0': 'eeg_ids'}, inplace=True)
total_ids_map.set_index('eeg_ids', inplace=True, drop=True)

# ---------------------- group1: data that are already analyzed in the first round
chunck1 = pd.read_csv(op.join(docs_dir, 'elig_par'))
chunck1 = chunck1['0'].tolist()
chunck1 = [i for i in chunck1 if i.isdigit()]
# sessions that trigger has not worked normally (the last one does not have only Stimulus 2,
# the forth only contains the triggers for the first trial)
imp_trg = [2152714, 2160311, 215711, 2152014, 2160314, 21052511, 2131811, 2131814] 
[chunck1.remove(str(i)) for i in imp_trg]

# %%
# ---------------------- group2: finding data that are not analyzed in chucnk1 + data that their marker description is
# not as it should but the durations between markers are correct.
# # get the id of the participants who did not come to hypnotizability session
bh_data = pd.read_excel(op.join(docs_dir, 'data_with_hypnotizability.xlsx'), header=1)
bh_data.set_index('index', inplace=True)
ind_1 = bh_data[pd.isna(bh_data.hypnotizability_total)].index

# finding the corresponding eeg ids of the above behavioral ids
group2 = [total_ids_map.index[total_ids_map['index'] == i].tolist()[0] for i in ind_1]
group2.append('plb-hyp-live2131111')
group2.append('plb-hyp-live21031114')

# impaired triggers in group2
imp_trg2 = [2151211, 2151114, 21051811, 21052814, 21060111, 216811]

# %%
# inspect the distance between triggers
[imp_trg.append(i) for i in imp_trg2]

diffs = {}
for key in imp_trg:
    diffs[key] = []

markers_code = {'baseline_1': 334.0, 'intro_1': 108.0, 'intro_2': 103.0, 'induction_1': 337.0,
                'induction_2': 336.0, 'induction_3': 331.0, 'induction_4': 363.0, 'experience': 416.0,
                'baseline_2': 343.0}

# Open data, find the relation between triggers, and save them in adocument
for sub_id in imp_trg:
    fname = op.join(eeg_dir, f'{sub_id}.vhdr')
    raw = mne.io.read_raw_brainvision(fname)
    onsets = raw.annotations.onset
    diffs[sub_id] = np.diff(onsets).tolist()

diffs_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in diffs.items()]))
diffs_df = round(diffs_df)
diffs_df

# %%
# --------------------------- group3: the duration between the data is not ok
# the duration between triggers is mostly fine:
# from my investigation in the last part data will be divided in two groups:
# 1. The duration of the first baseline is ok
inact_baselines = []
for col in diffs_df.columns:
    temp = diffs_df.loc[:, col].dropna().astype('int')
    checker = [True for i in temp if i == 334]
    if any(checker):
        inact_baselines.append(col)

# Check if this group can be analyzed by referencing the last trigger (Stimulus 9):

# 2. duration of first baseline is not ok: should be estimated by the behavioral data
# [imp_trg.remove(i) for i in inact_baselines]
imp_1base = [2152714, 2160311, 2160314, 21052511, 2151211, 21051811]

# 3. 216811 that the duration of the first experience is not ok
# and the 2152014 that has trigger for only the first

# --------------------------- group4: software crash, bathroom break, dipoles was detected in the cap
['plb-hyp-live213914', 'plb-hyp-live2139142', 2152014]
