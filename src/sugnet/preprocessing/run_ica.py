"""
run ica and detect biophysiological components automatically using template
"""

import numpy as np
import mne
from mne.preprocessing import ICA, read_ica, corrmap
from mne import Report
from pathlib import Path


def run_ica(raw,
            sub,
            task,
            filter_beforeICA=False,
            n_components=None,
            random_state=97,
            show_plot=False,
            threshold=0.85,
            report=True):

    ica_path = f'data/ica/fitted_icas/sub-{sub}_ses-01_task-{task}_ica.fif'
    # open templates
    eog_ica, eog_3rd_ica, eog_inds = _open_templates()

    # ica
    ica = ICA(n_components, random_state=random_state)
    if Path(ica_path).exists():
        ica = read_ica(ica_path)

    else:
        raw_filt = raw.copy().filter(1, None) if filter_beforeICA else raw
        ica.fit(raw_filt)

    # create list of icas
    icas_eog = [eog_ica]
    icas_eog.append(ica)

    # detect eog components
    eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw)  # noqa
    _, eog_scores = ica.find_bads_eog(raw, ch_name=['EOG1', 'EOG2', 'Fpz'])
    # we know that there is two components in the eog template so:
    [corrmap(icas_eog, template=(0, eog_inds[i]), threshold=threshold, label='blink', plot=show_plot)
     for i in range(2)]
    # detecting oculomotor activity using third compenents
    icas_eog.append(eog_3rd_ica)
    eog_inds.append(7)  # use components 7 from 3rd_ica as a template:
    corrmap(icas_eog, template=(2, eog_inds[2]), threshold=threshold, label='oculomotor', plot=show_plot)

    if 'blink' in ica.labels_.keys():
        eog_comps = ica.labels_['blink']
    else:
        eog_comps = []

    ica.exclude = eog_comps
    print(f'EOG COMPONENTS: {ica.labels_}')

    if report:

        # save report and save ica
        report = Report(title=f'{sub}-ICA Report')
        report.add_ica(
            ica=ica,
            picks=ica.exclude,
            title='ICA cleaning',
            inst=raw,
            # eog_evoked=eog_epochs.average(),
            # eog_scores=eog_scores,
            n_jobs=-1
        )
        # save reports and icas
        report.save(f'data/ica/reports/sub-{sub}_ses-01_task-{task}_report-ica.html',
                    open_browser=False, overwrite=True)

    if not Path(ica_path).exists():
        ica.save(ica_path)

    # apply ica
    print(f'Applying ICA for sub-{sub}...')
    ica.apply(raw, exclude=ica.exclude)

    return raw


def _open_templates(fname_eog_ica='data/ica/templates/eog-ica.fif',
                    fname_eog_3rd_ica='data/ica/templates/eog_3rd-ica.fif',
                    fname_eog_template='data/ica/templates/eog_template.npy',
                    ):
    # read ica and template
    eog_ica = read_ica(fname_eog_ica)
    eog_3rd_ica = read_ica(fname_eog_3rd_ica)
    eog_inds = np.load(fname_eog_template, allow_pickle=True)
    eog_inds = eog_inds[0]

    return eog_ica, eog_3rd_ica, eog_inds
