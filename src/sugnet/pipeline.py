
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


def _query_csv(path: Path,
               subject_condition: np.ndarray):
    """Query the given csv file for the given subject conditions.
    """

    data = pd.read_csv(path, index_col=0)
    data['bids_id'] = data['bids_id'].apply(int).apply(lambda x: str(x).rjust(2, '0'))
    data = data.query('condition.str.contains("experience")')
    all_idx = data[['bids_id', 'procedure']].agg(''.join, axis=1)
    valid_idx = all_idx.to_frame('idx').query('idx in @subject_condition').index
    data.drop(columns=['hypnosis_depth', 'procedure', 'description', 'session', 'condition',
                       'bids_id'], errors='ignore', inplace=True)
    return data.loc[valid_idx]


def _extract_features(subjects: np.ndarray,
                      kind: str,
                      frequency_band: str,
                      power_types='periodic',
                      data_dir=Path('data/classification_datasets'),
                      **kwargs) -> np.ndarray:
    """Extract features from the given subjects.

    Args:
    power_types: in ['periodic', 'nonperiodic' 'iaf', 'all'] effective only when kind is 'power source'.
    """

    subject_condition = pd.DataFrame(subjects).agg(''.join, axis=1).to_list()

    if kind.lower() == 'chance':
        n_features = kwargs.get('n_features', 4)
        X = np.random.rand((len(subjects), n_features))
        return X

    elif kind.lower() == 'power source':
        path = data_dir / 'power_source.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        # TODO: add statement when frequency_band is 'all' and power_types is 'periodic'
        if frequency_band != 'all' and power_types == 'periodic':
            col_names = [col for col in data.columns if frequency_band in col]
        elif power_types == 'nonperiodic':
            col_names = [col for col in data.columns if 'exponent' in col or 'offset' in col]
        elif power_types == 'iaf':
            col_names = [col for col in data.columns if 'IAF' in col]
        return data.fillna(0)[col_names]

    elif kind.lower() == 'power sensor':
        path = data_dir / 'power_sensor.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        # check power_types is valid otherwise raise error
        assert power_types in ['decibel', 'absolute']
        if frequency_band != 'all':
            if power_types == 'decibel':
                col_names = [col for col in data.columns if frequency_band in col and 'decibel' in col]
            elif power_types == 'absolute':
                col_names = [col for col in data.columns if frequency_band in col and 'decibel' not in col]
        return data[col_names]

    # TODO: this is a temporary solution
    elif kind.lower() == 'power sensor relative to sham':
        path = data_dir / 'power_sensor_relative_to_sham.csv'
        data = _query_csv(path, subject_condition)
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    elif kind.lower() == 'correlation source':
        path = data_dir / 'correlation_source.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    elif kind.lower() == 'correlation sensor':
        path = data_dir / 'correlation_sensor.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    elif kind.lower() == 'plv source':
        path = data_dir / 'plv_source.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    elif kind.lower() == 'plv sensor':
        path = data_dir / 'plv_sensor.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    elif kind.lower() == 'wpli sensor':
        path = data_dir / 'wpli_sensor.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    elif kind.lower() == 'wpli source':
        path = data_dir / 'wpli_source.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    raise ValueError(f'Unknown feature kind: {kind}')


class FeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self,
                 kind: str = 'power source',
                 frequency_band: str = 'all',
                 power_types='periodic',
                 data_dir=Path('data/classification_datasets'),
                 **kwargs):
        self.kind = kind
        self.frequency_band = frequency_band
        self.power_types = power_types
        self.data_dir = data_dir
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X_ = _extract_features(X,
                                    kind=self.kind,
                                    frequency_band=self.frequency_band,
                                    power_types=self.power_types,
                                    data_dir=self.data_dir,
                                    **self.kwargs)

        return self.X_

    def get_feature_names_out(self,
                              feature_names_in: np.ndarray) -> np.ndarray:
        return self.X_.columns.values


if __name__ == '__main__':
    # test
    _extract_features(np.array([['01', 'whitenoise'],
                               ['01', 'confusion'],
                               ['02', 'confusion'],
                               ['02', 'embedded']]),
                      kind='correlation sensor')
