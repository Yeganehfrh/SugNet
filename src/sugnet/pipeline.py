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
    data.drop(columns=['hypnosis_depth', 'procedure', 'description', 'session', 'condition'],
              errors='ignore', inplace=True)
    data = data.loc[valid_idx]
    return data.set_index('bids_id'), valid_idx


def _extract_features(subjects: np.ndarray,
                      kind: str,
                      frequency_band: str,
                      power_types='periodic',
                      data_dir=Path('data/classification_datasets'),
                      calculate_diff=False,
                      X_diff=None,
                      **kwargs) -> np.ndarray:
    """Extract features from the given subjects.

    Args:
    power_types: in ['periodic', 'nonperiodic' 'iaf', 'all'] effective only when kind is
    'power source'.
    """

    assert kind.lower() in ['power source', 'power sensor', 'power sensor relative to sham',
                            'correlation source', 'correlation sensor', 'wpli sensor', 'wpli source',
                            'power sensor real relative to sham']

    subject_condition = pd.DataFrame(subjects).agg(''.join, axis=1).to_list()
    fname = '_'.join(kind.lower().split(' ')) + '.csv'
    path = data_dir / fname
    data, valid_idx = _query_csv(path, subject_condition)
    col_names = data.columns

    if kind.lower() == 'chance':
        n_features = kwargs.get('n_features', 4)
        X = np.random.rand((len(subjects), n_features))
        return X

    elif kind.lower() == 'power sensor':
        assert power_types in ['decibel', 'absolute']
        if frequency_band != 'all':
            if power_types == 'decibel':
                col_names = [col for col in data.columns if frequency_band in col and 'decibel' in col]
            elif power_types == 'absolute':
                col_names = [col for col in data.columns if frequency_band in col and 'decibel' not in col]
    else:
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]

    if calculate_diff:
        X_diff = pd.DataFrame(X_diff).agg(''.join, axis=1).to_list()
        df, _ = _query_csv(path, X_diff)
        df_ = data[col_names] - df[col_names]
        return df_.set_index(valid_idx, drop=True)

    return data[col_names].set_index(valid_idx, drop=True)


class FeatureExtractor(TransformerMixin, BaseEstimator):
    def __init__(self,
                 kind: str = 'power source',
                 frequency_band: str = 'all',
                 power_types='periodic',
                 data_dir=Path('data/classification_datasets'),
                 calculate_diff: bool = False,
                 X_diff: np.ndarray = None,
                 **kwargs):
        self.kind = kind
        self.frequency_band = frequency_band
        self.power_types = power_types
        self.data_dir = data_dir
        self.calculate_diff = calculate_diff
        self.X_diff = X_diff
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X_ = _extract_features(X,
                                    kind=self.kind,
                                    frequency_band=self.frequency_band,
                                    power_types=self.power_types,
                                    data_dir=self.data_dir,
                                    calculate_diff=self.calculate_diff,
                                    X_diff=self.X_diff,
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
