"""Reproduce the selected walk-forward gate from prepared point-in-time inputs."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.models.gate import flag_top
from src.models.splits import label_end_dates, make_folds

KEY = ['date', 'ticker']
LAST_DATE = pd.Timestamp('2025-12-31')


def sample_pairs(train, seed, count):
    rng = np.random.default_rng(seed)
    y = train['y_crash'].eq(1).to_numpy()
    positives, negatives, weights = [], [], []
    for group in train.groupby('date', sort=True).indices.values():
        positive, negative = group[y[group]], group[~y[group]]
        if not len(positive) or not len(negative):
            continue
        take = min(count, len(negative))
        sampled = np.concatenate([rng.choice(negative, size=take, replace=False) for _ in positive])
        positives.append(np.repeat(positive, take))
        negatives.append(sampled)
        weights.append(np.full(len(sampled), 1 / len(sampled)))
    pair_weights = np.concatenate(weights)
    return np.concatenate(positives), np.concatenate(negatives), pair_weights / pair_weights.mean()


def score_fold(train, validation, test, config):
    columns = config['feature_columns']
    train = train.sort_values(KEY)
    fill = train[columns].median()
    if fill.isna().any():
        raise ValueError('The training fold contains a wholly missing feature')
    positive, negative, weights = sample_pairs(train, config['training_seed'], config['max_negatives_per_positive'])
    train_values = train[columns].fillna(fill).to_numpy(dtype='float64')
    difference = train_values[positive] - train_values[negative]
    signed = np.vstack((difference, -difference))
    pair_labels = np.repeat([1, 0], len(positive))
    ranker = LogisticRegression(fit_intercept=False, max_iter=config['max_iter'])
    ranker.fit(signed, pair_labels, sample_weight=np.tile(weights, 2))

    validation_values = validation[columns].fillna(fill).to_numpy(dtype='float64')
    logits = ranker.decision_function(validation_values)
    calibration = LogisticRegression(max_iter=config['max_iter'])
    calibration.fit(logits.reshape(-1, 1), validation['y_crash'].astype(int))
    if calibration.coef_[0, 0] <= 0:
        raise ValueError('Calibration would reverse the ordering of crash risk')
    test_values = test[columns].fillna(fill).to_numpy(dtype='float64')
    test_logits = ranker.decision_function(test_values)
    return calibration.predict_proba(test_logits.reshape(-1, 1))[:, 1]


def run_walk_forward(features, labels, ends, days, pressure, config, prereg):
    horizon = prereg['labels']['main_horizon_days']
    if not labels['h'].eq(horizon).all() or not labels['y_crash'].isin([0, 1]).all():
        raise ValueError('Expected binary labels for the main horizon')
    if any(pd.to_datetime(frame['date']).max() > LAST_DATE for frame in (features, labels, ends, pressure)):
        raise ValueError('This historical replay accepts signal dates through 2025 only')
    if days.max() > LAST_DATE or any(pd.to_datetime(ends[col]).max() > LAST_DATE for col in ('nominal_end', 'actual_end')):
        raise ValueError('The calendar and label ends must stop at the historical cutoff')
    if features.duplicated(KEY).any():
        raise ValueError('Feature date and ticker must be unique')

    latest = label_end_dates(ends, horizon)
    folds = make_folds(features['date'].unique(), days, h=horizon,
                       test_years=prereg['splits']['walk_forward_test_years'],
                       first_train_year=prereg['data']['first_train_signal_year'], actual_end_dates=latest)
    labelled = features.merge(labels[KEY + ['y_crash']], on=KEY, validate='one_to_one')
    results = []
    with threadpool_limits(limits=2, user_api='blas'), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for fold in folds:
            train = labelled.loc[labelled['date'].isin(fold.train)]
            years = fold.val.year >= fold.test_year - config['calibration_validation_years']
            validation = labelled.loc[labelled['date'].isin(fold.val[years])]
            test = features.loc[features['date'].isin(fold.test)]
            probabilities = score_fold(train, validation, test, config)
            results.append(test[KEY].assign(crash_prob=probabilities))
            print(f'{fold.test_year}: {len(test):,} predictions', flush=True)

    scores = pd.concat(results, ignore_index=True).merge(pressure, on=KEY, how='left', validate='one_to_one')
    if len(pressure) != len(scores) or not scores['pressure_prob'].between(0, 1).all():
        raise ValueError('The pressure file must have one out-of-sample probability for each test key')
    weight = config['pressure_probability_weight']
    scores['crash_prob'] = (1 - weight) * scores['crash_prob'] + weight * scores.pop('pressure_prob')
    scores['crash_flag'] = flag_top(scores, 'crash_prob', fraction=config['weekly_flag_fraction']).astype('int8')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'labels', 'ends', 'calendar', 'pressure', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load((HERE / 'config.yaml').read_text(encoding='utf-8'))
    prereg = yaml.safe_load((ROOT / 'configs/prereg.yaml').read_text(encoding='utf-8'))
    metrics = json.loads((HERE / 'metrics.json').read_text(encoding='utf-8'))
    for role, digest in metrics['replay_inputs_sha256'].items():
        with getattr(args, role).open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError(f'Archived input SHA-256 does not match: {role}')
    features = pd.read_parquet(args.features, columns=KEY + config['feature_columns'])
    labels = pd.read_parquet(args.labels, columns=KEY + ['h', 'y_crash'])
    ends = pd.read_parquet(args.ends, columns=KEY + ['h', 'nominal_end', 'actual_end'])
    calendar = pd.read_parquet(args.calendar, columns=['date'])
    pressure = pd.read_parquet(args.pressure, columns=KEY + ['pressure_prob'])
    scores = run_walk_forward(features, labels, ends, pd.DatetimeIndex(calendar['date']), pressure, config, prereg)
    scores.to_parquet(args.output, index=False)


if __name__ == '__main__':
    main()
