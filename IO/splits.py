"""Shared evaluation splits (every model: Qtome's train_finetune.py, baselines' train_baseline.py, analysis tools).

A cell's split comes from configs/finetune_protocols.json: protocol name -> split block, plus per-dataset settings
(_dataset_split: Compass's sessions and few-shot fractions). make_runs turns a split block into runs (train trial
indices + evaluation groups); SPLITS lists the split types. Model-specific parts of a protocol (Qtome's head and
optimiser) live with the model (configs/Qtome/protocol_heads.json).
"""
import json
import os
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold

from IO.preprocessing import cache_suffix

PROTOCOLS = 'configs/finetune_protocols.json'


def protocol_split(name, datasets, path=PROTOCOLS):
    """Protocol name -> its split block for these finetune datasets: the protocol's split, then each dataset's
    _dataset_split settings on top (train_fraction only for fewshot splits)."""
    raw = json.load(open(path))
    table = {k: v for k, v in raw.items() if not k.startswith('_')}
    if name not in table:
        raise ValueError(f"unknown finetune protocol {name!r}, known: {sorted(table)} ({path})")
    split = json.loads(json.dumps(table[name]))
    for ds in datasets:
        for k, v in raw.get('_dataset_split', {}).get(ds, {}).items():
            if k != 'train_fraction' or split.get('type') == 'fewshot':
                split[k] = v
    return split


def all_subjects(data_root):
    with open(os.path.join(data_root, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    all_available_subjects = list(meta.get('data_structure', {}).keys())
    try:
        return sorted([int(s) for s in all_available_subjects])
    except ValueError:
        return sorted(all_available_subjects)



def resolve_subjects(entry, pool):
    """Subject entry -> list of pool subjects: 'all' | ['all'] | explicit list | {'random': n, 'seed': s}."""
    if entry in ('all', ['all']):
        return list(pool)
    if isinstance(entry, dict):
        return sorted(random.Random(entry.get('seed', 42)).sample(list(pool), entry['random']))
    by_str = {str(s): s for s in pool}
    missing = [s for s in entry if str(s) not in by_str]
    if missing:
        raise ValueError(f"subjects {missing} are not in the pool {sorted(by_str, key=str)}")
    return [by_str[str(s)] for s in entry]


def load_sessions(config, ds_args, pool, subject_data):
    """Per-trial session index (0 = the subject's first recorded session) for the pool, read
    from the compiled cache's 'session' array (MoabbLoader datasets); None if any subject's
    cache predates it. Both sources keep each subject's trials in cache order, so the
    subject's array maps onto its rows of subject_data directly."""
    pp = config['preprocess_params']
    suffix = cache_suffix(pp['sample_freq'], pp['bandpass_filter'],
                          pp.get('pre_event_seconds', 0.0), pp.get('post_event_seconds', 0.0))
    session = np.zeros(len(subject_data), dtype=np.int64)
    for s in pool:
        z = np.load(os.path.join(ds_args['dataset_path'], 'cache', f'{s}_{suffix}.npz'))
        if 'session' not in z:
            return None
        rows = np.flatnonzero(subject_data == int(s))
        assert len(rows) == len(z['session']), f"subject {s}: {len(rows)} trials vs {len(z['session'])} cached sessions"
        session[rows] = z['session']
    return session


# ---------- split types: training_params.finetune.split = {"type": <name>, ...} ----------
# Every type takes `sessions` (keep only those session indices, 0 = each subject's first recorded
# session -- EEG-FM-Compass uses session 0 for MI/P300) and `seed`; the within-subject types also
# take `purge` (drop eval trials within that many recording positions of a train trial, for
# datasets whose neighbouring trials overlap in time). A new split pattern is a new function
# plus a SPLITS entry.
#   loso           every subject held out once (subject_kfold with n_folds = number of subjects)
#   subject_kfold  n_folds folds of whole subjects, subjects shuffled by seed; train_subjects
#                  optionally limits the training pool
#   eval_subjects  one run: eval_subjects (list, dict of named groups, {"random": n, "seed": s})
#                  vs the rest (or train_subjects)
#   kfold          per subject, n_folds shuffled stratified folds (optimistic: neighbouring
#                  trials land on both sides)
#   blocked_kfold  per subject, n_folds contiguous chronological blocks
#   fewshot        per subject, per class the first ceil(train_fraction * n) trials in recording
#                  order train, the rest evaluate (Compass within-subject calibration)

def _inter_runs(evals, pool, train_pool, trials, disjoint_check):
    runs = []
    for name, groups in evals:
        ev_subs = {s for v in groups.values() for s in v}
        train_subs = [s for s in (train_pool if train_pool is not None else pool) if s not in ev_subs]
        if not train_subs or any(not v for v in groups.values()):
            raise ValueError(f"run {name}: empty training set or evaluation group")
        if disjoint_check and train_pool is not None and set(train_pool) & ev_subs:
            raise ValueError(f"run {name}: train_subjects and evaluation subjects overlap: {sorted(set(train_pool) & ev_subs)}")
        runs.append(dict(name=name, train=trials(train_subs), train_subjects=[str(s) for s in train_subs],
                         eval={g: {str(s): trials([s]) for s in v} for g, v in groups.items()}))
    return runs


def _subject_kfold(split, pool, trials, labels, seed):
    k = int(split['n_folds'])
    if not 2 <= k <= len(pool):
        raise ValueError(f"n_folds must be in [2, {len(pool)}], got {k}")
    subs = list(pool)
    random.Random(seed).shuffle(subs)
    train_pool = resolve_subjects(split['train_subjects'], pool) if 'train_subjects' in split else None
    return _inter_runs([(f'fold{i}', {'heldout': subs[i::k]}) for i in range(k)], pool, train_pool, trials, False)


def _loso(split, pool, trials, labels, seed):
    return _subject_kfold({**split, 'n_folds': len(pool)}, pool, trials, labels, seed)


def _eval_subjects(split, pool, trials, labels, seed):
    ev = split['eval_subjects']
    ev = {'heldout': ev} if not isinstance(ev, dict) or 'random' in ev else ev
    train_pool = resolve_subjects(split['train_subjects'], pool) if 'train_subjects' in split else None
    return _inter_runs([('main', {g: resolve_subjects(v, pool) for g, v in ev.items()})], pool, train_pool, trials, True)


def _per_subject(folds_of):
    """Within-subject type: folds_of(split, idx, labels, seed) -> [(name, train, eval)] per subject."""
    def make(split, pool, trials, labels, seed):
        purge, runs = int(split.get('purge', 0)), []
        for s in pool:
            idx = trials([s])                      # recording order
            for name, tr, va in folds_of(split, idx, labels, seed):
                if purge:
                    pos = np.searchsorted(idx, tr)
                    near = np.zeros(len(idx), dtype=bool)
                    for d in range(-purge, purge + 1):
                        near[np.clip(pos + d, 0, len(idx) - 1)] = True
                    va = va[~near[np.searchsorted(idx, va)]]
                runs.append(dict(name=f'{s}_{name}', train=tr, train_subjects=[str(s)],
                                 eval={'heldout': {str(s): va}}))
        return runs
    return make


def _kfold_folds(split, idx, labels, seed):
    skf = StratifiedKFold(n_splits=int(split['n_folds']), shuffle=True, random_state=seed)
    return [(f'fold{i}', np.sort(idx[tr]), np.sort(idx[va])) for i, (tr, va) in enumerate(skf.split(idx, labels[idx]))]


def _blocked_folds(split, idx, labels, seed):
    return [(f'fold{i}', np.setdiff1d(idx, b), b) for i, b in enumerate(np.array_split(idx, int(split['n_folds'])))]


def _fewshot_folds(split, idx, labels, seed):
    f = float(split['train_fraction'])
    tr = np.concatenate([ic[:max(1, int(np.ceil(f * len(ic))))]
                         for ic in (idx[labels[idx] == c] for c in np.unique(labels[idx]))])
    return [('fewshot', np.sort(tr), np.setdiff1d(idx, tr))]


_COMMON = {'type', 'sessions', 'seed'}
SPLITS = {   # name -> (make_runs function, required keys, optional keys)
    'loso':          (_loso, set(), set()),
    'subject_kfold': (_subject_kfold, {'n_folds'}, {'train_subjects'}),
    'eval_subjects': (_eval_subjects, {'eval_subjects'}, {'train_subjects'}),
    'kfold':         (_per_subject(_kfold_folds), {'n_folds'}, {'purge'}),
    'blocked_kfold': (_per_subject(_blocked_folds), {'n_folds'}, {'purge'}),
    'fewshot':       (_per_subject(_fewshot_folds), {'train_fraction'}, {'purge'}),
}


def make_runs(split, pool, subject_data, labels, session=None):
    """split block -> runs [{name, train, train_subjects, eval}], see SPLITS above."""
    if split.get('type') not in SPLITS:
        raise ValueError(f"split.type must be one of {sorted(SPLITS)}, got {split!r}")
    make, required, optional = SPLITS[split['type']]
    if required - set(split) or set(split) - required - optional - _COMMON:
        raise ValueError(f"split type {split['type']!r} takes {sorted(required)} (required) and "
                         f"{sorted(optional | _COMMON - {'type'})} (optional), got {sorted(set(split) - {'type'})}")
    keep = np.ones(len(subject_data), dtype=bool)
    if split.get('sessions') is not None:
        if session is None:
            raise ValueError("split.sessions needs a cache with per-trial sessions -- recompile the dataset")
        keep = np.isin(session, split['sessions'])
    for s in pool:
        if not keep[subject_data == int(s)].any():
            raise ValueError(f"subject {s} has no trials in sessions {split.get('sessions')}")

    def trials(subjects):
        return np.flatnonzero(np.isin(subject_data, [int(x) for x in subjects]) & keep)

    return make(split, pool, trials, labels, split.get('seed', 42))
