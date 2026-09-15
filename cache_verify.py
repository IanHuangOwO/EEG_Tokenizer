import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.signal

from IO.loader import resolve_dataset_loader
from IO.preprocessing import BandpassResample, cache_suffix


def load_cache(dataset_path, subject_id, suffix):
    path = os.path.join(dataset_path, 'cache', f'{subject_id}_{suffix}.npz')
    if not os.path.exists(path):
        return None, path
    return np.load(path), path


def check_subject(dataset_path, subject_id, data_metadata, data_structure,
                   sample_freq, bandpass_filter, suffix, deep=False,
                   pre_event_seconds=0.0, post_event_seconds=0.0):
    """Returns a list of problem strings — empty means the cached subject looks correct.
    Checks: cache exists, shapes/dtypes consistent, labels in-range, no NaN/Inf, no
    dead (zero-variance) channels, that the bandpass filter actually attenuated content
    above the cutoff (spectral rolloff), and valid_start/valid_end's own internal
    consistency (present, right length, in-bounds, and the padded/real split they claim
    actually matches what's in `data` — see below). --deep additionally re-runs the raw
    loader + BandpassResample fresh and diffs byte-for-byte against the cache, catching
    a cache that's stale relative to the current loader/compile code."""
    problems = []
    npz, path = load_cache(dataset_path, subject_id, suffix)
    if npz is None:
        return [f"missing cache file: {path}"]

    data, labels = npz['data'], npz['labels']
    if data.ndim != 3:
        return [f"data has {data.ndim} dims, expected 3 (N, C, T): shape={data.shape}"]
    N, C, T = data.shape

    c_expected = data_metadata.get('channels', {}).get('count', 0)
    if C != c_expected:
        problems.append(f"channel count {C} != metadata.channels.count {c_expected}")
    if N != len(labels):
        problems.append(f"trial count mismatch: data N={N} vs labels N={len(labels)}")

    n_classes = data_metadata.get('targets', {}).get('count')
    if n_classes is not None and N > 0 and (labels.min() < 0 or labels.max() >= n_classes):
        problems.append(f"labels out of range [0,{n_classes}): got min={labels.min()} max={labels.max()}")

    if not np.isfinite(data).all():
        problems.append("data contains NaN/Inf")

    if N > 0:
        per_channel_std = data.std(axis=(0, 2))
        dead = np.where(per_channel_std < 1e-8)[0]
        if len(dead) > 0:
            problems.append(f"channels with ~zero variance (possible zero-pad/misindex): {dead.tolist()}")

    # valid_start/valid_end (see IO/loader.py's cut_event_window / cache_compile.py):
    # marks real vs zero-padded content per trial. Absent entirely means an older cache
    # from before this existed -- only a real problem once pre/post_event_seconds is
    # actually in use (0/0 means every loader falls back to full validity anyway, so an
    # old-format cache is equivalent, not stale).
    has_valid_range = 'valid_start' in npz and 'valid_end' in npz
    if not has_valid_range and (pre_event_seconds or post_event_seconds):
        problems.append("cache predates valid_start/valid_end (pre/post_event_seconds is "
                         "configured, but this cache has neither key) — rerun cache_compile.py")
    elif has_valid_range:
        vs, ve = npz['valid_start'], npz['valid_end']
        if len(vs) != N or len(ve) != N:
            problems.append(f"valid_start/valid_end length {len(vs)}/{len(ve)} != trial count N={N}")
        elif N > 0:
            if (vs < 0).any() or (ve > T).any() or (vs > ve).any():
                bad = np.where((vs < 0) | (ve > T) | (vs > ve))[0]
                problems.append(f"valid_start/valid_end out of [0,{T}] or start>end for trial(s): "
                                 f"{bad[:5].tolist()}{'...' if len(bad) > 5 else ''}")
            else:
                # The metadata's claimed padded region should actually BE zero in `data`,
                # and its claimed real region should NOT be (a mismatch either way means
                # the valid_start/valid_end written at compile time no longer describes
                # what's actually in this array -- e.g. a resample-rescale bug, or a stale
                # cache mixing an old data array with new metadata).
                bad_pad, bad_real, any_pad = [], [], False
                for i in range(N):
                    s, e = int(vs[i]), int(ve[i])
                    if s > 0 and np.abs(data[i, :, :s]).max() > 1e-6:
                        bad_pad.append(i)
                    if e < T and np.abs(data[i, :, e:]).max() > 1e-6:
                        bad_pad.append(i)
                    if e > s and np.abs(data[i, :, s:e]).max() < 1e-6:
                        bad_real.append(i)
                    any_pad = any_pad or s > 0 or e < T
                if bad_pad:
                    problems.append(f"valid_start/valid_end claims padding but data is nonzero there, "
                                     f"trial(s): {sorted(set(bad_pad))[:5]}{'...' if len(set(bad_pad)) > 5 else ''}")
                if bad_real:
                    problems.append(f"valid_start:valid_end region is all-zero (degenerate real content), "
                                     f"trial(s): {bad_real[:5]}{'...' if len(bad_real) > 5 else ''}")
                if any_pad:
                    frac = float(((vs > 0) | (ve < T)).mean())
                    print(f"  [info] {frac*100:.0f}% of trials have some padding (real content < {T} samples)")

    # Spectral rolloff: bandpass should have suppressed power above h_freq.
    l_freq, h_freq = bandpass_filter['l_freq'], bandpass_filter['h_freq']
    h_eff = min(h_freq, sample_freq / 2 - 1e-6)
    if N > 0 and T >= 16:
        sample = data[:min(N, 5)].reshape(-1, T)
        freqs, psd = scipy.signal.welch(sample, fs=sample_freq, axis=-1, nperseg=min(T, 256))
        psd_mean = psd.mean(axis=0)
        in_band = (freqs >= l_freq) & (freqs <= h_eff)
        out_band = freqs > h_eff + 5  # comfortably past the cutoff
        if in_band.any() and out_band.any():
            in_power, out_power = psd_mean[in_band].mean(), psd_mean[out_band].mean()
            if out_power > in_power * 0.5:
                problems.append(f"weak bandpass rolloff: in-band power {in_power:.4g} vs "
                                 f"out-of-band (>{h_eff + 5:.1f}Hz) power {out_power:.4g} — filter may not be applied")

    if deep:
        fs_orig = data_metadata['acquisition']['sample_frequency']
        loader_cls = resolve_dataset_loader(dataset_path)
        loader_config = {
            'dataset_params': {'dataset_path': dataset_path, 'pre_event_seconds': pre_event_seconds,
                                'post_event_seconds': post_event_seconds},
            'data_metadata': data_metadata,
            'data_structure': data_structure,
        }
        loader = loader_cls(config=loader_config, subject_id=subject_id,
                             desired_channel_indices=list(range(c_expected)))
        subject_data = loader.get_subject_data()
        if subject_data is None:
            problems.append("deep check: raw loader returned no data for this subject")
        else:
            transform = BandpassResample(original_freq=fs_orig, sample_freq=sample_freq,
                                          l_freq=l_freq, h_freq=h_freq)
            recomputed = transform(subject_data['data']).numpy()
            if recomputed.shape != data.shape:
                problems.append(f"deep check: recompute shape {recomputed.shape} != cache shape {data.shape} "
                                 f"— cache is stale, rerun cache_compile.py")
            elif not np.allclose(recomputed, data, atol=1e-4):
                max_diff = np.abs(recomputed - data).max()
                problems.append(f"deep check: recompute differs from cache (max abs diff={max_diff:.4g}) "
                                 f"— cache is stale, rerun cache_compile.py")
            if not np.array_equal(subject_data['labels'].astype(np.int64), labels):
                problems.append("deep check: recomputed labels differ from cached labels")

    return problems


def check_dataset(ds_name, ds_args, sample_freq, bandpass_filter, suffix, n_subjects, deep,
                   pre_event_seconds=0.0, post_event_seconds=0.0):
    """Checks the first n_subjects of one dataset, returns (ds_name, {sub_id: problems})."""
    dataset_path = ds_args['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    data_metadata = metadata.get('data_metadata', {})
    data_structure = metadata.get('data_structure', {})
    subjects = list(data_structure.keys())[:n_subjects]

    results = {}
    for sub_id in subjects:
        results[sub_id] = check_subject(dataset_path, sub_id, data_metadata, data_structure,
                                         sample_freq, bandpass_filter, suffix, deep=deep,
                                         pre_event_seconds=pre_event_seconds,
                                         post_event_seconds=post_event_seconds)
    return ds_name, len(data_structure), results


def _print_dataset_result(ds_name, n_total, results, deep):
    print(f"Checking {ds_name}: {len(results)}/{n_total} subject(s){' [deep]' if deep else ''}...")
    any_problems = False
    for sub_id, problems in results.items():
        if problems:
            any_problems = True
            print(f"  [{ds_name} S{sub_id}] FAIL:")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"  [{ds_name} S{sub_id}] OK")
    return any_problems


def main():
    parser = argparse.ArgumentParser(description='Sanity-check compiled per-subject .npz caches '
                                                   '(shape/labels/dead-channels/bandpass-rolloff, '
                                                   'optionally a full raw-vs-cache diff).')
    parser.add_argument('--config', type=str, default='config/compile.json')
    parser.add_argument('--dataset', type=str, default=None,
                         help='Limit to one dataset name (default: every dataset in the config).')
    parser.add_argument('--subjects', type=int, default=3,
                         help='Check the first N subjects per dataset (default 3).')
    parser.add_argument('--deep', action='store_true',
                         help='Also re-run the raw loader + BandpassResample and diff against the '
                              'cache byte-for-byte — slower, catches a stale cache.')
    parser.add_argument('--workers', type=int, default=1,
                         help='Datasets to check in parallel, one process per dataset (default 1, '
                              'sequential). Mainly helps --deep with a large --subjects count.')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    cp = config['compile_params']
    suffix = cache_suffix(cp['sample_freq'], cp['bandpass_filter'],
                           cp.get('pre_event_seconds', 0.0), cp.get('post_event_seconds', 0.0))
    names = [args.dataset] if args.dataset else list(cp['datasets'].keys())

    pre_event_seconds = cp.get('pre_event_seconds', 0.0)
    post_event_seconds = cp.get('post_event_seconds', 0.0)

    any_problems = False
    if args.workers <= 1:
        for ds_name in names:
            ds_name, n_total, results = check_dataset(
                ds_name, cp['datasets'][ds_name], cp['sample_freq'], cp['bandpass_filter'],
                suffix, args.subjects, args.deep, pre_event_seconds, post_event_seconds)
            any_problems |= _print_dataset_result(ds_name, n_total, results, args.deep)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(check_dataset, ds_name, cp['datasets'][ds_name], cp['sample_freq'],
                            cp['bandpass_filter'], suffix, args.subjects, args.deep,
                            pre_event_seconds, post_event_seconds)
                for ds_name in names
            ]
            for future in as_completed(futures):
                ds_name, n_total, results = future.result()
                any_problems |= _print_dataset_result(ds_name, n_total, results, args.deep)

    if any_problems:
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == '__main__':
    main()
