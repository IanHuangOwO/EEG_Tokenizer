import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from IO.loader import resolve_dataset_loader
from IO.preprocessing import BandpassResample, cache_suffix


def compile_dataset(ds_name: str, ds_args: dict, sample_freq: float, bandpass_filter: dict,
                     pre_event_seconds: float = 0.0, post_event_seconds: float = 0.0):
    dataset_path = ds_args['dataset_path']
    meta_path = os.path.join(dataset_path, 'metadata.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    data_metadata = metadata.get('data_metadata', {})
    data_structure = metadata.get('data_structure', {})
    fs_orig = data_metadata['acquisition']['sample_frequency']
    c_native = data_metadata.get('channels', {}).get('count', 0)
    if not c_native:
        raise ValueError(f"{meta_path}: data_metadata.channels.count is missing/zero.")

    loader_cls = resolve_dataset_loader(dataset_path)
    transform = BandpassResample(
        original_freq=fs_orig, sample_freq=sample_freq,
        l_freq=bandpass_filter['l_freq'], h_freq=bandpass_filter['h_freq'],
    )
    suffix = cache_suffix(sample_freq, bandpass_filter, pre_event_seconds, post_event_seconds)
    resample_scale = sample_freq / fs_orig  # same ratio BandpassResample.resample uses

    loader_config = {
        # pre_event_seconds/post_event_seconds are a GLOBAL compile-time setting (unlike
        # dataset_path), injected into every dataset's dataset_params uniformly -- most
        # loaders never reference them (see e.g. datas/EEGMMIdb/loader.py, which
        # deliberately opts out even though it receives them) and are unaffected.
        'dataset_params': {**ds_args, 'pre_event_seconds': pre_event_seconds,
                            'post_event_seconds': post_event_seconds},
        'data_metadata': data_metadata,
        'data_structure': data_structure,
    }

    cache_dir = os.path.join(dataset_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    n_written = 0
    for sub_id in data_structure.keys():
        loader = loader_cls(config=loader_config, subject_id=sub_id,
                             desired_channel_indices=list(range(c_native)))
        subject_data = loader.get_subject_data()
        if subject_data is None:
            print(f"  [{ds_name} S{sub_id}] no data, skipped.")
            continue

        data = transform(subject_data['data']).numpy()
        # Rescale NATIVE-rate valid_ranges (see IO/loader.py's get_subject_data) to the
        # compiled rate with the SAME ratio BandpassResample used above, so
        # valid_start/valid_end index correctly into `data`, not the pre-resample array.
        new_T = data.shape[-1]
        valid_start = np.array([min(int(vs * resample_scale), new_T) for vs, _ in subject_data['valid_ranges']],
                                dtype=np.int64)
        valid_end = np.array([min(int(round(ve * resample_scale)), new_T) for _, ve in subject_data['valid_ranges']],
                              dtype=np.int64)
        # Re-zero padding AFTER the bandpass filter, not just before it (cut_event_window
        # already zeroed it pre-filter) -- sosfiltfilt is zero-phase but NOT zero-leakage
        # across a hard real-to-zero discontinuity: a low l_freq (slow filter, long
        # settling time) rings for dozens of samples on either side of that edge, leaving
        # real-magnitude artifacts in what's supposed to be padding (measured: comparable
        # mean-abs to the real region, not a tiny residue -- caught by cache_verify.py's
        # valid_start/valid_end checks). Without this, those patches are ordinary
        # (unmasked, loss-included) model input -- valid_start/valid_end only gates
        # masking ELIGIBILITY (see IO/dataset.py's PretrainDataset._build_valid_masks),
        # nothing excludes them from being fed to the model or from the recon loss, so
        # the model would be trained to reconstruct filter ringing as if it were signal.
        for i in range(len(data)):
            data[i, :, :valid_start[i]] = 0.0
            data[i, :, valid_end[i]:] = 0.0
        out_path = os.path.join(cache_dir, f"{sub_id}_{suffix}.npz")
        np.savez(out_path, data=data.astype(np.float32), labels=subject_data['labels'].astype(np.int64),
                 valid_start=valid_start, valid_end=valid_end)
        print(f"  [{ds_name} S{sub_id}] {data.shape} -> {out_path}")
        n_written += 1

    print(f"[{ds_name}] {n_written}/{len(data_structure)} subjects compiled.\n")


def main():
    parser = argparse.ArgumentParser(description='Compile raw EEG datasets into per-subject '
                                                   'bandpass+resample-baked .npz caches.')
    parser.add_argument('--config', type=str, default='config/compile.json')
    parser.add_argument('--workers', type=int, default=1,
                         help='Datasets to compile in parallel (each dataset runs in its own '
                              'process — independent output dirs, no shared state). 1 = sequential.')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    compile_params = config['compile_params']
    sample_freq = compile_params['sample_freq']
    bandpass_filter = compile_params['bandpass_filter']
    pre_event_seconds = compile_params.get('pre_event_seconds', 0.0)
    post_event_seconds = compile_params.get('post_event_seconds', 0.0)
    datasets = compile_params['datasets']

    failed = []
    if args.workers <= 1:
        for ds_name, ds_args in datasets.items():
            print(f"Compiling {ds_name} ({ds_args['dataset_path']})...")
            try:
                compile_dataset(ds_name, ds_args, sample_freq, bandpass_filter,
                                 pre_event_seconds, post_event_seconds)
            except Exception as e:
                print(f"[{ds_name}] FAILED: {e}\n")
                failed.append(ds_name)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(compile_dataset, ds_name, ds_args, sample_freq, bandpass_filter,
                            pre_event_seconds, post_event_seconds): ds_name
                for ds_name, ds_args in datasets.items()
            }
            for future in as_completed(futures):
                ds_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[{ds_name}] FAILED: {e}\n")
                    failed.append(ds_name)

    if failed:
        print(f"Done with failures: {failed}")
    else:
        print("Done, all datasets compiled.")


if __name__ == '__main__':
    main()
