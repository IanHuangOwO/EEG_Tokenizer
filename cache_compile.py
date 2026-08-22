import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from IO.loader import resolve_dataset_loader
from IO.preprocessing import BandpassResample, cache_suffix


def compile_dataset(ds_name: str, ds_args: dict, sample_freq: float, bandpass_filter: dict):
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
    suffix = cache_suffix(sample_freq, bandpass_filter)

    loader_config = {
        'dataset_params': ds_args,
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
        out_path = os.path.join(cache_dir, f"{sub_id}_{suffix}.npz")
        np.savez(out_path, data=data.astype(np.float32), labels=subject_data['labels'].astype(np.int64))
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
    datasets = compile_params['datasets']

    failed = []
    if args.workers <= 1:
        for ds_name, ds_args in datasets.items():
            print(f"Compiling {ds_name} ({ds_args['dataset_path']})...")
            try:
                compile_dataset(ds_name, ds_args, sample_freq, bandpass_filter)
            except Exception as e:
                print(f"[{ds_name}] FAILED: {e}\n")
                failed.append(ds_name)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(compile_dataset, ds_name, ds_args, sample_freq, bandpass_filter): ds_name
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
