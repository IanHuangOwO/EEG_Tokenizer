"""Stamp-amplitude cache for the frozen backbone (finetune restructure, sub-project B).
Pipeline stage between the compiled data (cache_dataset.py) and the finetune head, hence a root script:
    python cache_feature.py --config config/config.template.json

The backbone never changes during finetuning, so its stamp amplitudes are computed once per
(checkpoint, dataset, preprocessing) and stored next to the backbone:
    <backbone run folder>/feature_cache/<dataset>/<key>/<subject>.npz
Only stamp features use it (StampExtractor output); raw features never touch the backbone."""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from IO.dataset import build_dataset_from_config
from IO.loader import get_standard_coords
from IO.preprocessing import cache_suffix, slice_patches
from model.factory import load_backbone
from model.MeSAE.MeSAE_modules import StampExtractor


def _fingerprint(path):
    """[size, mtime_ns] of a file, or None if missing."""
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def _mne_version():
    try:
        import mne
        return mne.__version__ if get_standard_coords('Cz') is not None else 'installed-but-unusable'
    except ImportError:
        return 'none'   # coordinates fall back to the flat metadata polar values


def cache_key(config, dataset_name, keep, checkpoint_path):
    """Folder key: everything that changes the amplitudes of a given subject file."""
    ds_args = config['dataset_params']['finetune'][dataset_name]
    parts = dict(
        ckpt=[os.path.basename(checkpoint_path), *_fingerprint(checkpoint_path)],
        keep=[int(k) for k in keep],
        preprocess=config.get('preprocess_params', {}),
        dataset={k: v for k, v in ds_args.items() if k != 'subject_to_use'},
        metadata=_fingerprint(os.path.join(ds_args['dataset_path'], 'metadata.json')),
        montages=_fingerprint(os.path.join('config', 'montages.json')),
        mne=_mne_version(),
    )
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _data_path(config, dataset_name, subject):
    pp = config['preprocess_params']
    suffix = cache_suffix(pp['sample_freq'], pp['bandpass_filter'],
                          pp.get('pre_event_seconds', 0.0), pp.get('post_event_seconds', 0.0))
    return os.path.join(config['dataset_params']['finetune'][dataset_name]['dataset_path'],
                        'cache', f"{subject}_{suffix}.npz")


def _is_current(path, data_fp):
    """True if the stored subject file exists and was built from the current compiled data file."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as z:
            return json.loads(str(z['meta']))['data'] == data_fp
    except Exception:
        return False


@torch.no_grad()
def _build_subject(config, dataset_name, subject, backbone, device, batch_size, path):
    sub_cfg = json.loads(json.dumps(config))
    sub_cfg['dataset_params']['finetune'] = {
        dataset_name: {**config['dataset_params']['finetune'][dataset_name], 'subject_to_use': [subject]}}
    base = build_dataset_from_config(sub_cfg, mode='finetune').base_dataset
    pp = config['preprocess_params']
    patch_len = pp.get('patch_length', 100)
    patch_stride = pp.get('patch_stride', patch_len)
    valid = base.all_valid_channels[0]
    channel_idx = torch.nonzero(valid).flatten().tolist()
    coords, vlen = base.all_coords[0], int(base.all_valid_length[0])
    extractor = StampExtractor(backbone, channel_idx).to(device).eval()
    out = []
    for i in range(0, len(base.data), batch_size):
        x = base.data[i:i + batch_size]                                   # [b, C, T]
        xp, _ = slice_patches(x, patch_len, patch_stride)                 # [b, C, N', L]
        b, P = xp.shape[0], xp.shape[2]
        amp = extractor(xp.to(device), coords.unsqueeze(0).expand(b, -1, -1).to(device),
                        torch.arange(P, device=device).unsqueeze(0).expand(b, P),
                        valid.unsqueeze(0).expand(b, -1).to(device))      # [b, N', Cv, S, 2] fp32
        out.append(amp.cpu())
    amp = torch.cat(out)
    if not torch.isfinite(amp).all() or amp.abs().max() >= 6e4:
        raise ValueError(f"subject {subject}: stamp amplitudes are not finite or exceed the fp16 range "
                         f"(max abs {amp.abs().max().item():.3g})")
    data_fp = _fingerprint(_data_path(config, dataset_name, subject))
    np.savez(path, amp=amp.half().numpy(), labels=base.labels.numpy().astype(np.int64),
             valid_length=np.full(len(amp), vlen, dtype=np.int64), channel_idx=np.asarray(channel_idx, dtype=np.int64),
             keep=extractor.keep.cpu().numpy().astype(np.int64), meta=np.array(json.dumps({'data': data_fp})))
    return amp.shape


def get_stamp_cache(config, dataset_name, subjects, device=None, batch_size=64):
    """Build any missing or stale per-subject file and return the cache folder."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = config['training_params']['finetune']['pretrained_checkpoint']
    backbone = load_backbone(config).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    keep = StampExtractor(backbone, [0]).keep.cpu().tolist()   # alive stamps do not depend on the channels
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    folder = os.path.join(run_dir, 'feature_cache', dataset_name, cache_key(config, dataset_name, keep, ckpt))
    os.makedirs(folder, exist_ok=True)
    for sub in subjects:
        sub = str(sub)
        path = os.path.join(folder, f'{sub}.npz')
        if _is_current(path, _fingerprint(_data_path(config, dataset_name, sub))):
            continue
        shape = _build_subject(config, dataset_name, sub, backbone, device, batch_size, path)
        print(f"  [feature_cache] built {dataset_name} subject {sub}: amp {tuple(shape)} -> {path}")
    return folder


class CachedStampDataset(Dataset):
    """Stamp amplitudes of the given subjects, in RAM. See the plan's Interfaces section."""
    def __init__(self, folder, subjects):
        parts = []
        for s in subjects:
            with np.load(os.path.join(folder, f'{s}.npz')) as z:
                parts.append({k: z[k] for k in ('amp', 'labels', 'valid_length', 'channel_idx', 'keep')})
        for p in parts[1:]:
            assert np.array_equal(p['channel_idx'], parts[0]['channel_idx']), "subjects have different real-channel sets"
            assert np.array_equal(p['keep'], parts[0]['keep']), "subjects were cached with different alive stamps"
        self.amp = torch.from_numpy(np.concatenate([p['amp'] for p in parts]))
        self.labels = torch.from_numpy(np.concatenate([p['labels'] for p in parts])).long()
        self.valid_length = torch.from_numpy(np.concatenate([p['valid_length'] for p in parts])).long()
        self.subject_data = torch.cat([torch.full((len(p['labels']),), int(s), dtype=torch.long)
                                       for s, p in zip(subjects, parts)])
        self.channel_idx = parts[0]['channel_idx'].tolist()
        self.keep = parts[0]['keep'].tolist()
        self.num_patches, self.num_channels, self.num_stamps = self.amp.shape[1], self.amp.shape[2], self.amp.shape[3]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.amp[i], self.labels[i], self.valid_length[i]


def _subjects(ds_args):
    subs = ds_args['subject_to_use']
    if subs in (['all'], 'all'):
        with open(os.path.join(ds_args['dataset_path'], 'metadata.json'), encoding='utf-8') as f:
            ids = list(json.load(f)['data_structure'].keys())
        try:
            return sorted(ids, key=int)
        except ValueError:
            return sorted(ids)
    return [str(s) for s in subs]


def main():
    ap = argparse.ArgumentParser(description="Build the stamp-amplitude cache for every finetune dataset in a config.")
    ap.add_argument('--config', default='config/config.template.json')
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = json.load(f)
    for name, ds_args in config['dataset_params']['finetune'].items():
        folder = get_stamp_cache(config, name, _subjects(ds_args), batch_size=args.batch_size)
        print(f"{name}: cache ready in {folder}")


if __name__ == '__main__':
    main()
