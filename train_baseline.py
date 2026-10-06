"""Train a from-scratch baseline (model/<Name>/, no pretrained backbone) on one finetune cell, through the same caches,
splits (configs/finetune_protocols.json via IO/splits.py) and result format as train_finetune.py, so a baseline cell
sits next to a Qtome cell in summarize_runs. Settings: configs/<Name>/settings.json (Compass's, per split mode, with
per-dataset overrides). One seed per run; scored every epoch on the evaluation trials (tail = mean of the last 10
epochs, last = Compass's number).

    python train_baseline.py --model EEGNet --dataset BNCI2014004 --protocol mi_fewshot [--seeds 1 2 3] [--set k=v]
    -> output/EEGNet/finetune/compass/<dataset>_<loso|fewshot>_seed<k>/artifacts/{config.json, group_eval.json}
    --tag NAME writes to output/EEGNet/finetune/<NAME>/ instead (variants, e.g. --set settings.norm='"ztrial"')
"""
import argparse, copy, glob, json, math, os

import mne
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

from model.EEGNet.EEGNet import EEGNet
from IO.splits import make_runs, protocol_split
from tools.analysis import apply_overrides

BASELINES = {'EEGNet': EEGNet}
# deterministic cuDNN: without it two runs of one seed differ (EEGNet's convolutions picked non-deterministic
# algorithms; 2026-10-06), and refactors here are checked by bit-identical reruns
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
CACHE_FS = 200.0
CACHE_GLOB = 'datas/finetune/{ds}/cache/*_fs200_bp0.5-100.0_pre1_post4.npz'


def load_cache(ds):
    """Every subject's compiled trials, concatenated in subject order (cache order kept within a subject: few-shot
    splits are chronological). Volts -> microvolts. End padding of the cache window is cut when it is uniform."""
    xs, ys, ss, subs = [], [], [], []
    for f in sorted(glob.glob(CACHE_GLOB.format(ds=ds)), key=lambda p: int(os.path.basename(p).split('_')[0])):
        d = np.load(f)
        x = d['data'] * 1e6
        if 'valid_end' in d.files and len(np.unique(d['valid_end'])) == 1:
            x = x[..., :int(d['valid_end'][0])]
        xs.append(x); ys.append(d['labels']); ss.append(d['session'])
        subs.append(np.full(len(x), int(os.path.basename(f).split('_')[0])))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(subs)


def preprocess(x, s):
    """Compass Preprocessor order: resample -> trim / pad (repeat from the start) to time_length -> 4th-order
    Butterworth filtfilt band -> optional common average reference."""
    x = mne.filter.resample(x.astype(np.float64), down=CACHE_FS / s['sample_freq'], verbose=False)
    n = int(s['time_length'] * s['sample_freq'])
    x = np.concatenate([x] * math.ceil(n / x.shape[-1]), -1)[..., :n]
    nyq = s['sample_freq'] / 2
    b, a = signal.butter(4, [s['band'][0] / nyq, s['band'][1] / nyq], btype='band')
    x = signal.filtfilt(b, a, x, axis=-1)
    if s['norm'] == 'car':
        x = x - x.mean(1, keepdims=True)
    elif s['norm'] == 'ztrial':      # Qtome's input normalisation: one mean / std per trial over channels and samples
        x = (x - x.mean((1, 2), keepdims=True)) / (x.std((1, 2), keepdims=True) + 1e-8)
    return np.ascontiguousarray(x, dtype=np.float32)


def cosine_schedule(base, final, epochs, iters, warmup):
    """Compass utils/optimizer.cosine_scheduler: linear warm-up from 0, then cosine to `final`, one value per step."""
    w = np.linspace(0, base, warmup * iters) if warmup else np.array([])
    n = epochs * iters - len(w)
    return np.concatenate([w, [final + 0.5 * (base - final) * (1 + math.cos(math.pi * i / n)) for i in range(n)]])


def train_run(model_cls, x, y, run, n_classes, s, seed, device):
    """One run (a loso fold or one subject's few-shot split) -> per-subject {tail, last, kappa_tail, kappa_last}."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_cls(n_classes, x.shape[1], x.shape[2], dropout=s['dropout']).to(device)
    params = list(model.named_parameters())
    if s['filter_bias_and_bn']:
        groups = [{'params': [p for n, p in params if p.ndim <= 1 or n.endswith('.bias')], 'weight_decay': 0.},
                  {'params': [p for n, p in params if p.ndim > 1 and not n.endswith('.bias')], 'weight_decay': s['weight_decay']}]
        opt = torch.optim.Adam(groups, lr=s['lr'], eps=1e-8)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=s['lr'], weight_decay=s['weight_decay'], eps=1e-8)
    tr = run['train']
    weight = None
    if s['class_weights']:
        cnt = np.bincount(y[tr], minlength=n_classes).astype(float)
        weight = torch.tensor(len(tr) / (n_classes * np.maximum(cnt, 1)), dtype=torch.float, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)
    Xtr, Ytr = torch.from_numpy(x[tr]).to(device), torch.from_numpy(y[tr]).long().to(device)
    iters = math.ceil(len(tr) / s['batch_size'])
    lr = cosine_schedule(s['lr'], s['min_lr'], s['epochs'], iters, min(s['warmup_epochs'], s['epochs']))
    gen = torch.Generator().manual_seed(seed)
    evals = {sub: idx for g in run['eval'].values() for sub, idx in g.items()}
    hist = {sub: [] for sub in evals}
    step = 0
    for _ in range(s['epochs']):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        for i in range(iters):
            for pg in opt.param_groups:
                pg['lr'] = lr[step]
            b = perm[i * s['batch_size']:(i + 1) * s['batch_size']]
            opt.zero_grad()
            crit(model(Xtr[b]), Ytr[b]).backward()
            opt.step()
            step += 1
        model.eval()
        with torch.no_grad():
            for sub, idx in evals.items():
                pred = torch.cat([model(torch.from_numpy(x[idx[i:i + 512]]).to(device)).argmax(1).cpu()
                                  for i in range(0, len(idx), 512)]).numpy()
                hist[sub].append((balanced_accuracy_score(y[idx], pred), cohen_kappa_score(y[idx], pred)))
    out = {}
    for sub, h in hist.items():
        h = np.array(h)
        out[str(sub)] = dict(tail=float(h[-10:, 0].mean()), last=float(h[-1, 0]), kappa_tail=float(h[-10:, 1].mean()),
                             kappa_last=float(h[-1, 1]), n_trials=int(len(evals[sub])))
    return out


def main():
    ap = argparse.ArgumentParser(description='Train a from-scratch baseline on one finetune cell')
    ap.add_argument('--model', default='EEGNet', choices=sorted(BASELINES))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--protocol', required=True, help='a configs/finetune_protocols.json entry (shared split)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override a setting (dotted path into {"settings": ..., "split": ...}), JSON value')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='compass', help='head-label folder under output/<model>/finetune/')
    a = ap.parse_args()

    split = protocol_split(a.protocol, [a.dataset])     # the shared split of a Qtome cell (sessions, fraction, purge)
    mode = 'loso' if split['type'] in ('loso', 'subject_kfold') else 'fewshot'
    presets = json.load(open(f'configs/{a.model}/settings.json'))
    settings = {**presets[mode], **presets.get('cells', {}).get(a.dataset, {}).get(mode, {})}
    run_cfg = apply_overrides({'settings': settings, 'split': split}, a.set)
    s, split = run_cfg['settings'], run_cfg['split']

    x, y, session, subject_data = load_cache(a.dataset)
    x = preprocess(x, s)
    n_classes = int(y.max()) + 1
    pool = sorted(np.unique(subject_data).tolist())
    runs = make_runs(split, pool, subject_data, y, session)
    for seed in a.seeds:
        base = f'output/{a.model}/finetune/{a.tag}/{a.dataset}_{mode}_seed{seed}/artifacts'
        os.makedirs(base, exist_ok=True)
        json.dump({'model': a.model, 'dataset': a.dataset, 'protocol': a.protocol, 'seed': seed, **run_cfg},
                  open(f'{base}/config.json', 'w'), indent=2)
        result = {}
        for run in runs:
            subs = train_run(BASELINES[a.model], x, y, run, n_classes, s, seed, a.device)
            result[run['name']] = {'train_subjects': run['train_subjects'], 'epochs': s['epochs'], 'tail_epochs': 10,
                                   'groups': {'heldout': {
                                       'subjects': subs, 'n_subjects': len(subs),
                                       'mean_tail': float(np.mean([v['tail'] for v in subs.values()])),
                                       'mean_last': float(np.mean([v['last'] for v in subs.values()])),
                                       'mean_kappa_tail': float(np.mean([v['kappa_tail'] for v in subs.values()])),
                                       'mean_kappa_last': float(np.mean([v['kappa_last'] for v in subs.values()]))}}}
            json.dump(result, open(f'{base}/group_eval.json', 'w'), indent=2)       # partial results survive
        last = np.mean([v['last'] for r in result.values() for v in r['groups']['heldout']['subjects'].values()])
        print(f'{a.model} {a.dataset} {mode} seed {seed}: last {100 * last:.2f}', flush=True)


if __name__ == '__main__':
    main()
