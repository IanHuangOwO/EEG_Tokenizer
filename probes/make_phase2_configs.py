"""Write the Phase 2 finetune configs (docs/superpowers/plans/2026-09-21-phase2-runs.md).
Usage: python probes/make_phase2_configs.py [--out config/phase2] [--epochs 50]
(--epochs is the default; EPOCHS below overrides it per dataset)
Each config = config/config.json with the finetune dataset, head, and split replaced."""
import argparse, copy, json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C1 = dict(input='stamp_induced', pool_channel='spatial:8', pool_time='learned:2', dropout=0.5,
          freeze_backbone=True)
HEADS = {
    'c1': C1,
    'c3': dict(C1, include_advance=True),
    'c4': dict(C1, evoked_rank=2),
    'raw': dict(input='raw', pool_channel='spatial:8', pool_time='trial', dropout=0.5,
                freeze_backbone=True),
}
EPOCHS = {'inria': 30}  # shorter to save GPU time; convergence is checked afterwards
# (dataset dir, short name, subject_groups json or None, task, split, head tags)
PLAN = [
    ('EEGMMIdb', 'eegmmidb', 'eegmmidb', 'mi', 'groups', ['c1', 'raw']),
    ('BETA_4s', 'beta4s', 'beta4s', 'ssvep', 'groups', ['c1', 'c3']),
    ('Inria_Train', 'inria', None, 'erp', 'kfold4', ['c1', 'c4', 'raw']),
]


def build(base, ds, short, groups_key, task, split, tag, epochs):
    cfg = copy.deepcopy(base)
    cfg['dataset_params']['finetune'] = {
        ds: {'dataset_path': f'datas/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}}
    cfg['model_params']['MeSAE']['finetune'] = dict(HEADS[tag], task=task)
    tp = cfg['training_params']['finetune']
    tp.pop('cv_folds', None)
    tp.update(model_name=f'mesae_p2_{short}_{tag}', epochs=EPOCHS.get(short, epochs),
              split_mode='subject_groups')
    if split == 'groups':
        g = json.load(open(os.path.join(REPO, 'config', 'subject_groups', f'{groups_key}.json')))
        tp['subject_group_runs'] = [{'name': 'main', 'train': g['train'], 'eval': g['eval']}]
    else:
        tp['subject_kfold'] = 4  # subject_kfold_seed defaults to 42: same folds for every head
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'config', 'phase2'))
    ap.add_argument('--epochs', type=int, default=50)
    a = ap.parse_args()
    base = json.load(open(os.path.join(REPO, 'config', 'config.json')))
    os.makedirs(a.out, exist_ok=True)
    for ds, short, groups_key, task, split, tags in PLAN:
        for tag in tags:
            cfg = build(base, ds, short, groups_key, task, split, tag, a.epochs)
            name = f'{short}_{tag}.json'
            with open(os.path.join(a.out, name), 'w', newline='\n') as f:
                json.dump(cfg, f, indent=1)
                f.write('\n')
            print('wrote', os.path.join(a.out, name))


if __name__ == '__main__':
    main()
