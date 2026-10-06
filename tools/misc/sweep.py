"""
Expand a sweep file into queue jobs (tools/misc/run_queue.py plan lines) -- one base config plus
--set overrides per job, so a grid is one small file instead of hundreds of configs.

Sweep file (JSON):
    {
      "script": "train_finetune.py",                          # or train_pretrain.py
      "config": "configs/runs/<backbone>/finetune/learned/BNCI2014001_loso.json",
      "set":    {"training_params.finetune.epochs": 50},     # fixed for every job
      "cases":  {"mi_loso":    {"dataset_params.finetune": {...}, "training_params.finetune.split": {...}},
                 "mi_fewshot": {...}},                       # optional, each case's own overrides
      "grid":   {"training_params.finetune.learning_rate": [0.001, 0.003],
                 "training_params.finetune.weight_decay":  [0.01, 0.1]},
      "name":   {"training_params.finetune.learning_rate": "lr",   # optional short names for job ids
                 "training_params.finetune.weight_decay":  "wd"},
      "output_path": "<backbone>/finetune/tune/{case}/{job}", # {case} / {job} filled in
      "derived": {"training_params.finetune.min_learning_rate": "training_params.finetune.learning_rate / 10"}
    }
Every combination of case x grid becomes one job `<case>/<job>`; "derived" values are Python
expressions over the job's own keys (the dotted names are available as-is). Optional
"backbones": [...] repeats the whole sweep per backbone, replacing '{backbone}' anywhere in it
(config path, output_path, set values); job names get a '<backbone>/' prefix.

    python -m tools.misc.sweep configs/Qtome/sweeps/tune_base_s1.json >> output/queue/tune.plan
"""
import itertools
import json
import shlex
import sys


def expand(sweep):
    if 'backbones' in sweep:                  # one copy of the sweep per backbone, '{backbone}' filled in
        rest = json.dumps({k: v for k, v in sweep.items() if k != 'backbones'})
        for b in sweep['backbones']:
            for name, cmd in expand(json.loads(rest.replace('{backbone}', b))):
                yield f'{b}/{name}', cmd
        return
    grid = sweep.get('grid', {})
    keys = list(grid)
    short = sweep.get('name', {})
    for case, case_sets in (sweep.get('cases') or {'': {}}).items():
        for values in itertools.product(*(grid[k] for k in keys)):
            job_vals = dict(zip(keys, values))
            job = '_'.join(f'{short.get(k, k.rsplit(".", 1)[-1])}{v:g}' if isinstance(v, (int, float))
                           else f'{short.get(k, k.rsplit(".", 1)[-1])}{v}' for k, v in job_vals.items()) or 'base'
            sets = {**sweep.get('set', {}), **case_sets, **job_vals}
            env = {k.replace('.', '__'): v for k, v in sets.items()}
            for k, expr in sweep.get('derived', {}).items():
                for name in sorted(sets, key=len, reverse=True):          # dotted key -> identifier
                    expr = expr.replace(name, name.replace('.', '__'))
                sets[k] = eval(expr, {}, env)
            if 'output_path' in sweep:
                mode = 'pretrain' if 'pretrain' in sweep['script'] else 'finetune'
                sets[f'training_params.{mode}.output_path'] = sweep['output_path'].format(case=case, job=job)
            name = f'{case}/{job}' if case else job
            cmd = ['python', sweep['script'], '--config', sweep['config']] + \
                  [a for k, v in sets.items() for a in ('--set', f'{k}={json.dumps(v)}')]
            yield name, ' '.join(shlex.quote(c) for c in cmd)


def main():
    sweep = json.load(open(sys.argv[1]))
    for name, cmd in expand(sweep):
        print(f'job {name} :: {cmd}')


if __name__ == '__main__':
    main()
