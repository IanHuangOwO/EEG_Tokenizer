"""Parse EEG-FM-Compass's per-dataset appendix tables (pdftotext -layout of docs/papers/2601.17883_EEG-FM-Compass.pdf)
-> best entry per (dataset, metric, scenario, group), group = specialist / FM full fine-tune / FM linear probe.
Each table lists, per scenario (LOSO, then few-shot), specialist models, then foundation models fully fine-tuned, then
the same foundation models linearly probed: a block ends where a model name repeats. Writes compass_appendix.json."""
import json, re, subprocess, sys

TABLES = {('BNCI2014001', 'acc'): 'TABLE XV:', ('BNCI2014001', 'kappa'): 'TABLE XVI:',
          ('BNCI2014004', 'acc'): 'TABLE XVII:', ('BNCI2014004', 'kappa'): 'TABLE XVIII:',
          }   # BNCI2014008 (Table XXII) has no kappa and no MI-only models: its Table V-VI values are used as is
txt = subprocess.run(['pdftotext', '-layout', 'docs/papers/2601.17883_EEG-FM-Compass.pdf', '-'],
                     capture_output=True).stdout.decode('utf-8', 'replace').replace('\f', '')
lines = txt.split('\n')
starts = [i for i, l in enumerate(lines) if re.match(r'\s*TABLE [IVXL]+:', l)]

out = {}
for (ds, metric), tag in TABLES.items():
    i0 = next(i for i in starts if lines[i].strip().startswith(tag))
    i1 = next(i for i in starts if i > i0)
    rows, seen, group, scen = [], set(), 'specialist', 'loso'
    for l in lines[i0:i1]:
        m = re.search(r'(?:^|\s{2,})([A-Za-z][\w+.\-]*(?:-[A-Za-z]+)?)\s+(-?\d+\.\d+(?:\s+-?\d+\.\d+)+)(?:±(\d+\.\d+))?\s*$', l)
        if not m:
            continue
        name, nums = m.group(1), m.group(2).split()
        if name == 'CSP+LDA' and seen:                       # second scenario starts
            scen, group, seen = 'fewshot', 'specialist', set()
        if name in seen:
            group = 'fm_linear'                              # the same FMs again: linear probing
            seen = set()
        elif group == 'specialist' and name in ('BENDR', 'BIOT', 'LaBraM'):
            group = 'fm_full'
        seen.add(name)
        rows.append((scen, group, name, float(nums[-1])))
    for scen in ('loso', 'fewshot'):
        for g in ('specialist', 'fm_full', 'fm_linear'):
            cand = [(v, n) for s, gg, n, v in rows if s == scen and gg == g]
            if cand:
                v, n = max(cand)
                out.setdefault(ds, {}).setdefault(metric, {}).setdefault(scen, {})[g] = [v, n]
                out[ds][metric][scen].setdefault('_all', {})[g] = sorted(cand, reverse=True)[:3]
json.dump(out, open('docs/reports/2026-09-29-combined-head/compass_appendix.json', 'w'), indent=1)
for ds in out:
    for metric in out[ds]:
        for scen in ('loso', 'fewshot'):
            r = out[ds][metric][scen]
            print(ds, metric, scen, {g: r[g] for g in ('specialist', 'fm_full', 'fm_linear') if g in r})
