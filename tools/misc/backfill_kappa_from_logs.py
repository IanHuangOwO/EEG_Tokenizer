"""One-off backfill: kappa_tail/kappa_last were added to train_finetune.py's
group_eval.json schema mid-session (this run's own session, 2026-09-23), so every
finetune run that already finished has NO kappa in its group_eval.json even though its
own artifacts/train_*.log printed a per-epoch pooled kappa the whole time (_metrics'
'[Val  ] ... | kappa: X.XXXX' line). Reconstructs kappa_tail/kappa_last from that log
instead of requiring a rerun.

Why the log's POOLED (whole held-out group) kappa is a faithful per-subject value here:
every run this repo has so far uses n_subjects=1 per fold (intra_subject: one subject's
own held-out CV fold; inter_subject/LOSO: one held-out subject) -- see
tools/analysis/group_summary.py's docstring and train_finetune.py's run['eval'] shape --
so the pooled-over-the-eval-set kappa for one fold's tag IS that fold's one subject's
kappa, not an average blurring several subjects together.

Tag matching: a log's "--- [TAG] Epoch e/E Summary ---" line uses
TAG = f"{dataset_name}_{run_name}" (train_finetune.py:393), while group_eval.json's own
top-level keys are just run_name ("<subject>_fold<k>"). Matched by TAG.endswith('_' +
run_name) instead of reconstructing dataset_name (several datasets were renamed between
when these logs were written and now -- BCICIV2a->BNCI2014001 etc -- so the log's own
prefix may not match today's directory name).

Idempotent: skips a run_name that already has kappa_tail. Skips (prints, doesn't crash)
any run_name with no matching tag/log found. Patches group_eval.json in place -- purely
additive (only adds kappa_tail/kappa_last/mean_kappa_tail/mean_kappa_last keys, never
touches tail/last), so re-running this script is harmless.

Usage: python tools/misc/backfill_kappa_from_logs.py [glob]
  glob defaults to 'output/*/finetune/*/*/artifacts/group_eval.json'
"""
import glob
import json
import os
import re
import sys

EPOCH_RE = re.compile(r'--- \[(?P<tag>.+?)\] Epoch (?P<epoch>\d+)/(?P<total>\d+) Summary ---')
VAL_RE = re.compile(r'\[Val\s*\].*?\bkappa: (?P<kappa>[-\d.]+)')


def parse_log_kappa(log_path):
    """-> {tag: (total_epochs, [(epoch, kappa), ...])}. Each Epoch-Summary line is
    immediately followed (within a couple lines) by its own '[Val  ]' metrics line --
    pairs them by simple sequential scan (matches train_finetune.py's own logger.info
    call order exactly, see run_one)."""
    runs = {}
    pending = None
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                pending = (m.group('tag'), int(m.group('epoch')), int(m.group('total')))
                continue
            if pending is not None:
                m = VAL_RE.search(line)
                if m:
                    tag, epoch, total = pending
                    runs.setdefault(tag, (total, []))[1].append((epoch, float(m.group('kappa'))))
                    pending = None
    return runs


def backfill_one(group_eval_path):
    run_dir = os.path.dirname(os.path.dirname(group_eval_path))  # .../artifacts/.. -> run dir
    artifacts_dir = os.path.dirname(group_eval_path)
    logs = sorted(glob.glob(os.path.join(artifacts_dir, 'train_*.log')))
    if not logs:
        print(f"  [skip] {group_eval_path}: no train_*.log in {artifacts_dir}")
        return
    log_path = logs[-1]  # newest by timestamped filename -- matches what actually
    # produced this group_eval.json (an earlier rerun's log would've been overwritten
    # by group_eval.json's own last write, same file, so only the latest log is live)
    log_runs = parse_log_kappa(log_path)

    with open(group_eval_path) as f:
        data = json.load(f)

    n_patched, n_skipped, n_missing = 0, 0, 0
    for run_name, run in data.items():
        for g_name, g in run.get('groups', {}).items():
            for subj, v in g['subjects'].items():
                if 'kappa_tail' in v:
                    n_skipped += 1
                    continue
                tag = next((t for t in log_runs if t.endswith('_' + run_name)), None)
                if tag is None:
                    n_missing += 1
                    continue
                total, series = log_runs[tag]
                tail_start = max(0, total - 10)
                tail_vals = [k for e, k in series if e > tail_start]
                if not tail_vals:
                    n_missing += 1
                    continue
                v['kappa_tail'] = float(sum(tail_vals) / len(tail_vals))
                v['kappa_last'] = float(series[-1][1])
                n_patched += 1
            kt = [v['kappa_tail'] for v in g['subjects'].values() if 'kappa_tail' in v]
            kl = [v['kappa_last'] for v in g['subjects'].values() if 'kappa_last' in v]
            if kt:
                g['mean_kappa_tail'] = float(sum(kt) / len(kt))
                g['mean_kappa_last'] = float(sum(kl) / len(kl))

    if n_patched:
        with open(group_eval_path, 'w') as f:
            json.dump(data, f, indent=2)
    print(f"  {group_eval_path}: patched={n_patched} already_had_kappa={n_skipped} "
          f"no_log_match={n_missing}  (log={os.path.basename(log_path)})")


if __name__ == '__main__':
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'output/*/finetune/*/*/artifacts/group_eval.json'
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no group_eval.json matched {pattern!r}")
        sys.exit(1)
    for p in paths:
        backfill_one(p)
