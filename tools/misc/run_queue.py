"""
Run a plan of jobs with a parallel limit and a CPU-thread cap, resumable. Replaces the ad-hoc bash
chains: this machine is CPU-bound, so every job gets OMP/MKL thread caps, and a queue's state
lives under output/queue/ (not /tmp), so a reboot or a kill loses nothing but the jobs in flight.

Plan file, one entry per line:
    job <name> :: <shell command>      # run from the repo root; <name> may contain '/'
    wait                               # barrier: every earlier job must finish first; stops the queue
                                       # if any job failed
    # comment
The file is re-read after each job starts, so jobs can be appended or reordered while it runs
(already-started jobs are never restarted). Job state: <state>/<name>.done|.failed|.log.
Rerunning the same plan skips .done jobs (delete a .done file to redo it; .failed jobs rerun).
Exits 1 if any job failed.

    python -m tools.misc.run_queue output/queue/tune.plan --max-parallel 2 --threads 8
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime


def read_plan(path):
    entries = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line == 'wait':
            entries.append(('wait', None))
        elif line.startswith('job ') and ' :: ' in line:
            name, cmd = line[4:].split(' :: ', 1)
            entries.append((name.strip(), cmd.strip()))
        else:
            raise ValueError(f'bad plan line: {line!r}')
    return entries


def log(msg):
    print(f'{datetime.now():%F %T} {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan')
    ap.add_argument('--max-parallel', type=int, default=1, dest='max_parallel')
    ap.add_argument('--threads', type=int, default=None, help='OMP/MKL threads per job (default: unset)')
    ap.add_argument('--state', default=None, help='default: output/queue/<plan file stem>/')
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    state = args.state or os.path.join(root, 'output', 'queue', os.path.splitext(os.path.basename(args.plan))[0])
    env = dict(os.environ)
    if args.threads:
        env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))

    running, started, failed, i = {}, set(), [], 0          # name -> Popen
    def reap():
        for name, p in list(running.items()):
            if p.poll() is not None:
                tag = 'done' if p.returncode == 0 else 'failed'
                open(os.path.join(state, f'{name}.{tag}'), 'w').write(f'{p.returncode}\n')
                log(f'{tag:6} {name} (exit {p.returncode})')
                if p.returncode != 0:
                    failed.append(name)
                del running[name]

    log(f'queue {args.plan}: max_parallel={args.max_parallel} threads={args.threads} state={state}')
    while True:
        entries = read_plan(args.plan)                    # re-read: appended jobs get picked up
        if i >= len(entries):
            break
        name, cmd = entries[i]
        if name == 'wait':
            while running:
                reap(); time.sleep(5)
            if failed:
                break
            i += 1
            continue
        if name in started or os.path.exists(os.path.join(state, f'{name}.done')):
            i += 1
            continue
        while len(running) >= args.max_parallel:
            reap(); time.sleep(5)
        path = os.path.join(state, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(f'{path}.failed'):
            os.remove(f'{path}.failed')
        argv = shlex.split(cmd)
        if argv and argv[0] == 'python':                  # the queue's own interpreter (the eeg_fm env)
            argv[0] = sys.executable
        running[name] = subprocess.Popen(argv, cwd=root, env=env,
                                         stdout=open(f'{path}.log', 'w'), stderr=subprocess.STDOUT)
        started.add(name)
        log(f'start  {name}')
        i += 1
    while running:
        reap(); time.sleep(5)
    if failed:
        log(f'queue ended: {len(failed)} failed ({", ".join(failed)})')
        sys.exit(1)
    log('queue finished')


if __name__ == '__main__':
    main()
