"""
Resumable raw-data downloads for new datasets, tracked per dataset in <dest>/FETCH_STATUS.json.

    python -m tools.misc.fetch_datasets nemar <nemar id> <dest dir> [--with-derivatives]
    python -m tools.misc.fetch_datasets moabb <MOABB class> <dest dir>
    python -m tools.misc.fetch_datasets status          # every datas/*/*/FETCH_STATUS.json as a table

nemar: files from data.nemar.org's manifest (every file with size, sha256 and a direct URL) into <dest>/raw/<path>;
a file already on disk with the right size and checksum is skipped, a new one is written to .part, checked, then
renamed. BIDS derivatives are skipped unless asked for. moabb: every subject through MOABB into <dest>/raw/ (the
layout IO/loader.py's MoabbLoader reads), retried on connection drops. Both only download: metadata, loader and
compile are the dataset-adding steps that follow (docs/agents/adding-a-dataset.md). Exit code 1 if anything failed.
"""
import glob
import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RETRIES = 8
WORKERS = 8
socket.setdefaulttimeout(300)  # urlretrieve has no timeout of its own: a dead connection hung NMT_Clinical for hours


def _status(dest, **kw):
    os.makedirs(dest, exist_ok=True)
    p = os.path.join(dest, 'FETCH_STATUS.json')
    s = json.load(open(p)) if os.path.exists(p) else {}
    s.update(kw, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    tmp = p + '.tmp'
    json.dump(s, open(tmp, 'w'), indent=1)
    os.replace(tmp, p)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _get_json(url):
    for a in range(RETRIES):
        try:
            return json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as e:
            print(f'  retry {a + 1} {url}: {e!r}', flush=True)
            time.sleep(20 * (a + 1))
    raise RuntimeError(f'cannot fetch {url}')


def fetch_nemar(ds_id, dest, derivatives=False):
    info = _get_json(f'https://data.nemar.org/{ds_id}/')
    ver = next(v for v in info['versions'] if v['version'] == info['latest'])
    files = _get_json(f"https://data.nemar.org{ver['manifest_url']}")
    if not derivatives:
        files = [f for f in files if not f['path'].startswith('derivatives/')]
    total = sum(f['size'] for f in files)
    _status(dest, source='nemar', id=ds_id, version=info['latest'], state='downloading', files_total=len(files),
            bytes_total=total)
    todo = [f for f in files if not (os.path.exists(p := os.path.join(dest, 'raw', f['path']))
                                     and os.path.getsize(p) == f['size'])]
    done_b, failed, n_done = total - sum(f['size'] for f in todo), [], len(files) - len(todo)
    lock = threading.Lock()

    def one(f):
        out = os.path.join(dest, 'raw', f['path'])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for a in range(RETRIES):
            try:
                for url in (f['url'], f['bytes_url']):
                    try:
                        urllib.request.urlretrieve(url, out + '.part')
                        break
                    except Exception:
                        if url == f['bytes_url']:
                            raise
                if os.path.getsize(out + '.part') != f['size'] or \
                        (f.get('checksum_algorithm') == 'sha256' and _sha256(out + '.part') != f['checksum']):
                    raise IOError('size/checksum mismatch')
                os.replace(out + '.part', out)
                return True
            except Exception as e:
                print(f"  {f['path']} attempt {a + 1}: {e!r}", flush=True)
                time.sleep(15 * (a + 1))
        return False

    # ponytail: NEMAR's S3 gives ~0.35 MB/s per connection, ~2.2 MB/s over 8; datasets stay sequential (the queue),
    # files within one dataset go 8 at a time.
    with ThreadPoolExecutor(WORKERS) as ex:
        for f, ok in zip(todo, ex.map(one, todo)):
            with lock:
                if ok:
                    done_b += f['size']
                    n_done += 1
                else:
                    failed.append(f['path'])
                if (n_done + len(failed)) % 50 == 0:
                    _status(dest, files_done=n_done, bytes_done=done_b, failed=len(failed))
                    print(f'  {ds_id}: {n_done}/{len(files)} files, {done_b / 1e9:.1f}/{total / 1e9:.1f} GB', flush=True)
    _status(dest, files_done=n_done, bytes_done=done_b, failed=len(failed),
            failed_files=failed[:50], state='failed' if failed else 'complete')
    return not failed


def fetch_moabb_raw(cls, dest):
    sys.path.insert(0, os.getcwd())
    from IO.loader import moabb_dataset
    ds = moabb_dataset(cls, dest)
    raw_dir = os.path.abspath(os.path.join(dest, 'raw'))
    subs = list(ds.subject_list)
    _status(dest, source='moabb', id=cls, state='downloading', subjects_total=len(subs))
    failed = []
    for n, s in enumerate(subs):
        for a in range(RETRIES):
            try:
                ds.data_path(s, path=raw_dir)
                break
            except Exception as e:
                print(f'  subject {s} attempt {a + 1}: {e!r}', flush=True)
                time.sleep(30)
        else:
            failed.append(s)
        _status(dest, subjects_done=n + 1 - len(failed), failed=len(failed), failed_subjects=failed)
        print(f'  {cls}: subject {s} ({n + 1}/{len(subs)})', flush=True)
    _status(dest, state='failed' if failed else 'complete')
    return not failed


def status():
    rows = []
    for p in sorted(glob.glob('datas/*/*/FETCH_STATUS.json')):
        s = json.load(open(p))
        prog = (f"{s.get('bytes_done', 0) / 1e9:.1f}/{s.get('bytes_total', 0) / 1e9:.1f} GB" if s.get('source') == 'nemar'
                else f"{s.get('subjects_done', 0)}/{s.get('subjects_total', '?')} subjects")
        rows.append(f"| {os.path.dirname(p)} | {s.get('source')} {s.get('id')} | {s.get('state')} | {prog} | "
                    f"{s.get('failed', 0)} | {s.get('updated')} |")
    print('| dataset | source | state | progress | failed | updated |\n|---|---|---|---|---|---|\n' + '\n'.join(rows))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['status']:
        status()
        sys.exit(0)
    kind, ident, dest = a[0], a[1], a[2]
    os.makedirs(dest, exist_ok=True)
    ok = fetch_nemar(ident, dest, '--with-derivatives' in a) if kind == 'nemar' else fetch_moabb_raw(ident, dest)
    sys.exit(0 if ok else 1)
