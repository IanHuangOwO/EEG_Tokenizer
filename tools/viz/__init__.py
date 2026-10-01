"""
viz/ — plotting only (extract.py, stamp_plots.py, topomap.py, timeseries.py, codebook.py,
snapshot.py). Config/model/dataset orchestration helpers live in analysis/, not here.
"""
import re

_ROWS = ['Fp', 'AF', 'F', 'FC', 'C', 'CP', 'P', 'PO', 'O', 'I']
_ROW_ALIAS = {'FT': 'FC', 'T': 'C', 'TP': 'CP'}


def mirror_order(channel_names):
    """Standard channel order for channel-axis plots (EEGLAB-style mirrored): left hemisphere front to back (lateral
    to medial within an electrode row), the midline front to back, then the right hemisphere back to front (medial to
    lateral), so homologous channels sit symmetric about the midline block. Names outside the 10-10 / 10-20 system
    keep their given order at the end. Returns indices into channel_names."""
    left, mid, right, other = [], [], [], []
    for i, n in enumerate(channel_names):
        m = re.fullmatch(r'([A-Za-z]+?)(\d+|z|Z)', str(n))
        prefix = m and _ROW_ALIAS.get(m.group(1), m.group(1))
        if not m or prefix not in _ROWS:
            other.append(i)
            continue
        row, num = _ROWS.index(prefix), m.group(2)
        if num in 'zZ':
            mid.append((row, i))
        elif int(num) % 2:
            left.append((row, -int(num), i))
        else:
            right.append((-row, int(num), i))
    return [k[-1] for k in sorted(left)] + [k[-1] for k in sorted(mid)] + [k[-1] for k in sorted(right)] + other


if __name__ == '__main__':
    names = ['Cz', 'C3', 'C4', 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'T8', 'O1', 'O2', 'Oz', 'P9', 'P10', 'EOG']
    got = [names[i] for i in mirror_order(names)]
    assert got == ['Fp1', 'F7', 'F3', 'T7', 'C3', 'P9', 'O1', 'Fz', 'Cz', 'Oz', 'O2', 'P10', 'C4', 'T8', 'F4', 'F8',
                   'Fp2', 'EOG'], got
    print('mirror_order ok')
