"""Resumable fetch through MOABB into raw/ (MOABB's own layout). Re-run to continue;
an already-complete file is skipped. 54 subjects x 2 sessions, ~0.6 GB each."""
import os
import time
from moabb.datasets import Lee2019_MI

raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
for s in Lee2019_MI().subject_list:
    for attempt in range(10):  # wasabi mirror drops connections (read timeouts)
        try:
            Lee2019_MI().data_path(s, path=raw)
            break
        except Exception as e:
            print("subject", s, "attempt", attempt, "failed:", e, flush=True)
            time.sleep(30)
    print("subject", s, "ok", flush=True)
