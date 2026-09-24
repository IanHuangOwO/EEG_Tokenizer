# SHU_MI -- password-locked, user fetching it themselves (2026-09-24)

SHU Multi-session motor imagery dataset (Ma J, Yang B, et al., Shanghai University),
figshare 19228725, CC BY 4.0. 25 subjects x 5 sessions (2-3 days apart), 100 trials
per session, left- vs right-hand MI, 4 s MI period per trial. BIDS-style metadata,
32ch EEG at 250 Hz (check task-motorimagery_channels.tsv / _eeg.json).

`./fetch.sh` (resumable) downloads everything except `mat_files.zip` (same data as the
EDFs, 1.4 GB) and `code_files.zip`. The events .tsv, channels and participants files
are plain. `edf_files.zip` and `events.zip` entries are AES-encrypted (zip method 99):
the password comes from the dataset author, Prof. Yang Banghua
(yangbanghua@shu.edu.cn), per the record description.

No loader yet -- write gen_metadata.py/loader.py once the password is in hand
(unzip with `7z x -p<password> edf_files.zip`).
