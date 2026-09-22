# TUAB -- gated, not fetched (2026-09-22)

Temple University Abnormal EEG Corpus. Table 14: 2383 subjects, 21ch, 250Hz,
10s trials, 53604 trials, normal/abnormal classification.

Access: registration form (isip.piconepress.com/projects/nedc/forms/
tuh_eeg.pdf), email signed copy to help@nedcdata.org, subject "Download The
TUH EEG Corpus". Approval ~24-48h, then ssh key + rsync (no plain
username/password). Rsync path once approved:
data/tuh_eeg/tuh_eeg_abnormal/v3.0.1

No public mirror found. User needs to submit the form; scaffolding
(gen_metadata.py/loader.py) can be written once access is granted or
against the corpus's documented format if useful sooner.
