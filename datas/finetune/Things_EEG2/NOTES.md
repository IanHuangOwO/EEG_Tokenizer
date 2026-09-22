# Things-EEG2 -- skipped for now (2026-09-22)

Open access, no gate: NEMAR nm000232 (nemar.org/dataset/nm000232), S3-backed,
range-resumable HTTPS/DataLad/git-annex.

10 subjects, 4 sessions each, 63ch (10-10), 1000 Hz, RSVP (5 Hz, 200ms SOA).
~32,540 train trials (16,540 distinct images) + ~16,000 test trials (200
held-out images, 80 repeats each). Task: 200-way image retrieval/matching,
NOT a simple N-class label like every other dataset in datas/finetune/ --
would need real pipeline changes (image-identity label scheme, no existing
multi-way-retrieval support in this repo) before a loader makes sense.

Also large: 241.5 GB total.

Skipped by user request 2026-09-22, revisit as its own task later.
