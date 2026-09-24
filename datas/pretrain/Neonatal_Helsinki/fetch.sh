#!/bin/bash
# Resumable: re-run to continue. Zenodo record 4940267 (86 files, 4.34 GB).
cd "$(dirname "$0")/raw"
paste - - < urls.txt | while read url out; do
  wget -q -c -O "${out#out=}" "$url" || echo "FAILED ${out#out=}"
done
echo "done $(ls *.edf | wc -l) edf"
