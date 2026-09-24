#!/bin/bash
# Resumable. figshare 19228725; skips mat_files.zip (same data as edf_files.zip) and code_files.zip.
cd "$(dirname "$0")/raw"
while read url name; do wget -q -c -O "$name" "$url" || echo "FAILED $name"; done < urls.txt
echo done
