#!/bin/bash
# Stamp hidden diagnostic (docs/cards/2026-09-29-stamp-hidden.md): stats + loso / few-shot ridge on u, and the z
# baselines that are missing. One step per python call; every step's exit code is logged.
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/stamp_hidden
log() { echo "$(date '+%F %T') $*" | tee -a $D/run.log; }
fails=0
step() {  # step <backbone> <out json> <python call>
  local bb=$1 out=output/$1/pretrain/analysis/$2
  if [ -f $out ]; then log "skip $bb $2 (exists)"; return; fi
  OMP_NUM_THREADS=16 $PY -c "
import json
from tools.analysis.ridge_probe import *
c = json.load(open('output/$bb/pretrain/artifacts/config.json'))
ck = 'output/$bb/pretrain/checkpoint/last.pth'
$3
" >> $D/$bb.log 2>&1 && log "done $bb $2" || { log "FAILED $bb $2 (exit $?)"; fails=$((fails+1)); }
}
step mesae_small_p50_s16_s1 ridge_probe.json "ridge_probe(c, ck, 'output/mesae_small_p50_s16_s1/pretrain/analysis/ridge_probe.json')"
for bb in mesae_tiny_p50_s16_s1 mesae_tiny_p50_s16_s2 mesae_tiny_p50_s16_s3 mesae_small_p50_s16_s1; do
  step $bb stamp_hidden_stats.json "stamp_hidden_stats(c, ck, 'output/$bb/pretrain/analysis/stamp_hidden_stats.json')"
  step $bb ridge_probe_stamp_hidden.json "ridge_probe(c, ck, 'output/$bb/pretrain/analysis/ridge_probe_stamp_hidden.json', feature='stamp_hidden')"
  step $bb fewshot_ridge.json "fewshot_ridge(c, ck, 'output/$bb/pretrain/analysis/fewshot_ridge.json')"
  step $bb fewshot_ridge_stamp_hidden.json "fewshot_ridge(c, ck, 'output/$bb/pretrain/analysis/fewshot_ridge_stamp_hidden.json', feature='stamp_hidden')"
done
log "ALL DONE, $fails failure(s)"
exit $fails
