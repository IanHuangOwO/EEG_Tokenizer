#!/bin/bash
# Test A (2026-09-29): does pooling pairs of patch-50 z tokens (39 -> 19 per window, stride 50 like patch 100)
# reproduce patch 100's MI few-shot gain? Starts after the patch-75 finetune driver exits.
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/testA
log() { echo "$(date '+%F %T') $*" | tee -a $D/run.log; }
log "wait for the patch-75 finetune driver"
while pgrep -f "bash output/queue/patch75/finetune.sh" >/dev/null; do sleep 60; done
log "patch-75 driver gone; start"
fails=0
for bb in mesae_tiny_notrial_s1 mesae_tiny_p50_s16_s2 mesae_tiny_p50_s16_s3; do
  OMP_NUM_THREADS=8 $PY -c "
import json, torch
from tools.analysis.ridge_probe import ridge_probe
c = json.load(open('output/$bb/pretrain/artifacts/config.json'))
ridge_probe(c, 'output/$bb/pretrain/checkpoint/last.pth', 'output/$bb/pretrain/analysis/ridge_probe_pool2.json', pool=2)
" >> $D/ridge_$bb.log 2>&1 || { log "FAILED ridge $bb"; fails=$((fails+1)); }
  $PY $D/ft_plan.py $bb > $D/ft_$bb.plan
  $PY -m tools.misc.run_queue $D/ft_$bb.plan --max-parallel 2 --threads 6 --state $D/ft_$bb >> $D/run.log 2>&1 || { log "FAILED finetunes $bb"; fails=$((fails+1)); }
  log "done $bb"
done
log "ALL DONE, $fails failure(s)"
exit $fails
