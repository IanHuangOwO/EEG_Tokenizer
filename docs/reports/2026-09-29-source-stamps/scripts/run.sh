#!/bin/bash
# Source-stamps pilot (docs/cards/2026-09-29-source-stamps.md): pretrain K4 + baseline stamp head on p50_s2 in
# parallel; then backbone_eval + topographies + z probe / stamp head finetunes on K4. Exit codes logged per step.
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/source_stamps
K4=mesae_tiny_p50_s16_k4_s2
log() { echo "$(date '+%F %T') $*" | tee -a $D/run.log; }
fails=0
log "start pretrain $K4 + baseline stamp head on mesae_tiny_p50_s16_s2"
OMP_NUM_THREADS=10 $PY train_pretrain.py --config configs/runs/$K4/pretrain.json > $D/pretrain.log 2>&1 &
PRE=$!
$PY $D/ft_plan.py mesae_tiny_p50_s16_s2 cw_stamp > $D/base.plan
$PY -m tools.misc.run_queue $D/base.plan --max-parallel 1 --threads 6 --state $D/base >> $D/run.log 2>&1 \
  && log "done baseline stamp head" || { log "FAILED baseline stamp head"; fails=$((fails+1)); }
wait $PRE; rc=$?
if [ $rc -ne 0 ] || [ ! -f output/$K4/pretrain/checkpoint/last.pth ]; then
  log "FAILED pretrain (exit $rc)"; tail -5 $D/pretrain.log | tee -a $D/run.log; exit 1
fi
log "done pretrain"
OMP_NUM_THREADS=10 $PY analysis_pretrain.py --run $K4 --panel backbone_eval >> $D/analysis.log 2>&1 \
  && log "done backbone_eval" || { log "FAILED backbone_eval"; fails=$((fails+1)); }
$PY $D/topo.py $K4 >> $D/analysis.log 2>&1 && log "done topographies" || { log "FAILED topographies"; fails=$((fails+1)); }
for head in patch_probe cw_stamp; do
  $PY $D/ft_plan.py $K4 $head > $D/k4_$head.plan
  $PY -m tools.misc.run_queue $D/k4_$head.plan --max-parallel 2 --threads 6 --state $D/k4_$head >> $D/run.log 2>&1 \
    && log "done K4 $head" || { log "FAILED K4 $head"; fails=$((fails+1)); }
done
log "ALL DONE, $fails failure(s)"
exit $fails
