#!/bin/bash
# After the tiny combined-head driver exits: if the P300 loso gate passes, run z probe, stamp head and combined head
# on mesae_small_p50_s16_s1 (parents rerun too, so all three use the current code and caches).
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/combined
log() { echo "$(date '+%F %T') [small] $*" | tee -a $D/run.log; }
while pgrep -f "bash output/queue/combined/run.sh" > /dev/null; do sleep 30; done
grep -q "ALL DONE, 0 failure" $D/run.log || { log "tiny driver did not finish cleanly; small run skipped"; exit 1; }
$PY $D/p300_gate.py >> $D/run.log 2>&1 || { log "P300 gate failed; small run skipped"; exit 0; }
fails=0
for head in combined patch_probe cw_stamp; do
  $PY $D/ft_plan.py mesae_small_p50_s16_s1 $head > $D/small_$head.plan
  $PY -m tools.misc.run_queue $D/small_$head.plan --max-parallel 2 --threads 6 --state $D/small_$head >> $D/run.log 2>&1 \
    && log "done $head" || { log "FAILED $head"; fails=$((fails+1)); }
done
log "ALL DONE, $fails failure(s)"
exit $fails
