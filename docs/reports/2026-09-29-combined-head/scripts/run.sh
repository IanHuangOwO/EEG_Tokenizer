#!/bin/bash
# Combined head (stamp_power + latent_signed) on the 3 patch-50 seeds, plus the stamp-head baseline on seed 3.
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/combined
log() { echo "$(date '+%F %T') $*" | tee -a $D/run.log; }
fails=0
run() {  # run <backbone> <head>
  $PY $D/ft_plan.py $1 $2 > $D/$1_$2.plan
  $PY -m tools.misc.run_queue $D/$1_$2.plan --max-parallel 2 --threads 6 --state $D/$1_$2 >> $D/run.log 2>&1 \
    && log "done $1 $2" || { log "FAILED $1 $2"; fails=$((fails+1)); }
}
for bb in mesae_tiny_p50_s16_s1 mesae_tiny_p50_s16_s2 mesae_tiny_p50_s16_s3; do run $bb combined; done
run mesae_tiny_p50_s16_s3 cw_stamp
log "ALL DONE, $fails failure(s)"
exit $fails
