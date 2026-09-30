#!/bin/bash
# Heads on small seeds 2-3: combined, z probe, stamp head; 6 cells x finetune seeds 1-3. P300 loso jobs first.
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/small_heads
: > $D/all.plan
for bb in mesae_small_p50_s16_s2 mesae_small_p50_s16_s3; do
  for head in combined patch_probe cw_stamp; do
    $PY $D/ft_plan.py $bb $head | sed "s/^job /job ${bb}_${head}_/" >> $D/all.plan
  done
done
{ grep "BNCI2014008_loso" $D/all.plan; grep -v "BNCI2014008_loso" $D/all.plan; } > $D/ordered.plan
$PY -m tools.misc.run_queue $D/ordered.plan --max-parallel 2 --threads 6 --state $D/state >> $D/run.log 2>&1
rc=$?
echo "$(date '+%F %T') ALL DONE exit $rc" >> $D/run.log
exit $rc
