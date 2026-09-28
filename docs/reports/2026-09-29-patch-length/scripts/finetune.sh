#!/bin/bash
# Finetune driver for the patch-length card. Seed-1 backbones first (their checkpoints exist); each new backbone
# waits for its pretrain job's .done (or .failed -> skipped) in output/queue/patchlen/pretrain/. Per backbone:
# 18 z-probe finetunes (2 in parallel) then backbone_eval / ridge_probe / attention_range for new backbones.
# A failed backbone does not stop the others; the exit code counts failures.
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/patchlen
log() { echo "$(date '+%F %T') $*" | tee -a $D/finetune.log; }
fails=0
for bb in mesae_tiny_notrial_s1 mesae_tiny_p100_s16_s1 mesae_tiny_p100_s16_s2 mesae_tiny_p100_s16_s3 mesae_tiny_p50_s16_s2 mesae_tiny_p50_s16_s3; do
  case $bb in mesae_tiny_notrial_s1|mesae_tiny_p100_s16_s1) ;;
    *) log "wait for pretrain $bb"
       until [ -e $D/pretrain/$bb.done ] || [ -e $D/pretrain/$bb.failed ]; do sleep 60; done
       if [ -e $D/pretrain/$bb.failed ]; then log "SKIP $bb: pretrain failed"; fails=$((fails+1)); continue; fi ;;
  esac
  $PY $D/ft_plan.py $bb > $D/ft_$bb.plan || { log "FAILED ft_plan $bb"; fails=$((fails+1)); continue; }
  case $bb in mesae_tiny_notrial_s1|mesae_tiny_p100_s16_s1) ;;
    *) echo "job analysis :: python analysis_pretrain.py --run $bb --panel backbone_eval --panel ridge_probe --panel attention_range --device cpu" >> $D/ft_$bb.plan ;;
  esac
  log "finetunes $bb"
  $PY -m tools.misc.run_queue $D/ft_$bb.plan --max-parallel 2 --threads 6 --state $D/ft_$bb >> $D/finetune.log 2>&1
  rc=$?; log "done $bb (exit $rc)"; [ $rc = 0 ] || fails=$((fails+1))
done
log "ALL DONE, $fails backbone(s) with failures"
exit $fails
