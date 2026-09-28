#!/bin/bash
# p100 seed-1 analyses: remaining pretrain panels now; finetune analysis (vs patch-50 seed 1) once both
# backbones' 18 z-probe finetunes have a .done/.failed file.
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/patchlen
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$PY analysis_pretrain.py --run mesae_tiny_p100_s16_s1 --panel stamp_templates --panel stamp_duplicates \
    --panel stamp_distribution --panel snapshot --panel codebook --device cpu --max-trials 100 > $D/p100_s1_pretrain_analysis.log 2>&1
pre=$?; echo "pretrain analysis exit $pre"
n() { ls $D/ft_$1/*.done $D/ft_$1/*.failed 2>/dev/null | wc -l; }
until [ "$(n mesae_tiny_notrial_s1)" -ge 18 ] && [ "$(n mesae_tiny_p100_s16_s1)" -ge 18 ]; do
  pgrep -f "bash output/queue/patchlen/finetune.sh" >/dev/null || { echo "finetune driver gone before seed-1 finetunes finished"; exit 1; }
  sleep 60
done
$PY analysis_finetune.py --group p50=mesae_tiny_notrial_s1 --group p100=mesae_tiny_p100_s16_s1 --ref p50 \
    --head patch_probe --panel summary --panel report --panel time_weights --panel probe_maps --device cpu --out output/reports/patch_s1 > $D/p100_s1_finetune_analysis.log 2>&1
ft=$?; echo "finetune analysis exit $ft"
exit $(( pre + ft ))
