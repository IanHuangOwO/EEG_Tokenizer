#!/bin/bash
# Compass trial windows: recompile BNCI2014001 / 004 / 008 (verification runs after each), then the 36
# finetunes per backbone (tiny winner, small) as label 'cw', then the old-vs-new comparison.
set -o pipefail
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/overnight
LOG=$D/windows.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
fail() { log "FAILED: $*"; exit 1; }
for DS in BNCI2014001 BNCI2014004 BNCI2014008; do
  log "compile $DS"
  OMP_NUM_THREADS=8 $PY cache_dataset.py --config configs/compile.json --dataset $DS > $D/compile_$DS.log 2>&1 \
    || fail "compile $DS (see $D/compile_$DS.log)"
done
FT_LABEL=cw $PY $D/ft_plan.py mesae_tiny_notrial_s1 mesae_small_graded_s1 > $D/ft_cw.plan || fail "ft_plan"
log "finetunes (72 jobs)"
$PY -m tools.misc.run_queue $D/ft_cw.plan --max-parallel 2 --threads 8 --state $D/ft_cw >> $LOG 2>&1 || fail "finetunes (see $D/ft_cw/)"
$PY $D/compare_windows.py mesae_tiny_notrial_s1 mesae_small_graded_s1 >> $LOG 2>&1 || fail "compare_windows.py"
log "DONE: output/reports/overnight/compass_windows.md"
