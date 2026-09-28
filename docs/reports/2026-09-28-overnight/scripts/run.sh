#!/bin/bash
# Overnight pipeline (2026-09-28), unattended. Every step checks the previous one's exit code and
# stops on failure; progress goes to output/queue/overnight/overnight.log.
#   A. tiny finest-skip run + analyses (its quick preset; band-error backbone_eval for no-trial / rankfix)
#   B. decide.py: finest vs graded by the pre-set card -> tiny winner
#   C. the winner's recipe on the small corpus (window_fraction 0.2), one pretrain, while the tiny
#      winner's 36 finetunes run (probe + stamp head, loso + few-shot, 3 datasets, finetune seeds 1-3)
#   D. small: quick analysis on the tiny corpus's held-out windows + the same 36 finetunes
#   E. summarize_small.py: the scale-up card
set -o pipefail
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer || exit 1
PY=/home/mamechin/anaconda3/envs/eeg_fm/bin/python
D=output/queue/overnight
LOG=$D/overnight.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
fail() { log "FAILED: $*"; exit 1; }

# ---- A
F=mesae_tiny_finestskip_s1
log "A1 pretrain $F"
$PY train_pretrain.py --config configs/runs/$F/pretrain.json > output/$F/pretrain.log 2>&1 || fail "pretrain $F (exit $?)"
[ -f output/$F/pretrain/checkpoint/last.pth ] || fail "no checkpoint for $F"
log "A2 analyses"
cat > $D/A2.plan <<EOF
job finest_quick :: python analysis_pretrain.py --run $F --preset quick --device cpu
job notrial_eval :: python analysis_pretrain.py --run mesae_tiny_notrial_s1 --panel backbone_eval --device cpu
job rankfix_eval :: python analysis_pretrain.py --run mesae_tiny_rankfix_s1 --panel backbone_eval --device cpu
EOF
$PY -m tools.misc.run_queue $D/A2.plan --max-parallel 2 --threads 8 --state $D/A2 >> $LOG 2>&1 || fail "A2 analyses (see $D/A2/*.log)"

# ---- B
log "B decide"
WIN=$($PY $D/decide.py | tail -1) || fail "decide.py"
log "B winner: $WIN"
TAG=$([ "$WIN" = "$F" ] && echo finestskip || echo graded)
S=mesae_small_${TAG}_s1

# ---- C
mkdir -p configs/runs/$S output/$S
$PY - "$WIN" "$S" <<'EOF' || fail "small config"
import json, sys
win, name = sys.argv[1], sys.argv[2]
c = json.load(open(f'configs/runs/{win}/pretrain.json'))
c['model_params']['MeSAE']['pretrain']['loss'].pop('mse_trial_weight', None)
c['preprocess_params']['window_fraction'] = 0.2
c['training_params']['pretrain'].update(model_name=name, output_path=f'{name}/pretrain', seed=1)
json.dump(c, open(f'configs/runs/{name}/pretrain.json', 'w'), indent=2)
EOF
$PY $D/ft_plan.py $WIN > $D/ft_tiny.plan || fail "ft_plan tiny"
log "C finetunes $WIN (36 jobs, background) + pretrain $S"
( $PY -m tools.misc.run_queue $D/ft_tiny.plan --max-parallel 1 --threads 6 --state $D/ft_tiny >> $D/ft_tiny.log 2>&1; echo $? > $D/ft_tiny.exit ) &
FT_PID=$!
$PY train_pretrain.py --config configs/runs/$S/pretrain.json > output/$S/pretrain.log 2>&1 || fail "pretrain $S (exit $?)"
[ -f output/$S/pretrain/checkpoint/last.pth ] || fail "no checkpoint for $S"

# ---- D
log "D analysis $S (tiny held-out windows) + finetunes $S"
$PY analysis_pretrain.py --run $S --preset quick --device cpu --config $D/eval_tiny_windows.json > $D/analysis_$S.log 2>&1 \
  || fail "analysis $S (see $D/analysis_$S.log)"
wait $FT_PID
[ "$(cat $D/ft_tiny.exit)" = 0 ] || fail "finetunes $WIN (see $D/ft_tiny.log)"
$PY $D/ft_plan.py $S > $D/ft_small.plan || fail "ft_plan small"
$PY -m tools.misc.run_queue $D/ft_small.plan --max-parallel 2 --threads 8 --state $D/ft_small >> $D/ft_small.log 2>&1 \
  || fail "finetunes $S (see $D/ft_small.log)"

# ---- E
log "E scale-up card"
$PY $D/summarize_small.py $WIN $S >> $LOG 2>&1 || fail "summarize_small.py"
log "DONE: output/reports/overnight/decision.md, output/reports/overnight/small_card.md"
