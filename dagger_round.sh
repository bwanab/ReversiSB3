#!/bin/zsh
# One DAgger round played with search (see CLAUDE.md, "Train of reasoning" steps 11-15):
#   1. collect positions a model reaches when it plays with pruned search (plus --explore
#      randomness), vs itself and Edax 4-8, from random and named openings
#   2. label them with Edax at TEACHER_DEPTH, every move scored
#   3. fine-tune BASE (a BC model) on all label files (labels_d*_all.npz)
#   4. evaluate: top move vs Edax 1-3, depth-5 top-3 pruned search vs Edax 2-8 (balanced and named
#      openings, eval_batch.py), and a head-to-head vs REF
#
# Usage: [N=positions] [EMPTIES=min-max] ./dagger_round.sh OUT BASE PLAY_MODEL TEACHER_DEPTH [COLLECT_SEARCH_DEPTH] [REF]
#   e.g. ./dagger_round.sh r256x12_sdag1 r256x12_bc r256x12_dagger 14 3 r256x12_dagger
#        N=300000 EMPTIES=21-40 ./dagger_round.sh r256x12_mid1 r256x12_bc r256x12_sdag1 16 3 r256x12_sdag1
#   N: positions to collect (default 1,000,000); EMPTIES: only collect positions in this range
# Steps whose output exists are skipped. Log: dagger_OUT.log. Needs the Edax server for step 4.

set -u
OUT=$1; BASE=$2; PLAY=$3; TD=$4; SD=${5:-3}; REF=${6:-$PLAY}
N=${N:-1000000}; RANGE=""
[ -n "${EMPTIES:-}" ] && RANGE="--min-empties ${EMPTIES%-*} --max-empties ${EMPTIES#*-}"
LOG=dagger_${OUT}.log
POS=positions_${OUT}.npy
LAB=labels_d${TD}_${OUT}_all.npz
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

say "DAgger round $OUT: play $PLAY with depth-$SD search, $N positions${EMPTIES:+ with $EMPTIES empties}, Edax depth-$TD labels, train from $BASE"
if [ ! -f $POS ]; then
  say "1. collect"
  uv run python collect_positions.py -m $PLAY -n $N -o $POS --search-depth $SD --search-top-k 3 \
      --explore 0.2 --opponents self:0.3,4:0.15,5:0.15,6:0.2,8:0.2 --exclude labels_d*_all.npz ${=RANGE} 2>&1 \
      | grep -E "Saved|Error|Traceback" | tee -a $LOG
  [ -f $POS ] || { say "collect failed"; exit 1; }
fi
if [ ! -f $LAB ]; then
  say "2. label at depth $TD"
  uv run python label_positions.py --positions $POS -n $N --depth $TD --every-move -w 12 -o $LAB 2>&1 \
      | grep -E "Saved|Error|Traceback" | tee -a $LOG
  [ -f $LAB ] || { say "labeling failed"; exit 1; }
fi
if [ ! -f models/${OUT}_CNN_test.zip ]; then
  say "3. train on $(ls labels_d*_all.npz | tr '\n' ' ')"
  uv run python edax_train.py --labels labels_d*_all.npz --base $BASE --model $OUT --train both \
      --policy-target graded --epochs 6 -lr 5e-5 --val-frac 0.02 2>&1 \
      | grep -E "dropped|train /|start:|epoch|saved|Error|Traceback" | tee -a $LOG
  [ -f models/${OUT}_CNN_test.zip ] || { say "training failed"; exit 1; }
fi
pgrep -f edax_server.py >/dev/null || { say "Edax server not running; skipping evaluation"; exit 0; }
say "4. evaluate"
for spec in "balanced:-e 200 --start-positions balanced_openings.npy" "named:-e 150 --start-positions opening_positions.npy"; do
  op=${spec%%:*}; args=${spec#*:}
  uv run python eval_batch.py -m ${OUT}_CNN_test -p 1,2,3 ${=args} 2>&1 | grep "Black wins" \
      | while read -r line; do say "eval $op | $line"; done
  uv run python eval_batch.py -m ${OUT}_CNN_test -p 2,3,4,5,6,7,8 ${=args} --search-depth 5 --search-top-k 3 \
      --search-prune-all 2>&1 | grep "Black wins" | while read -r line; do say "eval $op | $line"; done
done
h1=$(uv run python sb-play.py -m ${OUT}_CNN_test -r models/${REF}_CNN_test -o Model -e 400 --random-opening 8 --seed 42 2>&1 | grep "Model:")
h2=$(uv run python sb-play.py -m ${OUT}_CNN_test -r models/${REF}_CNN_test -o Model -e 300 --start-positions opening_positions.npy 2>&1 | grep "Model:")
say "h2h $OUT vs $REF | random: ${h1#*score } | named: ${h2#*score }"
say "round $OUT done"
