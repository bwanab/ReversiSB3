#!/bin/zsh
# Teacher swap: relabel existing positions with Egaroucid and retrain, changing only the teacher
# (see CLAUDE.md, "Train of reasoning" step 23):
#   1. pick N distinct positions in an empties range from the Edax label files (labels_d*_all.npz)
#   2. label them with Egaroucid at LEVEL, every move scored
#   3. train BASE (a BC model) on the new labels first, then all Edax label files: edax_train.py keeps
#      the first copy of a repeated position, so the Egaroucid labels replace Edax's for those positions
#   4. evaluate with MCTS (step 26) vs Edax 8-12 and Egaroucid 4-10 (balanced and
#      named openings) and head-to-head vs REF
#
# Usage: [N=positions] [EMPTIES=min-max] ./teacher_round.sh OUT BASE LEVEL REF
#   e.g. N=500000 EMPTIES=21-40 ./teacher_round.sh r256x12_egt1 r256x12_bc 12 r256x12_mid1
# Steps whose output exists are skipped. Log: teacher_OUT.log. Needs the Edax server for step 4.

set -u
OUT=$1; BASE=$2; LEVEL=$3; REF=$4
N=${N:-500000}; EMPTIES=${EMPTIES:-21-40}
LOG=teacher_${OUT}.log
POS=positions_${OUT}.npy
LAB=labels_eg${LEVEL}_${OUT}_all.npz
STRONG="--mcts-sims 400 --solve-empties 18 --leaf-solve-empties 16"   # step 26 (was the step-20 negamax)
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

say "teacher round $OUT: $N positions with $EMPTIES empties relabeled by Egaroucid level $LEVEL, train from $BASE"
if [ ! -f $POS ]; then
  say "1. select positions"
  uv run python select_positions.py --labels labels_d*_all.npz --min-empties ${EMPTIES%-*} \
      --max-empties ${EMPTIES#*-} -n $N -o $POS 2>&1 | tee -a $LOG
  [ -f $POS ] || { say "select failed"; exit 1; }
fi
if [ ! -f $LAB ]; then
  say "2. label with Egaroucid level $LEVEL"
  uv run python label_positions.py --positions $POS -n $N --teacher egaroucid --depth $LEVEL -w 12 -o $LAB 2>&1 \
      | grep -E "Saved|Error|Traceback" | tee -a $LOG
  [ -f $LAB ] || { say "labeling failed"; exit 1; }
fi
if [ ! -f models/${OUT}_CNN_test.zip ]; then
  say "3. train on $LAB first, then $(ls labels_d*_all.npz | tr '\n' ' ')"
  uv run python edax_train.py --labels $LAB labels_d*_all.npz --base $BASE --model $OUT --train both \
      --policy-target graded --epochs 6 -lr 5e-5 --val-frac 0.02 2>&1 \
      | grep -E "dropped|train /|start:|epoch|saved|Error|Traceback" | tee -a $LOG
  [ -f models/${OUT}_CNN_test.zip ] || { say "training failed"; exit 1; }
fi
pgrep -f "edax_server.py" >/dev/null || [ -S /tmp/edax_server.sock ] || { say "Edax server not running; skipping evaluation"; exit 0; }
say "4. evaluate (MCTS 400)"
for spec in "balanced:-e 200 --start-positions balanced_openings.npy" "named:-e 150 --start-positions opening_positions.npy"; do
  op=${spec%%:*}; args=${spec#*:}
  uv run python eval_batch.py -m ${OUT}_CNN_test -p 8,9,10,12 ${=args} ${=STRONG} 2>&1 | grep "Black wins" \
      | while read -r line; do say "eval $op | $line"; done
  uv run python eval_batch.py -m ${OUT}_CNN_test -o egaroucid -p 4,6,8,10 ${=args} ${=STRONG} 2>&1 | grep "Black wins" \
      | while read -r line; do say "eval $op | $line"; done
done
h1=$(uv run python sb-play.py -m ${OUT}_CNN_test -r models/${REF}_CNN_test -o Model -e 400 --start-positions balanced_openings.npy 2>&1 | grep "Model:")
h2=$(uv run python sb-play.py -m ${OUT}_CNN_test -r models/${REF}_CNN_test -o Model -e 300 --start-positions opening_positions.npy 2>&1 | grep "Model:")
say "h2h $OUT vs $REF | balanced: ${h1#*score } | named: ${h2#*score }"
say "teacher round $OUT done"
