#!/bin/zsh
# Full training course for a ResNet from scratch, using what worked so far (see CLAUDE.md):
#   1. BC on the human/Edax games dataset (planes, symmetry augmentation, cosine LR)
#   2. Fine-tune on Edax depth-12 every-move labels (graded policy + Edax-score value)
#   3. Evaluate: top move and depth-2 top-3 search vs Edax, from random and named openings,
#      plus a head-to-head against a reference model
#   4. One DAgger round: collect positions the model reaches, label them, redo stage 2 on all
#      labels, evaluate again
# No PPO (it degraded best play) and no standard-opening training (memorization).
#
# Usage: ./train_course.sh NAME CHANNELS BLOCKS [REFERENCE_MODEL]
#   e.g. ./train_course.sh r192x10 192 10 edax_dagger1
# Models: models/NAME_bc_CNN_test.zip, NAME_edax, NAME_dagger. Log: course_NAME.log
# Stages whose model already exists are skipped, so the script can be rerun after a failure.
# Needs the Edax server for evaluations (uv run python edax_server.py).

set -u
NAME=$1; CH=$2; BL=$3; REF=${4:-edax_dagger1}
LABELS=(labels_d12_100k_all.npz labels_d12_1m_all.npz labels_d12_rest_all.npz labels_d12_dagger1_all.npz)
LOG=course_${NAME}.log
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
have() { [ -f models/$1_CNN_test.zip ]; }

evaluate() {   # evaluate MODEL
  local m=$1
  pgrep -f edax_server.py >/dev/null || { say "Edax server not running; skipping evaluation of $m"; return; }
  for mode in top search; do
    local opts=""; [ $mode = search ] && opts="--search-depth 2 --search-top-k 3"
    for d in 1 2 3 4; do
      [ $mode = top ] && [ $d = 4 ] && continue
      local r1=$(uv run python sb-play.py -m ${m}_CNN_test -e 200 -o Edax -p $d --random-opening 8 --seed 42 ${=opts} 2>&1 | grep -i "black wins")
      local r2=$(uv run python sb-play.py -m ${m}_CNN_test -e 150 -o Edax -p $d --start-positions opening_positions.npy ${=opts} 2>&1 | grep -i "black wins")
      say "eval $m | $mode | edax-$d | random: ${r1#*: } | named: ${r2#*: }"
    done
  done
  local h1=$(uv run python sb-play.py -m ${m}_CNN_test -r models/${REF}_CNN_test -o Model -e 400 --random-opening 8 --seed 42 2>&1 | grep "Model:")
  local h2=$(uv run python sb-play.py -m ${m}_CNN_test -r models/${REF}_CNN_test -o Model -e 300 --start-positions opening_positions.npy 2>&1 | grep "Model:")
  say "h2h $m vs $REF | random: ${h1#*score } | named: ${h2#*score }"
}

edax_stage() {  # edax_stage OUTPUT LABEL_FILES...
  local out=$1; shift
  uv run python edax_train.py --labels "$@" --base ${NAME}_bc --model $out --train both \
      --policy-target graded --epochs 6 -lr 5e-5 --val-frac 0.02 2>&1 \
      | grep -E "dropped|train /|start:|epoch|saved|Error|Traceback" | tee -a $LOG
}

say "course $NAME: ResNet ${CH}x${BL}, reference $REF"

if ! have ${NAME}_bc; then
  say "stage 1: BC"
  uv run python bc_train.py --dataset combined_bc_dataset.pkl --model ${NAME}_bc --epochs 20 \
      --arch resnet --channels $CH --blocks $BL --augment --lr-schedule cosine 2>&1 \
      | grep -aE "Parameters|Summary|Val Acc|Error|Traceback" | tee -a $LOG
  have ${NAME}_bc || { say "stage 1 failed"; exit 1; }
fi

if ! have ${NAME}_edax; then
  say "stage 2: Edax labels"
  edax_stage ${NAME}_edax $LABELS
  have ${NAME}_edax || { say "stage 2 failed"; exit 1; }
  say "stage 3: evaluate ${NAME}_edax"
  evaluate ${NAME}_edax
fi

if ! have ${NAME}_dagger; then
  say "stage 4: DAgger round"
  pos=dagger_${NAME}_positions.npy; lab=labels_d12_dagger_${NAME}_all.npz
  [ -f $pos ] || uv run python collect_positions.py -m ${NAME}_edax -n 1000000 -o $pos --exclude $LABELS 2>&1 \
      | grep -E "Saved|Error|Traceback" | tee -a $LOG
  [ -f $lab ] || uv run python label_positions.py --positions $pos -n 1000000 --depth 12 --every-move -w 12 -o $lab 2>&1 \
      | grep -E "Saved|Error|Traceback" | tee -a $LOG
  [ -f $lab ] || { say "stage 4 labeling failed"; exit 1; }
  edax_stage ${NAME}_dagger $LABELS $lab
  have ${NAME}_dagger || { say "stage 4 training failed"; exit 1; }
  evaluate ${NAME}_dagger
fi
say "course $NAME done"
