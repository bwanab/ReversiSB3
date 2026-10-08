#!/bin/zsh
# Self-teacher round: the network learns from its own search (CLAUDE.md step 28):
#   1. MCTS self-play with BASE: every searched position with its visit distribution, root value and
#      game outcome (selfplay_mcts.py)
#   2. fine-tune BASE on those targets mixed with a replay of Edax labels (mcts_train.py)
#   3. evaluate: search regret on the held-out positions (bench_search.py, OUT and BASE), MCTS 400 vs
#      Edax 10/12/14 and Egaroucid 10/12 (balanced and named openings), network-only head-to-head
#
# Usage: [N=positions] [W=workers] [GAMES=per worker] [EPOCHS=2] [LR=2e-5] [Z=0.5] [REPLAY=1000000] \
#        ./self_teacher_round.sh OUT BASE
#   e.g. N=400000 ./self_teacher_round.sh r256x12_st1 r256x12_mid1
# Steps whose output exists are skipped. Log: selfteach_OUT.log. Needs the Edax server for step 3.

set -u
OUT=$1; BASE=$2
N=${N:-400000}; W=${W:-1}; GAMES=${GAMES:-512}; EPOCHS=${EPOCHS:-2}; LR=${LR:-2e-5}; Z=${Z:-0.5}
REPLAY=${REPLAY:-1000000}
LOG=selfteach_${OUT}.log
SP=selfplay_${OUT}.npz
MCTS="--mcts-sims 400 --solve-empties 18 --leaf-solve-empties 16"
cd "$(dirname "$0")"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

say "self-teacher round $OUT: $N MCTS self-play positions from $BASE ($W worker(s) x $GAMES games), fine-tune $EPOCHS epochs lr $LR z-weight $Z, $REPLAY Edax replay"
if [ ! -f $SP ]; then
  say "1. self-play"
  uv run python selfplay_mcts.py -m $BASE -n $N -o $SP -w $W --games $GAMES 2>&1 | grep -E "Saved|Error|Traceback|failed" | tee -a $LOG
  [ -f $SP ] || { say "self-play failed"; exit 1; }
fi
if [ ! -f models/${OUT}_CNN_test.zip ]; then
  say "2. fine-tune"
  uv run python mcts_train.py --mcts $SP --replay labels_d*_all.npz --replay-n $REPLAY --base $BASE --model $OUT \
      --epochs $EPOCHS -lr $LR --z-weight $Z 2>&1 | grep -E "positions|start:|epoch|saved|Error|Traceback" | tee -a $LOG
  [ -f models/${OUT}_CNN_test.zip ] || { say "training failed"; exit 1; }
fi
say "3. evaluate"
uv run python bench_search.py -m ${BASE}_CNN_test --labels bench/heldout_d14.npz --config top --config mcts:400:1.0 2>&1 \
    | grep -E "^(top|mcts)" | while read -r line; do say "bench $BASE | $line"; done
uv run python bench_search.py -m ${OUT}_CNN_test --labels bench/heldout_d14.npz --config top --config mcts:400:1.0 2>&1 \
    | grep -E "^(top|mcts)" | while read -r line; do say "bench $OUT | $line"; done
pgrep -f "edax_server.py" >/dev/null || [ -S /tmp/edax_server.sock ] || { say "Edax server not running; skipping games"; exit 0; }
for spec in "balanced:-e 200 --start-positions balanced_openings.npy" "named:-e 150 --start-positions opening_positions.npy"; do
  op=${spec%%:*}; args=${spec#*:}
  uv run python eval_batch.py -m ${OUT}_CNN_test -p 10,12,14 ${=args} ${=MCTS} 2>&1 | grep "Black wins" \
      | while read -r line; do say "eval $op | $line"; done
  uv run python eval_batch.py -m ${OUT}_CNN_test -o egaroucid -p 10,12 ${=args} ${=MCTS} 2>&1 | grep "Black wins" \
      | while read -r line; do say "eval $op | $line"; done
done
h1=$(uv run python sb-play.py -m ${OUT}_CNN_test -r models/${BASE}_CNN_test -o Model -e 400 --start-positions balanced_openings.npy 2>&1 | grep "Model:")
h2=$(uv run python sb-play.py -m ${OUT}_CNN_test -r models/${BASE}_CNN_test -o Model -e 300 --start-positions opening_positions.npy 2>&1 | grep "Model:")
say "h2h (network only) $OUT vs $BASE | balanced: ${h1#*score } | named: ${h2#*score }"
say "self-teacher round $OUT done"
